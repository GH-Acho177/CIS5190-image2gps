import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from geopy.distance import geodesic
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

DATASET_ID = "Acho177/CIS5190"

BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
STEP_SIZE = 5
GAMMA = 0.1
SAVE_PATH = "resnet_gps_regressor.pth"

class GPSImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, lat_mean=None, lat_std=None, lon_mean=None, lon_std=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

        # Compute mean and std from the dataframe if not provided
        self.latitude_mean = lat_mean if lat_mean is not None else np.mean(np.array(self.hf_dataset['Latitude']))
        self.latitude_std = lat_std if lat_std is not None else np.std(np.array(self.hf_dataset['Latitude']))
        self.longitude_mean = lon_mean if lon_mean is not None else np.mean(np.array(self.hf_dataset['Longitude']))
        self.longitude_std = lon_std if lon_std is not None else np.std(np.array(self.hf_dataset['Longitude']))

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Extract data
        example = self.hf_dataset[idx]

        # Load and process the image
        image = example['image']
        latitude = example['Latitude']
        longitude = example['Longitude']
        # image = image.rotate(-90, expand=True)
        if self.transform:
            image = self.transform(image)

        # Normalize GPS coordinates
        latitude = (latitude - self.latitude_mean) / self.latitude_std
        longitude = (longitude - self.longitude_mean) / self.longitude_std
        gps_coords = torch.tensor([latitude, longitude], dtype=torch.float32)

        return image, gps_coords

def build_transforms():
    """
    Train-time transforms with augmentation + inference-time transforms.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Random crop and resize to 224x224
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(degrees=15),  # Random rotation between -15 and 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Validation / test: no augmentation
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, inference_transform

def build_dataloaders():
    """
    Loads HF dataset, wraps as GPSImageDataset, and returns:
        train_loader, val_loader, (lat_mean, lat_std, lon_mean, lon_std)
    """
    print(f"Loading dataset from Hugging Face: {DATASET_ID}")
    dataset_train = load_dataset(DATASET_ID, split="train")
    dataset_val = load_dataset(DATASET_ID, split="validation")
    dataset_test = load_dataset(DATASET_ID, split="test")
    #dataset_test = load_dataset("gydou/released_img", split="train")

    print("Train split:", dataset_train)
    print("Val split:", dataset_val)
    print("Test split:", dataset_test)

    train_transform, inference_transform = build_transforms()

    # Training dataset computes its own statistics
    train_dataset = GPSImageDataset(
        hf_dataset=dataset_train,
        transform=train_transform,
    )

    # Grab stats from train dataset
    lat_mean = train_dataset.latitude_mean
    lat_std = train_dataset.latitude_std
    lon_mean = train_dataset.longitude_mean
    lon_std = train_dataset.longitude_std

    # Validation dataset uses train stats
    val_dataset = GPSImageDataset(
        hf_dataset=dataset_val,
        transform=inference_transform,
        lat_mean=lat_mean,
        lat_std=lat_std,
        lon_mean=lon_mean,
        lon_std=lon_std
    )

    test_dataset = GPSImageDataset(
        hf_dataset=dataset_test,
        transform=inference_transform,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Quick sanity check
    for images, gps_coords in train_dataloader:
        print(images.size(), gps_coords.size())
        break

    return train_dataloader, val_dataloader, test_dataloader, (lat_mean, lat_std, lon_mean, lon_std)

def build_model(device):
    """
    ResNet18 → 2D regression head (lat, lon).
    """
    # Load the pre-trained ResNet18 model
    resnet = models.resnet18(pretrained=False)

    # Modify the last fully connected layer to output 2 values (latitude and longitude)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, 2)  # Latitude and Longitude

    # Not freeze pre-trained weights (excluding the final layer)
    for param in resnet.parameters():
        param.requires_grad = True

    for param in resnet.fc.parameters():
        param.requires_grad = True

    resnet = resnet.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    # If not fine tuning
    # optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.001)
    # If fine tuning
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)

    # Add a learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    return resnet, criterion, optimizer, scheduler

def train_and_validate(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    scheduler,
    stats,
    device,
    num_epochs=NUM_EPOCHS,
):
    print("Start Training!")

    lat_mean, lat_std, lon_mean, lon_std = stats

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, gps_coords in train_dataloader:
            images, gps_coords = images.to(device), gps_coords.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, gps_coords)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        baseline_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, gps_coords in val_dataloader:
                images = images.to(device)
                gps_coords = gps_coords.to(device)

                batch_size = gps_coords.size(0)
                total_samples += batch_size

                # Model predictions
                outputs = model(images)

                # Denormalize predictions and actual GPS coordinates
                preds_denorm = outputs.cpu().numpy() * np.array([lat_std, lon_std]) + np.array([lat_mean, lon_mean])
                actuals_denorm = gps_coords.cpu().numpy() * np.array([lat_std, lon_std]) + np.array(
                    [lat_mean, lon_mean])

                # Compute geodesic distances between predictions and actuals
                for pred, actual in zip(preds_denorm, actuals_denorm):
                    distance = geodesic((actual[0], actual[1]), (pred[0], pred[1])).meters
                    val_loss += distance ** 2  # Squared distance

                # Baseline predictions
                baseline_preds = np.array([lat_mean, lon_mean])

                # Compute geodesic distances between baseline preds and actuals
                for actual in actuals_denorm:
                    distance = geodesic((actual[0], actual[1]), (baseline_preds[0], baseline_preds[1])).meters
                    baseline_loss += distance ** 2  # Squared distance

        # Compute average losses
        val_loss /= total_samples
        baseline_loss /= total_samples

        # Compute RMSE
        val_rmse = np.sqrt(val_loss)
        baseline_rmse = np.sqrt(baseline_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss (meters^2): {val_loss:.2f}, Baseline Loss (meters^2): {baseline_loss:.2f}")
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Validation RMSE (meters): {val_rmse:.2f}, Baseline RMSE (meters): {baseline_rmse:.2f}")

    print("Training complete!\n")


def test(model,
    test_dataloader,
    stats,
    device):

    print("Start Testing!")

    all_preds = []
    all_actuals = []
    distances = []
    model.eval()

    lat_mean, lat_std, lon_mean, lon_std = stats

    with torch.no_grad():
        for images, gps_coords in test_dataloader:
            images, gps_coords = images.to(device), gps_coords.to(device)

            outputs = model(images)

            # Denormalize predictions and actual values
            preds = outputs.cpu() * torch.tensor([lat_std, lon_std]) + torch.tensor([lat_mean, lon_mean])
            actuals = gps_coords.cpu() * torch.tensor([lat_std, lon_std]) + torch.tensor([lat_mean, lon_mean])

            all_preds.append(preds)
            all_actuals.append(actuals)

            for pred, actual in zip(preds, actuals):
                distance = geodesic((actual[0], actual[1]), (pred[0], pred[1])).meters
                distances.append(distance)

    # Concatenate all batches
    all_preds = torch.cat(all_preds).numpy()
    all_actuals = torch.cat(all_actuals).numpy()

    # Compute error metrics
    mae = mean_absolute_error(all_actuals, all_preds)
    rmse = mean_squared_error(all_actuals, all_preds)#, squared=False)

    print(f'Mean Absolute Error: {mae}')
    print(f'Root Mean Squared Error: {rmse}')

    avg_distance = sum(distances) / len(distances)
    print(f"Avg Distance: {avg_distance}")
    print("Testing Complete!")

    return all_preds, all_actuals

def plot_predictions(all_preds, all_actuals, stats):
    """
    visualize predicted vs actual GPS points.
    """

    lat_mean, lat_std, lon_mean, lon_std = stats

    all_preds_denorm = all_preds * np.array([lat_std, lon_std]) + np.array([lat_mean, lon_mean])
    all_actuals_denorm = all_actuals * np.array([lat_std, lon_std]) + np.array([lat_mean, lon_mean])

    plt.figure(figsize=(10, 5))

    # Plot actual points
    plt.scatter(all_actuals_denorm[:, 1], all_actuals_denorm[:, 0], label='Actual', color='blue', alpha=0.6)

    # Plot predicted points
    plt.scatter(all_preds_denorm[:, 1], all_preds_denorm[:, 0], label='Predicted', color='red', alpha=0.6)

    # Draw lines connecting actual and predicted points
    for i in range(len(all_actuals_denorm)):
        plt.plot(
            [all_actuals_denorm[i, 1], all_preds_denorm[i, 1]],
            [all_actuals_denorm[i, 0], all_preds_denorm[i, 0]],
            color='gray', linewidth=0.5
        )

    plt.legend()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Actual vs. Predicted GPS Coordinates with Error Lines')
    plt.show()

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_dataloader, val_dataloader, test_dataloader, stats = build_dataloaders()

    # Model
    model, criterion, optimizer, scheduler = build_model(device)

    # Train + validate
    train_and_validate(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        scheduler,
        stats,
        device,
        num_epochs=NUM_EPOCHS,
    )

    all_preds, all_actuals = test(model, test_dataloader, stats, device)

    # Save model weights
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model weights saved to: {SAVE_PATH}")

    # visualization
    try:
        plot_predictions(all_preds, all_actuals, stats)
    except Exception as e:
        print(f"Plotting failed (non-fatal): {e}")

if __name__ == "__main__":
    main()