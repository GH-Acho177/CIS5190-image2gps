import os
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from geopy.distance import geodesic
from torch.optim.lr_scheduler import StepLR

from model import Model

# CONFIG
CSV_PATH = "data/metadata.csv"
SAVE_PATH = "model.pt"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
STEP_SIZE = 5
GAMMA = 0.1
VAL_SPLIT = 0.2
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# DATASET
class GPSDataset(Dataset):
    """
    Loads image_path, Latitude, Longitude from CSV.
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])

        from PIL import Image
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        target = torch.tensor([lat, lon], dtype=torch.float32)
        return img, target

# TRANSFORMS
def build_transforms():
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    return train_transform, val_transform

# DATA LOADERS
def build_dataloaders(csv_path, batch_size, val_split):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = ["image_path", "Latitude", "Longitude"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    # coordinate sanity
    df = df[df["Latitude"].between(-90, 90)]
    df = df[df["Longitude"].between(-180, 180)]
    df = df.reset_index(drop=True)

    # train/val split
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    split = int(len(df) * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")

    # stats (from train only)
    lat_mean = float(train_df["Latitude"].mean())
    lon_mean = float(train_df["Longitude"].mean())
    lat_std = float(train_df["Latitude"].std() or 1.0)
    lon_std = float(train_df["Longitude"].std() or 1.0)

    train_transform, val_transform = build_transforms()

    train_ds = GPSDataset(train_df, transform=train_transform)
    val_ds = GPSDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    stats = (lat_mean, lon_mean, lat_std, lon_std)
    return train_loader, val_loader, stats

# MODEL
def build_model(device, stats):
    """
    Build the leaderboard-wrapped model and set normalization stats.
    We train the inner backbone (wrapper.model) to predict normalized coords.
    """
    lat_mean, lon_mean, lat_std, lon_std = stats

    wrapper = Model()
    wrapper.to(device)
    wrapper.train()

    # Fill normalization buffers so they are saved in state_dict
    with torch.no_grad():
        wrapper.lat_mean.fill_(lat_mean)
        wrapper.lon_mean.fill_(lon_mean)
        wrapper.lat_std.fill_(lat_std)
        wrapper.lon_std.fill_(lon_std)

    backbone = wrapper.model  # inner ResNet18

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(backbone.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    return wrapper, backbone, criterion, optimizer, scheduler

# TRAIN + VALIDATE
def train_and_validate(
    wrapper,
    backbone,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    stats,
    device,
    num_epochs,
):
    lat_mean, lon_mean, lat_std, lon_std = stats
    mean_t = torch.tensor([lat_mean, lon_mean], dtype=torch.float32, device=device)
    std_t = torch.tensor([lat_std, lon_std], dtype=torch.float32, device=device)

    best_rmse = float("inf")

    print("\n===== START TRAINING =====\n")

    for epoch in range(1, num_epochs + 1):

        # ---------------------- TRAIN ----------------------
        wrapper.train()
        epoch_loss = 0.0

        for images, gps in train_loader:
            images = images.to(device)
            gps = gps.to(device)   # degrees

            # normalize labels
            gps_norm = (gps - mean_t) / std_t

            optimizer.zero_grad()
            preds = backbone(images)         # normalized outputs
            loss = criterion(preds, gps_norm)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        avg_train = epoch_loss / max(1, len(train_loader))
        print(f"[Epoch {epoch}/{num_epochs}] Train MSE (norm²): {avg_train:.6f}")

        # ---------------------- VAL ----------------------
        wrapper.eval()
        val_sq_dist = 0.0
        baseline_sq_dist = 0.0
        total = 0

        with torch.no_grad():
            for images, gps in val_loader:
                images = images.to(device)
                gps = gps.to(device)              # degrees
                gps_np = gps.cpu().numpy()

                preds_norm = backbone(images)     # normalized
                preds_deg = preds_norm * std_t + mean_t  # de-normalize to degrees
                preds_np = preds_deg.cpu().numpy()

                for p, a in zip(preds_np, gps_np):
                    d_val = geodesic((a[0], a[1]), (p[0], p[1])).meters
                    d_base = geodesic((a[0], a[1]), (lat_mean, lon_mean)).meters

                    val_sq_dist += d_val**2
                    baseline_sq_dist += d_base**2
                    total += 1

        if total > 0:
            val_mse = val_sq_dist / total
            base_mse = baseline_sq_dist / total
            val_rmse = math.sqrt(val_mse)
            base_rmse = math.sqrt(base_mse)
        else:
            val_rmse = float("inf")
            base_rmse = float("inf")

        print(f"[Epoch {epoch}] Val RMSE (m): {val_rmse:.2f}, Baseline RMSE (m): {base_rmse:.2f}")

        # save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(wrapper.state_dict(), SAVE_PATH)
            print(f" → Saved best model: RMSE = {best_rmse:.2f} m\n")
        else:
            print(" (No improvement)\n")

    print("===== TRAINING FINISHED =====")
    print(f"Best validation RMSE: {best_rmse:.2f} m")

def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, stats = build_dataloaders(
        CSV_PATH,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
    )

    wrapper, backbone, criterion, optimizer, scheduler = build_model(device, stats)

    train_and_validate(
        wrapper,
        backbone,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        stats,
        device,
        num_epochs=NUM_EPOCHS,
    )

if __name__ == "__main__":
    main()
