import math
import os
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from preprocess import prepare_data
from model import Model
import matplotlib.pyplot as plt

from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CUSTOMENAME = "smooth_haversine_loss_mix"

_transform = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop(224),                # consistent framing
    T.ColorJitter(0.1,0.1,0.1),       # mild lighting noise
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class Img2GPSDataset(Dataset):
    """
    Wraps tensors returned by prepare_data into a standard PyTorch Dataset.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_path = self.X[idx]
        img = Image.open(img_path).convert("RGB")
        img = _transform(img)
        return img, self.y[idx]


def plot_gt_only(gt):
    """
    Plot only the ground-truth GPS points.
    """
    gt = gt.cpu().numpy()  # ensure numpy

    plt.figure(figsize=(5, 5))
    plt.scatter(gt[:, 1], gt[:, 0], color="blue", alpha=0.7, s=25, label="GT")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.close()

def plot_predictions_vs_gt(gt, pred, epoch):
    """
    gt: Tensor (N, 2)
    pred: Tensor (N, 2)
    Saves a scatter plot of predicted vs ground-truth coordinates.
    """
    gt = gt.numpy()
    pred = pred.numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(gt[:, 1], gt[:, 0], label="Ground Truth", alpha=0.6)  # lon vs lat
    plt.scatter(pred[:, 1], pred[:, 0], label="Predicted", alpha=0.6)

    for i in range(gt.shape[0]):
        gt_lat, gt_lon = gt[i]
        pred_lat, pred_lon = pred[i]

        # Draw a thin line connecting GT → Pred
        plt.plot(
            [gt_lon, pred_lon],
            [gt_lat, pred_lat],
            color="gray",
            linewidth=0.6,
            alpha=0.7,
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.title(f"Epoch {epoch+1}: Pred vs GT")
    plt.savefig(f"plots/data_epoch_{epoch+1}_{CUSTOMENAME}.png", dpi=200)
    plt.close()

def haversine_loss(pred, target, lat_center, lon_center, scale):
    """
    pred, target: (B, 2) normalized lat/lon tensors
    convert to degrees, compute Haversine distance per sample, return mean loss
    """
    # Recover degrees
    pred_lat = lat_center + pred[:, 0] * scale
    pred_lon = lon_center + pred[:, 1] * scale
    pred_lat = pred_lat.clamp(min=lat_center - 0.01, max=lat_center + 0.01)
    pred_lon = pred_lon.clamp(min=lon_center - 0.01, max=lon_center + 0.01)

    gt_lat   = lat_center + target[:, 0] * scale
    gt_lon   = lon_center + target[:, 1] * scale

    # Convert to radians
    lat1 = torch.deg2rad(pred_lat)
    lon1 = torch.deg2rad(pred_lon)
    lat2 = torch.deg2rad(gt_lat)
    lon2 = torch.deg2rad(gt_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    a = a.clamp(min=1e-8, max=1 - 1e-8)
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    R = 6371000.0  # meters
    dist = R * c

    return dist.mean()

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for X, y in dataloader:
        X = X.to(model.device)
        y = y.to(model.device)

        optimizer.zero_grad()

        preds = model.backbone(X)

        base = haversine_loss(preds, y, model.lat_center_buf.item(), model.lon_center_buf.item(),
                              model.scale_buf.item())
        smooth = 0.001 * torch.nn.functional.l1_loss(preds, y)
        loss = base + smooth

        # loss = criterion(preds, y)
        # loss = haversine_loss(preds, y, model.lat_center_buf.item(), model.lon_center_buf.item(),
        #                     model.scale_buf.item())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes distance between two GPS points in meters.
    Input: lat/lon in degrees.
    """
    R = 6371000  # Earth radius in meters

    # Convert to radians
    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) *
         math.sin(dlon / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def validate(model, dataloader, criterion, lat_center, lon_center, scale):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(model.device)
            y = y.to(model.device)

            preds = model.backbone(X)

            all_preds.append(preds.cpu())
            y_deg = y.clone()
            y_deg[:, 0] = lat_center + y[:, 0] * scale
            y_deg[:, 1] = lon_center + y[:, 1] * scale
            all_targets.append(y_deg.cpu())

            # loss = criterion(preds, y)
            #loss = haversine_loss(preds, y, lat_center, lon_center, scale)
            base = haversine_loss(preds, y, model.lat_center_buf.item(), model.lon_center_buf.item(),
                                  model.scale_buf.item())
            smooth = 0.001 * torch.nn.functional.l1_loss(preds, y)
            loss = base + smooth


            total_loss += loss.item() * X.size(0)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    preds_deg = all_preds.clone()
    preds_deg[:, 0] = lat_center + all_preds[:, 0] * scale
    preds_deg[:, 1] = lon_center + all_preds[:, 1] * scale

    dists = []
    for i in range(all_preds.size(0)):
        lat_pred, lon_pred = preds_deg[i].tolist()
        lat_gt, lon_gt = all_targets[i].tolist()

        d = haversine_distance(lat_pred, lon_pred, lat_gt, lon_gt)
        dists.append(d)

    avg_dist = sum(dists) / len(dists)

    return total_loss / len(dataloader.dataset), preds_deg, all_targets, avg_dist


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    X_train, y_train = prepare_data(args.train_csv)
    lat_center = y_train[:, 0].mean().item()
    lon_center = y_train[:, 1].mean().item()

    lat_std = y_train[:, 0].std().item()
    lon_std = y_train[:, 1].std().item()

    scale = ((lat_std + lon_std) / 2)

    y_train_norm = y_train.clone()
    y_train_norm[:, 0] = (y_train[:, 0] - lat_center) / scale
    y_train_norm[:, 1] = (y_train[:, 1] - lon_center) / scale

    dataset = Img2GPSDataset(X_train, y_train_norm)

    # Optional validation
    if args.val_csv:
        X_val, y_val = prepare_data(args.val_csv)

        y_val_norm = y_val.clone()
        y_val_norm[:, 0] = (y_val[:, 0] - lat_center) / scale
        y_val_norm[:, 1] = (y_val[:, 1] - lon_center) / scale

        val_dataset = Img2GPSDataset(X_val, y_val_norm)
    else:
        val_dataset = None

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

    # Build model
    model = Model()
    model = model.to(device)

    model.lat_center_buf[0] = lat_center
    model.lon_center_buf[0] = lon_center
    model.scale_buf[0] = scale

    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("plots", exist_ok=True)

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

        if val_dataset:
            val_loss, val_preds, val_gts, val_dist = validate(
                model, val_loader, criterion,
                lat_center, lon_center, scale
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss (norm): {train_loss:.6f} | "
                f"Val Loss (norm): {val_loss:.6f} | "
                f"Val Avg Dist (m): {val_dist:.2f}"
            )

            plot_gt_only(val_gts)
            plot_predictions_vs_gt(val_gts, val_preds, epoch)
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}")

        # Create a fresh model instance for saving
        model_to_save = Model()
        model_to_save.load_state_dict(model.state_dict())

        # Copy buffer values
        model_to_save.lat_center_buf[0] = lat_center
        model_to_save.lon_center_buf[0] = lon_center
        model_to_save.scale_buf[0] = scale

        # Save
        model_to_save = model_to_save.to("cpu")

        torch.save(model_to_save.state_dict(), f"Model/model_epoch_{epoch+1}_{CUSTOMENAME}.pt")
        print(f"Saved model weights t Model/model_{epoch+1}_{CUSTOMENAME}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True, help="Path to training metadata.csv")
    parser.add_argument("--val_csv", type=str, default=None, help="Optional validation metadata.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)

    args = parser.parse_args()
    main(args)
