import pandas as pd
import torch
from typing import Any, List, Tuple
from PIL import Image
import torchvision.transforms as T

def prepare_data(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Template preprocessing for leaderboard.

    Requirements:
    - Must read the provided data path at `path`.
    - Must return a tuple (X, y):
        X: a list of model-ready inputs (these must match what your model expects in predict(...))
        y: a list of ground-truth labels aligned with X (same length)

    Notes:
    - The evaluation backend will call this function with the shared validation data
    - Ensure the output format (types, shapes) of X matches your model's predict(...) inputs.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Check required columns
    required = ["image_path", "Latitude", "Longitude"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Filter valid coordinate ranges
    df = df[df["Latitude"].between(-90, 90)]
    df = df[df["Longitude"].between(-180, 180)]
    df = df.reset_index(drop=True)

    # Define model preprocessing
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    X = []
    y = []

    # Load each image
    for _, row in df.iterrows():
        img_path = row["image_path"]

        try:
            img = Image.open(img_path).convert("RGB")
            tensor_img = transform(img)

            X.append(tensor_img)
            y.append([float(row["Latitude"]), float(row["Longitude"])])

        except Exception as e:
            print(f"[WARN] Failed to load image: {img_path} ({e})")
            continue

    return X, y