import os
import pandas as pd
from torchvision import transforms as T

import torch
from typing import Any, List, Tuple

# Image settings (matches standard ResNet expectations)
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _infer_column_name(candidates, df_columns):
    """
    Find the first column name in df_columns that matches one of candidates.
    Raises a clear error if nothing matches.
    """
    for c in candidates:
        if c in df_columns:
            return c
    raise ValueError(f"None of the candidate columns {candidates} found in CSV. "
                     f"Available columns: {list(df_columns)}")

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

    img_col = _infer_column_name(
        ["image_path", "filepath", "image", "path", "file_name"],
        df.columns,
    )
    lat_col = _infer_column_name(
        ["Latitude", "latitude", "lat"],
        df.columns,
    )
    lon_col = _infer_column_name(
        ["Longitude", "longitude", "lon"],
        df.columns,
    )

    images = []
    targets = []

    base_dir = os.path.dirname(os.path.abspath(path))

    for idx, row in df.iterrows():
        img_path = row[img_col]

        # Support both absolute and relative paths
        if not os.path.isabs(img_path):
            img_path = os.path.join(base_dir, img_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image file not found for row {idx}: {img_path}"
            )

        # # Load image and ensure RGB
        # with Image.open(img_path) as img:
        #     if img.mode != "RGB":
        #         img = img.convert("RGB")
        #     img_tensor = _transform(img)

        lat = float(row[lat_col])
        lon = float(row[lon_col])

        images.append(img_path)
        targets.append([lat, lon])

    # Stack into tensors
    #X = torch.stack(images, dim=0)  # (N, 3, 224, 224)
    y = torch.tensor(targets, dtype=torch.float32)  # (N, 2)

    X = images
    return X, y