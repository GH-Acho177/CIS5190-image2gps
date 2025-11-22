import os
import pandas as pd
import torch
from typing import Tuple
import cv2

# normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

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
    def find_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        raise KeyError(f"Missing required column in: {cands}")

    image_col = find_col(["image_path", "filepath", "image", "path", "file_name"])
    lat_col   = find_col(["Latitude", "latitude", "lat"])
    lon_col   = find_col(["Longitude", "longitude", "lon"])

    # Filter valid coordinate ranges
    df = df[df[lat_col].between(-90, 90)]
    df = df[df[lon_col].between(-180, 180)]
    df = df.reset_index(drop=True)

    X_list = []
    y_list = []

    base_dir = os.path.dirname(path)

    # Load each image
    for _, row in df.iterrows():

        img_path = row[image_col]
        img_path = os.path.join(base_dir, img_path)
        img_path = os.path.normpath(img_path)

        # if not os.path.exists(img_path):
        #     print(f"[WARN] Missing: {img_path}")
        #     continue

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read: {img_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        t = torch.from_numpy(rgb).float() / 255.0
        t = t.permute(2, 0, 1)

        t = (t - MEAN) / STD

        X_list.append(t)
        y_list.append([float(row[lat_col]), float(row[lon_col])])

    # if len(X_list) == 0:
    #     raise RuntimeError("No valid images found!")

    X = torch.stack(X_list, dim=0)  # (N, 3, 224, 224)
    y = torch.tensor(y_list, dtype=torch.float32)  # (N, 2)

    return X, y
