import numpy as np
import pandas as pd
import torch
from typing import Any, List, Tuple
import os, glob
from PIL import Image, ImageStat
import piexif
import torchvision.transforms as T

def dms_to_dd(dms, ref):
    """Convert degree/minute/second to decimal degrees."""
    deg = dms[0][0] / dms[0][1]
    minute = dms[1][0] / dms[1][1]
    sec = dms[2][0] / dms[2][1]
    dd = deg + minute/60 + sec/3600
    if ref in (b'S', b'W', 'S', 'W'):
        dd = -dd
    return dd

def extract_gps(img_path):
    """Extract (lat, lon) from EXIF. Return None if missing."""
    try:
        img = Image.open(img_path)
        exif_data = img.info.get("exif", b"")
        if not exif_data:
            return None

        exif_dict = piexif.load(exif_data)
        gps = exif_dict.get("GPS", {})

        lat = gps.get(2)
        lon = gps.get(4)
        lat_ref = gps.get(1)
        lon_ref = gps.get(3)

        if lat and lon and lat_ref and lon_ref:
            return (
                dms_to_dd(lat, lat_ref),
                dms_to_dd(lon, lon_ref)
            )
    except:
        pass
    return None

def is_vague(img, blur_threshold=20, brightness_range=(10, 245), min_edge_ratio=0.002):
    try:
        # Convert to grayscale
        gray = img.convert("L")
        stat = ImageStat.Stat(gray)

        # (1) brightness check
        if not (brightness_range[0] <= stat.mean[0] <= brightness_range[1]):
            return True

        arr = np.array(gray, dtype=np.float32)

        # (2) blur check: Laplacian variance (via gradient)
        lap_var = np.var(np.gradient(arr))
        if lap_var < blur_threshold:
            return True

        # (3) edge density check
        edges = np.abs(np.gradient(arr)).sum(axis=0)
        edge_ratio = (edges > 50).mean()
        if edge_ratio < min_edge_ratio:
            return True

        return False
    except:
        return True

def create_metadata(img_root, output_csv="metadata.csv"):
    """
    Scan folder, extract GPS, filter vague images, and save metadata.csv.
    """
    print(f"Scanning images under: {img_root}")

    rows = []
    image_paths = glob.glob(os.path.join(img_root, "**", "*.*"), recursive=True)

    for p in image_paths:
        if not p.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Extract GPS
        gps = extract_gps(p)
        if gps is None:
            continue
        lat, lon = gps

        # Load image
        try:
            img = Image.open(p).convert("RGB")
        except:
            continue

        # Quality filter
        if is_vague(img):
            continue

        # Add row
        rows.append({
            "image_path": os.path.basename(p),
            "Latitude": lat,
            "Longitude": lon
        })

    # Build dataframe
    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid GPS + non-vague images found!")
        return

    # Filter valid coordinate ranges
    df = df[df["Latitude"].between(-90, 90)]
    df = df[df["Longitude"].between(-180, 180)]
    df = df.reset_index(drop=True)

    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved filtered metadata → {output_csv}")
    print(df.head())

if __name__ == "__main__":
    create_metadata(
        img_root="extracted_images",
        output_csv="extracted_images/metadata.csv"
    )