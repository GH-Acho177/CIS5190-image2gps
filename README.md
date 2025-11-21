# CIS5190 – Image2GPS Project

This repository contains a complete pipeline for predicting **GPS coordinates (latitude, longitude)** directly from images.  

---

## Quick Start

### 1. Add your dataset

Place all raw `.jpg`, `.jpeg`, or `.png` files inside: data/photos/

Make sure images contain EXIF GPS information.

---

### 2. Generate metadata

Run the preprocessing script:

    python preprocess_metadata.py


This will create:

    data/metadata.csv

with the following columns:

- image_path  
- Latitude  
- Longitude  

Images missing GPS metadata or failing the blur/brightness checks are removed.

---

### 3. Train the model

Train the ResNet18 model using:



    python train.py


After training, the model weights will be saved to:



    model.pt


---

### 4. Evaluate

Use the Project A evaluator with:

    python eval_project_a.py
    --model model.py
    --preprocess preprocess.py
    --csv data/metadata.csv
    --weights model.pt


This prints:

- number of examples  
- average inference time  
- MAE / RMSE in degrees  
- RMSE in meters  
- average distance error in meters  

---

## Model Overview

The model is a modified ResNet18:



ResNet18
└── fc → Linear(512 → 2) # outputs (latitude, longitude)


Key properties:

- No pretrained weights (weights=None)  
- MSE loss on normalized GPS coordinates  
- Standard image augmentations during training  
- Outputs real lat/lon values after de-normalization  

---

## File Descriptions

| File | Purpose                                                           |
|------|-------------------------------------------------------------------|
| preprocess_metadata.py | Extract EXIF GPS, apply blur filtering, and generate metadata.csv |
| preprocess.py | Resize and normalize images for the evaluator                     |
| model.py | Leaderboard-compatible inference model                            |
| train.py | Full training pipeline including normalization and validation     |
| eval_project_a.py | Evaluator                                                         |
| metadata.csv | Generated metadata used for both training and evaluation          |

---

## License

MIT License.