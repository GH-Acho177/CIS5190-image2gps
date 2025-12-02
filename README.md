# CIS5190 – Image2GPS Project

This repository contains a complete pipeline for predicting **GPS coordinates (latitude, longitude)** directly from images.  

---

## Quick Start

### 1. Add your dataset

Place all raw `.jpg`, `.jpeg`, or `.png` files inside: data/

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



    python train.py --train_csv clean/clean_metadata.csv --epochs 20 --val_csv test/metadata.csv


After training, the model weights will be saved to:



    /Model/


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

## File Descriptions

| File | Purpose                                                           |
|------|-------------------------------------------------------------------|
| preprocess_metadata.py | Extract EXIF GPS, apply blur filtering, and generate metadata.csv |
| preprocess.py | Resize and normalize images for the evaluator                     |
| model.py | Leaderboard-compatible inference model                            |
| train.py | Full training pipeline including normalization and validation     |
| resnet.py | ResNet Implementation (ResNet50, ResNet101)|
| eval_project_a.py | Evaluator                                                         |
| metadata.csv | Generated metadata used for both training and evaluation          |

---

## License

MIT License.
