# 🪖 EdgeHelmet-AI - Real-Time Helmet Detection System
A real-time helmet detection system built with **YOLOv8** for safety compliance monitoring. Detects whether individuals are wearing helmets in images, videos, and live webcam feeds.

---
## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Dataset Preparation](#1-dataset-preparation)
  - [2. Training](#2-training)
  - [3. Image Detection](#3-image-detection)
  - [4. Video Detection](#4-video-detection)
  - [5. Live Webcam Detection](#5-live-webcam-detection)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Training Configuration](#training-configuration)
- [License](#license)
---

## Overview
EdgeHelmet-AI is designed to monitor safety helmet compliance in real-time scenarios such as:
- 🏗️ **Construction sites** — detecting hard hats on workers
- 🏍️ **Road safety** — detecting motorcycle/bike helmets
- 🏭 **Industrial environments** — safety gear monitoring
The system uses **YOLOv8 Nano** for fast inference on edge devices with limited GPU resources (~4GB VRAM).
---

## Features
- ✅ **2-Class Detection** — `helmet` (wearing) and `no_helmet` (bare head)
- ✅ **Multiple Input Modes** — Image, video file, and live webcam
- ✅ **Real-time Alerts** — Color-coded bounding boxes (🟢 green = helmet, 🔴 red = no helmet)
- ✅ **Live Dashboard** — Frame counter, detection count, and safety status overlay
- ✅ **Screenshot Capture** — Save frames during live detection (press `s`)
- ✅ **Auto-save Video** — Processed video output with detections overlaid
- ✅ **Edge Optimized** — Runs on YOLOv8n for fast inference on limited hardware
---

## Dataset
The model is trained on a **combined dataset** from two sources:
| Dataset | Images | Source Labels | Mapped To |
|---------|--------|---------------|-----------|
| **BikesHelmets** | 766 | `With Helmet`, `Without Helmet` | `helmet` (0), `no_helmet` (1) |
| **Hard Hat Workers** | ~5,000 | `helmet`, `head`, `person` | `helmet` (0), `no_helmet` (1), *skipped* |
### Label Unification
Both datasets use Pascal VOC XML annotations. The `convert_dataset.py` script unifies different label names into a common 2-class YOLO format:
```
"With Helmet" / "helmet"    → Class 0 (helmet)
"Without Helmet" / "head"   → Class 1 (no_helmet)
"person"                    → Skipped (full-body bbox, not relevant)
```

### Dataset Statistics
| Split | Images | Labels |
|-------|--------|--------|
| **Train** | 4,734 | 4,734 |
| **Validation** | 1,278 | 1,278 |
| **Total** | **6,012** | **6,012** |
### Annotation Distribution
| Label | Object Count |
|-------|-------------|
| `helmet` (With Helmet + helmet) | 19,928 |
| `no_helmet` (Without Helmet + head) | 6,274 |
| `person` (skipped) | 751 |
| **Total used** | **26,202** |
---
## Model Architecture
| Property | Value |
|----------|-------|
| **Base Model** | YOLOv8 Nano (`yolov8n.pt`) |
| **Input Size** | 640×640 |
| **Classes** | 2 (`helmet`, `no_helmet`) |
| **Optimizer** | AdamW |
| **Framework** | Ultralytics YOLOv8 |
---

## Installation
### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended, ~4GB VRAM minimum)
### Setup
```bash
# Clone the repository
git clone https://github.com/Mahir-bot-2003/EdgeHelmet-AI.git
cd EdgeHelmet-AI
# Create virtual environment
python -m venv .venv
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
# Install dependencies
pip install ultralytics opencv-python
```
---
## Usage
### 1. Dataset Preparation
Convert Pascal VOC XML annotations to YOLO format:
```bash
# Check available labels in your annotations
python check_labels.py
# Convert XML annotations → YOLO format + train/val split
python convert_dataset.py
```
This creates the `NEWDS/` directory with the YOLO-formatted dataset.
### 2. Training
```bash
python train.py
```
Training runs for up to 100 epochs with early stopping (patience=15). Results are saved to `runs/detect/helmet_combined_v2/`.
### 3. Image Detection
```bash
python image_detect.py
```
<img width="540" height="360" alt="result_test" src="https://github.com/user-attachments/assets/39f6242c-dbc0-4806-bdd1-5fcb61960374" />

### 4. Video Detection
```bash
python video_detect.py
```
Available in runs/predicts

### 5. Live Webcam Detection

```bash
python live_detect.py
```
**Controls:**
- `q` — Quit
- `s` — Save screenshot
**On-screen info:**
- 🟢 Green box — Helmet detected
- 🔴 Red box — No helmet detected
- Top bar — Frame count + per-frame detection stats
- Bottom bar — Safety alert status

## Project Structure
```
EdgeHelmet-AI/
├── images/                    # Raw images (BikesHelmets + hard_hat_workers)
├── annotations/               # Pascal VOC XML annotations
├── NEWDS/                     # Converted YOLO dataset
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── helmet.yaml            # YOLO dataset config
├── runs/                      # Training outputs
│   └── detect/
│       └── helmet_combined_v2/
│           ├── weights/
│           │   ├── best.pt    # Best model weights
│           │   └── last.pt    # Latest checkpoint
│           ├── results.csv    # Epoch-by-epoch metrics
│           ├── results.png    # Training curves plot
│           ├── confusion_matrix.png
│           └── ...
├── train.py                   # Training script
├── convert_dataset.py         # XML → YOLO format converter
├── check_labels.py            # Annotation label inspector
├── image_detect.py            # Single image detection
├── video_detect.py            # Video file detection
├── live_detect.py             # Live webcam detection
├── combine_datasets.py        # Dataset merger utility
├── EVALUATION_METRICS.md      # Detailed evaluation metrics
├── README.md                  # This file
├── yolov8n.pt                 # YOLOv8 Nano pretrained weights
└── yolov8s.pt                 # YOLOv8 Small pretrained weights
```
---
## Training Configuration
```yaml
Model:          YOLOv8 Nano
Epochs:         100 (early stopping patience: 15)
Image Size:     640×640
Batch Size:     4
Optimizer:      AdamW
Learning Rate:  0.001 → 0.00001 (cosine decay)
Weight Decay:   0.0005
# Data Augmentation
Mosaic:         1.0
Mixup:          0.1
HSV Hue:        0.015
HSV Saturation: 0.7
HSV Value:      0.4
Rotation:       ±10°
Translation:    ±10%
Scale:          ±50%
Horizontal Flip: 50%
```
---
## License
This project is for educational and safety research purposes.
Dataset sources:
- [BikesHelmets Dataset](https://www.kaggle.com/) — CC BY 4.0
- [Hard Hat Workers Dataset](https://universe.roboflow.com/) — CC BY 4.0
