# Real-Time Spatio-Temporal Action Recognition on Edge Devices 🥊

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![YOLO](https://img.shields.io/badge/YOLOv8-Pose%20Estimation-00FFFF)
![EdgeAI](https://img.shields.io/badge/Edge%20AI-Raspberry%20Pi%20%2F%20Jetson-76B900)

## Overview
This repository contains a full-stack Machine Learning pipeline designed to perform real-time action recognition (specifically combat sports/striking) on highly constrained edge hardware. 

Instead of processing heavy, high-resolution video frames directly, this project utilizes a **Hybrid Architecture**:
1. **Vision Front-End (Spatial):** A lightweight YOLOv8-Nano pose estimation model extracts 17 COCO skeletal keypoints.
2. **Sequence Back-End (Temporal):** A custom PyTorch neural network (LSTM/Transformer) processes the coordinate time-series to classify the action.

By compressing dense pixel data into lightweight coordinate tensors, the system achieves significant inference speedups suitable for deployment on an NVIDIA Jetson Orin Nano or Raspberry Pi 5.

## 🏗️ System Architecture

### 1. Automated Data Engineering Pipeline
* Utilizes the Kaggle API to ingest the 7GB UCF101 dataset.
* Automatically parses, filters, and isolates balanced sets of target classes (e.g., `Boxing`, `WalkingWithDog`).
* Runs batch feature extraction, converting `.avi` video files into sequential `.csv` files containing `(X, Y, Confidence)` vectors for all 17 joints per frame.

### 2. Spatio-Temporal Classification Model
* Custom `torch.utils.data.Dataset` designed to ingest variable-length coordinate CSVs.
* PyTorch-based sequence model designed to recognize biomechanical patterns over time windows.

### 3. Edge Deployment & Hardware Optimization *(In Progress)*
* Benchmarking CPU vs. NPU/GPU inference.
* Target hardware: Raspberry Pi 5 (Hailo-8L) / NVIDIA Jetson Orin Nano (TensorRT).

## 📂 Project Structure
```text
├── dataset_videos/      # Raw video data (ignored by git)
├── dataset_csvs/        # Extracted sequential keypoints (ignored by git)
├── auto_pipeline.py     # Automated dataset ingestion and balancing script
├── extract_batch.py     # YOLOv8-Pose batch feature extraction pipeline
├── dataset.py           # PyTorch Custom Dataset loader
├── model.py             # PyTorch Sequence Model architecture (WIP)
└── README.md