# Real-Time Human Action Recognition (HAR)
**Skeleton-based LSTM Pipeline utilizing YOLOv8 and PyTorch**

## 🚀 Overview
This project implements an end-to-end deep learning pipeline for real-time human action recognition. The system uses **YOLOv8-Pose** for spatial feature extraction and a custom **LSTM (Long Short-Term Memory)** network to classify temporal sequences of movement.



## 🛠️ Technical Stack
* **Computer Vision:** YOLOv8-Pose (Ultralytics)
* **Deep Learning Framework:** PyTorch
* **Data Processing:** Pandas, NumPy
* **Dataset:** KTH Action Recognition Dataset (Boxing & Walking)
* **Hardware Acceleration:** NVIDIA GeForce RTX 3050 Ti (CUDA)

## 📁 Project Structure
- `setup_kth.py`: Automated dataset downloader and organizer via `kagglehub`.
- `extract_batch.py`: Batch processor converting raw video into 51-dimensional coordinate CSVs.
- `dataset.py`: Optimized PyTorch Dataset using **RAM Preloading** and **Sliding Window** augmentation.
- `model.py`: Architecture definition for the LSTM Neural Network.
- `train.py`: Training loop with validation tracking and model checkpointing.
- `live_inference.py`: Real-time webcam application with 30-frame temporal smoothing.

## 🧠 Pipeline Logic
1. **Extraction:** YOLOv8 identifies 17 keypoints per frame. Each keypoint has $(x, y, c)$ values, totaling 51 features per frame.
2. **Windowing:** The `dataset.py` slices videos into overlapping **30-frame windows** (approx. 1 second of movement).
3. **Training:** The LSTM learns the transition of these coordinates over time to distinguish a "Punch" from "Walking."
4. **Optimization:** Data is preloaded into System RAM to ensure the GPU is never bottlenecked by Disk I/O.



## 📈 Performance
- **Target Accuracy:** >90% on KTH Validation Set.
- **Inference Latency:** <30ms per frame (Real-time).
- **GPU Utilization:** Optimized via CUDA for both feature extraction and training.

## ⚙️ How to Run
1. **Prepare Environment:**
   ```bash
   pip install torch torchvision ultralytics pandas kagglehub opencv-python

2. **Download & Extract Features:**

    ```Bash
    python setup_kth.py
    python extract_batch.py

3. **Train Model:**

    ```Bash
    python train.py
4. **Run Real-Time Inference:**

    ```Bash
    python live_inference.py
## 🔮 Future Work
 * [ ] Implement Global Coordinate Normalization (making the model distance-invariant).

 *[ ] Add Delta-Velocity Features to improve high-speed action detection.

*[ ] Expand dataset classes to include Waving, Clapping, and Static.

*[ ] Explore ST-GCN (Spatio-Temporal Graph Convolutional Networks) for higher topological accuracy.