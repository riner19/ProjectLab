
# Combat AI Referee: Real-Time Strike Detection on Edge Hardware

![BME Project Laboratory](https://img.shields.io/badge/BME-Project_Laboratory-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C)
![Ultralytics](https://img.shields.io/badge/YOLO-Pose_Estimation-00FFFF)
![Edge AI](https://img.shields.io/badge/Edge-NVIDIA_Jetson-76B900)

## Overview
The **Combat AI Referee** is a lightweight, purely vision-based pipeline designed to classify six distinct offensive boxing actions (Jab, Cross, Lead Hook, Rear Hook, Lead Uppercut, Rear Uppercut) alongside one defensive state (Idle/Guard) in real-time. 

Designed explicitly for edge hardware (specifically the NVIDIA Jetson Orin Nano), the architecture discards computationally heavy LSTMs and ST-GCNs in favor of an asynchronous, multithreaded 1D Temporal Convolutional Network (TCN) pipeline that runs at a stable **~30 FPS** while staying well within edge VRAM and thermal constraints.

## Key Technical Features
* **High-Speed Spatial Extraction:** Utilizes the NMS-free `YOLO26-nano` pose estimator to extract 17 anatomical keypoints. Applies a strict Mid-Hip Origin Translation to guarantee scale and translation invariance.
* **Supervised Negative Class Integration:** Resolves severe combat sports dataset imbalance by supervised extraction of 'Idle/Guard' states, preventing continuous false positives during live inference.
* **Custom 1D-Causal Architecture (`EdgeBoxingTCN`):** A highly optimized PyTorch model utilizing depthwise separable causal convolutions (dilation rates of 1, 2, 4, 8) and dynamic kinematic feature expansion (velocity and acceleration computed directly on the GPU).
* **Algorithmic Class Balancing:** Implements a `WeightedRandomSampler` and Multi-Class Focal Loss (gamma = 2.0) to penalize the massive majority 'Idle' class and force the optimizer to learn complex minority strikes.
* **O(1) Multithreaded Edge Inference:** Uses a producer-consumer paradigm with a `collections.deque` temporal sliding window, decoupling camera hardware from the GPU inference engine for zero-lag kinematic classification.

## Repository Structure
* `data_extractor.py`: Extracts spatial-temporal 34-D tensors from raw monocular RGB videos using YOLO26n and mid-hip translation.
* `extract_dedicated_idle.py`: Mines and normalizes the pure defensive ('Idle/Guard') states for the negative class dataset.
* `train_tcn.py`: Contains the `EdgeBoxingTCN` architecture and the complete training loop (AdamW, Cosine Annealing, Focal Loss).
* `evaluate_model.py`: Generates the confusion matrix and detailed classification report on the validation split.
* `test_inference.py`: The live, multithreaded edge deployment script featuring the kinematic debouncing state-machine.
* `/datasets/`: Directory holding raw `RGB_videos_720p/` and `Annotations_30fps/` (requires user population).

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/combat-ai-referee.git](https://github.com/yourusername/combat-ai-referee.git)
   cd combat-ai-referee



2. **Install dependencies:**
```bash
pip install torch torchvision torchaudio ultralytics opencv-python pandas numpy tqdm matplotlib seaborn scikit-learn

```


*(Note: For Jetson deployment, ensure you are using the NVIDIA-provided PyTorch wheels compiled with TensorRT/CUDA support).*
3. **Download YOLO Weights:**
Ensure `yolo26n-pose.pt` is in the root directory.

## Usage Pipeline

### 1. Data Extraction

Populate your `/datasets/RGB_videos_720p/` and `/datasets/Annotations_30fps/` folders. Run the extraction scripts to build the `.pt` tensors:

```bash
python data_extractor.py
python extract_dedicated_idle.py

```

### 2. Model Training

Train the temporal network. This script will load the tensors, dynamically apply the `WeightedRandomSampler`, and output `edge_boxing_tcn.pth`.

```bash
python train_tcn.py

```

### 3. Validation

Evaluate your trained model and generate a Seaborn confusion matrix heat map:

```bash
python evaluate_model.py

```

### 4. Real-Time Inference

Deploy the trained model on edge hardware or webcam. Edit `VIDEO_PATH` in the script to use a local video or `0` for live webcams.

```bash
python test_inference.py

```



## Author

**Rinat Yerkinbek**
*Budapest University of Technology and Economics (BME)*
*Project Laboratory*

```

