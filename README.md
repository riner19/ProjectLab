# Ref-Brain: Combat Sports AI Referee

**Ref-Brain** is an embedded AI system designed to automatically detect, classify, and score combat sports strikes from 2D video feeds. It utilizes a decoupled, lightweight architecture optimized for edge devices (specifically the NVIDIA Jetson Nano via TensorRT FP16). 

Instead of relying on computationally expensive 3D-CNNs, Ref-Brain extracts physical keypoints using an NMS-free pose model and processes the temporal biomechanics using a Bidirectional LSTM.

## 🧠 System Architecture

The pipeline is split into two distinct phases to maximize frame rates on constrained hardware:

### Phase 1: Spatial Extraction & Physics Engineering (`yolo26n-pose`)
Raw video frames are passed through Ultralytics YOLO26n-pose. To overcome the "2D Spatial Ambiguity" problem (where a hook and a cross look identical from a side-profile), the pipeline extracts 17 skeletal keypoints and engineers explicit combat physics into the data stream:
1. **Bounding Box Normalization:** Makes the model scale-invariant to fighters moving closer/further from the camera.
2. **Wrist Velocities ($dx, dy$):** Frame-by-frame delta tracking to distinguish linear strikes (Cross) from lateral strikes (Hook).
3. **Elbow Joint Angles:** Dot-product calculations of the Shoulder-Elbow-Wrist to differentiate bent-arm vs. straight-arm strikes.
4. **Hip-to-Wrist Distance:** Provides spatial context to separate lead-arm strikes (Jabs) from rear-arm strikes (Crosses).

**Output Vector:** A 59-dimensional array per frame (51 raw normalized coordinates + 8 physics features).

### Phase 2: Temporal Sequence Modeling (`Bi-LSTM`)
The 59-D feature vectors are buffered into fixed 30-frame sliding windows and fed into a lightweight Bidirectional LSTM. The model uses forward and backward temporal passes to understand the wind-up, execution, and retraction phases of a strike.

## 📂 Directory Structure

Ensure your project is structured exactly like this before running the pipeline:

```text
HAR/
│
├── datasets/
│   ├── resized/                 # Pre-resized .mp4 files (1080p max)
│   ├── Annotation_files/        # .xlsx files (Start_Frame, Ending_Frame, Class)
│   └── Skeleton_data/           # Auto-generated .npy tensor arrays
│
├── models/                      # Auto-generated .pth weight files
│   └── best_referee_bilstm.pth
│
├── tools/ (or root)
│   ├── data_extractor.py        # ETL pipeline (Video -> 59-D Tensors)
│   ├── model.py                 # PyTorch Bi-LSTM Architecture definition
│   ├── train.py                 # Training loop with dynamic class weights
│   └── evaluate.py              # Confusion matrix & precision/recall diagnostics
│
├── yolo26n-pose.pt              # YOLO pose weights
└── README.md
```

## ⚙️ Installation & Requirements

Requires Python 3.8+ and a CUDA-capable GPU for training.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics pandas numpy opencv-python scikit-learn tqdm openpyxl
```

## 🚀 Pipeline Execution

### 1. Data Extraction
Run the ETL pipeline to process the videos, extract the physics-enhanced skeletons, and generate the training arrays. This script features an auto-purge protocol that deletes old/stale data to prevent tensor mismatches.

```bash
python data_extractor.py
```
*Expected Output:* `Vector dimensions per frame: 59`

### 2. Model Training
Train the Ref-Brain Bi-LSTM. The script automatically handles Train/Val splitting (80/20) and computes balanced class weights to penalize the network for ignoring rare strikes.

```bash
python train.py
```
*Note:* The script includes a hardware assertion check to guarantee it is loading the 59-D physics data. It will save the highest validation accuracy weights to `models/best_referee_bilstm.pth`.

### 3. Model Diagnostics
If validation accuracy plateaus, run the evaluation script to generate a classification report and Pandas-formatted Confusion Matrix. This is critical for identifying biomechanical overlaps (e.g., the "Cross Bias").

```bash
python evaluate.py
```

## 📊 Current Classification Classes
Currently configured for a 6-class active strike environment:
1. Cross
2. Jab
3. Lead Hook
4. Lead Uppercut
5. Rear Hook
6. Rear Uppercut

*Note for Future Deployment:* Before transitioning to continuous live-video inference, a 7th "Idle/Movement" class must be injected into the dataset to prevent False Positives during non-combat movement.