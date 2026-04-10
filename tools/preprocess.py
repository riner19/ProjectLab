import cv2
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

VIDEO_PATH = PROJECT_DIR / 'data' / 'raw_videos' / 'shadow_boxing_nate.mp4'
OUTPUT_CSV = PROJECT_DIR / 'data' / 'processed_sequences' / 'shadow_boxing_nate_labelled.csv'
MODEL_VARIANT = PROJECT_DIR / 'tools' / 'yolo26n-pose.pt'

# BIOMECHANIC THRESHOLDS (Normalized to Torso Size)
EXTENSION_THRESH = 0.015  # Arm straightening speed
GUARD_BREAK_THRESH = 0.04  # Wrist moving away from nose
MIN_PUNCH_DURATION = 3  # Min frames to count as a "strike" to avoid jitter

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO(str(MODEL_VARIANT)).to(device)

cap = cv2.VideoCapture(str(VIDEO_PATH))
# Start when the action begins (Example: skip intro)
cap.set(cv2.CAP_PROP_POS_MSEC, 120000)

csv_data = []
frame_idx = 0
punch_buffer = 0  # Temporal smoothing

# Biomechanic Memory
prev_ext_l, prev_ext_r = None, None
prev_guard_l, prev_guard_r = None, None

print(f"🚀 Auto-Extracting Combat Logic on {device}...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # NMS-free inference is faster here
    results = yolo_model(frame, verbose=False)
    annotated_frame = results[0].plot()

    current_label = 0
    row_data = {'frame_idx': frame_idx, 'label': 0}

    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        # 1. Focus on the main boxer
        boxes = results[0].boxes.xywh.cpu().numpy()
        main_idx = (boxes[:, 2] * boxes[:, 3]).argmax()
        kpts = results[0].keypoints.data[main_idx].cpu().numpy()

        # 2. Reference Points (Nose: 0, L_Shld: 5, R_Shld: 6, L_Wrist: 9, R_Wrist: 10, L_Hip: 11, R_Hip: 12)
        nose = kpts[0][:2]
        shld_l, shld_r = kpts[5][:2], kpts[6][:2]
        wrist_l, wrist_r = kpts[9][:2], kpts[10][:2]
        hip_l, hip_r = kpts[11][:2], kpts[12][:2]

        # 3. Scaling & Normalization
        torso_size = np.linalg.norm((shld_l + shld_r) / 2 - (hip_l + hip_r) / 2) + 1e-6
        mid_hip = (hip_l + hip_r) / 2

        # 4. Physics Metrics
        curr_guard_l = np.linalg.norm(wrist_l - nose) / torso_size
        curr_guard_r = np.linalg.norm(wrist_r - nose) / torso_size
        curr_ext_l = np.linalg.norm(wrist_l - shld_l) / torso_size
        curr_ext_r = np.linalg.norm(wrist_r - shld_r) / torso_size

        if prev_ext_l is not None:
            # Velocity calculation
            v_ext = max(curr_ext_l - prev_ext_l, curr_ext_r - prev_ext_r)
            v_break = max(curr_guard_l - prev_guard_l, curr_guard_r - prev_guard_r)

            # Trigger Logic with Temporal Smoothing
            if v_ext > EXTENSION_THRESH or v_break > GUARD_BREAK_THRESH:
                punch_buffer = MIN_PUNCH_DURATION  # Hold the '1' label for a few frames

            if punch_buffer > 0:
                current_label = 1
                punch_buffer -= 1

        # Update Memory
        prev_ext_l, prev_ext_r = curr_ext_l, curr_ext_r
        prev_guard_l, prev_guard_r = curr_guard_l, curr_guard_r

        # 5. Export Flattened Keypoints
        row_data['label'] = current_label
        for j in range(17):
            row_data[f'kp{j}_x'] = (kpts[j, 0] - mid_hip[0]) / torso_size
            row_data[f'kp{j}_y'] = (kpts[j, 1] - mid_hip[1]) / torso_size
            row_data[f'kp{j}_v'] = kpts[j, 2]

    csv_data.append(row_data)
    frame_idx += 1

    # Visual Feedback
    color = (0, 0, 255) if current_label == 1 else (0, 255, 0)
    cv2.putText(annotated_frame, f"STATE: {'STRIKE' if current_label else 'IDLE'}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Auto-Ref Data Gen', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

if csv_data:
    pd.DataFrame(csv_data).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Success. Created {OUTPUT_CSV} with auto-labeled biomechanics.")
else:
    print("⚠️ No frames were processed, so no CSV was created.")