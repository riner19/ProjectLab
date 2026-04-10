import os
import glob
import re
import shutil
import numpy as np
import pandas as pd
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm

# --- Local Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR if "HAR" in os.path.basename(BASE_DIR) else os.path.dirname(BASE_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

VIDEO_DIR = os.path.join(DATASETS_DIR, "resized")
ANNOTATION_DIR = os.path.join(DATASETS_DIR, "Annotation_files")
OUTPUT_DIR = os.path.join(DATASETS_DIR, "Skeleton_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading YOLO26 on {device}...")
# Ensure you are pointing to your yolo26n-pose.pt file
pose_model = YOLO("yolo26n-pose.pt").to(device)

CLASS_MAP = {
    "Cross": 0, "Jab": 1, "Lead Hook": 2,
    "Lead Uppercut": 3, "Rear Hook": 4,
    "Rear Uppercut": 5, "Idle": 6
}
SEQ_LENGTH = 30


def calculate_angle(a, b, c):
    """Calculates 2D angle between 3 points. 'b' is the vertex (e.g., elbow)."""
    ba = a - b
    bc = c - b

    # Avoid division by zero if keypoints are missing [0,0]
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-5 or norm_bc < 1e-5:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    # Normalize angle from 0-180 degrees down to 0-1 range for the neural network
    return np.degrees(angle) / 180.0


def extract_physics_features(curr_kpts, prev_kpts):
    """
    Computes biomechanical features: Joint Angles, Velocities, and Hip Distances.
    YOLO Pose Indices:
    L_Shoulder=5, R_Shoulder=6 | L_Elbow=7, R_Elbow=8
    L_Wrist=9, R_Wrist=10 | L_Hip=11, R_Hip=12
    """
    # 1. Elbow Angles (Shoulder -> Elbow -> Wrist)
    l_angle = calculate_angle(curr_kpts[5][:2], curr_kpts[7][:2], curr_kpts[9][:2])
    r_angle = calculate_angle(curr_kpts[6][:2], curr_kpts[8][:2], curr_kpts[10][:2])

    # 2. Wrist Velocities (dx, dy)
    l_vel = curr_kpts[9][:2] - prev_kpts[9][:2]
    r_vel = curr_kpts[10][:2] - prev_kpts[10][:2]

    # 3. Wrist to Hip Distances (Arm Extension)
    l_dist = np.linalg.norm(curr_kpts[9][:2] - curr_kpts[11][:2])
    r_dist = np.linalg.norm(curr_kpts[10][:2] - curr_kpts[12][:2])

    return np.array([l_angle, r_angle, l_vel[0], l_vel[1], r_vel[0], r_vel[1], l_dist, r_dist])


def get_normalized_keypoints(frame):
    results = pose_model(frame, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return np.zeros((17, 3))

    box = results[0].boxes.xyxy[0].cpu().numpy()
    keypoints = results[0].keypoints.data[0].cpu().numpy()

    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return np.zeros((17, 3))

    normalized_kpts = np.zeros((17, 3))
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:
            normalized_kpts[i] = [
                (x - x_min) / width,
                (y - y_min) / height,
                conf
            ]
    return normalized_kpts


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def process_single_pair(video_path, excel_path, base_name):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()

    if 'Class' not in df.columns:
        print(f"Skipping {base_name}: 'Class' column not found.")
        return [], []

    cap = cv2.VideoCapture(video_path)
    X_local, y_local = [], []

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"  Extracting {base_name}", leave=False):
        start_frame = int(row['Start_Frame'])
        end_frame = int(row['Ending_Frame'])
        label = CLASS_MAP.get(row['Class'], -1)

        if label == -1: continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        sequence = []
        prev_kpts = np.zeros((17, 3))

        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret: break

            # HD Downscaling
            h, w = frame.shape[:2]
            if w > 1920 or h > 1080:
                scale = min(1920 / w, 1080 / h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            curr_kpts = get_normalized_keypoints(frame)

            # If first frame of sequence, prev_kpts = curr_kpts (velocity is 0)
            if frame_idx == start_frame:
                prev_kpts = curr_kpts.copy()

            physics_features = extract_physics_features(curr_kpts, prev_kpts)

            # Concatenate the 51 raw coordinates with the 8 physics features
            full_frame_vector = np.concatenate((curr_kpts.flatten(), physics_features))
            sequence.append(full_frame_vector)

            prev_kpts = curr_kpts.copy()

        if len(sequence) < SEQ_LENGTH:
            padding = [np.zeros(59)] * (SEQ_LENGTH - len(sequence))  # Note the 59!
            sequence.extend(padding)
        else:
            sequence = sequence[:SEQ_LENGTH]

        X_local.append(sequence)
        y_local.append(label)

    cap.release()
    return X_local, y_local


def main():
    print("Starting Physics-Enhanced Extraction Phase...")
    all_X, all_y = [], []
    excel_files = glob.glob(os.path.join(ANNOTATION_DIR, "*.xlsx"))
    excel_files.sort(key=natural_sort_key)

    for excel_path in tqdm(excel_files, desc="Processing Videos"):
        base_name = os.path.splitext(os.path.basename(excel_path))[0]
        video_path = os.path.join(VIDEO_DIR, f"{base_name}.mp4")

        if not os.path.exists(video_path):
            tqdm.write(f"Warning: Video missing for {base_name}. Skipping.")
            continue

        X_local, y_local = process_single_pair(video_path, excel_path, base_name)
        all_X.extend(X_local)
        all_y.extend(y_local)

    final_X = np.array(all_X)
    final_y = np.array(all_y)

    x_path = os.path.join(OUTPUT_DIR, "X_data.npy")
    y_path = os.path.join(OUTPUT_DIR, "y_data.npy")
    np.save(x_path, final_X)
    np.save(y_path, final_y)

    print("\nExtraction finished.")
    print(f"Total sequences extracted: {len(final_X)}")
    print(f"Vector dimensions per frame: {final_X.shape[2]} (Should be 59)")


if __name__ == "__main__":
    main()