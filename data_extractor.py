import os
import glob
import re
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

# Pointing to your pre-resized videos folder
VIDEO_DIR = os.path.join(DATASETS_DIR, "resized")
ANNOTATION_DIR = os.path.join(DATASETS_DIR, "Annotation_files")
OUTPUT_DIR = os.path.join(DATASETS_DIR, "Skeleton_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading YOLO26 on {device}...")
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

TARGET_TRACK_ID = None


def get_normalized_keypoints(frame):
    global TARGET_TRACK_ID

    # Use .track() instead of direct inference to assign persistent IDs
    results = pose_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    if not results or len(results[0].boxes) == 0 or results[0].boxes.id is None:
        return np.zeros((17, 3))

    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    keypoints_batch = results[0].keypoints.data.cpu().numpy()

    target_idx = -1

    # Lock onto the first detected ID and stick with it for the whole video
    if TARGET_TRACK_ID is None and len(track_ids) > 0:
        TARGET_TRACK_ID = track_ids[0]

    # Find our target fighter in the current frame
    if TARGET_TRACK_ID in track_ids:
        target_idx = track_ids.index(TARGET_TRACK_ID)
    else:
        # If the target is temporarily lost, fallback to the most confident detection
        target_idx = 0

    box = boxes[target_idx]
    keypoints = keypoints_batch[target_idx]

    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return np.zeros((17, 3))

    normalized_kpts = np.zeros((17, 3))
    for i, (x, y, conf) in enumerate(keypoints):
        # Keep your 0.3 confidence threshold to drop blurry limbs
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
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    X_local, y_local = [], []

    # 1. Map out all annotated intervals to find the "gaps"
    annotated_intervals = []
    for index, row in df.iterrows():
        label = CLASS_MAP.get(row['Class'], -1)
        if label != -1:
            annotated_intervals.append((int(row['Start_Frame']), int(row['Ending_Frame']), label))

    # Sort by start frame to process sequentially
    annotated_intervals.sort(key=lambda x: x[0])

    # 2. Helper function to extract a specific frame range
    def extract_sequence(start_f, end_f, label_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        sequence = []
        prev_kpts = np.zeros((17, 3))

        for frame_idx in range(start_f, end_f + 1):
            ret, frame = cap.read()
            if not ret: break

            curr_kpts = get_normalized_keypoints(frame)
            if frame_idx == start_f:
                prev_kpts = curr_kpts.copy()

            physics_features = extract_physics_features(curr_kpts, prev_kpts)
            full_frame_vector = np.concatenate((curr_kpts.flatten(), physics_features))
            sequence.append(full_frame_vector)
            prev_kpts = curr_kpts.copy()

        # FIXED PADDING: Repeat the last known frame state instead of zeroing out
        if len(sequence) < SEQ_LENGTH:
            last_frame_state = sequence[-1] if len(sequence) > 0 else np.zeros(59)
            padding = [last_frame_state] * (SEQ_LENGTH - len(sequence))
            sequence.extend(padding)
        else:
            sequence = sequence[:SEQ_LENGTH]

        return sequence

    # 3. Extract logic: Mine both Strikes and Idle Gaps
    current_frame = 0
    for start_f, end_f, label in tqdm(annotated_intervals, desc=f"  Extracting {base_name}", leave=False):

        # --- IDLE MINING (Class 6) ---
        # If there is a gap of at least SEQ_LENGTH (30 frames) before the next strike,
        # extract chunks of it as "Idle" data.
        gap_size = start_f - current_frame
        if gap_size >= SEQ_LENGTH:
            # Step by SEQ_LENGTH to avoid overlapping idle frames
            for idle_start in range(current_frame, start_f - SEQ_LENGTH + 1, SEQ_LENGTH):
                idle_seq = extract_sequence(idle_start, idle_start + SEQ_LENGTH - 1, 6)
                if len(idle_seq) == SEQ_LENGTH:
                    X_local.append(idle_seq)
                    y_local.append(6)  # 6 is "Idle"

        # --- STRIKE EXTRACTION ---
        strike_seq = extract_sequence(start_f, end_f, label)
        if len(strike_seq) == SEQ_LENGTH:
            X_local.append(strike_seq)
            y_local.append(label)

        # Move the cursor past this strike
        current_frame = end_f + 1

    cap.release()
    return X_local, y_local


def main():
    print("Starting Physics-Enhanced Extraction Phase...")

    # AUTO-PURGE: Guarantee the old 51-D data is deleted before writing new data
    x_save_path = os.path.join(OUTPUT_DIR, "X_data.npy")
    y_save_path = os.path.join(OUTPUT_DIR, "y_data.npy")

    if os.path.exists(x_save_path):
        os.remove(x_save_path)
        print("  -> Purged old X_data.npy")
    if os.path.exists(y_save_path):
        os.remove(y_save_path)
        print("  -> Purged old y_data.npy")

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

    np.save(x_save_path, final_X)
    np.save(y_save_path, final_y)

    print("\nExtraction finished.")
    print(f"Total sequences extracted: {len(final_X)}")
    print(f"Vector dimensions per frame: {final_X.shape[2]} (Should be 59)")


if __name__ == "__main__":
    main()