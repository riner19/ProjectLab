import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm

# --- Dynamic Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

RGB_DIR = os.path.join(DATASET_DIR, "RGB_videos_720p")
ANNOTATION_DIR = os.path.join(DATASET_DIR, "Annotations_30fps")
OUTPUT_DIR = os.path.join(DATASET_DIR, "Skeleton_data_1")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Hardware & Model Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Hardware Check: {device}")

model_path = os.path.join(BASE_DIR, "yolo26n-pose.pt")
if not os.path.exists(model_path):
    print("Warning: yolo26n-pose.pt not found. Falling back to yolo11n-pose.pt...")
    model_path = "yolo11n-pose.pt"

pose_model = YOLO(model_path).to(device)

# --- CONSTANTS ---
CLASS_MAP = {
    "Cross": 0, "Jab": 1, "Lead Hook": 2,
    "Lead Uppercut": 3, "Rear Hook": 4,
    "Rear Uppercut": 5
}
SEQ_LENGTH = 30


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-5 or norm_bc < 1e-5: return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle) / 180.0


def extract_physics_features(curr_kpts, prev_kpts):
    l_angle = calculate_angle(curr_kpts[5][:2], curr_kpts[7][:2], curr_kpts[9][:2])
    r_angle = calculate_angle(curr_kpts[6][:2], curr_kpts[8][:2], curr_kpts[10][:2])

    l_vel = curr_kpts[9][:2] - prev_kpts[9][:2]
    r_vel = curr_kpts[10][:2] - prev_kpts[10][:2]

    l_hip_dx = curr_kpts[9][0] - curr_kpts[11][0]
    l_hip_dy = curr_kpts[9][1] - curr_kpts[11][1]

    r_hip_dx = curr_kpts[10][0] - curr_kpts[12][0]
    r_hip_dy = curr_kpts[10][1] - curr_kpts[12][1]

    shoulder_twist = curr_kpts[5][0] - curr_kpts[6][0]
    l_reach = np.linalg.norm(curr_kpts[9][:2] - curr_kpts[5][:2])
    r_reach = np.linalg.norm(curr_kpts[10][:2] - curr_kpts[6][:2])

    return np.array([
        l_angle, r_angle, l_vel[0], l_vel[1], r_vel[0], r_vel[1],
        l_hip_dx, l_hip_dy, r_hip_dx, r_hip_dy,
        shoulder_twist, l_reach, r_reach
    ])


def get_normalized_keypoints(frame):
    # OPTIMIZATION: half=True enables FP16 on your RTX 3050 Ti for a ~30% speed boost.
    # Removed tracker="bytetrack.yaml" as it breaks when skipping frames.
    results = pose_model(frame, verbose=False, half=True)

    if not results or len(results[0].boxes) == 0:
        return np.zeros((17, 3))

    # OPTIMIZATION: Robustly grab the largest person on screen instead of relying on fragile track IDs
    boxes = results[0].boxes.xywh.cpu().numpy()
    main_idx = (boxes[:, 2] * boxes[:, 3]).argmax()
    keypoints = results[0].keypoints.data[main_idx].cpu().numpy()

    # --- EGOCENTRIC NORMALIZATION ---
    neck_x = (keypoints[5][0] + keypoints[6][0]) / 2.0
    neck_y = (keypoints[5][1] + keypoints[6][1]) / 2.0
    neck = np.array([neck_x, neck_y])

    hip_x = (keypoints[11][0] + keypoints[12][0]) / 2.0
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2.0
    mid_hip = np.array([hip_x, hip_y])

    torso_length = np.linalg.norm(neck - mid_hip)
    if torso_length < 10.0:
        torso_length = 10.0

    normalized_kpts = np.zeros((17, 3))

    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:
            norm_x = (x - mid_hip[0]) / torso_length
            norm_y = (y - mid_hip[1]) / torso_length
            normalized_kpts[i] = [norm_x, norm_y, conf]

    return normalized_kpts


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def process_annotated_pair(video_path, csv_path, base_name):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if 'Class' not in df.columns: return [], [], []

    df['Class'] = df['Class'].str.title()

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- OPTIMIZATION: Pre-calculate all required windows and needed frames ---
    windows = []
    needed_frames = set()

    for index, row in df.iterrows():
        label = CLASS_MAP.get(row['Class'], -1)
        if label != -1:
            start_f = int(row['Start_Frame'])
            end_f = int(row['Ending_Frame'])
            duration = end_f - start_f

            if duration <= SEQ_LENGTH * 1.5:
                centers = [(start_f + end_f) // 2]
            else:
                centers = range(start_f + 15, end_f - 15, 15)

            for center_f in centers:
                win_start = max(0, center_f - (SEQ_LENGTH // 2))
                win_end = win_start + SEQ_LENGTH

                if win_end > total_video_frames:
                    win_end = total_video_frames
                    win_start = max(0, total_video_frames - SEQ_LENGTH)

                windows.append((win_start, win_end, label))
                # Log exactly which frames we need to read from the video
                for f in range(win_start, win_end):
                    needed_frames.add(f)

    if not needed_frames:
        cap.release()
        return [], [], []

    # --- OPTIMIZATION: Single-Pass Chronological Read ---
    min_f = min(needed_frames)
    max_f = max(needed_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, min_f)
    current_f = min_f
    frame_cache = {}  # Store skeletons in fast RAM

    # Fast-forward reading loop
    with tqdm(total=(max_f - min_f), desc=f"  YOLO Scanning: {base_name}", leave=False) as pbar:
        while cap.isOpened() and current_f <= max_f:
            ret, frame = cap.read()
            if not ret: break

            if current_f in needed_frames:
                frame_cache[current_f] = get_normalized_keypoints(frame)

            current_f += 1
            pbar.update(1)

    cap.release()

    # --- RECONSTRUCTION: Build the Tensors from RAM ---
    X_local, y_local, groups_local = [], [], []

    for win_start, win_end, label in windows:
        sequence = []
        prev_kpts = np.zeros((17, 3))

        for f in range(win_start, win_end):
            if f in frame_cache:
                curr_kpts = frame_cache[f]

                # Fetch prev_kpts chronologically from cache for physics
                if f == win_start or (f - 1) not in frame_cache:
                    prev_kpts = curr_kpts.copy()
                else:
                    prev_kpts = frame_cache[f - 1]

                physics = extract_physics_features(curr_kpts, prev_kpts)
                sequence.append(np.concatenate((curr_kpts.flatten(), physics)))
            else:
                break  # Frame couldn't be loaded

        # Edge Padding to 64 Dimensions
        if len(sequence) > 0:
            last_valid_frame = sequence[-1]
            while len(sequence) < SEQ_LENGTH:
                sequence.append(last_valid_frame)
        else:
            while len(sequence) < SEQ_LENGTH:
                sequence.append(np.zeros(64))

        X_local.append(sequence)
        y_local.append(label)
        groups_local.append(base_name)

    return X_local, y_local, groups_local


def main():
    print("Starting Unified 6-Class Local Extraction Pipeline (Optimized)...")

    if not os.path.exists(DATASET_DIR):
        print(f"Error: {DATASET_DIR} not found. Please check your folder structure.")
        return

    x_save_path = os.path.join(OUTPUT_DIR, "X_data.npy")
    y_save_path = os.path.join(OUTPUT_DIR, "y_data.npy")
    groups_save_path = os.path.join(OUTPUT_DIR, "groups_data.npy")

    all_X, all_y, all_groups = [], [], []

    csv_files = glob.glob(os.path.join(ANNOTATION_DIR, "*.csv"))
    csv_files.sort(key=natural_sort_key)

    for csv_path in tqdm(csv_files, desc="Total Progress"):
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        video_path = os.path.join(RGB_DIR, f"{base_name}.mp4")

        if os.path.exists(video_path):
            X_loc, y_loc, g_loc = process_annotated_pair(video_path, csv_path, base_name)
            all_X.extend(X_loc)
            all_y.extend(y_loc)
            all_groups.extend(g_loc)
        else:
            print(f"\nMissing video pair for annotation: {video_path}")

    if len(all_X) > 0:
        final_X = np.array(all_X, dtype=np.float32)
        final_y = np.array(all_y, dtype=np.int64)
        final_groups = np.array(all_groups)

        np.save(x_save_path, final_X)
        np.save(y_save_path, final_y)
        np.save(groups_save_path, final_groups)

        print("\n" + "=" * 40)
        print("EXTRACTION COMPLETE")
        print(f"Total Sequences: {len(final_X)}")
        print(f"Vector Dimensions: {final_X.shape[2]} (Target: 64)")
        print(f"Groups Tracked: {len(np.unique(final_groups))} Unique Videos")
        print(f"Saved to: {OUTPUT_DIR}")
        print("=" * 40)
    else:
        print("\nExtraction failed: No matching video/CSV pairs found or processed.")


if __name__ == "__main__":
    main()