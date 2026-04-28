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
# Resolves paths relative to the location of data_extractor.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

# Updated folder names based on your project structure
RGB_DIR = os.path.join(DATASET_DIR, "RGB_videos_720p")
ANNOTATION_DIR = os.path.join(DATASET_DIR, "Annotations_30fps")
OUTPUT_DIR = os.path.join(DATASET_DIR, "Skeleton_data_1")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Hardware & Model Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Hardware Check: {device}")

model_path = os.path.join(BASE_DIR, "yolo26n-pose.pt")
if not os.path.exists(model_path):
    print("Warning: yolo26n-pose.pt not found in root. Falling back to standard yolo11n-pose.pt...")
    model_path = "yolo11n-pose.pt"

pose_model = YOLO(model_path).to(device)

# --- CONSTANTS ---
CLASS_MAP = {
    "Cross": 0, "Jab": 1, "Lead Hook": 2,
    "Lead Uppercut": 3, "Rear Hook": 4,
    "Rear Uppercut": 5, "Guard": 6, "Idle": 7
}
SEQ_LENGTH = 30
TARGET_TRACK_ID = None


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
    l_hip_dist = np.linalg.norm(curr_kpts[9][:2] - curr_kpts[11][:2])
    r_hip_dist = np.linalg.norm(curr_kpts[10][:2] - curr_kpts[12][:2])
    shoulder_twist = curr_kpts[5][0] - curr_kpts[6][0]
    l_reach = np.linalg.norm(curr_kpts[9][:2] - curr_kpts[5][:2])
    r_reach = np.linalg.norm(curr_kpts[10][:2] - curr_kpts[6][:2])

    return np.array([
        l_angle, r_angle, l_vel[0], l_vel[1], r_vel[0], r_vel[1],
        l_hip_dist, r_hip_dist, shoulder_twist, l_reach, r_reach
    ])


def get_normalized_keypoints(frame):
    global TARGET_TRACK_ID
    results = pose_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    if not results or len(results[0].boxes) == 0 or results[0].boxes.id is None:
        return np.zeros((17, 3))

    track_ids = results[0].boxes.id.int().cpu().tolist()
    keypoints_batch = results[0].keypoints.data.cpu().numpy()

    if TARGET_TRACK_ID is None and len(track_ids) > 0:
        TARGET_TRACK_ID = track_ids[0]

    target_idx = track_ids.index(TARGET_TRACK_ID) if TARGET_TRACK_ID in track_ids else 0
    keypoints = keypoints_batch[target_idx]

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
    global TARGET_TRACK_ID
    TARGET_TRACK_ID = None

    # Switched to read_csv
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if 'Class' not in df.columns: return [], [], []

    # Ensure class consistency
    # Ensure class consistency (Title Case keeps the capital H in Hook)
    df['Class'] = df['Class'].str.title()

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    X_local, y_local, groups_local = [], [], []
    annotated_intervals = []

    for index, row in df.iterrows():
        label = CLASS_MAP.get(row['Class'], -1)
        # Allows all classes through, including Guard (6) and Idle (7)
        if label != -1:
            annotated_intervals.append((int(row['Start_Frame']), int(row['Ending_Frame']), label))

    annotated_intervals.sort(key=lambda x: x[0])

    for start_f, end_f, label in tqdm(annotated_intervals, desc=f"  Extracting: {base_name}", leave=False):
        duration = end_f - start_f

        # Chunking logic for long sequences (Guard/Idle) vs short strikes
        if duration <= SEQ_LENGTH * 1.5:
            centers = [(start_f + end_f) // 2]
        else:
            centers = range(start_f + 15, end_f - 15, 15)

        for center_f in centers:
            half_window = SEQ_LENGTH // 2
            win_start = max(0, center_f - half_window)
            win_end = win_start + SEQ_LENGTH

            if win_end > total_video_frames:
                win_end = total_video_frames
                win_start = max(0, total_video_frames - SEQ_LENGTH)

            cap.set(cv2.CAP_PROP_POS_FRAMES, win_start)

            sequence = []
            prev_kpts = np.zeros((17, 3))

            for frame_idx in range(win_start, win_end):
                ret, frame = cap.read()
                if not ret: break

                curr_kpts = get_normalized_keypoints(frame)
                if frame_idx == win_start: prev_kpts = curr_kpts.copy()

                physics = extract_physics_features(curr_kpts, prev_kpts)
                sequence.append(np.concatenate((curr_kpts.flatten(), physics)))
                prev_kpts = curr_kpts.copy()

            while len(sequence) < SEQ_LENGTH:
                sequence.append(np.zeros(62))

            X_local.append(sequence)
            y_local.append(label)
            groups_local.append(base_name)

    cap.release()
    return X_local, y_local, groups_local


def main():
    print("Starting Unified 8-Class Local Extraction Pipeline...")

    if not os.path.exists(DATASET_DIR):
        print(f"Error: {DATASET_DIR} not found. Please check your folder structure.")
        return

    x_save_path = os.path.join(OUTPUT_DIR, "X_data.npy")
    y_save_path = os.path.join(OUTPUT_DIR, "y_data.npy")
    groups_save_path = os.path.join(OUTPUT_DIR, "groups_data.npy")

    all_X, all_y, all_groups = [], [], []

    print("\n[Phase 1] Processing All Annotated Actions (Classes 0-7)...")
    # Using the updated paths to find your CSVs and MP4s
    csv_files = glob.glob(os.path.join(ANNOTATION_DIR, "*.csv"))
    csv_files.sort(key=natural_sort_key)

    for csv_path in tqdm(csv_files, desc="Processing Videos"):
        # Match the base name of the CSV (e.g. V1_with_guard) to the MP4.
        # If your MP4 is named differently (e.g. V1.mp4), you may need to adjust the replacement logic here.
        base_name = os.path.splitext(os.path.basename(csv_path))[0]

        # If your CSV has a suffix like "_with_guard" that the video doesn't have, uncomment the line below:
        # video_base_name = base_name.replace("_with_guard", "")
        # video_path = os.path.join(RGB_DIR, f"{video_base_name}.mp4")

        video_path = os.path.join(RGB_DIR, f"{base_name}.mp4")

        if os.path.exists(video_path):
            X_loc, y_loc, g_loc = process_annotated_pair(video_path, csv_path, base_name)
            all_X.extend(X_loc)
            all_y.extend(y_loc)
            all_groups.extend(g_loc)
        else:
            print(f"\nMissing video pair for annotation: {video_path}")

    # No Phase 2 or 3 needed!

    # Save Final Outputs
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
        print(f"Vector Dimensions: {final_X.shape[2]} (Target: 62)")
        print(f"Groups Tracked: {len(np.unique(final_groups))} Unique Videos")
        print(f"Saved to: {OUTPUT_DIR}")
        print("=" * 40)
    else:
        print("\nExtraction failed: No matching video/CSV pairs found or processed.")


if __name__ == "__main__":
    main()