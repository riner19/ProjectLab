import os
import cv2
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

# Defined Class Mapping based on the target architecture
CLASS_MAP = {
    'Idle/Guard': 0,
    'Jab': 1,
    'Cross': 2,
    'Lead Hook': 3,
    'Rear Hook': 4,
    'Lead Uppercut': 5,
    'Rear Uppercut': 6
}


def extract_spatial_temporal_tensors(video_dir, csv_dir, model_path="yolo26n-pose.pt", T_max=30):
    """
    Main execution function to process videos and output PyTorch tensors.
    """
    # device agnostic code
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"[*] Initializing memory-safe pipeline on device: {device.upper()}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate YOLO26 model: {e}")

    all_punches_features = []
    all_punches_labels = []

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    # tqdm bar
    for csv_file in tqdm(csv_files, desc="Processing CSVs", position=0):
        base_name = os.path.splitext(csv_file)[0]
        video_filename = f"{base_name}.mp4"

        csv_path = os.path.join(csv_dir, csv_file)
        video_path = os.path.join(video_dir, video_filename)

        if not os.path.exists(video_path):
            tqdm.write(f"Warning: Expected video asset {video_filename} not found. Skipping.")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            tqdm.write(f"Failed to read metadata file ({csv_file}): {e}")
            continue

        required_cols = ['Start_Frame', 'Ending_Frame', 'Class']
        if not all(col in df.columns for col in required_cols):
            continue

        df['Class'] = df['Class'].astype(str).str.strip().str.title()
        punches = df.sort_values(by='Start_Frame').to_dict('records')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            tqdm.write(f"Codec Error: Unable to open video stream for {video_path}")
            continue

        current_frame_idx = 0
        active_punches = []
        punches_completed = 0

        # inner tqdm bar
        pbar = tqdm(total=len(punches), desc=f"Extracting {video_filename}", position=1, leave=False)

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # EOF

            # State Evaluation: Do any new punches initiate?
            for p in punches:
                if p['Start_Frame'] == current_frame_idx:
                    active_punches.append({
                        'punch_data': p,
                        'frames': []
                    })

            # Append current frame matrix to active punches
            for active in active_punches:
                active['frames'].append(frame)

            # State Evaluation: Do any active punches terminate?
            still_active = []
            for active in active_punches:
                if current_frame_idx >= active['punch_data']['Ending_Frame']:

                    frames = active['frames']
                    raw_label = active['punch_data']['Class']

                    if raw_label in CLASS_MAP:
                        target_label_idx = CLASS_MAP[raw_label]
                        punch_sequence = []

                        # running batch inferece
                        results = model.predict(frames, verbose=False, device=device)

                        for result in results:
                            if not hasattr(result, 'keypoints') or result.keypoints is None or len(
                                    result.keypoints.xyn) == 0:
                                punch_sequence.append(np.zeros(34, dtype=np.float32))  # <-- FLATTENED TO 34!
                                continue

                            kpts_xyn = result.keypoints.xyn[0].cpu().numpy()
                            kpts_conf = result.keypoints.conf[0].cpu().numpy()

                            left_hip_conf, right_hip_conf = kpts_conf[11], kpts_conf[12]

                            if left_hip_conf >= 0.5 and right_hip_conf >= 0.5:
                                mid_hip_x = (kpts_xyn[11, 0] + kpts_xyn[12, 0]) / 2.0
                                mid_hip_y = (kpts_xyn[11, 1] + kpts_xyn[12, 1]) / 2.0
                            elif left_hip_conf >= 0.5:
                                mid_hip_x, mid_hip_y = kpts_xyn[11, 0], kpts_xyn[11, 1]
                            elif right_hip_conf >= 0.5:
                                mid_hip_x, mid_hip_y = kpts_xyn[12, 0], kpts_xyn[12, 1]
                            else:
                                mid_hip_x, mid_hip_y = 0.5, 0.5

                            translated_kpts = np.zeros((17, 2))

                            for i in range(17):
                                if kpts_conf[i] >= 0.5:
                                    translated_kpts[i, 0] = kpts_xyn[i, 0] - mid_hip_x
                                    translated_kpts[i, 1] = kpts_xyn[i, 1] - mid_hip_y
                                else:
                                    translated_kpts[i, 0] = 0.0
                                    translated_kpts[i, 1] = 0.0

                            punch_sequence.append(translated_kpts.flatten())

                        # Format and Normalize Sequence
                        punch_array = np.array(punch_sequence)
                        T_actual = punch_array.shape[0]
                        normalized_sequence = np.zeros((T_max, 34), dtype=np.float32)

                        if T_actual >= T_max:
                            normalized_sequence = punch_array[:T_max, :]
                        else:
                            normalized_sequence[:T_actual, :] = punch_array

                        normalized_sequence = normalized_sequence.transpose()

                        all_punches_features.append(normalized_sequence)
                        all_punches_labels.append(target_label_idx)

                    # update progress bar
                    punches_completed += 1
                    pbar.update(1)

                else:
                    still_active.append(active)

            active_punches = still_active
            current_frame_idx += 1

            if punches_completed == len(punches):
                break

        pbar.close()
        cap.release()

    if len(all_punches_features) == 0:
        print("\nPipeline Warning: No valid punches were extracted from the dataset.")
        return torch.empty((0, 34, T_max)), torch.empty((0,))

    final_features_np = np.stack(all_punches_features, axis=0)
    final_labels_np = np.array(all_punches_labels)

    X_tensor = torch.tensor(final_features_np, dtype=torch.float32).contiguous()
    y_tensor = torch.tensor(final_labels_np, dtype=torch.long)

    print(f"\n[*] Pipeline Execution Complete.")
    print(f"[*] Feature Tensor (X) Shape: {X_tensor.shape} | Represents (N, C, T)")
    print(f"[*] Label Tensor (y) Shape: {y_tensor.shape} | Represents (N,)")

    return X_tensor, y_tensor


if __name__ == "__main__":
    X, y = extract_spatial_temporal_tensors(
        video_dir='datasets/RGB_videos_720p/',
        csv_dir='datasets/Annotations_30fps/',
        model_path='yolo26n-pose.pt',
        T_max=30
    )

    if X.shape[0] > 0:
        torch.save(X, 'datasets/processed_X_features.pt')
        torch.save(y, 'datasets/processed_y_labels.pt')
        print("[*] Successfully saved processed tensors to disk.")