import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

IDLE_VIDEO_PATHS = [
    'guard/guard1.mp4',
    'guard/guard2.mp4'
]
MODEL_PATH = 'yolo26n-pose.pt'
T_MAX = 30
BATCH_SIZE = 32


def extract_dedicated_idle_videos():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Extracting Supervised Idle Data on: {device.upper()}")

    model = YOLO(MODEL_PATH)
    all_idle_features = []

    # looping through videos, I will add more in the future, currently only 2 videos
    for video_path in IDLE_VIDEO_PATHS:
        if not os.path.exists(video_path):
            print(f"\n[!] Warning: Could not find '{video_path}'. Skipping.")
            continue

        video_name = os.path.basename(video_path)
        print(f"\n[*] Processing Video: {video_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[!] Could not open video at {video_path}")
            continue

        # tqdm progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc=f"Extracting {video_name}")

        raw_sequences = []
        batch_frames = []

        while True:
            ret, frame = cap.read()
            if not ret: break

            batch_frames.append(frame)
            pbar.update(1)

            # batches of 32 frames
            if len(batch_frames) == BATCH_SIZE or pbar.n == total_frames:
                results = model.predict(batch_frames, verbose=False, device=device)

                for result in results:
                    # 1. Human detection
                    if not hasattr(result, 'keypoints') or result.keypoints is None or len(result.keypoints.xyn) == 0:
                        raw_sequences.append(np.zeros(34, dtype=np.float32))
                        continue

                    kpts_xyn = result.keypoints.xyn[0].cpu().numpy()
                    kpts_conf = result.keypoints.conf[0].cpu().numpy()

                    # 2. Mid Hip Translation
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

                    translated_kpts = np.zeros((17, 2), dtype=np.float32)

                    for i in range(17):
                        if kpts_conf[i] >= 0.5:
                            translated_kpts[i, 0] = kpts_xyn[i, 0] - mid_hip_x
                            translated_kpts[i, 1] = kpts_xyn[i, 1] - mid_hip_y
                        else:
                            translated_kpts[i, 0] = 0.0
                            translated_kpts[i, 1] = 0.0

                    raw_sequences.append(translated_kpts.flatten())

                batch_frames = []

        cap.release()
        pbar.close()

        # 3.
        print(f"[*] Slicing {video_name} timeline into 30-frame chunks...")
        for i in range(0, len(raw_sequences) - T_MAX, T_MAX):
            window = np.array(raw_sequences[i: i + T_MAX], dtype=np.float32)

            # Matrix Transpose: (Time, Channels) -> (Channels, Time) -> (34, 30)
            window = window.transpose()
            all_idle_features.append(window)

    # 4. Check if we extracted any sequences
    if len(all_idle_features) == 0:
        print("\n[!] No sequences extracted from any of the videos.")
        return

    # 5. Materialize and Save Tensors
    final_features_np = np.stack(all_idle_features, axis=0)
    final_labels_np = np.zeros(len(all_idle_features), dtype=np.longlong)  # Class 0 = Idle

    X_idle = torch.tensor(final_features_np, dtype=torch.float32).contiguous()
    y_idle = torch.tensor(final_labels_np, dtype=torch.long)

    print(f"\n[*] Success! Extracted a total of {len(all_idle_features)} pure Idle sequences across all videos.")
    torch.save(X_idle, 'datasets/processed_X_idle.pt')
    torch.save(y_idle, 'datasets/processed_y_idle.pt')


if __name__ == "__main__":
    extract_dedicated_idle_videos()