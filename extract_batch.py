import cv2
import os
import pandas as pd
import torch
from ultralytics import YOLO

print(f"CUDA Available: {torch.cuda.is_available()}")

print("Loading YOLOv8 Pose Model...")
yolo_model = YOLO('yolov8n-pose.pt')

if torch.cuda.is_available():
    yolo_model.to('cuda')
    print("YOLO successfully moved to GPU!")
else:
    print("WARNING: YOLO is stuck on CPU!")

# Make sure these folders match your actual setup
VIDEO_DIR = "raw_videos"
OUTPUT_CSV_DIR = "custom_dataset_new"

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
CLASSES = {"idle": 0, "punch": 1}
total_videos_processed = 0

for class_name, label_value in CLASSES.items():
    class_folder = os.path.join(VIDEO_DIR, class_name)

    if not os.path.exists(class_folder):
        print(f"Warning: Folder '{class_folder}' not found. Skipping...")
        continue

    video_files = [f for f in os.listdir(class_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    print(f"\nFound {len(video_files)} videos in '{class_name}' folder.")

    for video_file in video_files:
        video_path = os.path.join(class_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        # Get video dimensions for normalization
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        csv_data = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame, device=0, verbose=False)
            row_data = {'frame_number': frame_idx, 'label': label_value}

            # Check if any person is detected
            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:

                # --- FIX 1: Find the main person (largest bounding box) ---
                boxes = results[0].boxes.xywh.cpu().numpy()  # x, y, width, height
                areas = boxes[:, 2] * boxes[:, 3]  # width * height
                main_person_idx = areas.argmax()  # index of the largest box

                # Extract keypoints for the main person only
                kpts = results[0].keypoints.data[main_person_idx].cpu().numpy()

                for j, kp in enumerate(kpts):
                    # --- FIX 2: Normalize coordinates ---
                    row_data[f'kp_{j}_x'] = kp[0] / frame_width if frame_width > 0 else 0
                    row_data[f'kp_{j}_y'] = kp[1] / frame_height if frame_height > 0 else 0
                    row_data[f'kp_{j}_conf'] = kp[2]
            else:
                # Fallback: fill with zeros if YOLO loses the person
                for j in range(17):
                    row_data[f'kp_{j}_x'] = 0.0
                    row_data[f'kp_{j}_y'] = 0.0
                    row_data[f'kp_{j}_conf'] = 0.0

            csv_data.append(row_data)
            frame_idx += 1

        cap.release()

        if csv_data:
            csv_filename = f"{class_name}_{video_file.split('.')[0]}.csv"
            csv_path = os.path.join(OUTPUT_CSV_DIR, csv_filename)

            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            total_videos_processed += 1
            print(f"Extracted: {csv_filename} ({frame_idx} frames)")

print(f"\nBatch extraction complete! {total_videos_processed} CSVs saved to '{OUTPUT_CSV_DIR}'.")