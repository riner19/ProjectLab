import cv2
import csv
import os
from ultralytics import YOLO

# 1. Initializing the YOLO model
print("Loading YOLOv8 Nano Pose Model...")
model = YOLO('yolov8n-pose.pt')

# 2. Define our classes and folders
categories = {'idle': 0, 'punch': 1}
base_input_folder = 'dataset_videos'
output_folder = 'dataset_csvs'

os.makedirs(output_folder, exist_ok=True)

# 3. Setup the CSV Header (17 joints: x, y, conf + 1 Label)
header = ['frame_number']
for i in range(17):
    header.extend([f'x{i}', f'y{i}', f'conf{i}'])
header.append('label')

# 4. Process both folders
for category_name, label_value in categories.items():
    folder_path = os.path.join(base_input_folder, category_name)

    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} not found. Skipping.")
        continue

    video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
    print(f"\nProcessing {len(video_files)} videos in '{category_name}' (Label: {label_value})...")

    for video_name in video_files:
        video_path = os.path.join(folder_path, video_name)
        csv_name = f"{category_name}_{video_name.replace('.avi', '.csv')}"
        csv_path = os.path.join(output_folder, csv_name)

        cap = cv2.VideoCapture(video_path)

        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)

                # If a person is found, extract their coordinates
                if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                    kpts = results[0].keypoints.data[0].cpu().numpy()
                    row = [frame_idx]
                    for kp in kpts:
                        row.extend([kp[0], kp[1], kp[2]])
                    row.append(label_value)
                    writer.writerow(row)
                else:
                    # If no person is found in this frame, fill with zeros to keep the timeline stable
                    row = [frame_idx] + [0.0] * 51
                    row.append(label_value)
                    writer.writerow(row)

                frame_idx += 1

        cap.release()
        print(f"   ✅ Saved: {csv_name} ({frame_idx} frames)")

print("\nBATCH EXTRACTION COMPLETE! All 100 CSVs are ready in the 'dataset_csvs' folder.")