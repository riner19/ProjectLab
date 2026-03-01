import kagglehub
import os
import shutil

# 1. Download the Dataset
print("Downloading UCF101 via KaggleHub")
dataset_path = kagglehub.dataset_download("pevogam/ucf101")
print(f"Download complete. Raw files are at: {dataset_path}")

# 2. Setup our organized local folders
base_folder = 'dataset_videos'
punch_folder = os.path.join(base_folder, 'punch')
idle_folder = os.path.join(base_folder, 'idle')

os.makedirs(punch_folder, exist_ok=True)
os.makedirs(idle_folder, exist_ok=True)

# 3. Search and Copy exactly 50 of each
boxing_count = 0
walking_count = 0
LIMIT = 50

print(f"search for {LIMIT} Boxing and {LIMIT} Walking videos...")

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.avi'):
            # Found a Boxing video
            if 'Boxing' in file and boxing_count < LIMIT:
                source_path = os.path.join(root, file)
                dest_path = os.path.join(punch_folder, file)
                shutil.copy2(source_path, dest_path)
                boxing_count += 1

            # Found a Walking (Idle) video
            elif 'v_WalkingWithDog' in file and walking_count < LIMIT:
                source_path = os.path.join(root, file)
                dest_path = os.path.join(idle_folder, file)
                shutil.copy2(source_path, dest_path)
                walking_count += 1

        # Stop searching if we have enough of both
        if boxing_count >= LIMIT and walking_count >= LIMIT:
            break
    if boxing_count >= LIMIT and walking_count >= LIMIT:
        break

print(f"Copied {boxing_count} videos to '{punch_folder}'")
print(f"Copied {walking_count} videos to '{idle_folder}'")
