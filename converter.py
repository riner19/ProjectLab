import pandas as pd
import os
import glob

# --- Configuration ---
CSV_DIR = "datasets/Annotation_files_2"
OUTPUT_DIR = "datasets/Annotations_30fps"

fps_map = {
    "V1_with_guard": 24.0,
    "V2_with_guard": 23.98,
    "V3_with_guard": 23.98,
    "V4_with_guard": 23.98,
    "V5_with_guard": 29.97,
    "V6_with_guard": 23.98,
    "V7_with_guard": 29.97,
    "V8_with_guard": 29.97,
    "V9_with_guard": 23.97,
    "V10_with_guard": 25

}

TARGET_FPS = 30.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))

for file_path in csv_files:
    base_name = os.path.splitext(os.path.basename(file_path))[0]


    original_fps = fps_map.get(base_name, 30.0)


    ratio = TARGET_FPS / original_fps


    df = pd.read_csv(file_path)


    df['Start_Frame'] = (df['Start_Frame'] * ratio).round().astype(int)
    df['Ending_Frame'] = (df['Ending_Frame'] * ratio).round().astype(int)


    save_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")
    df.to_csv(save_path, index=False)

    print(f"Converted {base_name}: {original_fps}fps -> {TARGET_FPS}fps (Ratio: {ratio:.3f})")

print("All CSVs successfully scaled!")