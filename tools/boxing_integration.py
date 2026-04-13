import os
import pandas as pd
import numpy as np

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SKELETON_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'Skeleton_data')
ANNOTATION_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'Annotation_files')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_sequences')
OUTPUT_X = os.path.join(OUTPUT_DIR, 'boxvi_X.npy')
OUTPUT_Y = os.path.join(OUTPUT_DIR, 'boxvi_y.npy')

# --- CLASS MAP ---
CLASS_MAP = {
    'jab': 1, 'cross': 2,
    'lead_hook': 3, 'hook': 3, 'leadhook': 3,
    'rear_hook': 4, 'rearhook': 4,
    'lead_uppercut': 5, 'uppercut': 5, 'leadupper': 5,
    'rear_uppercut': 6, 'rearupper': 6
}


def normalize_sequence(punch_clip):
    """
    Normalizes a single pre-cut clip:

    """
    normalized_punch = []

    for frame in punch_clip:
        f_flat = np.array(frame).flatten()

        # Ensure minimum joint data is present
        if len(f_flat) >= 51:
            f_safe = f_flat[:51].reshape(17, 3)
        elif len(f_flat) >= 34:
            f_safe = f_flat[:34].reshape(17, 2)
        else:
            continue

            # Calculate biomechanical anchors
        mid_hip = (f_safe[11][:2] + f_safe[12][:2]) / 2.0
        shld_center = (f_safe[5][:2] + f_safe[6][:2]) / 2.0
        torso_size = float(np.linalg.norm(shld_center - mid_hip) + 1e-6)

        frame_feats = []
        for joint in f_safe:
            nx = (joint[0] - mid_hip[0]) / torso_size
            ny = (joint[1] - mid_hip[1]) / torso_size
            # Binary confidence to match dataset standard
            nv = 1.0 if (len(joint) > 2 and float(joint[2]) > 0.4) else 0.0
            frame_feats.extend([nx, ny, nv])

        normalized_punch.append(frame_feats)

    # Reject completely corrupted clips
    if not normalized_punch:
        return None

    # Standardize length to exactly 25 frames
    if len(normalized_punch) > 25:
        normalized_punch = normalized_punch[:25]
    else:
        # Edge Padding: repeat the last valid position
        last_valid = normalized_punch[-1]
        while len(normalized_punch) < 25:
            normalized_punch.append(last_valid)

    # velocity
    enhanced_punch = []
    for i in range(25):
        current_pos = normalized_punch[i]
        if i == 0:
            velocity = [0.0] * 51
        else:
            velocity = [current_pos[j] - normalized_punch[i - 1][j] for j in range(51)]

        # 51 Positions + 51 Velocities = 102 Features
        enhanced_punch.append(current_pos + velocity)

    return np.array(enhanced_punch, dtype=np.float32)


def process_boxing_vi():
    all_sequences, all_labels = [], []

    if not os.path.exists(SKELETON_DIR):
        print(f"Error: {SKELETON_DIR} not found.")
        return

    video_files = [f for f in os.listdir(SKELETON_DIR) if f.endswith('.npy')]

    for v_file in video_files:
        v_id = v_file.split('.')[0]
        xlsx_path = os.path.join(ANNOTATION_DIR, f"{v_id}.xlsx")
        npy_path = os.path.join(SKELETON_DIR, v_file)

        if not os.path.exists(xlsx_path):
            continue

        # Load Annotations
        df = pd.read_excel(xlsx_path)
        df.columns = [str(c).strip() for c in df.columns]

        # Load Pre-cut Clips
        raw_data = np.load(npy_path, allow_pickle=True)
        raw_data = np.squeeze(raw_data)

        class_col_idx = 2 if len(df.columns) > 2 else -1

        print(f"Processing: {v_id} | .npy clips: {len(raw_data)} | Excel rows: {len(df)}")

        video_count = 0
        dropped_shape = 0
        dropped_label = 0

        # Direct 1-to-1 Mapping Iteration
        for idx, row in df.iterrows():
            try:
                # Failsafe: Don't read past the available .npy clips
                if idx >= len(raw_data):
                    break

                # Extract Class Label
                label_raw = str(row.iloc[class_col_idx])
                label_str = label_raw.lower().strip().replace(' ', '_')
                final_label = CLASS_MAP.get(label_str)

                if final_label is None:
                    dropped_label += 1
                    continue

                # Pull the exact pre-cut clip directly by index
                punch_clip = raw_data[idx]

                # Normalize and Add Velocity
                processed_arr = normalize_sequence(punch_clip)

                # Ensure it is a perfect 102-dimension tensor
                if processed_arr is not None and processed_arr.shape == (25, 102):
                    all_sequences.append(processed_arr)
                    all_labels.append(final_label)
                    video_count += 1
                else:
                    dropped_shape += 1

            except Exception as e:
                dropped_shape += 1
                continue

        print(
            f"Integrated: {video_count} | Dropped (Shape): {dropped_shape} | Dropped (Label): {dropped_label}")

    # Save to Disk
    if all_sequences:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        X_final = np.array(all_sequences, dtype=np.float32)
        y_final = np.array(all_labels, dtype=np.int64)
        np.save(OUTPUT_X, X_final)
        np.save(OUTPUT_Y, y_final)
        print(f"\nSUCCESS: Saved {len(X_final)} true clips to {OUTPUT_X}")
        print(f"Final Tensor Shape: X = {X_final.shape}, y = {y_final.shape}")
    else:
        print("\nNo valid sequences found. Check your folder paths and Excel format.")


if __name__ == "__main__":
    process_boxing_vi()