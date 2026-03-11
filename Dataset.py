import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class PoseActionDataset(Dataset):
    def __init__(self, csv_dir, max_frames=30, overlap=15):
        self.csv_dir = csv_dir
        self.max_frames = max_frames
        self.step_size = max_frames - overlap  # How many frames to skip before the next window
        self.file_names = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        self.cached_data = []

        print(f"Preloading {len(self.file_names)} CSVs and generating sliding windows...")

        for file_name in self.file_names:
            file_path = os.path.join(self.csv_dir, file_name)

            try:
                df = pd.read_csv(file_path)
                if df.empty or len(df) < 5:  # Skip empty or tiny files
                    continue

                label = int(df['label'].iloc[0])
                features_df = df.drop(columns=['frame_number', 'label'], errors='ignore')
                features = features_df.values  # Shape: (total_frames, 51)

                total_frames = features.shape[0]

                # --- THE SLICING LOGIC ---
                if total_frames < self.max_frames:
                    # If the whole video is shorter than 30 frames, pad it once
                    padding = np.zeros((self.max_frames - total_frames, features.shape[1]))
                    window = np.vstack((features, padding))
                    self._add_to_cache(window, label)
                else:
                    # Slide the window across the video to get multiple samples
                    for start in range(0, total_frames - self.max_frames + 1, self.step_size):
                        window = features[start: start + self.max_frames, :]
                        self._add_to_cache(window, label)

            except Exception as e:
                print(f"⚠️ Error processing {file_name}: {e}")

        print(f"Created {len(self.cached_data)} total samples from {len(self.file_names)} files!")

    def _add_to_cache(self, window, label):
        features_tensor = torch.tensor(window, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        self.cached_data.append((features_tensor, label_tensor))

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        return self.cached_data[idx]