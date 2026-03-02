import torch
from torch.utils.data import Dataset
import pandas as pd
import os


class PoseActionDataset(Dataset):
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.file_names = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # 1. Read the CSV file
        file_path = os.path.join(self.csv_dir, self.file_names[idx])
        df = pd.read_csv(file_path)

        # 2. Separate features (the 51 coordinates) and the label
        #  to keep only the X, Y, Conf math
        features = df.drop(columns=['frame_number', 'label']).values
        label = df['label'].iloc[0]  # The label is the same for the whole video

        # 3. Convert to PyTorch Tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return features_tensor, label_tensor