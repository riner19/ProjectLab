import torch
from torch.utils.data import Dataset
import numpy as np


class PoseDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.longlong)

        # Simple Normalization: Ensure no extreme outliers
        # (Though your preprocess script already did torso-scaling)
        self.x = np.nan_to_num(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(np.array(self.y[idx]))