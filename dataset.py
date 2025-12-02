from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
import glob

class CT_Dataset(Dataset):
    def __init__(self, low_paths, high_paths):
        self.low_paths = low_paths
        self.high_paths = high_paths

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low = cv2.imread(self.low_paths[idx], cv2.IMREAD_UNCHANGED)
        high = cv2.imread(self.high_paths[idx], cv2.IMREAD_UNCHANGED)

        low = (low.astype(np.float32)/2048.0) *2 - 1
        high = (high.astype(np.float32)/2048.0) *2 - 1

        return torch.tensor(low)[None], torch.tensor(high)[None]
