import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read([1, 2, 3, 4, 5, 6])  # Read all 6 bands
        image = np.transpose(image, (1, 2, 0))  # Change from (bands, height, width) to (height, width, bands)

        with rasterio.open(self.label_paths[idx]) as src:
            label = src.read(1)  # Assuming labels are in the first band

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return sorted(file_paths)  # Ensure consistent ordering

# Example transform that can be used when creating the dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
