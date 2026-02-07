from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

# this class loads the ASL dataset from a CSV file
class ASLDataSet(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        # First column is label
        self.labels = self.data.iloc[:, 0].values.astype(np.int64)

        # Remaining columns are pixels
        self.images = self.data.iloc[:, 1:].values.astype(np.float32).reshape(-1, 28, 28)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # convert numpy array to PIL image
        img = Image.fromarray((img * 255).astype(np.uint8), mode='L') # Grayscale PIL image

        if self.transform:
            img = self.transform(img)

        return img, label