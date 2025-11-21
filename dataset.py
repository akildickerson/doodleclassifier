import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class DoodleDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform or transforms.Resize((28, 28))
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = image.float() / 255
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        return image, label
