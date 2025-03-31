import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

class DigitDataset(Dataset):
    def __init__(self, data_dir='data/numbers', cache_file='processed_data.npz', transform=None):
        self.cache_file = os.path.join(data_dir, cache_file)
        if not os.path.exists(self.cache_file):
            raise FileNotFoundError(f"Cached dataset not found at {self.cache_file}")
        
        cached_data = np.load(self.cache_file)
        self.images = cached_data['images']
        self.labels = cached_data['labels']
        self.transform = transform    
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

    @classmethod
    def create_dataloaders(cls, batch_size=32, train_split=0.8):
        dataset = cls()

        # Calculate split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_split, random_state=42)
        train_indices, val_indices = next(sss.split(dataset.images, dataset.labels))

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        train_dataset.dataset.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(40),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader
    
    @staticmethod
    def save_image(tensor_image, filepath, title="Image"):
        if tensor_image.ndim == 3:
            tensor_image = tensor_image.squeeze(0)
        plt.imshow(tensor_image.cpu().numpy(), cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.savefig(filepath)
        print(f"Image saved to {filepath}")