import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from src.image2latex.preprocessing.tokenizer import LaTeXTokenizer

class ImageLatexDataset(Dataset):
    def __init__(self, image_dir: Path, latex_file: Path, tokenizer: LaTeXTokenizer, transform=None):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.samples = self._load_samples(latex_file)

    def _load_samples(self, latex_file: Path):
        if not latex_file.exists():
            raise FileNotFoundError(f"Latex file not found: {latex_file}")

        samples = []
        with open(latex_file, "r", encoding="utf-8") as f:
            for line in f:
                image_name, label = line.strip().split("\t")
                samples.append((image_name, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, latex_label = self.samples[idx]
        image_path = self.image_dir / image_name
        
        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image at {image_path}: {e}")

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label_tensor = torch.tensor(self.tokenizer.encode(latex_label), dtype=torch.long)
        return image, label_tensor