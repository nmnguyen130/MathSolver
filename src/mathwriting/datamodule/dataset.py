from pathlib import Path
from PIL import Image
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from src.mathwriting.preprocessing.tokenizer import LaTeXTokenizer

class MathWritingDataset(Dataset):
    """
    Dataset class for images and LaTeX labels rendered beforehand.
    Expects a folder with PNG images and a labels.txt file.
    """
    def __init__(self, image_dir: Path, label_file: Path, tokenizer: LaTeXTokenizer, transform=None):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.samples = self._load_samples(label_file)

    def _load_samples(self, label_file: Path):
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")

        samples = []
        with open(label_file, "r", encoding="utf-8") as f:
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

        label_tensor = Tensor(self.tokenizer.encode(latex_label)).long()
        return image, label_tensor