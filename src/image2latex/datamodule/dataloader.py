import torch
import torch.nn.functional as F
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.image2latex.datamodule.dataset import ImageLatexDataset
from src.image2latex.preprocessing.tokenizer import LaTeXTokenizer

class ImageLatexDataManager:
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dir = self.data_dir / "train"
        self.valid_dir = self.data_dir / "valid"
        self.test_dir = self.data_dir / "test"

        # Check for file existence early
        self._check_data_dirs()

        self.train_transform = A.Compose([
            A.Compose([
                A.Affine(translate_percent=0, scale=(0.85, 1.0), rotate=1, border_mode=0,
                         interpolation=3, fill=(255, 255, 255), p=1),
                A.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                                 fill=(255, 255, 255), p=0.5),
            ], p=1),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            A.GaussNoise(std_range=(0.0392, 0.0392), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=(-0.2, 0), p=0.2),
            A.ImageCompression(quality_range=(90, 95), p=0.2),
            A.ToGray(p=1.0),
            A.Normalize(mean=(0.7931, 0.7931, 0.7931), std=(0.1738, 0.1738, 0.1738)),
            ToTensorV2()
        ])

        self.val_test_transform = A.Compose([
            A.ToGray(p=1.0),
            A.Normalize(mean=(0.7931, 0.7931, 0.7931), std=(0.1738, 0.1738, 0.1738)),
            ToTensorV2()
        ])

        # Initialize tokenizer and datasets
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.vocab_size = None
        self._setup() # Call setup during initialization

    def _check_data_dirs(self):
        for d in [self.train_dir, self.valid_dir]:
            if not (d / "latex.txt").exists():
                raise FileNotFoundError(f"latex.txt not found in {d}")

    def _load_labels(self, label_file: Path) -> list[str]:
        labels = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                _, label = line.strip().split("\t")
                labels.append(label)
        return labels

    def _setup(self):
        print("Setting up tokenizer...")
        train_labels = self._load_labels(self.train_dir / "latex.txt")
        self.tokenizer = LaTeXTokenizer()
        self.tokenizer.build_vocab(train_labels)
        self.vocab_size = len(self.tokenizer.vocab)
        print(f"Tokenizer built with vocab size: {self.vocab_size}")

        self.train_dataset = self._setup_dataset(self.train_dir, self.train_transform)
        self.val_dataset = self._setup_dataset(self.valid_dir, self.val_test_transform)
        if (self.test_dir / "latex.txt").exists():
            self.test_dataset = self._setup_dataset(self.test_dir, self.val_test_transform)

    def _setup_dataset(self, folder_dir: Path, transform):
        dataset = ImageLatexDataset(
            image_dir=folder_dir / "images",
            latex_file=folder_dir / "latex.txt",
            tokenizer=self.tokenizer,
            transform=transform
        )
        print(f"{folder_dir.stem.capitalize()} samples: {len(dataset)}")
        return dataset

    def collate_fn(self, batch, img_size=(224, 224)):
        images, latex_labels = zip(*batch)
        max_width, max_height = img_size

        # Create white background
        mean = 0.7931
        std = 0.1738
        bg_value = (1.0 - mean) / std
        src = torch.full((len(images), images[0].size(0), max_height, max_width),
                        bg_value, dtype=images[0].dtype, device=images[0].device)

        # Center and pad individual images
        for i, img in enumerate(images):
            _, img_h, img_w = img.size()

            # Chỉ resize nếu ảnh lớn hơn max_height hoặc max_width
            if img_h > max_height or img_w > max_width:
                scale = min(max_height / img_h, max_width / img_w)
                new_h = int(img_h * scale)
                new_w = int(img_w * scale)
                img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
            else:
                new_h, new_w = img_h, img_w  # Giữ nguyên kích thước nếu ảnh nhỏ

            # Center và pad ảnh
            pad_h_start = (max_height - new_h) // 2
            pad_w_start = (max_width - new_w) // 2
            pad_h_end = pad_h_start + new_h
            pad_w_end = pad_w_start + new_w
            src[i, :, pad_h_start:pad_h_end, pad_w_start:pad_w_end] = img

        # Pad label sequences
        pad_id = self.tokenizer.token_to_idx['<pad>']
        tgt = pad_sequence(latex_labels, batch_first=True, padding_value=pad_id).long()

        return src, tgt
    
    def get_dataloader(self, dataset_type: str):
        dataset_map = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset if hasattr(self, 'test_dataset') else None,
        }

        dataset = dataset_map.get(dataset_type)
        if not dataset:
            raise ValueError(f"Invalid dataset type: {dataset_type}.")

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=(dataset_type == 'train')
        )
    
if __name__ == "__main__":
    dataloader = ImageLatexDataManager(data_dir='./data/im2latex', batch_size=1)
    train_dataloader = dataloader.get_dataloader("train")
    for i, batch in enumerate(train_dataloader):
        src, tgt = batch
        print(src.shape, tgt.shape)
        break