import torch
import torch.nn.functional as F
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

from src.mathwriting.datamodule.dataset import MathWritingDataset
from src.mathwriting.preprocessing.tokenizer import LaTeXTokenizer

class MathWritingDataManager:
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

        self.train_dir = self.data_dir / "train_image"
        self.valid_dir = self.data_dir / "valid_image"
        self.test_dir = self.data_dir / "test_image"

        # Check for file existence early
        self._check_data_dirs()

        # Define image transformations
        self.train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=1,
                scale=(0.85, 1.0),
                translate=(0, 0),
                interpolation=transforms.InterpolationMode.BICUBIC,
                fill=255
            ),  # shift=0, rotate=±1°, scale=0.85~1
            transforms.ColorJitter(brightness=0.05, contrast=0.2),  # brightness ~5%, contrast ~[-20%, 0%]
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))], p=0.2),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738))
        ])

        self.val_test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738))
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
            if not (d / "labels.txt").exists():
                raise FileNotFoundError(f"labels.txt not found in {d}")
            
    def _load_labels(self, label_file: Path) -> list[str]:
        labels = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                _, label = line.strip().split("\t")
                labels.append(label)
        return labels

    def _setup(self):
        print("Setting up tokenizer...")
        train_labels = self._load_labels(self.train_dir / "labels.txt")
        self.tokenizer = LaTeXTokenizer()
        self.tokenizer.build_vocab(train_labels)
        self.vocab_size = len(self.tokenizer.vocab)
        print(f"Tokenizer built with vocab size: {self.vocab_size}")

        self.train_dataset = self._setup_dataset(self.train_dir, self.train_transform)
        self.val_dataset = self._setup_dataset(self.valid_dir, self.val_test_transform)
        if (self.test_dir / "labels.txt").exists():
            self.test_dataset = self._setup_dataset(self.test_dir, self.val_test_transform)

    def _setup_dataset(self, folder: Path, transform):
        dataset = MathWritingDataset(
            image_dir=folder / "images",
            label_file=folder / "labels.txt",
            tokenizer=self.tokenizer,
            transform=transform
        )
        print(f"{folder.stem.capitalize()} samples: {len(dataset)}")
        return dataset

    def collate_fn(self, batch, img_size=(224, 224)):
        images, latex_labels = zip(*batch)
        max_width, max_height = img_size

        # Create white background
        bg_value = 1.0
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