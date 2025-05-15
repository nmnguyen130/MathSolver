import torch
import torch.nn.functional as F
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.mathwriting.datamodule.dataset import MathWritingDataset
from src.mathwriting.datamodule.transforms import get_train_transform, get_val_test_transform
from src.mathwriting.preprocessing.tokenizer import LaTeXTokenizer

class MathWritingDataManager:
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        img_size: tuple = (224, 224),
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.img_size = img_size

        self.train_dir = self.data_dir / "train_image"
        self.valid_dir = self.data_dir / "valid_image"
        self.test_dir = self.data_dir / "test_image"

        # Check for file existence early
        self._check_data_dirs()

        self.mean = 0.7931
        self.std = 0.1738

        self.train_transform = get_train_transform(self.img_size, self.mean, self.std)
        self.val_test_transform = get_val_test_transform(self.img_size, self.mean, self.std)

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

    def collate_fn(self, batch):
        images, latex_labels = zip(*batch)

        # Stack images
        src = torch.stack(images)

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