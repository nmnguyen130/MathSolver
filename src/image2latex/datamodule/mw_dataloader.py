import torch
import torch.nn.functional as F
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

from src.image2latex.datamodule.dataset import ImageLatexDataset
from src.image2latex.preprocessing.tokenizer import LaTeXTokenizer

class MWImageLatexDataManager:
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        split_ratio: float = 0.9,  # tỷ lệ train
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.split_ratio = split_ratio
        self.seed = seed

        self.image_dir = self.data_dir / "images"
        self.latex_file = self.data_dir / "latex.txt"

        self._check_data_files()

        # Define image transformations
        self.train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=1,
                scale=(0.85, 1.0),
                translate=(0, 0),
                interpolation=transforms.InterpolationMode.BICUBIC,
                fill=255
            ),
            transforms.ColorJitter(brightness=0.05, contrast=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))], p=0.2),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738))
        ])

        self.val_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738))
        ])

        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.vocab_size = None

        self._setup()

    def _check_data_files(self):
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.latex_file.exists():
            raise FileNotFoundError(f"latex.txt not found: {self.latex_file}")

    def _load_labels(self, label_file: Path) -> list[str]:
        labels = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                _, label = line.strip().split("\t")
                labels.append(label)
        return labels

    def _setup(self):
        print("Setting up tokenizer...")
        all_labels = self._load_labels(self.latex_file)
        vocab_path = Path("src/image2latex/checkpoints/vocab.txt")
        self.tokenizer = LaTeXTokenizer(vocab_file=vocab_path)
        # self.tokenizer.build_vocab(all_labels)
        self.vocab_size = len(self.tokenizer.vocab)
        print(f"Tokenizer built with vocab size: {self.vocab_size}")

        full_dataset = ImageLatexDataset(
            image_dir=self.image_dir,
            latex_file=self.latex_file,
            tokenizer=self.tokenizer,
            transform=self.train_transform  # will override later for val
        )

        train_size = int(len(full_dataset) * self.split_ratio)
        val_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(self.seed)
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

        # Override transform for validation subset
        val_subset.dataset.transform = self.val_transform

        self.train_dataset = train_subset
        self.val_dataset = val_subset

        print(f"Total samples: {len(full_dataset)}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")

    def collate_fn(self, batch, img_size=(224, 224)):
        images, latex_labels = zip(*batch)
        max_width, max_height = img_size

        bg_value = 1.0
        src = torch.full((len(images), images[0].size(0), max_height, max_width),
                         bg_value, dtype=images[0].dtype, device=images[0].device)

        for i, img in enumerate(images):
            _, img_h, img_w = img.size()

            if img_h > max_height or img_w > max_width:
                scale = min(max_height / img_h, max_width / img_w)
                new_h = int(img_h * scale)
                new_w = int(img_w * scale)
                img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
            else:
                new_h, new_w = img_h, img_w

            pad_h_start = (max_height - new_h) // 2
            pad_w_start = (max_width - new_w) // 2
            pad_h_end = pad_h_start + new_h
            pad_w_end = pad_w_start + new_w
            src[i, :, pad_h_start:pad_h_end, pad_w_start:pad_w_end] = img

        pad_id = self.tokenizer.token_to_idx['<pad>']
        tgt = pad_sequence(latex_labels, batch_first=True, padding_value=pad_id).long()

        return src, tgt

    def get_dataloader(self, dataset_type: str):
        dataset_map = {
            "train": self.train_dataset,
            "val": self.val_dataset,
        }

        dataset = dataset_map.get(dataset_type)
        if dataset is None:
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
    dataloader = MWImageLatexDataManager(data_dir='./data/CROMHE', batch_size=1)
    train_dataloader = dataloader.get_dataloader("train")
    for i, batch in enumerate(train_dataloader):
        src, tgt = batch
        print(src.shape, tgt.shape)