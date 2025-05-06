import os
from typing import List, Dict
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer

from src.mathsolver.preprocessing.tokenizer import MathTokenizer
from src.mathsolver.datamodule.dataset import MathDataset

class MathSolverDataManager:
    def __init__(
        self,
        json_file: str,
        batch_size: int = 16,
        max_length: int = 512,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        self.json_file = json_file
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self._check_data_dirs()
        self._load_data()
        self._split_data()

    def _check_data_dirs(self):
        if not os.path.exists(self.json_file):
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")

    def _load_data(self):
        self.tokenizer = MathTokenizer()
        self.tokenizer.build_vocab(self.json_file)
        self.vocab_size = len(self.tokenizer.vocab)

        self.dataset = MathDataset(self.json_file, self.tokenizer, self.max_length)

    def _split_data(self, val_size: float = 0.2) -> None:
        dataset_size = len(self.dataset)
        val_size = int(val_size * dataset_size)
        train_size = dataset_size - val_size

        torch.manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.token_to_idx['<pad>']
        # Extract components
        input_ids = [item['input_ids'] for item in batch]
        target_ids = [item['target_ids'] for item in batch]

        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        target_ids = pad_sequence(target_ids, batch_first=True, padding_value=pad_id)

        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
        }

    def get_dataloader(self, type: str) -> DataLoader:
        dataset_map = {
            "train": self.train_dataset,
            "val": self.val_dataset,
        }

        dataset = dataset_map.get(type)
        if not dataset:
            raise ValueError(f"Invalid dataset type: {type}.")

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if type == 'train' else False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory
        )

    def decode(self, tensor: torch.Tensor) -> List[str]:
        decoded_labels = []
        for t in tensor:
            decoded_label = self.tokenizer.decode(t.tolist())  # Chuyển tensor thành list rồi decode
            decoded_labels.append(decoded_label)
        return decoded_labels
    
if __name__ == '__main__':
    dataloader = MathSolverDataManager(json_file='./data/mathsolver/math_dataset.json', batch_size=1, max_length=512)
    train_loader = dataloader.get_dataloader('train')

    for batch in train_loader:
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        
        input_decode = dataloader.decode(input_ids)
        target_decode = dataloader.decode(target_ids)
        
        print("Decoded Source IDs:")
        for src in input_decode:
            print(src)

        print("Decoded Target IDs:")
        for tgt in target_decode:
            print(tgt)

        break