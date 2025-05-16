import json
import os
import random
from typing import List, Dict
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.mathsolver.preprocessing.tokenizer import MathTokenizer
from src.mathsolver.datamodule.dataset import MathDataset

class MathSolverDataManager:
    def __init__(
        self,
        json_folder: str,
        batch_size: int = 16,
        max_length: int = 512,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_ratio: float = 0.15
    ):
        self.json_folder = json_folder
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_ratio = val_ratio

        self._check_data_dirs()
        self._load_data_and_split()

    def _check_data_dirs(self):
        if not os.path.exists(self.json_folder):
            raise FileNotFoundError(f"Folder not found: {self.json_folder}")

        # Kiểm tra có ít nhất 1 file JSON trong folder
        json_files = [f for f in os.listdir(self.json_folder) if f.endswith('.json')]
        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON files found in folder: {self.json_folder}")
        self.json_files = [os.path.join(self.json_folder, f) for f in json_files]

    def _load_data_and_split(self):
        all_samples = []
        for json_file in self.json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_samples.extend(data)

        random.seed(42)
        random.shuffle(all_samples)

        total_size = len(all_samples)
        val_size = int(total_size * self.val_ratio)
        train_data = all_samples[val_size:]
        val_data = all_samples[:val_size]
        
        self.tokenizer = MathTokenizer()
        self.tokenizer.build_vocab(all_samples)
        self.vocab_size = len(self.tokenizer.vocab)

        self.train_dataset = MathDataset(train_data, self.tokenizer, self.max_length)
        self.val_dataset = MathDataset(val_data, self.tokenizer, self.max_length)

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
    dataloader = MathSolverDataManager(json_folder='./data/mathsolver', batch_size=1, max_length=512)
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