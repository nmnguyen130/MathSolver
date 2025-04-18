import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

from src.shared.preprocessing.solution_tokenizer import SolutionTokenizer

class MathDataset(Dataset):
    def __init__(self, json_file: str, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = SolutionTokenizer()
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        all_texts = []
        for sample in self.data:
            all_texts.append(sample['latex_equation'])
            all_texts.append(sample['query'])
            all_texts.extend(sample['solution_steps'])
        self.tokenizer.build_vocab(all_texts)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        input_text = f"{sample['latex_equation']} [SEP] {sample['query']}"
        target_text = " ".join(sample['solution_steps'])
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length)
        target_ids = self.tokenizer.encode(target_text, max_length=self.max_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        input_attention_mask = (input_ids != self.tokenizer.token_to_idx["<pad>"]).long()
        target_attention_mask = (target_ids != self.tokenizer.token_to_idx["<pad>"]).long()
        return {
            'input_ids': input_ids,
            'attention_mask': input_attention_mask,
            'target_ids': target_ids,
            'target_attention_mask': target_attention_mask
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    target_attention_mask = torch.stack([item['target_attention_mask'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'target_ids': target_ids,
        'target_attention_mask': target_attention_mask
    }

def get_dataloader(json_file: str, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    dataset = MathDataset(json_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader

if __name__ == "__main__":
    json_file = "data/mathsolver/math_dataset.json"
    dataloader = get_dataloader(json_file, batch_size=4, shuffle=True, num_workers=0)
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Target IDs shape:", batch['target_ids'].shape)
        print("Sample input IDs:", batch['input_ids'][0][:10])
        print("Sample target IDs:", batch['target_ids'][0][:10])
        dataset = MathDataset(json_file)
        sample = dataset[0]
        input_text = dataset.tokenizer.decode(sample['input_ids'].tolist())
        target_text = dataset.tokenizer.decode(sample['target_ids'].tolist())
        print("Sample input decoded:", input_text)
        print("Sample target decoded:", target_text)
        break