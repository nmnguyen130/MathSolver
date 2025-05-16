from typing import Dict, List
import torch
from torch.utils.data import Dataset

from src.mathsolver.preprocessing.tokenizer import MathTokenizer

class MathDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: MathTokenizer, max_length: int = 512):
        self.data = data
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        latex_equation = sample.get("latex_equation", "")
        query = sample.get("query", "")
        solution_steps = sample.get("solution_steps", [])
        answer = sample.get("answer", "")

        input_ids, target_ids = self.tokenizer.encode(
            latex_equation=latex_equation,
            query=query,
            solution_steps=solution_steps,
            answer=answer
        )

        input_ids = input_ids[:self.max_length]
        target_ids = target_ids[:self.max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
        }