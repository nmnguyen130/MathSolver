import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import numpy as np

class SymbolDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # Create label mapping
        unique_labels = sorted(set(label for _, label in data))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        strokes, label = self.data[idx]
        
        # Convert multi-stroke data into a single sequence
        stroke_sequence = []
        for stroke in strokes:
            # Convert numpy array points to [x, y, pen_up]
            for i in range(len(stroke)):
                # Add pen_down (0) for all points except the last one
                stroke_sequence.append([float(stroke[i][0]), float(stroke[i][1]), 0])
            if len(stroke) > 0:  # Check length instead of truth value
                stroke_sequence[-1][2] = 1  # Set last point's pen_up to 1
        
        # Handle empty strokes case
        if len(stroke_sequence) == 0:
            stroke_sequence = [[0.0, 0.0, 1.0]]  # Add a dummy point if no strokes
            
        # Convert to tensor
        strokes_tensor = torch.tensor(stroke_sequence, dtype=torch.float32)
        label_idx = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        
        return strokes_tensor, label_idx
    
    def collate_fn(self, batch):
        # Separate strokes and labels
        strokes, labels = zip(*batch)
        
        # Pack the variable length sequences
        strokes_packed = rnn_utils.pad_sequence(strokes, batch_first=True)
        labels = torch.stack(labels)
        
        return strokes_packed, labels
    
    def get_num_classes(self):
        return len(self.label_to_idx)