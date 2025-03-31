import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from shared.preprocessing.inkml_loader import InkMLDatasetLoader
from shared.preprocessing.latex_tokenizer import LatexTokenizer

class MathExpressionDataset(Dataset):
    def __init__(self, ink_folder, vocab, tokenizer, subset_size=None):
        self.ink_folder = ink_folder
        self.vocab = vocab
        self.tokenizer = tokenizer

        ink_loader = InkMLDatasetLoader(ink_folder)
        self.data = ink_loader.load_data()

        if subset_size is not None and subset_size < len(self.data):
            self.data = self.data[:subset_size]  # Take a subset of the data if needed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        strokes, label = self.data[idx]

        stroke_tensors = []
        for stroke in strokes:
            xy = stroke[:2, :].T   # Extract x and y coordinates (num_points, 2)
            s = np.zeros((xy.shape[0], 1))  # Initialize stroke start indicator
            if xy.shape[0] > 0:
                s[0, 0] = 1  # Set the first point of the stroke to have s=1
            stroke_tensor = np.hstack((xy, s))  # Combine coordinates and stroke indicator (num_points, 3)
            stroke_tensors.append(stroke_tensor)
        
        # Stack all stroke tensors into a single tensor
        if stroke_tensors:
            all_points = np.vstack(stroke_tensors)
        else:
            all_points = np.zeros((0, 3))  # Handle case with no strokes
        
        points_tensor = torch.tensor(all_points, dtype=torch.float32)

        # Process label with LatexTokenizer
        tokens = self.tokenizer.tokenize(label)
        token_ids = [self.vocab['<sos>']] + \
                    [self.vocab.get(token, self.vocab['<unk>']) for token in tokens] + \
                    [self.vocab['<eos>']]
        label_tensor = torch.tensor(token_ids, dtype=torch.long)

        return points_tensor, label_tensor

def collate_fn(batch, vocab):
    points, labels = zip(*batch)
    # Pad sequences to the maximum length in the batch
    points_padded = pad_sequence(points, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=vocab['<pad>'])
    return points_padded, labels_padded

def create_data_loaders(root_dir, batch_size=32):
    train_dir = os.path.join(root_dir, "train")
    valid_dir = os.path.join(root_dir, "valid")
    test_dir = os.path.join(root_dir, "test")
    
    tokenizer = LatexTokenizer()

    # Load training data
    train_loader_raw = InkMLDatasetLoader(train_dir)
    train_data = train_loader_raw.load_data()
    vocab = tokenizer.build_vocab(train_data, tokenizer)

    # Create datasets for training, validation, and testing
    train_dataset = MathExpressionDataset(train_dir, vocab, tokenizer)
    valid_dataset = MathExpressionDataset(valid_dir, vocab, tokenizer)
    test_dataset = MathExpressionDataset(test_dir, vocab, tokenizer)

    # Create data loaders for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, vocab))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, vocab))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, vocab))

    return train_loader, valid_loader, test_loader, vocab

# Usage example:
if __name__ == "__main__":
    root_dir = "data/mathwriting-2024-excerpt"
    train_loader, valid_loader, test_loader, vocab = create_data_loaders(root_dir)

    print(f"Kích thước từ điển: {len(vocab)}")

    # Example of iterating through the training loader
    for points, labels in train_loader:
        print(f"Kích thước batch points: {points.shape}")
        print(f"Kích thước batch labels: {labels.shape}")
        break