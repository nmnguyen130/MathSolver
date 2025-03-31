import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.shared.preprocessing.inkml_loader import InkMLLoader
from src.symbol_detection.models.dataset import SymbolDataset
from src.symbol_detection.models.model import SymbolDetectionModule

ROOT_PATH = 'data/mathwriting-2024-excerpt'

class Trainer():
    def __init__(self):
        pass

if __name__ == '__main__':
    train_loader = InkMLLoader(os.path.join(ROOT_PATH, 'symbols'))
    train_data = train_loader.load_data()
    
    # Print sample data information
    # sample_strokes, sample_label = train_data[1]
    # print(f"Strokes: {sample_strokes}. \nLabel: {sample_label}")
    
    train_dataset = SymbolDataset(train_data)
    print(f"Number of classes: {train_dataset.get_num_classes()}")
    
    # Get sample batch to check shapes
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn  # Add custom collate function
    )
    sample_batch_strokes, sample_batch_labels = next(iter(train_dataloader))
    print(f"Batch strokes shape: {sample_batch_strokes.shape}")
    print(f"Batch labels shape: {sample_batch_labels.shape}")
    
    model = SymbolDetectionModule(
        input_size=3,  # x, y, pen_up coordinates
        hidden_size=256,
        output_size=train_dataset.get_num_classes(),
        num_layers=2
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for strokes, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(strokes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")