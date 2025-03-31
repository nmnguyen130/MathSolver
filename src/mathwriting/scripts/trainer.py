import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.mathwriting.models.model import Seq2Seq
from mathwriting.models.dataset import create_data_loaders

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate()
            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    def train_one_epoch(self):
        """Huấn luyện mô hình qua một epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Training')
        for batch_idx, (points, labels) in progress_bar:
            points, labels = points.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            output = self.model(points, labels)

            output = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def evaluate(self):
        """Đánh giá mô hình trên tập validation."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for points, labels in tqdm(self.val_loader, desc='Evaluating'):
                points, labels = points.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                output = self.model(points, labels)

                output = output.view(-1, output.size(-1))
                labels = labels.view(-1)

                loss = self.criterion(output, labels)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Checkpoint saved at {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")

    def visualize_batch(self, batch_idx=0):
        """Trực quan hóa một batch dữ liệu để thấy đầu vào và đầu ra."""
        self.model.eval()
        # Get the vocabulary index to token mapping
        idx_to_token = {idx: token for token, idx in self.train_loader.dataset.vocab.items()}
        
        with torch.no_grad():
            for i, (points, labels) in enumerate(self.train_loader):
                if i == batch_idx:
                    points, labels = points.to(self.device), labels.to(self.device)
                    output = self.model(points, labels)
                    pred = output.argmax(dim=2)
                    
                    for j in range(min(2, points.size(0))):  # In 2 mẫu đầu tiên trong batch
                        print(f"\nMẫu {j}:")
                        print(f"  Đầu vào (points): shape = {points[j].shape}")
                        
                        # Convert label indices to tokens
                        true_tokens = [idx_to_token[idx.item()] for idx in labels[j] if idx.item() != 0]  # Skip padding
                        pred_tokens = [idx_to_token[idx.item()] for idx in pred[j] if idx.item() != 0]  # Skip padding
                        
                        print(f"  Nhãn thực tế: {' '.join(true_tokens)}")
                        print(f"  Dự đoán: {' '.join(pred_tokens)}")
                        
                        # Trực quan hóa nét vẽ
                        self._plot_strokes(points[j].cpu().numpy())
                    break

    def _plot_strokes(self, points):
        """Vẽ nét vẽ từ tensor points."""
        plt.figure(figsize=(10, 5))
        # Reshape points to get x, y coordinates
        points = points.reshape(-1, 3)  # Reshape to (n_points, 3) where each row is [x, y, stroke_end]
        
        # Split into separate strokes based on the stroke_end flag
        current_stroke = []
        for point in points:
            current_stroke.append(point[:2])  # Only take x, y coordinates
            if point[2] == 1:  # If it's end of stroke
                if current_stroke:
                    stroke_points = np.array(current_stroke)
                    plt.plot(stroke_points[:, 0], stroke_points[:, 1], 'b-')
                current_stroke = []
        
        # Plot any remaining points in the last stroke
        if current_stroke:
            stroke_points = np.array(current_stroke)
            plt.plot(stroke_points[:, 0], stroke_points[:, 1], 'b-')
        
        plt.title("Nét vẽ tay")
        plt.gca().invert_yaxis()  # Đảo ngược trục y để giống nét vẽ thực tế
        plt.savefig('mathtrain.png')
        plt.close()

if __name__ == '__main__':
    # Thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "data/mathwriting-2024-excerpt"  # Thay đổi đường dẫn nếu cần

    # Use the new function to create data loaders
    train_loader, valid_loader, test_loader, vocab = create_data_loaders(root_dir)

    # Định nghĩa mô hình
    input_dim = 3  # (x, y, s) từ nét vẽ
    output_dim = len(vocab)  # Kích thước từ điển
    hidden_dim = 256
    model = Seq2Seq(input_dim, output_dim, hidden_dim).to(device)

    # Định nghĩa loss và optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Tạo Trainer
    trainer = Trainer(model, train_loader, valid_loader, criterion, optimizer, device)

    num_epochs = 10
    trainer.train(num_epochs)

    trainer.save_checkpoint('mathwriting_checkpoint.pth')

    # Trực quan hóa một batch dữ liệu
    print("\nTrực quan hóa batch đầu tiên:")
    trainer.visualize_batch(batch_idx=0)