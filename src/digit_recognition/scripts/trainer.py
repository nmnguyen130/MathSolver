import torch
import torch.nn as nn
import torch.optim as optim
from src.digit_recognition.scripts.digit_dataset import DigitDataset
from src.digit_recognition.scripts.evaluator import ModelEvaluator

class ModelTrainer:
    def __init__(self, model, batch_size=32, learning_rate=0.001, epochs=50, patience=5):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=2, factor=0.5)
        self.evaluator = ModelEvaluator(model, self.criterion)

    def prepare_data(self):
        return DigitDataset.create_dataloaders(
            batch_size=self.batch_size,
            train_split=0.8
        )
    
    def train(self):
        train_loader, val_loader = self.prepare_data()
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss, correct, total = 0, 0, 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # print(f"Inputs: {inputs[0]}")
                DigitDataset.save_image(inputs[0], filepath='image.png')
                break
                # print(f"Inputs: {inputs.shape}, targets: {targets.shape}")
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1}/{self.epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
            
            # Validation phase using evaluator
            val_metrics = self.evaluator.evaluate(val_loader)
            train_acc = 100. * correct / total
            
            print(f'\nEpoch {epoch+1}/{self.epochs}:')
            print(f'Train Loss: {train_loss/len(train_loader):.5f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_metrics["loss"]:.5f} | Val Acc: {val_metrics["accuracy"]:.2f}%\n')
            
            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                torch.save(self.model.state_dict(), 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping triggered. At epoch {epoch}")
                break

            self.scheduler.step(val_metrics["accuracy"])