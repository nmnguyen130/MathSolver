import torch
import torch.nn as nn

class ModelEvaluator:
    def __init__(self, model, criterion=None):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()

    def calculate_metrics(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        return loss.item(), correct

    def evaluate(self, data_loader):
        """
        Evaluate the model on the given data loader
        """
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss, correct = self.calculate_metrics(outputs, targets)
                
                total_loss += loss
                total += targets.size(0)
                correct += correct
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(data_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }