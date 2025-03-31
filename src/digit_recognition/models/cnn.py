import torch
import torch.nn as nn

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        # Convolutional block for feature extraction
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # Input: 28x28x1 -> Output: 28x28x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), # 28x28x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 14x14x32
            
            nn.Conv2d(32, 64, 3, padding=1), # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 7x7x64
        )

        # Classifier for digit prediction
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                 # Regularization
            nn.Linear(64 * 7 * 7, 512),      # Flatten: 7*7*64 = 3136 -> 512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)               # Output: 10 digit classes
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 3136)
        return self.classifier(x)