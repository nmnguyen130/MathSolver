import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            self.depthwise,
            self.pointwise,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)
    
class SimpleDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(ConvBlock(channels, growth_rate))
            channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)
    
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)
    
class CustomCNN(nn.Module):
    def __init__(self, in_channels=3, growth_rate=16, num_classes=512):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)

        self.block1 = SimpleDenseBlock(64, growth_rate, num_layers=4)
        self.trans1 = TransitionLayer(64 + 4 * growth_rate, 64)

        self.block2 = SimpleDenseBlock(64, growth_rate, num_layers=4)
        self.trans2 = TransitionLayer(64 + 4 * growth_rate, 64)

        self.block3 = SimpleDenseBlock(64, growth_rate, num_layers=4)
        self.trans3 = TransitionLayer(64 + 4 * growth_rate, 64)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.trans3(self.block3(x))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)