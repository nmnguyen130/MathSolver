import torch
import torch.nn as nn
import torch.nn.functional as F

class SymbolDetectionModule(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, output_size=10, num_layers=2, dropout=0.2):
        super(SymbolDetectionModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        output, (hidden, _) = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(hidden[-1])
        return out