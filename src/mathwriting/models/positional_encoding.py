import math
import torch
import torch.nn as nn

class PositionalEncoding1D(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 1000,
        temperature: float = 10000.0,
    ):
        super().__init__()

        # Generate position and dimension tensors for encoding
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        dim_t = torch.arange(0, d_model, 2, dtype=torch.float32)  # (d_model//2,)
        div_term = torch.exp(dim_t * (-math.log(temperature) / d_model))  # (d_model//2,)

        # Initialize and fill the positional encoding matrix with sine/cosine values
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        return self.dropout(x + self.pe[None, :seq_len, :])


class PositionalEncoding2D(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 30, # Default max height/width
        temperature: float = 10000.0,
    ):
        super().__init__()

        # Generate position and dimension tensors for 1D encoding
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        dim_t = torch.arange(0, d_model, 2, dtype=torch.float32)  # (d_model//2,)
        div_term = torch.exp(dim_t * (-math.log(temperature) / d_model))  # (d_model//2,)

        # Initialize and fill the 1D positional encoding matrix with sine/cosine values
        pe_1D = torch.zeros(max_len, d_model)
        pe_1D[:, 0::2] = torch.sin(position * div_term)
        pe_1D[:, 1::2] = torch.cos(position * div_term)

        # 2D encoding bằng outer addition (pe_y[i] + pe_x[j])
        pe_2D = torch.zeros(max_len, max_len, d_model)
        for i in range(d_model):
            pe_2D[:, :, i] = pe_1D[:, i].unsqueeze(1) + pe_1D[:, i].unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe_2D)

    def forward(self, x):
        """
        x shape: (batch, height, width, d_model)
        """
        batch, height, width, d_model = x.shape
        return self.dropout(x + self.pe[None, :height, :width, :])
