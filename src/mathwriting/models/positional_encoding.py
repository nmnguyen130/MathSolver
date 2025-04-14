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
        self.dropout = nn.Dropout(dropout)
        pe = self._compute_pe(d_model, max_len, temperature)
        self.register_buffer("pe", pe)

    @staticmethod
    def _compute_pe(d_model, max_len, temperature):
        # Generate position and dimension tensors for encoding
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        dim_t = torch.arange(0, d_model, 2, dtype=torch.float32)  # (d_model//2,)
        div_term = torch.exp(dim_t * (-math.log(temperature) / d_model))  # (d_model//2,)

        # Initialize and fill the positional encoding matrix with sine/cosine values
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        return self.dropout(x + self.pe[None, :seq_len, :])


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        batch, height, width, d_model = x.shape
        return self.dropout(x + self.pe[:, :height, :width, :])
