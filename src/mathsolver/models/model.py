import math
import torch
import torch.nn as nn

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
   
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len) - Token IDs
        Output: (batch_size, seq_len, d_model)
        Nhân với sqrt(d_model) để chuẩn hóa độ lớn embedding, đảm bảo tương thích
        với Positional Encoding và ổn định gradient trong attention.
        """
        return self.embedding(x) * math.sqrt(self.d_model)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, depth: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        return self.encoder(src, src_key_padding_mask=src_key_padding_mask)
    
class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, depth: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

# Full Model
class MathSolverModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, depth=3, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Encoder và Decoder
        self.encoder = TransformerEncoder(d_model, nhead, depth, mlp_dim, dropout)
        self.decoder = TransformerDecoder(d_model, nhead, depth, mlp_dim, dropout)

        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, src, tgt: torch.Tensor):
        src_key_padding_mask = src.eq(0)
        tgt_key_padding_mask = tgt.eq(0)

        src = self.token_embedding(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src, src_key_padding_mask)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        tgt = self.token_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask, tgt_key_padding_mask)

        return self.fc_out(output)