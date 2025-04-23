import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    
# Graph Encoder for Directed Graphs
class GraphEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=64):  # Reduced d_model
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.conv1 = GATConv(d_model, d_model // 2, heads=2, concat=True)  # GAT with multi-head attention
        self.conv2 = GATConv(d_model, d_model, heads=1, concat=True)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.embedding(x)  # (num_nodes, d_model)
        x = F.relu(self.conv1(x, edge_index, edge_attr))  # Use edge_attr for direction
        x = self.conv2(x, edge_index, edge_attr)
        return x  # (num_nodes, d_model)
    
# MathEmbedding: Embedding + Tokenizer Features
class MathEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, structure_dim: int = 7):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model - structure_dim, padding_idx=0)
        self.structure_linear = nn.Linear(structure_dim, structure_dim)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor, structure_features: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_embedding(tokens)  # (B, S, d_model - structure_dim)
        structure_emb = self.structure_linear(structure_features)  # (B, S, structure_dim)
        return torch.cat([token_emb, structure_emb], dim=-1)  # (B, S, d_model)

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
    
# Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)
        return x

# Decoder Block with cross-attention
class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x2, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(x2))

        x2, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + self.dropout(x2))

        x2 = self.ff(x)
        x = self.norm3(x + self.dropout(x2))
        return x

# Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return x

# Full Model
class MathSolverModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, d_ff=512, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.graph_encoder = GraphEncoder(vocab_size, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Encoder và Decoder
        self.encoder = TransformerEncoder(num_layers, d_model, nhead, d_ff, dropout)
        self.decoder = TransformerDecoder(num_layers, d_model, nhead, d_ff, dropout)

        # Output layer
        self.final_layer = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, graph_data: list[Data], tgt: torch.Tensor, query_ids: torch.Tensor):
        batch_size = tgt.size(0)
        device = tgt.device
        
        # Encode directed graph
        graph_embs = [self.graph_encoder(g.to(device)) for g in graph_data]
        max_nodes = max(emb.size(0) for emb in graph_embs)
        graph_emb = torch.zeros(batch_size, max_nodes, self.d_model, device=device)

        # Mã hóa query
        for i, emb in enumerate(graph_embs):
            graph_emb[i, :emb.size(0)] = emb
        query_emb = self.pos_encoder(self.token_embedding(query_ids) * math.sqrt(self.d_model))

        # Kết hợp
        encoder_input = torch.cat([graph_emb, query_emb], dim=1)

        tgt_key_padding_mask = tgt.eq(0)  # (B, T)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)# (T, T)
        tgt_emb = self.pos_encoder(self.token_embedding(tgt) * math.sqrt(self.d_model))

        memory = self.encoder(encoder_input)
        output = self.decoder(tgt_emb, memory, tgt_mask, tgt_key_padding_mask)
        return self.final_layer(output)