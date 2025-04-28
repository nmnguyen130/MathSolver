import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, 'Image size must be divisible by patch size'
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, depth=6, mlp_dim=1024, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, 
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, depth=4, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, 
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, tgt_mask=None):
        padding_mask = tgt.eq(0)  # True cho <pad>, False cho token hợp lệ
        tgt_embed = self.embedding(tgt)
        tgt_embed = self.pos_encoding(tgt_embed)
        output = self.decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask
        )
        return output
    
class ImageToLatexModel(nn.Module):
    def __init__(self, vocab_size, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=256, num_heads=8, depth=4, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim) * 0.02)  # +1 cho CLS token

        self.encoder = ViTEncoder(embed_dim, num_heads, depth, mlp_dim, dropout)
        self.decoder = TransformerDecoder(vocab_size, embed_dim, num_heads, depth, mlp_dim, dropout)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, img, tgt):
        features = self.patch_embedding(img)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(features.size(0), -1, -1)  # (B, 1, embed_dim)
        features = torch.cat((cls_tokens, features), dim=1)  # (B, num_patches+1, embed_dim)
        features = features + self.pos_embed  # add positional encoding
        memory = self.encoder(features)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.decoder(tgt, memory, tgt_mask)  # (B, tgt_len, vocab_size)
        output = self.fc_out(output)
        return output
    
    def generate(self, img, max_len=100, sos_idx=1, eos_idx=2):
        self.eval()
        with torch.no_grad():
            features = self.patch_embedding(img)
            cls_tokens = self.cls_token.expand(features.size(0), -1, -1).to(img.device)
            features = torch.cat((cls_tokens, features), dim=1)
            features = features + self.pos_embed.to(img.device)
            memory = self.encoder(features)

            tgt = torch.ones(img.size(0), 1, dtype=torch.long, device=img.device) * sos_idx
            output = []
            for _ in range(max_len):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(img.device)
                out = self.decoder(tgt, memory, tgt_mask)
                out = self.fc_out(out[:, -1, :]).argmax(-1).unsqueeze(1)
                output.append(out)
                tgt = torch.cat([tgt, out], dim=1)
        return torch.cat(output, dim=1)