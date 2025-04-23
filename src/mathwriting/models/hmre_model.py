import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create the divisor to scale the positions appropriately
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices for positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as a buffer (not trainable)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]
    
# class TransformerEncoder(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         x = self.encoder(x)
#         return self.norm(x)

# class VisionTransformer(nn.Module):
#     def __init__(self, img_size: int = 224, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 256, dim_feedforward: int = 512):
#         super().__init__()
#         self.patch_size = patch_size
#         self.num_patches = (img_size // patch_size) ** 2
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
#         self.encoder = TransformerEncoder(
#             embed_dim=embed_dim, num_heads=4, num_layers=3, dim_feedforward=dim_feedforward, dropout=0.1
#         )

#     def forward(self, x):
#         x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
#         x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
#         x = x + self.pos_embed
#         x = self.encoder(x)
#         return x
    
class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))  # Giảm kích thước đầu ra
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.proj = nn.Linear(7 * 7, embed_dim)

    def forward(self, x):
        x = self.cnn(x)  # (B, embed_dim, 7, 7)
        x = self.flatten(x).transpose(1, 2)  # (B, 49, embed_dim)
        x = self.proj(x)  # (B, 49, embed_dim)
        return x

class LaTeXDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 3, dim_feedforward: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features, tgt, tgt_mask):
        padding_mask = tgt.eq(0)  # True cho <pad>, False cho token hợp lệ
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_dim)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.decoder(
            tgt_emb,
            features,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask)
        output = self.norm(output)
        return self.fc_out(output)

class MathWritingModel(nn.Module):
    """
    Complete model for Math Writing task, combining image embedding, transformer encoder, and LaTeX decoder.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, dim_feedforward: int = 512):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=3, embed_dim=embed_dim)
        self.decoder = LaTeXDecoder(vocab_size=vocab_size, embed_dim=embed_dim, dim_feedforward=dim_feedforward)
        
    def generate_square_subsequent_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, img, tgt):
        features = self.encoder(img)  # (B, num_patches, embed_dim)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.decoder(features, tgt, tgt_mask)  # (B, tgt_len, vocab_size)
        return output

    @torch.no_grad()
    def greedy_search(self, src, tokenizer, max_len: int = 100, sos_idx: int = 1, eos_idx: int = 2):
        self.eval()
        features = self.encoder(src)
        batch_size = src.size(0)
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)
        output_tokens = [[] for _ in range(batch_size)]
        completed = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for step in range(1, max_len):
            tgt_mask = self.generate_square_subsequent_mask(step).to(src.device)[:step, :step]
            out = self.decoder(features, tgt, tgt_mask)
            next_token = out[:, -1].argmax(-1, keepdim=True)

            for i in range(batch_size):
                if not completed[i]:
                    output_tokens[i].append(next_token[i].item())
                    if next_token[i] == eos_idx:
                        completed[i] = True

            tgt = torch.cat([tgt, next_token], dim=1)
            if completed.all():
                break

        decoded = [tokenizer.decode(output) for output in output_tokens]
        return decoded

    @torch.no_grad()
    def beam_search(self, src, tokenizer, beam_width: int = 2, max_len: int = 50, sos_idx: int = 1, eos_idx: int = 2):
        self.eval()
        features = self.encoder(src)
        batch_size = src.size(0)
        max_mask = self.generate_square_subsequent_mask(max_len).to(src.device)
        final_outputs = []

        for b in range(batch_size):
            beams = [([sos_idx], 0.0)]
            completed_beams = []

            for step in range(1, max_len):
                candidates = []
                for seq, log_prob in beams:
                    if seq[-1] == eos_idx:
                        completed_beams.append((seq, log_prob))
                        continue

                    tgt = torch.tensor([seq], dtype=torch.long, device=src.device)
                    tgt_mask = max_mask[:len(seq), :len(seq)]
                    out = self.decoder(features[b:b+1], tgt, tgt_mask)
                    probs = out[:, -1].squeeze(0)
                    top_probs, top_tokens = probs.topk(beam_width)

                    for k in range(beam_width):
                        new_seq = seq + [top_tokens[k].item()]
                        new_log_prob = log_prob + top_probs[k].item()
                        candidates.append((new_seq, new_log_prob))

                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                if len(completed_beams) >= beam_width or not beams:
                    break

            completed_beams.extend(beams)
            best_seq = max(completed_beams, key=lambda x: x[1])[0]
            final_outputs.append(tokenizer.decode(best_seq))

        return final_outputs