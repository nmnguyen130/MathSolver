import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mathwriting.models.vit_encoder import PatchEmbedding, ViTEncoder
from src.mathwriting.models.swin_encoder import SwinEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout=0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Non-trainable buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, depth=3, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, tgt_mask=None):
        padding_mask = tgt.eq(0)  # True cho <pad>, False cho token hợp lệ
        tgt_embed = self.embedding(tgt) * math.sqrt(self.embed_dim)
        tgt_embed = self.pos_encoding(tgt_embed)

        output = self.decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask
        )
        output = self.norm(output)
        return output

class MathWritingModel(nn.Module):
    # Vit Encoder + Transformer Decoder
    # def __init__(self, vocab_size, img_size=224, patch_size=16, in_channels=3,
    #              embed_dim=256, num_heads=8, depth=3, mlp_dim=1024, dropout=0.1):
    #     super().__init__()
    #     self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
    #     self.encoder = ViTEncoder(embed_dim, num_heads, depth, mlp_dim, dropout)

    #     self.decoder = TransformerDecoder(vocab_size, embed_dim, num_heads, depth, mlp_dim, dropout)
    #     self.fc_out = nn.Linear(embed_dim, vocab_size)

    # Swin Encoder + Transformer Decoder
    def __init__(self, vocab_size, img_size=224, patch_size=4, in_channels=3,
                 embed_dim=64, num_heads=[2, 4, 8], depths=[1, 1, 2], mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.encoder = SwinEncoder(img_size, patch_size, in_channels, embed_dim, depths, num_heads, window_size=7, mlp_ratio=3.0, dropout=dropout)
        self.decoder = TransformerDecoder(vocab_size, embed_dim=256, num_heads=8, depth=3, mlp_dim=mlp_dim, dropout=dropout)
        self.fc_out = nn.Linear(256, vocab_size)

    def generate_square_subsequent_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, img, tgt):
        # Vit Encoder
        # memory = self.patch_embedding(img)  # (B, num_patches+1, embed_dim)
        # memory = self.encoder(memory)  # (B, num_patches+1, embed_dim)

        # Swin Encoder
        memory = self.encoder(img)  # (B, num_patches, 256)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.decoder(tgt, memory, tgt_mask)  # (B, tgt_len, embed_dim)
        output = self.fc_out(output)  # (B, tgt_len, vocab_size)
        return output

    def generate(self, img, max_len=100, sos_idx=1, eos_idx=2):
        self.eval()
        with torch.no_grad():
            # Vit Encoder
            # memory = self.patch_embedding(img)  # (B, num_patches+1, embed_dim)
            # memory = self.encoder(memory)  # (B, num_patches+1, embed_dim)

            # Swin Encoder
            memory = self.encoder(img)  # (B, num_patches, 256)

            tgt = torch.ones(img.size(0), 1, dtype=torch.long, device=img.device) * sos_idx
            output = []
            for _ in range(max_len):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(img.device)
                out = self.decoder(tgt, memory, tgt_mask)
                out = self.fc_out(out[:, -1, :]).argmax(-1).unsqueeze(1)
                output.append(out)
                tgt = torch.cat([tgt, out], dim=1)

                if (out == eos_idx).all():
                    break
        return torch.cat(output, dim=1)
    
    def beam_search(self, img, max_len=100, sos_idx=1, eos_idx=2, beam_width=3):
        self.eval()
        with torch.no_grad():
            # Vit Encoder
            # memory = self.patch_embedding(img)  # (B, num_patches+1, embed_dim)
            # memory = self.encoder(memory)  # (B, num_patches+1, embed_dim)

            # Swin Encoder
            memory = self.encoder(img)  # (B, num_patches, 256)
            batch_size = img.size(0)
            device = img.device

            # Khởi tạo beams: mỗi mẫu có danh sách các tuple [(seq, log_prob)]
            beams = [[([sos_idx], 0.0)] for _ in range(batch_size)]
            completed = [[] for _ in range(batch_size)]

            for _ in range(max_len):
                all_new_beams = [[] for _ in range(batch_size)]
                for b in range(batch_size):
                    for seq, log_prob in beams[b]:
                        # Nếu chuỗi kết thúc (eos_idx), thêm vào completed
                        if seq[-1] == eos_idx:
                            completed[b].append((seq, log_prob))
                            continue
                        # Dự đoán token tiếp theo
                        tgt = torch.tensor([seq], dtype=torch.long, device=device)
                        tgt_mask = self.generate_square_subsequent_mask(len(seq)).to(device)
                        out = self.decoder(tgt, memory[b:b+1], tgt_mask)
                        logits = self.fc_out(out[:, -1, :]).log_softmax(-1)
                        topk_log_probs, topk_ids = logits[0].topk(beam_width)

                        # Tạo các beam mới
                        for i in range(beam_width):
                            new_seq = seq + [topk_ids[i].item()]
                            new_log_prob = log_prob + topk_log_probs[i].item()
                            all_new_beams[b].append((new_seq, new_log_prob))

                    # Cập nhật beams: chọn beam_width chuỗi tốt nhất
                    beams[b] = sorted(all_new_beams[b], key=lambda x: x[1], reverse=True)[:beam_width]

                # Dừng nếu tất cả mẫu đã có ít nhất beam_width chuỗi hoàn thành
                if all(len(completed[b]) >= beam_width for b in range(batch_size)):
                    break

            # Chọn chuỗi tốt nhất cho mỗi mẫu
            outputs = []
            for b in range(batch_size):
                if completed[b]:
                    best_seq = max(completed[b], key=lambda x: x[1])[0]
                else:
                    best_seq = max(beams[b], key=lambda x: x[1])[0]
                outputs.append(best_seq[1:])  # Bỏ sos_idx

            # Chuyển thành tensor
            max_len = max(len(seq) for seq in outputs)
            result = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            for b, seq in enumerate(outputs):
                result[b, :len(seq)] = torch.tensor(seq, device=device)
            return result