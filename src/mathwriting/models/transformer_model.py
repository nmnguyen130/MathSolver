import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mathwriting.models.custom_cnn import SimpleDenseBlock, TransitionLayer
from src.mathwriting.models.positional_encoding import PositionalEncoding1D, PositionalEncoding2D

class Permute(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
    
class ImageEncoder(nn.Module):
    def __init__(self, d_model=256, dropout=0.1):
        super().__init__()
        self.init_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        self.block1 = SimpleDenseBlock(64, 16, num_layers=3)
        self.trans1 = TransitionLayer(64 + 3 * 16, 64)

        self.block2 = SimpleDenseBlock(64, 16, num_layers=3)
        self.trans2 = TransitionLayer(64 + 3 * 16, 64)

        self.block3 = SimpleDenseBlock(64, 16, num_layers=3)
        self.trans3 = TransitionLayer(64 + 3 * 16, 64)

        self.reduce_conv = nn.Conv2d(64, d_model, kernel_size=1)

        self.encoder = nn.Sequential(
            Permute(0, 2, 3, 1),                         # (B, H, W, C)
            PositionalEncoding2D(d_model, dropout),     # Add position
            nn.Flatten(1, 2),                            # (B, H*W, C)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.trans3(self.block3(x))
        x = self.reduce_conv(x)
        x = self.encoder(x)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, dropout, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding1D(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, features, tgt, tgt_mask):
        # tgt: (batch_size, seq_len)
        padding_mask = tgt.eq(0)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt_emb = self.pos_encoding(tgt_emb)

        # Transformer decoder
        out = self.transformer_decoder(
            tgt_emb,
            features,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask,
        )

        return self.fc_out(out)

class ImageToLatexModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.2, num_layers=3):
        super().__init__()
        self.encoder = ImageEncoder(d_model, dropout)
        self.decoder = TransformerDecoder(vocab_size, d_model, nhead, dim_feedforward, dropout, num_layers)
        self.vocab_size = vocab_size

    def forward(self, src, tgt, tgt_mask=None):
        """
        Args:
            src (Tensor): input image tensor of shape (B, 3, H, W)
            tgt (Tensor): input token sequence for decoder (B, T)
            tgt_mask (Tensor, optional): causal mask for decoder
            tgt_key_padding_mask (Tensor, optional): padding mask for target

        Returns:
            Tensor: output logits of shape (B, T, vocab_size)
        """
        features = self.encoder(src)  # (B, H*W, d_model)
        out = self.decoder(features, tgt, tgt_mask)  # (B, T, vocab_size)
        return out
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask.to(next(self.parameters()).device)
    
    def compute_loss(self, src, tgt_input, tgt_output, tgt_mask):
        """
        Computes cross-entropy loss between predicted output and ground-truth.

        Args:
            src (Tensor): input image (B, 3, H, W)
            tgt_input (Tensor): input token sequence with <BOS> (B, T)
            tgt_output (Tensor): target sequence shifted by 1 (B, T)
            tgt_mask (Tensor, optional): decoder mask

        Returns:
            Tensor: scalar loss value
        """
        out = self.forward(src, tgt_input, tgt_mask)  # (B, T, vocab_size)
        return F.cross_entropy(out.reshape(-1, self.vocab_size), tgt_output.reshape(-1), ignore_index=0)
    
    @torch.no_grad()
    def greedy_decode(self, src, tokenizer, max_len=256, bos_token_id=1, eos_token_id=2):
        """
        Greedy decoding: generates tokens one-by-one based on highest probability.

        Args:
            src (Tensor): input image (B, 3, H, W)
            tokenizer: tokenizer with `.decode()` method
            max_len (int): maximum sequence length
            bos_token_id (int): start token ID
            eos_token_id (int): end token ID

        Returns:
            List[str]: list of decoded LaTeX strings (batch size)
        """
        self.eval()
        memory = self.encoder(src)  # Encode image to memory (B, H*W, d_model)
        batch_size = src.size(0)

        # Start with <BOS> token
        tgt = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=src.device)

        for i in range(1, max_len):
            # Create causal mask for current sequence
            tgt_mask = torch.triu(torch.ones(i, i, dtype=torch.bool), diagonal=1)

            # Decode one step
            out = self.decoder(tgt, memory, tgt_mask)  # (B, T, vocab_size)
            next_token = out[:, -1].argmax(-1, keepdim=True)  # Get token with highest score

            # Append next token
            tgt = torch.cat([tgt, next_token], dim=1)

            # Stop if all sequences have produced <EOS>
            if (next_token == eos_token_id).all():
                break

        # Decode token sequences to LaTeX strings
        return [tokenizer.decode(seq.tolist()) for seq in tgt]