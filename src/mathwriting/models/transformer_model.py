import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights

from src.mathwriting.models.positional_encoding import PositionalEncoding1D, PositionalEncoding2D

class Permute(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
    
class ImageEncoder(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        # Load DenseNet pretrained, remove the last classifier
        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(densenet.features.children()))  # only CNN part

        # Convert feature map (B, 1024, H, W) → (B, d_model, H, W)
        self.reduce_conv = nn.Conv2d(in_channels=1024, out_channels=d_model, kernel_size=1)

        # Positional encoding + flatten
        self.encoder = nn.Sequential(
            Permute(0, 2, 3, 1),                         # (B, H, W, C)
            PositionalEncoding2D(d_model, dropout),     # Add position
            nn.Flatten(1, 2),                            # (B, H*W, C)
        )

    def forward(self, x):
        # x shape: (B, 3, H, W)
        x = self.backbone(x)          # → (B, 1024, H, W)
        x = self.reduce_conv(x)       # → (B, d_model, H, W)
        x = self.encoder(x)           # → (B, H*W, d_model)
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

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt: (batch_size, seq_len)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt_emb = self.pos_encoding(tgt_emb)

        # Transformer decoder
        out = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return self.fc_out(out)

class ImageToLatexModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=6):
        super().__init__()
        self.encoder = ImageEncoder(d_model, dropout)
        self.decoder = TransformerDecoder(vocab_size, d_model, nhead, dim_feedforward, dropout, num_layers)
        self.vocab_size = vocab_size

    def forward(self, src, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            src (Tensor): input image tensor of shape (B, 3, H, W)
            tgt (Tensor): input token sequence for decoder (B, T)
            tgt_mask (Tensor, optional): causal mask for decoder
            tgt_key_padding_mask (Tensor, optional): padding mask for target

        Returns:
            Tensor: output logits of shape (B, T, vocab_size)
        """
        memory = self.encoder(src)  # (B, H*W, d_model)
        out = self.decoder(tgt, memory, tgt_mask, tgt_key_padding_mask)  # (B, T, vocab_size)
        return out
    
    def compute_loss(self, src, tgt_input, tgt_output, tgt_mask=None):
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
    
    def generate_square_subsequent_mask(self, sz):
        """
        Creates a causal mask for transformer decoding.

        Args:
            sz (int): sequence length

        Returns:
            Tensor: (sz, sz) mask with -inf above the diagonal
        """
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
        return mask.to(next(self.parameters()).device)
    
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
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))

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