import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class PositionalEncoding(nn.Module):
    """
    Positional encoding for LaTeX token sequences.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        
        # Create a tensor to hold positional encodings of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Position tensor for each word
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

class ImagePatchEmbedding(nn.Module):
    """
    Convert image into patches and embed them as vectors.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional layer to generate patch embeddings
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable positional embeddings for each patch
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        return x

class TransformerEncoder(nn.Module):
    """
    Transformer encoder that processes image features.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.encoder(x)  # (B, num_patches, embed_dim)
        return self.norm(x)

class LaTeXDecoder(nn.Module):
    """
    Decoder that generates LaTeX sequence from image features.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 256, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features, tgt, tgt_mask):
        tgt = self.embedding(tgt) * math.sqrt(tgt.size(-1))
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, features, tgt_mask=tgt_mask)  # (B, tgt_len, embed_dim)
        output = self.norm(output)
        return self.fc_out(output)

class MathWritingModel(nn.Module):
    """
    Complete model for Math Writing task, combining image embedding, transformer encoder, and LaTeX decoder.
    """
    def __init__(self, vocab_size: int, img_size: int = 224, embed_dim: int = 256):
        super().__init__()
        self.patch_embed = ImagePatchEmbedding(img_size=img_size, embed_dim=embed_dim)
        self.encoder = TransformerEncoder(embed_dim=embed_dim)
        self.decoder = LaTeXDecoder(vocab_size=vocab_size, embed_dim=embed_dim)
        
    def generate_square_subsequent_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, img, tgt):
        features = self.patch_embed(img)  # (B, num_patches, embed_dim)
        features = self.encoder(features)  # (B, num_patches, embed_dim)
        
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.decoder(features, tgt, tgt_mask)  # (B, tgt_len, vocab_size)
        return output

    @torch.no_grad()
    def greedy_search(self, src, tokenizer, max_len: int = 100, sos_idx: int = 1, eos_idx: int = 2):
        self.eval()
        features = self.patch_embed(src)
        features = self.encoder(features)
        
        batch_size = src.size(0)
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)
        completed = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        max_mask = self.generate_square_subsequent_mask(max_len).to(src.device)
        output_tokens = [[] for _ in range(batch_size)]
        
        for step in range(1, max_len):
            tgt_mask = max_mask[:step, :step]
            out = self.decoder(features, tgt, tgt_mask)
            probs = F.softmax(out[:, -1], dim=-1)  # Xác suất các token
            next_token = probs.argmax(-1).unsqueeze(1)
        
            for i in range(batch_size):
                if not completed[i]:
                    output_tokens[i].append(next_token[i].item())
                    if next_token[i] == eos_idx:
                        completed[i] = True
            
            # Concatenate the selected token to the target sequence for next iteration
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Check if all sequences have reached the EOS token
            if completed.all():
                break
        
        decoded = [tokenizer.decode(output) for output in output_tokens]
        return decoded
    
    @torch.no_grad()
    def beam_search(self, src, tokenizer, beam_width: int = 3, max_len: int = 100, sos_idx: int = 1, eos_idx: int = 2):
        self.eval()
        features = self.patch_embed(src)
        features = self.encoder(features)
        
        batch_size = src.size(0)
        max_mask = self.generate_square_subsequent_mask(max_len).to(src.device)

        final_outputs = []

        for b in range(batch_size):
            # Initialize the beam for sample b
            beams = [([sos_idx], 0.0)]  # (sequence of tokens, log probability)
            completed_beams = []
            
            for step in range(1, max_len):
                candidates = []
                
                # Iterate through each current beam
                for seq, log_prob in beams:
                    if seq[-1] == eos_idx:
                        completed_beams.append((seq, log_prob))
                        continue
                    
                    # Prepare input for the decoder
                    tgt = torch.tensor([seq], dtype=torch.long, device=src.device)  # (1, T)
                    tgt_mask = max_mask[:len(seq), :len(seq)]
                    
                    # Compute probabilities
                    out = self.decoder(features[b:b+1], tgt, tgt_mask)  # (1, T, vocab_size)
                    probs = F.log_softmax(out[:, -1], dim=-1).squeeze(0)  # (vocab_size,)
                    
                    # Get the top k tokens
                    top_probs, top_tokens = probs.topk(beam_width)
                    
                    # Add new candidates
                    for k in range(beam_width):
                        new_seq = seq + [top_tokens[k].item()]
                        new_log_prob = log_prob + top_probs[k].item()
                        candidates.append((new_seq, new_log_prob))
                
                # Select the top beam_width candidates
                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                beams = candidates
                
                # If all beams are complete, stop
                if len(completed_beams) >= beam_width or not beams:
                    break
            
            # Add any unfinished beams to completed_beams
            completed_beams.extend(beams)
            
            # Choose the best sequence
            best_seq = max(completed_beams, key=lambda x: x[1])[0]
            final_outputs.append(tokenizer.decode(best_seq))
        
        return final_outputs