import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mathwriting.models.custom_cnn import SimpleDenseBlock, TransitionLayer, SEBlock
from src.mathwriting.models.positional_encoding import PositionalEncoding1D, PositionalEncoding2D

class Permute(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
    
class ImageEncoder(nn.Module):
    def __init__(self, d_model=256, growth_rate=16, num_layers=4, dropout=0.1):
        super().__init__()
        self.init_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        self.block1 = SimpleDenseBlock(64, growth_rate, num_layers=num_layers)
        self.trans1 = TransitionLayer(64 + num_layers * growth_rate, 64)

        self.block2 = SimpleDenseBlock(64, growth_rate, num_layers=num_layers)
        self.trans2 = TransitionLayer(64 + num_layers * growth_rate, 64)

        self.block3 = SimpleDenseBlock(64, growth_rate, num_layers=num_layers)
        self.trans3 = TransitionLayer(64 + num_layers * growth_rate, 64)

        self.se_block = SEBlock(64)
        self.reduce_conv = nn.Conv2d(64, d_model, kernel_size=1)

        self.encoder = nn.Sequential(
            Permute(0, 2, 3, 1),                    # (B, C, H', W') -> (B, H', W', C)
            PositionalEncoding2D(d_model, dropout), # Maintains shape
            nn.Flatten(1, 2),                       # (B, H', W', C) -> (B, H'*W', C)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.trans3(self.block3(x))
        x = self.se_block(x)
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
        """
        Args:
            features (Tensor): Encoded image features of shape (B, H'*W', d_model).
            tgt (Tensor): Target token sequence of shape (B, T).
            tgt_mask (Tensor): Causal mask of shape (T, T).

        Returns:
            Tensor: Output logits of shape (B, T, vocab_size).
        """
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
        self.encoder = ImageEncoder(d_model=d_model, dropout=dropout)
        self.decoder = TransformerDecoder(vocab_size, d_model, nhead, dim_feedforward, dropout, num_layers)
        self.vocab_size = vocab_size

    def forward(self, src, tgt, tgt_mask=None):
        """
        Args:
            src (Tensor): Input image tensor of shape (B, 3, H, W).
            tgt (Tensor): Input token sequence of shape (B, T).
            tgt_mask (Tensor, optional): Causal mask of shape (T, T).

        Returns:
            Tensor: output logits of shape (B, T, vocab_size)
        """
        features = self.encoder(src)  # (B, H'*W', d_model)
        out = self.decoder(features, tgt, tgt_mask)  # (B, T, vocab_size)
        return out
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask.to(next(self.parameters()).device)
    
    def compute_loss(self, src, tgt_input, tgt_output, tgt_mask):
        """
        Computes cross-entropy loss between predicted output and ground-truth.

        Args:
            src (Tensor): Input image of shape (B, 3, H, W).
            tgt_input (Tensor): Input token sequence with <BOS> of shape (B, T).
            tgt_output (Tensor): Target token sequence shifted by 1 of shape (B, T).
            tgt_mask (Tensor): Causal mask of shape (T, T).

        Returns:
            Tensor: Scalar cross-entropy loss.
        """
        out = self.forward(src, tgt_input, tgt_mask)  # (B, T, vocab_size)
        return F.cross_entropy(out.reshape(-1, self.vocab_size), tgt_output.reshape(-1), ignore_index=0)
    
    @torch.no_grad()
    def greedy_decode(self, src, tokenizer, max_len=256, bos_token_id=1, eos_token_id=2):
        """
        Greedy decoding: generates tokens one-by-one based on highest probability.

        Args:
            src (Tensor): Input image of shape (B, 3, H, W).
            tokenizer: Tokenizer with `.decode()` method to convert token IDs to strings.
            max_len (int): Maximum sequence length.

        Returns:
            List[str]: List of decoded LaTeX strings, one per batch element.
        """
        self.eval()
        features = self.encoder(src)
        batch_size = src.size(0)
        tgt = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=src.device)
        completed = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        max_mask = self.generate_square_subsequent_mask(max_len).to(src.device)
        outputs = [[] for _ in range(batch_size)]
        
        # Lưu xác suất và chuỗi token từng bước
        probs_list = []
        tokens_list = []
        for step in range(1, max_len):
            tgt_mask = max_mask[:step, :step]
            out = self.decoder(features, tgt, tgt_mask)
            probs = F.softmax(out[:, -1], dim=-1)  # Xác suất các token
            next_token = probs.argmax(-1)
            
            # Lưu xác suất và token
            probs_list.append(probs.cpu().numpy())
            tokens_list.append(next_token.cpu().numpy())
            
            for i in range(batch_size):
                if not completed[i]:
                    outputs[i].append(next_token[i].item())
                    if next_token[i] == eos_token_id:
                        completed[i] = True
            
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            
            if completed.all():
                break
        
        decoded = [tokenizer.decode(output) for output in outputs]
        return decoded, probs_list, tokens_list
    
    @torch.no_grad()
    def beam_decode(self, src, tokenizer, beam_width=3, max_len=256, bos_token_id=1, eos_token_id=2):
        """
        Performs beam search to generate LaTeX sequences from the input image.

        Args:
            src (Tensor): Input image, shape (B, 3, H, W).
            tokenizer: Tokenizer with a decode() method.
            beam_width (int): Number of candidate sequences to keep at each step.
            max_len (int): Maximum length of the generated sequence.

        Returns:
            List[str]: A list of the best LaTeX sequences for each sample in the batch.
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        features = self.encoder(src)  # (B, H'*W', d_model)
        max_mask = self.generate_square_subsequent_mask(max_len).to(device)
        
        final_outputs = []
        
        for b in range(batch_size):
            # Initialize the beam for sample b
            beams = [([bos_token_id], 0.0)]  # (sequence of tokens, log probability)
            completed_beams = []
            
            for step in range(1, max_len):
                candidates = []
                
                # Iterate through each current beam
                for seq, log_prob in beams:
                    if seq[-1] == eos_token_id:
                        completed_beams.append((seq, log_prob))
                        continue
                    
                    # Prepare input for the decoder
                    tgt = torch.tensor([seq], dtype=torch.long, device=device)  # (1, T)
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