import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hàm xoay cặp phần tử bằng PyTorch thuần
def rotate_every_two(x):
    batch_size, h, w, d = x.size()
    assert d % 2 == 0
    x1 = x[:, :, :, :d//2]
    x2 = x[:, :, :, d//2:]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return x_rotated.view(batch_size, h, w, d)

# Image Rotary Embedding
class ImageRotaryEmbed(nn.Module):
    def __init__(self, d_model: int = 256, temperature: float = 10000, normalize: bool = False, scale: float = None):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        not_mask = ~mask
        embed_y = not_mask.cumsum(dim=1, dtype=torch.float32)
        embed_x = not_mask.cumsum(dim=2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            embed_y = embed_y / (embed_y[:, -1:, :] + eps) * self.scale
            embed_x = embed_x / (embed_x[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(0, self.half_d_model, 2, dtype=torch.float, device=x.device)
        inv_feq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))

        pos_x = torch.einsum("b h w, d -> b h w d", embed_x, inv_feq)
        pos_y = torch.einsum("b h w, d -> b h w d", embed_y, inv_feq)

        sin_x = pos_x.sin().repeat(1, 1, 1, 2)
        cos_x = pos_x.cos().repeat(1, 1, 1, 2)
        sin_y = pos_y.sin().repeat(1, 1, 1, 2)
        cos_y = pos_y.cos().repeat(1, 1, 1, 2)

        sin = torch.cat((sin_x, sin_y), dim=-1)
        cos = torch.cat((cos_x, cos_y), dim=-1)

        x = (x * cos) + (rotate_every_two(x) * sin)
        return x
  
class ViTBackbone(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=256, num_heads=8,
                 num_layers=3, mlp_dim=1024, dropout=0.1):
        super(ViTBackbone, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)

        x = x + self.pos_embedding
        x = self.transformer(x)  # (batch_size, num_patches + 1, embed_dim)
        return x

class PositionForestTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=3, mlp_dim=1024, max_depth=5, num_patches_h=14, num_patches_w=14):
        super(PositionForestTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.max_depth = max_depth
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w

        self.rotary_embed = ImageRotaryEmbed(d_model=embed_dim, normalize=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.depth_predictor = nn.Linear(embed_dim, max_depth)
        self.relative_pos_predictor = nn.Linear(embed_dim, 5)
        self.structure_classifier = nn.Linear(embed_dim, 5)

    def _build_forest(self, struct_probs, depth_probs, rel_pos_probs):
        batch_size = struct_probs.size(1)
        seq_len = struct_probs.size(0)

        depth_idx = torch.argmax(depth_probs, dim=-1)  # (seq_len, batch_size)
        rel_pos_idx = torch.argmax(rel_pos_probs, dim=-1)  # (seq_len, batch_size)

        depth_diff = depth_idx.unsqueeze(0) - depth_idx.unsqueeze(1)  # (seq_len, seq_len, batch_size)
        is_child = (depth_diff > 0).float()  # (seq_len, seq_len, batch_size)

        struct_prob = struct_probs[:, :, 0].unsqueeze(0)  # fraction prob (seq_len, 1, batch_size)
        rel_pos_prob = rel_pos_probs[:, :, 1:3].sum(dim=-1).unsqueeze(1)  # L + R (seq_len, 1, batch_size)
        adjacency = is_child * struct_prob * rel_pos_prob  # (seq_len, seq_len, batch_size)

        adjacency = adjacency / (adjacency.sum(dim=1, keepdim=True) + 1e-6)
        return adjacency

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1) - 1

        patches = x[:, 1:].view(batch_size, self.num_patches_h, self.num_patches_w, self.embed_dim)
        mask = torch.zeros(batch_size, self.num_patches_h, self.num_patches_w, dtype=torch.bool, device=x.device)

        patches = self.rotary_embed(patches, mask)
        patches = patches.view(batch_size, seq_len, self.embed_dim)
        x = torch.cat((x[:, :1], patches), dim=1)

        x = x.transpose(0, 1)
        x = self.transformer(x)

        x_patches = x[1:]  # (seq_len, batch_size, embed_dim)

        depth_logits = self.depth_predictor(x_patches)
        depth_probs = F.softmax(depth_logits, dim=-1)

        rel_pos_logits = self.relative_pos_predictor(x_patches)
        rel_pos_probs = F.softmax(rel_pos_logits, dim=-1)

        struct_logits = self.structure_classifier(x_patches)
        struct_probs = F.softmax(struct_logits, dim=-1)

        adjacency = self._build_forest(struct_probs, depth_probs, rel_pos_probs)
        forest_features = torch.bmm(adjacency.transpose(0, 2), x_patches.transpose(0, 1))
        forest_features = forest_features.transpose(0, 1)
        x_patches = x_patches + forest_features

        return x_patches

class PosFormerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=3, mlp_dim=1024, dropout=0.1):
        super(PosFormerDecoder, self).__init__()
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        pos_encoding = torch.zeros(1000, embed_dim)
        position = torch.arange(0, 1000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding, persistent=False)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.iac_mask = nn.Parameter(torch.zeros(1, 1, vocab_size), requires_grad=True)

    def forward(self, memory, tgt):
        tgt_embed = self.embedding(tgt) * torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float, device=tgt.device))
        tgt_len = tgt.size(0)
        tgt_embed = tgt_embed + self.pos_encoding[:tgt_len, :].unsqueeze(1)

        output = self.transformer_decoder(tgt_embed, memory)
        iac_weight = torch.sigmoid(self.iac_mask.expand(tgt_len, -1, -1))
        output = output * iac_weight
        output = self.fc_out(output)
        return output

class PosFormer(nn.Module):
    def __init__(self, vocab_size, in_channels=3, embed_dim=256, num_heads=8, num_layers=3, mlp_dim=1024, dropout=0.1, max_depth=5):
        super(PosFormer, self).__init__()
        self.backbone = ViTBackbone(
            image_size=224, patch_size=16, in_channels=in_channels, embed_dim=embed_dim,
            num_heads=num_heads, num_layers=num_layers, mlp_dim=mlp_dim, dropout=dropout
        )
        self.position_forest_transformer = PositionForestTransformer(
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, max_depth=max_depth, mlp_dim=mlp_dim
        )
        self.decoder = PosFormerDecoder(
            vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, mlp_dim=mlp_dim, dropout=dropout
        )

    def forward(self, x, tgt):
        features = self.backbone(x)
        memory = self.position_forest_transformer(features)
        output = self.decoder(memory, tgt)
        return output