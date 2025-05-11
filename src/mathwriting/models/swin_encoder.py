import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x
    
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Linear layers for query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B_, N, C = x.shape  # B_ = batch_size * num_windows, N = window_size^2, C = embed_dim
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Mỗi cái có shape (B_, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, num_heads, N, N)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            # Mask có shape (nW, N, N), cần lặp lại cho mỗi head và batch
            mask = mask.unsqueeze(0).unsqueeze(2)  # (1, nW, 1, N, N)
            attn = attn.masked_fill(mask, float('-inf'))  # Che các vị trí True
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=3.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape  # B = batch_size, L = num_patches, C = embed_dim
        assert L == H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        x_windows = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)

        attn_mask = None
        if self.shift_size > 0:
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = img_mask.view(-1, 1, self.window_size * self.window_size)
            attn_mask = mask_windows - mask_windows.transpose(1, 2)  # (nW, ws*ws, ws*ws)
            attn_mask = attn_mask != 0  # Boolean mask: True = che, False = không che

        x_windows = self.attn(x_windows, mask=attn_mask)

        x = x_windows.view(-1, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        x = x.view(B, H, W, C)

        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, (H // 2) * (W // 2), 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2

class SwinEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=64, depths=[1, 1, 2],
                 num_heads=[2, 4, 8], window_size=7, mlp_ratio=3.0, dropout=0.1):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_drop = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinBlock(
                    dim=int(embed_dim * 2 ** i_layer),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                ) for i in range(depths[i_layer])
            ])
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.layers.append(PatchMerging(dim=int(embed_dim * 2 ** i_layer)))

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.proj_out = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)), 256)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        x = self.pos_drop(x)
        B, L, C = x.shape
        H, W = int(L ** 0.5), int(L ** 0.5)

        for layer in self.layers:
            if isinstance(layer, PatchMerging):
                x, H, W = layer(x, H, W)
            else:
                for block in layer:
                    x = block(x, H, W)

        x = self.norm(x)
        x = self.proj_out(x)  # (B, num_patches, 256)
        return x