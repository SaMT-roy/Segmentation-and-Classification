import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        """
        x: (B, N, D)
        """
        B, N, D = x.shape

        qkv = self.qkv(x)                      # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, H, N, Hd)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                         # (B, H, N, Hd)
        out = out.transpose(1, 2).reshape(B, N, D)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
class MLP(nn.Module):  # SwiGLU
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=True)
        self.w_value = nn.Linear(dim, hidden_dim, bias=True)
        self.w_out = nn.Linear(hidden_dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))     # SiLU on gate
        value = self.w_value(x)
        x = gate * value
        x = self.w_out(x)
        x = self.drop(x)
        return x
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, N, D)
        # Calculate RMS: sqrt(mean(x^2))
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
    
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=3.0,
        attn_dropout=0.0,
        dropout=0.0
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads, attn_dropout, dropout
        )

        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(
            dim,
            int(dim * mlp_ratio),
            dropout
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # Pre-LN
        x = x + self.mlp(self.norm2(x))
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        x = self.proj(x)          # (B, D, H/P, W/P)
        H, W = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x, (H, W)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=3.0,
        dropout=0.0,
        attn_dropout=0.0,
        base_grid_size=(16, 16),  # reference size (e.g. 256/16)
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            patch_size, in_channels, embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        num_base_patches = base_grid_size[0] * base_grid_size[1]

        self.pos_embed = nn.Parameter(torch.zeros(1, num_base_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                attn_dropout,
                dropout
            )
            for _ in range(depth)
        ])

        self.norm = RMSNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def interpolate_pos_encoding(self, x, grid_hw):
        """
        x: (B, N+1, D)
        grid_hw: (H, W) patch grid
        """
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]

        H0 = W0 = int(patch_pos.shape[1] ** 0.5)
        patch_pos = patch_pos.reshape(1, H0, W0, -1).permute(0, 3, 1, 2)

        patch_pos = F.interpolate(
            patch_pos,
            size=grid_hw,
            mode="bicubic",
            align_corners=False
        )

        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, patch_pos.shape[1])
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward(self, x, masks=None):
        """
        x: (B, C, H, W)
        """
        B = x.shape[0]

        x, (H, W) = self.patch_embed(x)  # (B, N, D)

        if masks is not None:
            # Expand mask token to batch size
            mask_token = self.mask_token.expand(B, x.shape[1], -1)
            
            # Create a binary mask tensor of shape (B, N, 1)
            # masks is (B, N), make it (B, N, 1) to broadcast over Dimension D
            w = masks.unsqueeze(-1).type_as(x)
            
            # Where w is 1 (True), use mask_token. Where 0, use x.
            x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.interpolate_pos_encoding(x, (H, W))
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x
