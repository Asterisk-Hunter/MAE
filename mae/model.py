"""Masked Autoencoders (MAE) -- He et al., "Masked Autoencoders Are Scalable
Vision Learners", CVPR 2022 (arXiv:2111.06377).

Asymmetric design:
  * Encoder -- deep ViT that sees only the *visible* patches (~25% at a 0.75
    mask ratio), so attention cost drops by the square of the keep rate.
  * Decoder -- light ViT that receives the encoded visible tokens plus learnable
    ``[mask]`` tokens and predicts raw pixels for every patch.
  * Loss -- MSE on the masked patches only.

Config is tuned for small images (32x32 CIFAR-10) on a 6 GB GPU. Because a
16x16 patch grid would leave only 4 tokens on a 32x32 image (degenerate under a
high mask ratio), the default patch size here is 4 -> a 8x8 = 64 token grid.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


def sincos_2d(grid_size: int, dim: int) -> torch.Tensor:
    """Fixed 2D sine-cosine positional embeddings, shape (grid_size**2, dim)."""
    if dim % 4 != 0:
        raise ValueError("embed dim must be divisible by 4 for 2D sincos pos-emb")
    coords = torch.arange(grid_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    omega = 1.0 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)))
    return torch.cat(
        [
            torch.sin(xx.reshape(-1, 1) * omega),
            torch.cos(xx.reshape(-1, 1) * omega),
            torch.sin(yy.reshape(-1, 1) * omega),
            torch.cos(yy.reshape(-1, 1) * omega),
        ],
        dim=1,
    )


class PatchEmbed(nn.Module):
    """Non-overlapping patchify via stride == kernel convolution."""

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 192):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} not divisible by patch_size {patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_dim = patch_size * patch_size * in_chans
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                  # B, D, g, g
        return x.flatten(2).transpose(1, 2)  # B, N, D


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # scales by 1/sqrt(head_dim) internally
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedAutoencoderViT(nn.Module):
    """MAE with an asymmetric encoder/decoder and per-sample random masking."""

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 192, depth: int = 12, num_heads: int = 3,
                 decoder_embed_dim: int = 128, decoder_depth: int = 4,
                 decoder_num_heads: int = 4, mlp_ratio: float = 4.0,
                 norm_pix_loss: bool = False):
        super().__init__()
        self.config = dict(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
            norm_pix_loss=norm_pix_loss,
        )
        self.norm_pix_loss = norm_pix_loss

        # ---- encoder (operates on visible patches only) ----
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.register_buffer("enc_pos_embed", sincos_2d(self.patch_embed.grid_size, embed_dim),
                             persistent=False)
        self.enc_blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.enc_norm = nn.LayerNorm(embed_dim)

        # ---- decoder (sees all patches, including [mask] tokens) ----
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.register_buffer("dec_pos_embed", sincos_2d(self.patch_embed.grid_size, decoder_embed_dim),
                             persistent=False)
        self.dec_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)]
        )
        self.dec_norm = nn.LayerNorm(decoder_embed_dim)
        self.dec_pred = nn.Linear(decoder_embed_dim, self.patch_embed.patch_dim)

        self.apply(self._init_weights)
        trunc_normal_(self.mask_token, std=0.02)

    @staticmethod
    def default_config() -> dict:
        return dict(
            img_size=32, patch_size=4, in_chans=3,
            embed_dim=192, depth=12, num_heads=3,
            decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
            mlp_ratio=4.0, norm_pix_loss=False,
        )

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    # ------------------------------------------------------------------ ops
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, patch_size**2 * C)."""
        p = self.patch_embed.patch_size
        g = self.patch_embed.grid_size
        B, C = imgs.shape[:2]
        x = imgs.reshape(B, C, g, p, g, p)
        x = x.permute(0, 2, 4, 3, 5, 1)          # B, g, g, p, p, C
        return x.reshape(B, g * g, p * p * C)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_size**2 * C) -> (B, C, H, W)."""
        p = self.patch_embed.patch_size
        g = self.patch_embed.grid_size
        C = self.patch_embed.in_chans
        B = x.shape[0]
        x = x.reshape(B, g, g, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)          # B, C, g, p, g, p
        return x.reshape(B, C, g * p, g * p)

    def random_masking(self, x: torch.Tensor, mask_ratio: float,
                       generator: torch.Generator | None = None):
        """Per-sample random token masking.

        Returns the visible subset, the (B, N) mask (1 == masked), and
        ``ids_restore`` to un-shuffle the decoder sequence. Passing a seeded
        ``generator`` makes the permutation reproducible, which the demo uses to
        keep visible patches *nested* as the mask ratio is scrubbed upward.
        """
        B, N, D = x.shape
        len_keep = max(1, int(N * (1 - mask_ratio)))
        noise = torch.rand(B, N, device=x.device, generator=generator)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_visible, mask, ids_restore

    def forward_encoder(self, imgs: torch.Tensor, mask_ratio: float,
                        generator: torch.Generator | None = None):
        x = self.patch_embed(imgs) + self.enc_pos_embed
        x, mask, ids_restore = self.random_masking(x, mask_ratio, generator)
        for blk in self.enc_blocks:
            x = blk(x)
        return self.enc_norm(x), mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)
        B, L, D = x.shape
        N = self.patch_embed.num_patches
        mask_tokens = self.mask_token.expand(B, N - L, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = x + self.dec_pos_embed
        for blk in self.dec_blocks:
            x = blk(x)
        return self.dec_pred(self.dec_norm(x))

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                       # B, N
        return (loss * mask).sum() / mask.sum()

    # -------------------------------------------------------------- forward
    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75,
                generator: torch.Generator | None = None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, generator)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    @torch.no_grad()
    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Mean-pooled encoder features (B, embed_dim) for downstream probes."""
        x = self.patch_embed(imgs) + self.enc_pos_embed
        for blk in self.enc_blocks:
            x = blk(x)
        return self.enc_norm(x).mean(dim=1)

    @torch.no_grad()
    def reconstruct(self, imgs: torch.Tensor, mask_ratio: float = 0.75,
                    generator: torch.Generator | None = None):
        """Inference helper for visualization, all outputs in [0, 1] pixels.

        Returns ``(reconstruction, masked_input, mask)`` where ``mask`` is
        (B, N) with 1 marking a masked patch.
        """
        _, pred, mask = self.forward(imgs, mask_ratio, generator)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            pred = pred * (var + 1e-6) ** 0.5 + mean
        rec = self.unpatchify(pred).clamp(0, 1)
        masked_img = self.unpatchify(target * (1 - mask).unsqueeze(-1))
        return rec, masked_img, mask
