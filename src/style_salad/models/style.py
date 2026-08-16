import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        style_dim = int(config["style_dim"])
        self.mlp = nn.Sequential(
            nn.Linear(32, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, x, mask):
        if mask is None:
            raise ValueError("StyleMLP.forward requires a non-None mask.")
        style = self.mlp(x)
        valid = mask[:, :, None, None].to(style.dtype)
        style = style * valid
        numerator = style.sum(dim=(1, 2))
        denominator = valid.sum(dim=(1, 2)).clamp(min=1e-5)
        return numerator / denominator


class StyleAttnPool(nn.Module):
    """Same per-cell 4-layer MLP as StyleMLP, but pools T*J cells with a
    single learnable query (scaled dot-product attention, O(T*J) cost)
    instead of a uniform masked mean. Padded cells get score=-inf so they
    always receive zero weight, same invariant the mean-pool denominator
    enforced via masking."""

    def __init__(self, config):
        super().__init__()
        style_dim = int(config["style_dim"])
        self.mlp = nn.Sequential(
            nn.Linear(32, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
        )
        self.query = nn.Parameter(torch.randn(style_dim) * (style_dim ** -0.5))
        self.scale = style_dim ** -0.5

    def forward(self, x, mask):
        if mask is None:
            raise ValueError("StyleAttnPool.forward requires a non-None mask.")
        v = self.mlp(x)  # (B, T, J, D)
        B, T, J, D = v.shape

        scores = torch.einsum("btjd,d->btj", v, self.query) * self.scale  # (B, T, J)
        valid = mask[:, :, None].expand(B, T, J)
        scores = scores.masked_fill(~valid, float("-inf"))

        weights = F.softmax(scores.reshape(B, T * J), dim=1).reshape(B, T, J)
        weights = torch.nan_to_num(weights, nan=0.0)  # guard: a fully-masked sample would divide 0/0

        style = torch.einsum("btj,btjd->bd", weights, v)
        return style


class StyleSTAttn(nn.Module):
    """Lets T*J cells exchange information before pooling, instead of the
    fully independent per-cell MLP used by StyleMLP/StyleAttnPool.

    One factored temporal self-attention (fixed joint, T frames attend to
    each other) followed by one factored skeletal self-attention (fixed
    frame, 22 joints attend to each other) -- O(T^2*J + J^2*T) instead of a
    full O((T*J)^2) joint attention. Learned temporal/joint positional
    embeddings are added first because attention, like mean pooling, is
    permutation-invariant on its own and has no notion of order otherwise.
    Pooling after these two layers is the same masked mean pool as
    StyleMLP -- only the "do cells talk to each other first" part changes.

    use_temporal_attn / use_skeletal_attn (both default True) let either
    sub-layer be switched off for ablation -- e.g. temporal-only to check
    whether cross-token (skeletal) mixing specifically is what collapses
    style expression, vs. cross-frame (temporal) mixing.
    """

    def __init__(self, config):
        super().__init__()
        style_dim = int(config["style_dim"])
        num_heads = int(config.get("num_heads", 4))
        dropout = float(config.get("dropout", 0.0))
        self.max_len = int(config.get("max_len", 64))
        # NOT the 22 raw skeleton joints (that's vae_opt.joints_num, used for
        # FK/decoding elsewhere). SALAD's ST-VAE groups those into 7
        # structural tokens per frame -- confirmed empirically via
        # vae.encode(...).shape == (B, T, 7, latent_dim) -- and that grouped
        # axis is what StyleSTAttn's "skeletal" attention actually runs over.
        self.num_joints = int(config.get("num_joints", 7))
        self.use_temporal_attn = bool(config.get("use_temporal_attn", True))
        self.use_skeletal_attn = bool(config.get("use_skeletal_attn", True))
        if not (self.use_temporal_attn or self.use_skeletal_attn):
            raise ValueError("StyleSTAttn: at least one of use_temporal_attn/use_skeletal_attn must be True.")

        # Centering (DINO-style; a lighter cousin of BERT-whitening/W-MSE
        # full covariance whitening) against measured anisotropy: pooled
        # embeddings -- trained or not -- had ~0.8-0.997 pairwise cosine
        # similarity regardless of style label, i.e. nearly all of the 256
        # dims were spent on a shared direction supcon never gets to touch.
        # Subtracting a running batch mean before normalize removes that
        # shared direction so cosine similarity is computed on the residual,
        # actually-discriminative directions instead. affine=False: only
        # recenters (and unit-variances per-dim), no learned scale/shift --
        # keep this a fixed correction, not new capacity to fit with.
        self.use_centering = bool(config.get("use_centering", False))
        if self.use_centering:
            self.pool_norm = nn.BatchNorm1d(style_dim, affine=False)

        self.input_proj = nn.Linear(32, style_dim)
        self.temporal_pos = nn.Parameter(torch.zeros(self.max_len, style_dim))
        self.joint_pos = nn.Parameter(torch.zeros(self.num_joints, style_dim))
        nn.init.normal_(self.temporal_pos, std=0.02)
        nn.init.normal_(self.joint_pos, std=0.02)

        if self.use_temporal_attn:
            self.temporal_attn = nn.MultiheadAttention(style_dim, num_heads, dropout=dropout, batch_first=True)
            self.temporal_norm = nn.LayerNorm(style_dim)

        if self.use_skeletal_attn:
            self.skeletal_attn = nn.MultiheadAttention(style_dim, num_heads, dropout=dropout, batch_first=True)
            self.skeletal_norm = nn.LayerNorm(style_dim)

    def forward(self, x, mask):
        if mask is None:
            raise ValueError("StyleSTAttn.forward requires a non-None mask.")
        B, T, J, _ = x.shape
        if T > self.max_len:
            raise ValueError(f"StyleSTAttn: motion length {T} exceeds max_len={self.max_len}.")
        if J != self.num_joints:
            raise ValueError(
                f"StyleSTAttn: latent's structural axis has size {J}, but joint_pos was "
                f"built for num_joints={self.num_joints}. Pass the correct num_joints in "
                f"config (check vae.encode(...).shape[2] for the VAE actually in use)."
            )

        x = self.input_proj(x)                                     # (B,T,J,style_dim)
        x = x + self.temporal_pos[:T][None, :, None, :]             # broadcast over B,J
        x = x + self.joint_pos[None, None, :, :]                    # broadcast over B,T

        # ---- temporal self-attention: fix joint, let its T frames attend to each other ----
        if self.use_temporal_attn:
            xt = x.permute(0, 2, 1, 3).reshape(B * J, T, -1)         # (B*J, T, D)
            # True = ignore. Assumes every sample has >=1 valid frame (holds under
            # this dataset's min_frames filtering), so no row is fully masked.
            key_padding_mask = (~mask).repeat_interleave(J, dim=0)  # (B*J, T)
            attn_out, _ = self.temporal_attn(xt, xt, xt, key_padding_mask=key_padding_mask, need_weights=False)
            xt = self.temporal_norm(xt + attn_out)
            x = xt.reshape(B, J, T, -1).permute(0, 2, 1, 3)          # back to (B,T,J,D)

        # ---- skeletal self-attention: fix frame, let its structural tokens attend to each other ----
        if self.use_skeletal_attn:
            xs = x.reshape(B * T, J, -1)                             # (B*T, J, D)
            attn_out, _ = self.skeletal_attn(xs, xs, xs, need_weights=False)
            xs = self.skeletal_norm(xs + attn_out)
            x = xs.reshape(B, T, J, -1)

        # ---- same masked mean pool as StyleMLP ----
        valid = mask[:, :, None, None].to(x.dtype)
        x = x * valid
        numerator = x.sum(dim=(1, 2))
        denominator = valid.sum(dim=(1, 2)).clamp(min=1e-5)
        pooled = numerator / denominator

        if self.use_centering:
            pooled = self.pool_norm(pooled)  # subtract the shared/anisotropic direction

        return pooled


STYLE_REGISTRY = {"StyleMLP": StyleMLP, "StyleAttnPool": StyleAttnPool, "StyleSTAttn": StyleSTAttn}
