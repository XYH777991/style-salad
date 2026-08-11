import math

import torch
import torch.nn as nn


class HyperLoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rank = int(config["rank"])
        self.scale = float(config["scale"])
        self.style_dim = int(config["style_dim"])
        self.in_dim = int(config["in_dim"])
        self.out_dim = int(config["out_dim"])
        hidden_dim = int(config.get("hidden_dim", self.style_dim))

        self.project = nn.Sequential(
            nn.Linear(self.style_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.style_dim),
        )
        self.head_A = nn.Sequential(
            nn.LayerNorm(self.style_dim),
            nn.Linear(self.style_dim, self.style_dim),
            nn.GELU(),
            nn.Linear(self.style_dim, self.rank * self.in_dim),
        )
        self.head_B = nn.Sequential(
            nn.LayerNorm(self.style_dim),
            nn.Linear(self.style_dim, self.style_dim),
            nn.GELU(),
            nn.Linear(self.style_dim, self.out_dim * self.rank),
        )

        nn.init.kaiming_uniform_(self.head_A[-1].weight, a=math.sqrt(5))
        nn.init.zeros_(self.head_A[-1].bias)
        nn.init.zeros_(self.head_B[-1].weight)
        nn.init.zeros_(self.head_B[-1].bias)

        self.cache_enabled = False
        self._cached_style_key = None
        self._cached_A = None
        self._cached_B = None

    def enable_cache(self, enabled=True):
        self.cache_enabled = bool(enabled)
        if not self.cache_enabled:
            self.clear_cache()

    def clear_cache(self):
        self._cached_style_key = None
        self._cached_A = None
        self._cached_B = None

    @staticmethod
    def _style_cache_key(style):
        if style.requires_grad:
            return None
        return (style.data_ptr(), tuple(style.shape), style.device.type, style.device.index, str(style.dtype))

    def forward(self, style, len_mask=None):
        del len_mask
        style_key = self._style_cache_key(style) if self.cache_enabled else None
        if style_key is not None and style_key == self._cached_style_key:
            return self._cached_A, self._cached_B

        batch_size = style.shape[0]
        projected = self.project(style)
        A = self.head_A(projected).view(batch_size, self.rank, self.in_dim)
        B = self.head_B(projected).view(batch_size, self.out_dim, self.rank)

        if style_key is not None:
            self._cached_style_key = style_key
            self._cached_A = A.detach()
            self._cached_B = B.detach()
        return A, B


LORA_REGISTRY = {"HyperLoRA": HyperLoRA}
