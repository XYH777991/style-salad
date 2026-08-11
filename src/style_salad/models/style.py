import torch.nn as nn


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


STYLE_REGISTRY = {"StyleMLP": StyleMLP}
