import torch
import torch.nn as nn


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    """Identity on the forward pass; negates (and scales by lambda_) the
    gradient on the backward pass. Standard building block from
    domain-adversarial training (Ganin & Lempitsky, ICML 2015 / JMLR 2016)
    for making a representation invariant to whatever an attached classifier
    is trained to predict from it."""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.lambda_)


class ContentAdversary(nn.Module):
    """Predicts content_idx from the style embedding through a
    GradientReversal layer. self.net trains normally and gets as good as
    possible at this prediction; because of the GRL, the *upstream* style
    encoder receives the negated gradient instead and is pushed to make
    content unpredictable from `s` -- same mechanism as Fader Networks
    (Lample et al., NeurIPS 2017) applied to an arbitrary attribute, here
    applied to 100STYLE's content label instead of an image attribute."""

    def __init__(self, style_dim, num_classes, hidden_dim=128, lambda_=1.0):
        super().__init__()
        self.grl = GradientReversal(lambda_)
        self.net = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, style):
        return self.net(self.grl(style))
