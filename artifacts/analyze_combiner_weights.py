"""Diagnostic: does style_combiner's Linear(512, 256) suppress the
pure_style_encoder (s_pure) branch relative to the mixing style_encoder
branch (s_gen)?

Motivation (see STYLE_ENCODER_RESEARCH_LOG.md #16-17): the dual-encoder run
(reruns/ours_train_stattn_dualenc) still trails the StyleMLP baseline's
SRA_5 by ~6pt (81.10 vs 86.27). One hypothesis was that style_combiner --
which has a live gradient from the diffusion loss on the s_gen half but
only a detached input on the s_pure half -- could learn to lean on the
content-biased s_gen branch (better for reconstruction) and dilute the
discriminative s_pure signal in style_for_gen.

This script checks that hypothesis via the trained combiner weight matrix
alone (no data/forward pass needed). Usage:

    python artifacts/analyze_combiner_weights.py [checkpoint_path]
"""
import sys

import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/t2sm/reruns/ours_train_stattn_dualenc/epoch_0100.ckpt"


def main():
    sd = torch.load(CKPT, map_location="cpu")
    W = sd["style_combiner.weight"]  # (style_dim, 2*style_dim)
    b = sd["style_combiner.bias"]
    style_dim = W.shape[0]
    assert W.shape[1] == 2 * style_dim, f"unexpected style_combiner.weight shape {tuple(W.shape)}"

    # torch.cat([s_gen, s_pure.detach()], dim=1) in t2sm.py -> first half
    # multiplies s_gen (mixing branch, live gradient), second half
    # multiplies s_pure.detach() (pure branch, detached).
    W_gen, W_pure = W[:, :style_dim], W[:, style_dim:]

    fro_gen, fro_pure = W_gen.norm().item(), W_pure.norm().item()
    total = fro_gen + fro_pure

    row_gen, row_pure = W_gen.norm(dim=1), W_pure.norm(dim=1)
    gen_wins = int((row_gen > row_pure).sum().item())

    print(f"checkpoint: {CKPT}")
    print(f"||W_gen||_F  = {fro_gen:.4f}  ({100 * fro_gen / total:.1f}% of combined weight mass)")
    print(f"||W_pure||_F = {fro_pure:.4f}  ({100 * fro_pure / total:.1f}% of combined weight mass)")
    print(f"bias norm    = {b.norm().item():.4f}")
    print(f"output dims where |row_gen| > |row_pure|: {gen_wins}/{style_dim}")
    print(f"mean row norm gen={row_gen.mean().item():.4f}  pure={row_pure.mean().item():.4f}"
          f"  ratio gen/pure={(row_gen.mean() / row_pure.mean()).item():.3f}")

    if fro_pure > fro_gen:
        print("\n-> combiner weight mass leans toward s_pure, not s_gen: the "
              "'diffusion loss dilutes the pure branch via the combiner' "
              "hypothesis is NOT supported by weight norms alone.")
    else:
        print("\n-> combiner weight mass leans toward s_gen: consistent with "
              "the 'diffusion loss dilutes the pure branch' hypothesis.")


if __name__ == "__main__":
    main()
