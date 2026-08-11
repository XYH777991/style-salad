# Third-Party Notices

This repository includes a small amount of vendored utility code and depends on
external datasets, checkpoints, and model assets. The Style-SALAD source code is
licensed under the MIT License in `LICENSE`; third-party materials keep their
own licenses and terms.

## Vendored Code

- SALAD utility/model code is vendored from https://github.com/seokhyeonhong/salad
  for compatibility with the released `ours` configuration. A top-level SALAD
  license was not visible during this cleanup pass; confirm upstream license or
  permission before public release, and preserve upstream notices in copied files.
- SMooDi-related utility/evaluation code is derived from https://github.com/neu-vi/SMooDi,
  which is distributed under the MIT License.
- MLD-related utility code is derived from https://github.com/ChenFengYe/motion-latent-diffusion,
  which is distributed under the MIT License.

## External Runtime Assets

The following assets are not licensed by this repository unless explicitly noted:

- HumanML3D: obtain through the official HumanML3D project and follow its data
  terms. HumanML3D is derived from AMASS, so it is not redistributed directly by
  this repository.
- 100STYLE and retargeted 100STYLE: follow the original 100STYLE and SMooDi
  dataset terms.
- SALAD, MLD, and SMooDi pretrained checkpoints/evaluators: follow the upstream
  release terms for each asset.
- GloVe vocabulary/evaluator files and CLIP weights: follow their respective
  upstream licenses and model cards.

## Style-SALAD Materials

- Source code authored for Style-SALAD: MIT License, see `LICENSE`.
- Tracked Style-SALAD checkpoint: CC-BY-NC-ND-4.0, see
  `LICENSE-CHECKPOINTS.md`.
- Paper text, paper figures, and project-page media copied from the paper:
  CC-BY-NC-ND-4.0 unless stated otherwise.
