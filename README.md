# Style-SALAD

> **Status: release artifact**
>
> This repository contains the released Style-SALAD code path for the `ours`
> configuration, with large third-party assets kept outside git.

Style-SALAD generates stylized human motion from a content text prompt and a
reference style motion. This repository keeps the public tree lightweight: it
contains the code paths for the released `ours` configuration, plus the minimal
SALAD/MLD/SMooDi utility code required at runtime. Large datasets, checkpoints,
CLIP caches, generated videos, and metric outputs are intentionally excluded from
git.

## Setup

Install `ffmpeg` first if you want MP4 previews from generation:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Create the Python environment:

```bash
conda env create -f environment.yml
conda activate style-salad
```

If you manage PyTorch separately, install the CUDA build that matches your
machine first, then run:

```bash
pip install -r requirements.txt
pip install -e .
```

The CLIP text encoder is loaded through Hugging Face Transformers on first use
and is cached under `checkpoints/clip-vit-base-patch32/` by default. For offline
or cluster runs, pre-populate that cache before launching generation or training.

## Assets

`configs/ours.yaml` is the source of truth for paths. The Style-SALAD checkpoint
`checkpoints/t2sm/ours/epoch_0100.ckpt` is tracked in this repository. The small
100STYLE metadata files used by the released config are tracked too:
`dataset/100style/{100style_smoodi.json,10content_smoodi.json,100STYLE_name_dict_Filter.txt,Mean.npy,Std.npy,test_humanml.txt,test_100STYLE_Filter.txt}`.
The remaining large runtime assets are not tracked.

For generation and training with `configs/ours.yaml`, you need these external
assets:

```text
checkpoints/
  t2m/
    t2m_denoiser_vpred_vaegelu/opt.txt
    t2m_denoiser_vpred_vaegelu/model/net_best_fid.tar
    t2m_vae_gelu/opt.txt
    t2m_vae_gelu/model/net_best_fid.tar
    Comp_v6_KLD005/meta/{mean.npy,std.npy}
dataset/
  100style/
    new_joint_vecs/
```

Generation also needs CLIP weights. By default they are downloaded through
Hugging Face Transformers and cached under `checkpoints/clip-vit-base-patch32/`.
For offline or cluster runs, pre-populate that cache before launching.

Full evaluation additionally needs these external assets:

```text
checkpoints/
  t2m/
    Comp_v6_KLD01/meta/{mean.npy,std.npy}
    text_mot_match/model/finest.tar
  style_classifier/style_classifier.pt
glove/
  our_vab_data.npy
  our_vab_idx.pkl
  our_vab_words.pkl
dataset/
  100style/
    texts/
  humanml3d/
    new_joint_vecs/
    texts/
```

Evaluation asset roles:

- GloVe files turn HumanML3D caption tokens into word embeddings and POS one-hot
  vectors for the T2M text encoder.
- `Comp_v6_KLD01` mean/std renormalize generated and ground-truth motions into
  the feature normalization expected by the T2M evaluator.
- `text_mot_match/model/finest.tar` provides the pretrained T2M text, movement,
  and motion encoders used for matching score, R-precision, FID, and diversity.
- `style_classifier.pt`, `100STYLE_name_dict_Filter.txt`, and 100STYLE
  `Mean.npy`/`Std.npy` are used for style recognition accuracy.

Not read by the current `ours` entrypoints: `dataset/humanml3d/test.txt`,
`train_random_humanml.txt`, `valid_random_humanml.txt`, HumanML3D `Mean.npy` /
`Std.npy`, and 100STYLE `texts/` during generation/training.

Where to get assets:

| Asset | Source | Destination |
| --- | --- | --- |
| SALAD HumanML3D weights, VAE/denoiser opts, and T2M eval stats | [SALAD `download_t2m.sh`](https://github.com/seokhyeonhong/salad/blob/main/prepare/download_t2m.sh), which downloads this [Google Drive folder](https://drive.google.com/drive/folders/1YuDQCgc6RJ4WlR9vt_L34nkXChC7C98_?usp=drive_link) | `checkpoints/t2m/` |
| T2M evaluator fallback | [MLD `download_t2m_evaluators.sh`](https://github.com/ChenFengYe/motion-latent-diffusion/blob/main/prepare/download_t2m_evaluators.sh) | `checkpoints/t2m/text_mot_match/model/finest.tar` |
| GloVe evaluator vocabulary | [SALAD `download_glove.sh`](https://github.com/seokhyeonhong/salad/blob/main/prepare/download_glove.sh), which downloads this [Google Drive folder](https://drive.google.com/drive/folders/1C78gilIEImoXTzl3_KaY7EwJt4Yy1Rav?usp=drive_link) | `glove/` |
| HumanML3D motions/texts | [HumanML3D instructions](https://github.com/EricGuo5513/HumanML3D) | `dataset/humanml3d/{new_joint_vecs,texts}/` |
| Retargeted 100STYLE motions/texts | [SMooDi retargeted 100STYLE Google Drive folder](https://drive.google.com/drive/folders/1P_aQdSuiht3gh1kjGkK4KBt_9i9ARawy?usp=drive_link) | `dataset/100style/{new_joint_vecs,texts}/` |
| SMooDi style classifier | [SMooDi pretrained weights Google Drive folder](https://drive.google.com/drive/folders/12m_v_vybVeAQFkH9bP8wmJIxJhGoIJL1?usp=sharing); SMooDi config uses `./experiments/style_classifier.pt` | `checkpoints/style_classifier/style_classifier.pt` |
| CLIP text encoder | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32), auto-downloaded by Transformers unless pre-cached | `checkpoints/clip-vit-base-patch32/` |

See `docs/ASSETS.md` for the full file layout and source links.

## Generate

```bash
style-salad-generate \
  --config configs/ours.yaml \
  --ref_motion_id 031617 \
  --caption "a person walks forward" \
  --num_samples 8 \
  --output_length 68
```

This writes MP4 previews, `results.npy`, and `metadata.json` under
`outputs/ours/...`.

You can also use the wrapper script:

```bash
scripts/generate.sh --ref_motion_id 031617 --caption "a person walks forward"
```

## Evaluate

```bash
style-salad-evaluate --config configs/ours.yaml --csv_name metrics_ours.csv
```

Evaluation uses the Style-SALAD checkpoint, style classifier, T2M text-motion
evaluator, HumanML3D test split, and 100STYLE reference motions configured in
`configs/ours.yaml`. Results are appended to
`artifacts/evaluation/metrics_ours.csv` by default.

The paired evaluation split files are tracked at
`dataset/100style/test_humanml.txt` and
`dataset/100style/test_100STYLE_Filter.txt`.

## Train

```bash
style-salad-train --config configs/ours.yaml
```

Training requires a CUDA-capable GPU. It writes checkpoints to
`checkpoints/t2sm/ours/` and logs/plots under `outputs/ours/` and `artifacts/`.

## Style encoder research: an improved training recipe

`configs/ours.yaml` (above) is the released checkpoint, tracked as-is. This
fork also contains a from-scratch investigation into a real gap between
that released checkpoint and this codebase's own retrains of the same
recipe, plus a fix. Full writeup, every experiment tried (including the
ones that didn't work and why), and every raw config/log/eval CSV behind
it are in `artifacts/STYLE_ENCODER_RESEARCH_LOG.md`.

**Current recommended recipe**:
`configs/reruns/ours_train_stattn_dualenc_nodetach.yaml`. In short: a
`StyleSTAttn` mixing encoder (temporal + skeletal self-attention before
pooling) drives generation, fused via `style_combiner` with a separate,
non-mixing `pure_style_encoder` that trains directly on the contrastive
loss and is *not* protected from the diffusion objective's gradient
(`detach_from_generation: false`). Result, averaged over 4 seeds: SRA_5 =
87.88 ± 0.63, statistically tied with the hard-SupCon baseline's
87.12 ± 1.35 (z = 0.56), with FID also tied and a small R-Precision@3 cost
(z = −2.01).

This recipe is a config + reproducible training run, **not** a tracked
checkpoint — `checkpoints/t2sm/reruns/` is gitignored like every other
from-scratch retrain in this repo (see Assets above; only
`checkpoints/t2sm/ours/epoch_0100.ckpt` ships in git). To reproduce it:

```bash
style-salad-train --config configs/reruns/ours_train_stattn_dualenc_nodetach.yaml
style-salad-evaluate --config configs/reruns/ours_train_stattn_dualenc_nodetach.yaml \
  --csv_name metrics_dualenc_nodetach.csv
```

`configs/reruns/` also holds every other config from the investigation
(seed variants, every ablation that was tried and ruled out) if you want
to reproduce a specific research-log entry.

A related, separate finding: generated motion tempo (how fast a style
plays out) is not transferred by style guidance at all, regardless of
recipe — root cause and an experimental (opt-in, not default)
`tempo_guidance` sampling-time mitigation are documented in the same
research log and in `src/style_salad/models/t2sm.py`
(`_tempo_guidance_loss`).

## Acknowledgements

This repository vendors a minimal subset of SALAD and MLD/SMooDi utility code
needed by Style-SALAD. Please cite or acknowledge the upstream projects and
datasets used in your experiments, including SALAD, SMooDi, Motion Latent
Diffusion, HumanML3D, and 100STYLE.

## License

Style-SALAD source code is released under the MIT License; see `LICENSE`. The
tracked Style-SALAD checkpoint at `checkpoints/t2sm/ours/epoch_0100.ckpt` is
released under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (`CC-BY-NC-ND-4.0`); see `LICENSE-CHECKPOINTS.md`. Paper text,
paper figures, and project-page media copied from the paper are also
`CC-BY-NC-ND-4.0` unless stated otherwise.

Third-party datasets, checkpoints, CLIP/GloVe assets, and vendored utility code
keep their own licenses and citation requirements; see `THIRD_PARTY_NOTICES.md`.
