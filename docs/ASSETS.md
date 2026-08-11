# Runtime Assets

`configs/ours.yaml` is the source of truth for runtime paths. Keep large
datasets, pretrained backbones, evaluator assets, CLIP caches, generated videos,
and metric outputs outside git. The tracked runtime assets are the released
Style-SALAD checkpoint at `checkpoints/t2sm/ours/epoch_0100.ckpt` and the small
100STYLE metadata files under `dataset/100style/`.

## Source Links

- Style-SALAD project page: https://junhyukjeon.github.io/projects/style-salad/
- Style-SALAD repository: https://github.com/junhyukjeon/style-salad
- SALAD repository: https://github.com/seokhyeonhong/salad
- MLD repository: https://github.com/ChenFengYe/motion-latent-diffusion
- SMooDi repository: https://github.com/neu-vi/SMooDi
- HumanML3D repository: https://github.com/EricGuo5513/HumanML3D
- 100STYLE dataset page: https://www.ianxmason.com/100style/

## Download Links

| Needed asset | Upstream source | Expected local path |
| --- | --- | --- |
| SALAD HumanML3D pretrained weights, VAE/denoiser opts, and evaluator stats | SALAD provides `prepare/download_t2m.sh`, which downloads https://drive.google.com/drive/folders/1YuDQCgc6RJ4WlR9vt_L34nkXChC7C98_?usp=drive_link | `checkpoints/t2m/` |
| T2M evaluator checkpoint fallback | MLD provides `prepare/download_t2m_evaluators.sh`, which downloads `t2m.tar.gz` from https://drive.google.com/uc?id=1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8 | `checkpoints/t2m/text_mot_match/model/finest.tar` |
| GloVe evaluator vocabulary | SALAD provides `prepare/download_glove.sh`, which downloads https://drive.google.com/drive/folders/1C78gilIEImoXTzl3_KaY7EwJt4Yy1Rav?usp=drive_link | `glove/our_vab_*` |
| HumanML3D motions and captions | HumanML3D must be regenerated from the official project because AMASS-derived data is not redistributed directly: https://github.com/EricGuo5513/HumanML3D | `dataset/humanml3d/new_joint_vecs/`, `dataset/humanml3d/texts/` |
| Retargeted 100STYLE motions and captions | SMooDi provides the retargeted 100STYLE dataset at https://drive.google.com/drive/folders/1P_aQdSuiht3gh1kjGkK4KBt_9i9ARawy?usp=drive_link | `dataset/100style/new_joint_vecs/`, `dataset/100style/texts/` |
| SMooDi style classifier | SMooDi provides pretrained weights at https://drive.google.com/drive/folders/12m_v_vybVeAQFkH9bP8wmJIxJhGoIJL1?usp=sharing; their config points `PRETRAINED_STYLE` to `./experiments/style_classifier.pt` | `checkpoints/style_classifier/style_classifier.pt` |
| CLIP text encoder | Hugging Face model `openai/clip-vit-base-patch32`: https://huggingface.co/openai/clip-vit-base-patch32 | `checkpoints/clip-vit-base-patch32/` cache |

## Upstream Asset Pattern

- SALAD keeps large weights outside git and provides `gdown` scripts for
  HumanML3D/KIT pretrained weights, evaluator assets, and GloVe. This repo follows
  that pattern for the SALAD backbone assets.
- SMooDi publishes two Drive folders in its README: one for the retargeted
  100STYLE dataset and one for pretrained weights. The SMooDi config references
  the style classifier at `./experiments/style_classifier.pt`; for this repo, copy
  that checkpoint to `checkpoints/style_classifier/style_classifier.pt`.
- MLD provides dependency scripts for SMPL/CLIP/T2M evaluators plus a manual
  Google Drive folder for pretrained assets. This repo only needs the T2M
  evaluator layout used by the HumanML3D metrics.
- HumanML3D documents how to reproduce the dataset from AMASS-derived sources and
  lists the final `new_joint_vecs`, `texts`, and split-file structure. It does not
  directly redistribute HumanML3D because of the AMASS distribution policy.

## Generation And Training

Shared external assets required by both `style-salad-generate --config configs/ours.yaml`
and `style-salad-train --config configs/ours.yaml`:

```text
checkpoints/
  t2m/
    t2m_denoiser_vpred_vaegelu/opt.txt
    t2m_denoiser_vpred_vaegelu/model/net_best_fid.tar
    t2m_vae_gelu/opt.txt
    t2m_vae_gelu/model/net_best_fid.tar
    Comp_v6_KLD005/meta/mean.npy
    Comp_v6_KLD005/meta/std.npy
dataset/
  100style/
    new_joint_vecs/
```

The `100style_smoodi.json` and `10content_smoodi.json` metadata files are tracked
in this repo. They are derived from the filtered 47-style 100STYLE name
dictionary used by `configs/ours.yaml`.

Generation additionally loads the tracked Style-SALAD checkpoint:

```text
checkpoints/t2sm/ours/epoch_0100.ckpt
```

Training writes new checkpoints under `checkpoints/t2sm/ours/`; it does not need
the tracked `epoch_0100.ckpt` as an input.

Notes:

- `Comp_v6_KLD005` mean/std are the motion-feature normalization used for
  Style-SALAD/SALAD inputs and outputs during training and generation.
- The 100STYLE `texts/` directory is not read by the current generation/training
  dataset class; captions come from the content labels in code.
- CLIP is loaded through Hugging Face Transformers and cached under
  `checkpoints/clip-vit-base-patch32/` by default. Treat that cache as a runtime
  asset, especially for offline runs.
- MP4 generation requires `ffmpeg` to be installed on the system.

## Full Evaluation

Full `style-salad-evaluate --config configs/ours.yaml` additionally requires:

```text
checkpoints/
  t2m/
    Comp_v6_KLD01/meta/mean.npy
    Comp_v6_KLD01/meta/std.npy
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

The 100STYLE name dictionary, style mean/std files, and paired evaluation split
files are tracked in this repo.

Evaluation reads split files explicitly configured as:

```yaml
eval_humanml_split_file: ./dataset/100style/test_humanml.txt
eval_style_split_file: ./dataset/100style/test_100STYLE_Filter.txt
```

## Why These Evaluation Assets Exist

- `glove/our_vab_*`: `WordVectorizer` converts each HumanML3D caption token into
  a word embedding and POS one-hot vector. The T2M text encoder consumes these
  features when computing text-motion matching metrics.
- `Comp_v6_KLD01/meta/{mean.npy,std.npy}`: the generated motion and ground-truth
  HumanML3D motion are renormalized into the feature scale expected by the T2M
  evaluator before motion embeddings are computed.
- `text_mot_match/model/finest.tar`: this checkpoint provides the pretrained T2M
  text encoder, movement encoder, and motion encoder. Their embeddings are used
  for matching score, R-precision, FID, and diversity.
- `style_classifier/style_classifier.pt`: predicts the generated motion style;
  those logits are used for SRA, SRA@3, and SRA@5.
- `dataset/100style/Mean.npy`, `dataset/100style/Std.npy`, and
  `100STYLE_name_dict_Filter.txt`: put motions in the style-classifier feature
  scale and map style motion ids/classes to labels.
- 100STYLE and HumanML3D `texts/` plus `new_joint_vecs/`: provide the paired
  content text/motion and reference style motion used by the evaluation loader.

Not read by the current `ours` entrypoints:

```text
dataset/humanml3d/test.txt
dataset/humanml3d/train_random_humanml.txt
dataset/humanml3d/valid_random_humanml.txt
dataset/humanml3d/Mean.npy
dataset/humanml3d/Std.npy
dataset/100style/texts/ for generation or training
```

## Data Notes

HumanML3D must be obtained or regenerated through the official HumanML3D project
because its source AMASS data is not redistributed directly. For 100STYLE, this
repo expects HumanML3D-style `new_joint_vecs` features and text files. The
SMooDi release provides a retargeted 100STYLE dataset; alternatively, process the
original 100STYLE BVH files into HumanML3D features yourself.

## Ignored Local Assets

The following are intentionally ignored by git:

```text
artifacts/
outputs/
checkpoints/ runtime files except .gitkeep placeholders and epoch_0100.ckpt
dataset/ runtime files except .gitkeep placeholders and tracked 100STYLE metadata
glove/ runtime files except .gitkeep
external/ runtime files except .gitkeep
checkpoints/clip-vit-*/
configs/reruns/mirrored_supcon*.yaml
```
