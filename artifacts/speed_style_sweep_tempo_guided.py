"""Same 14-style sweep as speed_style_sweep.py, but with the new
tempo_guidance sampling-guidance term turned on (see memory:
style-salad-tempo-not-transferred and t2sm.py's _tempo_guidance_loss /
_mean_joint_speed): an inference-only, no-retraining attempt at fixing
tempo transfer by adding an explicit gradient-guidance objective (reusing
the same DDIM x0_hat mechanism as trajectory/keyframe guidance) that
nudges each generated sample toward the reference clip's own measured
speed, since HyperLoRA itself has no architectural pathway to do this via
the style vector alone.

Also calls set_normalization_stats(), which nothing in this codebase
currently does -- without it, _denormalize_motion (used by trajectory/
keyframe/tempo guidance) silently no-ops on already-normalized motion
instead of returning real-world-scale values, producing nonsense loss
targets. Found while sanity-checking this script's own tempo_target
against speed_style_sweep.py's independent computation (53.3 vs the
expected 0.031 -- a ~1700x mismatch that went away once this was called).

Usage:
    TEMPO_WEIGHT=1.0 python artifacts/speed_style_sweep_tempo_guided.py
"""
import os

import numpy as np
import torch
import yaml

from style_salad.data.dataset import Dataset100STYLE
from style_salad.models.t2sm import Text2StylizedMotion
from style_salad.utils.motion import recover_from_ric

CONFIG_PATH = "configs/reruns/ours_train_stattn_dualenc_nodetach.yaml"
NUM_SAMPLES = 4
OUTPUT_LENGTH = 120
NUM_INFERENCE_STEPS = 50
SEED = 42
TEMPO_WEIGHT = float(os.environ.get("TEMPO_WEIGHT", "1.0"))

# Spans "semantically slow/cautious", "semantically fast/urgent", and
# neutral/other styles so a systematic compression-toward-average-speed
# pattern (if present) shows up as a direction, not noise.
STYLES = [
    "Neutral", "Rushed", "Elated", "Roadrunner",       # fast / energetic
    "Depressed", "Old", "InTheDark", "Zombie",          # slow / cautious
    "DragLeftLeg", "Crouched", "Stiff", "OnToesCrouched",  # slow / effortful
    "Drunk", "Robot",                                    # other
]


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config["run_name"] = "reruns/ours_train_stattn_dualenc_nodetach"
    config["result_dir"] = f"{config['result_dir']}/ours_train_stattn_dualenc_nodetach"
    config["checkpoint_dir"] = f"{config['checkpoint_dir']}/{config['run_name']}"
    return config


def load_model(config, device):
    model = Text2StylizedMotion(config["model"]).to(device)
    ckpt_path = f"{config['checkpoint_dir']}/{config.get('checkpoint_name', 'latest.ckpt')}"
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()
    return model


def get_eval_window(ds_style, idx):
    meta = ds_style.items[idx]
    motion = ds_style.motion_cache[meta["motion_id"]]
    total_frames = int(motion.shape[0])
    feat_dim = int(motion.shape[1])
    unit = ds_style.unit_length
    motion_length = max(1, total_frames // unit) * unit if unit > 0 else total_frames
    if ds_style.max_frames is not None:
        motion_length = min(motion_length, ds_style.max_frames)
    motion_length = max(1, min(motion_length, total_frames))
    start = max(0, (total_frames - motion_length) // 2)
    window = motion[start:start + motion_length]
    if ds_style.max_frames is not None and motion_length < ds_style.max_frames:
        pad = torch.zeros(ds_style.max_frames - motion_length, feat_dim, dtype=window.dtype)
        window = torch.cat([window, pad], dim=0)
    content_key = ds_style.idx_to_content[meta["content_idx"]]
    caption = ds_style.content_prompts[content_key]
    return caption, window, motion_length


def find_first_index_for_style(ds_style, style_name):
    # ds_style.items is sorted globally by clip length (see dataset.py), so
    # picking the first match for a style silently picks its SHORTEST clip
    # -- for several styles that's a near-static few-frame hold (ref speed
    # ~0.0003), producing bogus 10x+ "generated is faster" ratios that are a
    # sampling artifact, not a real effect. Pick the median-length
    # non-mirrored clip instead, for a representative-length reference.
    candidates = [
        (i, item["length"]) for i, item in enumerate(ds_style.items)
        if not str(item["motion_id"]).startswith("M") and ds_style.idx_to_style[item["style_idx"]] == style_name
    ]
    if not candidates:
        candidates = [
            (i, item["length"]) for i, item in enumerate(ds_style.items)
            if ds_style.idx_to_style[item["style_idx"]] == style_name
        ]
    if not candidates:
        raise ValueError(f"No motion found for style '{style_name}'")
    candidates.sort(key=lambda x: x[1])
    return candidates[len(candidates) // 2][0]


def mean_joint_speed(joints, length):
    j = joints[:length]
    vel = np.linalg.norm(j[1:] - j[:-1], axis=-1)
    return float(vel.mean())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    set_seed(config["random_seed"])

    model = load_model(config, device)
    style_cfg = config["dataset_style"]
    ds_style = Dataset100STYLE(style_cfg)

    mean = torch.tensor(np.load(style_cfg["mean_path"]), dtype=torch.float32, device=device)
    std = torch.tensor(np.load(style_cfg["std_path"]), dtype=torch.float32, device=device)
    model.set_normalization_stats(mean, std)  # see module docstring -- required for tempo/trajectory/keyframe guidance

    print(f"tempo_guidance weight = {TEMPO_WEIGHT}")
    results = []
    for style_name in STYLES:
        idx = find_first_index_for_style(ds_style, style_name)
        caption, window, motion_length = get_eval_window(ds_style, idx)

        motions = window[:motion_length].unsqueeze(0).to(device)
        style_lengths = torch.tensor([motion_length], dtype=torch.long, device=device)
        motions_rep = motions.repeat(NUM_SAMPLES, 1, 1)
        style_lengths_rep = style_lengths.repeat(NUM_SAMPLES)
        captions = [caption] * NUM_SAMPLES
        gen_lengths = torch.full((NUM_SAMPLES,), OUTPUT_LENGTH, dtype=torch.long, device=device)

        guidance = {
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "tempo": {"weight": TEMPO_WEIGHT},
        }
        # NOT wrapped in torch.no_grad(): tempo/style/trajectory/keyframe
        # guidance need autograd enabled internally (z.requires_grad_(True)
        # + torch.autograd.grad on x0_hat each DDIM step) -- get_teaser_
        # current.py's real generate() call is likewise un-wrapped.
        stylized, _ = model.generate(motions_rep, captions, gen_lengths, style_lengths_rep, guidance=guidance)
        with torch.no_grad():
            reference = motions * std + mean
            stylized_real = stylized.detach() * std + mean
            joints_ref = recover_from_ric(reference, 22).detach().cpu().numpy()[0]
            joints_gen = recover_from_ric(stylized_real, 22).detach().cpu().numpy()

        ref_speed = mean_joint_speed(joints_ref, motion_length)
        gen_speeds = [mean_joint_speed(joints_gen[i], OUTPUT_LENGTH) for i in range(NUM_SAMPLES)]
        gen_mean = float(np.mean(gen_speeds))
        ratio = gen_mean / ref_speed if ref_speed > 0 else float("nan")

        results.append((style_name, ref_speed, gen_mean, ratio))
        print(f"{style_name:16s} ref={ref_speed:.5f}  gen_mean={gen_mean:.5f}  ratio={ratio:.2f}  "
              f"gen_each={['%.4f' % s for s in gen_speeds]}")

    ratios = [r[3] for r in results]
    print()
    print(f"n={len(ratios)}  mean ratio={np.mean(ratios):.2f}  median={np.median(ratios):.2f}  "
          f"faster-than-ref count={sum(r > 1.0 for r in ratios)}/{len(ratios)}")


if __name__ == "__main__":
    main()
