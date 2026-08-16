"""Same sweep as speed_style_sweep.py, but with style_for_gen overridden at
inference time to be the raw s_gen (StyleSTAttn/mixing branch) output
instead of style_combiner's fused (s_gen, s_pure) vector -- no retraining,
just monkeypatches style_combiner to drop the s_pure half.

Motivated by tempo_diagnostics.py: s_gen alone predicts a clip's own speed
much better (R^2=0.53) than the combined vector actually fed to HyperLoRA
(R^2=0.40) or s_pure alone (R^2=0.07). This checks whether that undiluted
signal actually produces better tempo matching in generation, or whether
HyperLoRA/guidance just doesn't act on tempo regardless of what's in the
style vector.

Usage:
    python artifacts/speed_style_sweep_sgen_only.py
"""
import numpy as np
import torch
import torch.nn as nn
import yaml

from style_salad.data.dataset import Dataset100STYLE
from style_salad.models.t2sm import Text2StylizedMotion
from style_salad.utils.motion import recover_from_ric

CONFIG_PATH = "configs/reruns/ours_train_stattn_dualenc_nodetach.yaml"
NUM_SAMPLES = 4
OUTPUT_LENGTH = 120
NUM_INFERENCE_STEPS = 50
SEED = 42

STYLES = [
    "Neutral", "Rushed", "Elated", "Roadrunner",
    "Depressed", "Old", "InTheDark", "Zombie",
    "DragLeftLeg", "Crouched", "Stiff", "OnToesCrouched",
    "Drunk", "Robot",
]


class DropPureBranch(nn.Module):
    """Wraps the REAL trained style_combiner Linear layer, but zeroes the
    s_pure half of its input before calling it -- unlike truncating the
    combiner entirely (bypassing the Linear layer, which was the first,
    flawed version of this test: HyperLoRA never saw an un-transformed
    s_gen during training, only W_gen@s_gen + W_pure@s_pure + b, so
    skipping the Linear layer put it in an out-of-distribution regime and
    confounded the result). Zeroing s_pure and still calling the trained
    Linear keeps HyperLoRA in the same output distribution/scale it
    trained on -- isolates "what if s_pure's contribution were zero"
    cleanly."""

    def __init__(self, orig_combiner):
        super().__init__()
        self.orig_combiner = orig_combiner

    def forward(self, x):
        style_dim = x.shape[1] // 2
        x = x.clone()
        x[:, style_dim:] = 0.0
        return self.orig_combiner(x)


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
    candidates = [
        (i, item["length"]) for i, item in enumerate(ds_style.items)
        if not str(item["motion_id"]).startswith("M") and ds_style.idx_to_style[item["style_idx"]] == style_name
    ]
    if not candidates:
        candidates = [
            (i, item["length"]) for i, item in enumerate(ds_style.items)
            if ds_style.idx_to_style[item["style_idx"]] == style_name
        ]
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
    model.style_combiner = DropPureBranch(model.style_combiner).to(device)  # zero s_pure's input, keep the trained Linear

    style_cfg = config["dataset_style"]
    ds_style = Dataset100STYLE(style_cfg)

    mean = torch.tensor(np.load(style_cfg["mean_path"]), dtype=torch.float32, device=device)
    std = torch.tensor(np.load(style_cfg["std_path"]), dtype=torch.float32, device=device)

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

        guidance = {"num_inference_steps": NUM_INFERENCE_STEPS}
        with torch.no_grad():
            stylized, _ = model.generate(motions_rep, captions, gen_lengths, style_lengths_rep, guidance=guidance)
            reference = motions * std + mean
            stylized_real = stylized * std + mean
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
