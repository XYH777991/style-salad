"""Two cheap, no-training diagnostics for the tempo-not-transferred finding
(see memory: style-salad-tempo-not-transferred). Both operate on real
100STYLE clips, batched, single model+dataset load.

1. VAE round-trip: does VAE.encode->decode alone (no diffusion, no style)
   preserve a clip's own speed? If not, the bottleneck is the VAE latent,
   not the style branch.

2. Linear probe: how well does each style representation (s_gen = raw
   StyleSTAttn/mixing branch, s_pure = StyleMLP/pure branch, combined =
   what actually drives generation) predict a clip's own reference speed?
   Ridge regression, 5-fold CV R^2. StyleMLP pools with a permutation-
   invariant masked mean over time -- structurally blind to frame order,
   so a low R^2 for s_pure specifically (vs s_gen, which has temporal
   self-attention + positional embeddings) would be a strong architectural
   explanation, not just "wasn't trained for it".

Usage:
    python artifacts/tempo_diagnostics.py
"""
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from style_salad.data.dataset import Dataset100STYLE
from style_salad.models.t2sm import Text2StylizedMotion
from style_salad.utils.motion import recover_from_ric

CONFIG_PATH = "configs/reruns/ours_train_stattn_dualenc_nodetach.yaml"
N_SAMPLES = 240
BATCH_SIZE = 32
SEED = 42


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
    return window, motion_length


def pick_sample_indices(ds_style, n):
    # Stride through the (length-sorted) items list for a spread across
    # both clip length and style, skipping mirrored duplicates.
    non_mirrored = [i for i, item in enumerate(ds_style.items) if not str(item["motion_id"]).startswith("M")]
    if len(non_mirrored) <= n:
        return non_mirrored
    stride = len(non_mirrored) / n
    return [non_mirrored[int(i * stride)] for i in range(n)]


def mean_joint_speed(joints, length):
    j = joints[:length]
    vel = np.linalg.norm(j[1:] - j[:-1], axis=-1)
    return float(vel.mean())


def frames_to_mask(num_frames):
    max_frames = int(num_frames.max())
    return torch.arange(max_frames, device=num_frames.device).expand(len(num_frames), max_frames) < num_frames.unsqueeze(1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    set_seed(SEED)

    model = load_model(config, device)
    style_cfg = config["dataset_style"]
    ds_style = Dataset100STYLE(style_cfg)

    mean = torch.tensor(np.load(style_cfg["mean_path"]), dtype=torch.float32, device=device)
    std = torch.tensor(np.load(style_cfg["std_path"]), dtype=torch.float32, device=device)

    indices = pick_sample_indices(ds_style, N_SAMPLES)
    print(f"sampling {len(indices)} clips")

    ref_speeds, vae_speeds = [], []
    s_gen_list, s_pure_list, combined_list = [], [], []

    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            windows, lengths = zip(*[get_eval_window(ds_style, i) for i in batch_idx])
            motion = torch.stack(windows).to(device)
            m_lens = torch.tensor(lengths, dtype=torch.long, device=device)

            len_mask = frames_to_mask(m_lens // 4)
            latent = model.vae.encode(motion)[0]
            len_mask = F.pad(len_mask, (0, latent.shape[1] - len_mask.shape[1]), mode="constant", value=False)
            latent = latent * len_mask[..., None, None].float()

            recon = model.vae.decode(latent)
            recon_real = recon * std + mean
            ref_real = motion * std + mean

            joints_ref = recover_from_ric(ref_real, 22).cpu().numpy()
            joints_recon = recover_from_ric(recon_real, 22).cpu().numpy()

            s_gen = F.normalize(model.style_encoder(latent, len_mask), dim=1)
            s_pure = F.normalize(model.pure_style_encoder(latent, len_mask), dim=1)
            combined = model.style_combiner(torch.cat([s_gen, s_pure.detach()], dim=1))

            for k in range(len(batch_idx)):
                L = lengths[k]
                ref_speeds.append(mean_joint_speed(joints_ref[k], L))
                vae_speeds.append(mean_joint_speed(joints_recon[k], L))
            s_gen_list.append(s_gen.cpu().numpy())
            s_pure_list.append(s_pure.cpu().numpy())
            combined_list.append(combined.cpu().numpy())

    ref_speeds = np.array(ref_speeds)
    vae_speeds = np.array(vae_speeds)
    s_gen_arr = np.concatenate(s_gen_list, axis=0)
    s_pure_arr = np.concatenate(s_pure_list, axis=0)
    combined_arr = np.concatenate(combined_list, axis=0)

    print("\n=== Diagnostic 1: VAE round-trip speed preservation ===")
    ratio = vae_speeds / np.clip(ref_speeds, 1e-8, None)
    corr = np.corrcoef(ref_speeds, vae_speeds)[0, 1]
    rho, pval = spearmanr(ref_speeds, vae_speeds)
    print(f"ref speed:   mean={ref_speeds.mean():.5f} std={ref_speeds.std():.5f}")
    print(f"VAE recon:   mean={vae_speeds.mean():.5f} std={vae_speeds.std():.5f}")
    print(f"ratio (recon/ref): mean={ratio.mean():.3f} median={np.median(ratio):.3f}  "
          f"(heavy right skew if mean >> median -- a few near-zero-speed refs blow the ratio up)")
    print(f"Pearson r(ref, recon) = {corr:.3f}   Spearman rho = {rho:.3f} (p={pval:.3f})")

    print("\n=== Diagnostic 2: linear probe, style embedding -> reference speed ===")
    # log-transform: speed is strictly positive and heavy-tailed (std~mean
    # in diag 1), so raw-scale R^2 gets dominated by a couple of outlier
    # clips landing in the test fold -- log scale is the natural one for a
    # multiplicative quantity like this and avoids that.  RidgeCV picks its
    # own regularization (256-dim features, ~190 train rows/fold -- fixed
    # alpha=10 undershoots this badly). Report out-of-fold R^2 AND Spearman
    # rank correlation (robust to whatever tail remains after the log).
    y = np.log(np.clip(ref_speeds, 1e-6, None))
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    alphas = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    for name, X in [("s_gen (StyleSTAttn, mixing)", s_gen_arr),
                     ("s_pure (StyleMLP, pure)", s_pure_arr),
                     ("combined (style_for_gen)", combined_arr)]:
        Xs = StandardScaler().fit_transform(X)
        model_cv = RidgeCV(alphas=alphas, cv=5)
        y_pred = cross_val_predict(model_cv, Xs, y, cv=kf)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        rho, pval = spearmanr(y, y_pred)
        print(f"{name:32s}  out-of-fold R^2(log speed) = {r2:.3f}   Spearman rho = {rho:.3f} (p={pval:.3f})")


if __name__ == "__main__":
    main()
