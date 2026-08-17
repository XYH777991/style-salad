import argparse
import math
import os
import random

# Cap host-side threading by default so concurrent training jobs do not
# oversubscribe the same CPU. These can still be overridden from the shell.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "1")

import numpy as np
import torch
import yaml
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from style_salad.utils.plot import plot_tsne

from style_salad.data.dataset import DATASET_REGISTRY
from style_salad.data.sampler import SAMPLER_REGISTRY
from style_salad.models.t2sm import Text2StylizedMotion
from style_salad.losses.losses import LOSS_REGISTRY


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file (YAML)')
    args = parser.parse_args()

    from pathlib import Path
    cfg_path = Path(args.config).resolve()

    with cfg_path.open('r') as f:
        config = yaml.safe_load(f)

    parts = cfg_path.parts
    if "configs" in parts:
        i = parts.index("configs")
        sub = Path(*parts[i+1:]).with_suffix("")
        run_name = str(sub).replace("\\", "/")
    else:
        run_name = cfg_path.stem

    config["run_name"] = run_name
    config["result_dir"] = os.path.join(config["result_dir"], run_name)
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], run_name)
    return config


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Style-SALAD training requires a CUDA-capable GPU.")
    device = torch.device("cuda")
    use_bf16 = (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)

    amp_enabled = use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else None

    print(f"use_bf16: {use_bf16}")
    print(f"amp_enabled: {amp_enabled}")

    config = load_config()
    set_seed(config["random_seed"])

    # Match Torch's thread pools to the default caps above unless the user
    # overrides them via environment variables at launch time.
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_num_interop_threads(int(os.environ["TORCH_NUM_INTEROP_THREADS"]))

    # Output directory
    os.makedirs(config["result_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["result_dir"], "valid"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("./artifacts/tensorboard", config["run_name"]))

    # Dataset
    dataset_cfg = config['dataset']
    dataset = DATASET_REGISTRY[dataset_cfg['class']](dataset_cfg)

    # Dataloader (custom sampler)
    sampler_cfg = config['sampler']
    sampler = SAMPLER_REGISTRY[sampler_cfg['class']](sampler_cfg, dataset)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    # t-SNE
    label_to_name_dict = dict(dataset.idx_to_style)
    tsne_every = int(config.get("tsne_every", 10))
    tsne_max_samples = int(config.get("tsne_max_samples", 1000))

    # Model
    model_cfg = dict(config['model'])
    model_cfg.pop("class", None)
    model = Text2StylizedMotion(model_cfg).to(device)
    style_names = [dataset.idx_to_style[i] for i in range(len(dataset.idx_to_style))]
    model.set_style_text_prior(style_names)
    # Without this, _denormalize_motion (used by the new loss_tempo, and by
    # trajectory/keyframe/tempo sampling guidance) silently no-ops on
    # still-normalized motion -- see memory: style-salad-tempo-not-transferred
    # and the same fix already applied in evaluate.py / get_teaser_current.py.
    model.set_normalization_stats(np.load(dataset_cfg["mean_path"]), np.load(dataset_cfg["std_path"]))

    # Optional: warm-start from an existing checkpoint instead of training
    # from scratch -- e.g. fine-tuning in the newly added TemporalGate
    # (models/transformer.py) onto an already-trained dual-encoder
    # checkpoint. strict=False because init_checkpoint may be missing keys
    # for newly added modules (they keep their own fresh init, e.g.
    # TemporalGate's zero-init last layer -> identity gate at step 0) or
    # have extra keys the current config doesn't use.
    init_checkpoint_path = config.get("init_checkpoint_path")
    init_sd = None
    if init_checkpoint_path:
        print(f"Warm-starting from checkpoint: {init_checkpoint_path}")
        init_sd = torch.load(init_checkpoint_path, map_location=device)
        missing, unexpected = model.load_state_dict(init_sd, strict=False)
        print(f"  loaded with {len(missing)} missing keys (expected for newly added modules), "
              f"{len(unexpected)} unexpected keys ignored")

    # Optional Phase 1: pretrain style_encoder with supcon ALONE, before the
    # diffusion objective ever gets a chance to compete for its parameters.
    # supcon.weight x4 (see metrics_ours_train_stattn_supcon4.csv) and a
    # detached split head (metrics_ours_train_stattn_splithead.csv) both
    # left SRA essentially unchanged -- suggesting supcon's gradient into
    # the trunk was never large enough to compete with the diffusion loss
    # in the first place, whether shared or isolated. This sidesteps that
    # entirely: for `pretrain_style_epochs` epochs (additional, not counted
    # against `epochs`), the denoiser is not even run -- style_encoder is
    # the ONLY thing being trained, with supcon as its ONLY loss. Phase 2
    # (below, unchanged) then proceeds exactly as every earlier experiment,
    # starting from this supcon-organized trunk instead of a random init.
    pretrain_style_epochs = int(config.get("pretrain_style_epochs", 0))
    if pretrain_style_epochs > 0:
        print(f"=== Phase 1: pretraining style_encoder with supcon only ({pretrain_style_epochs} epochs) ===")
        phase1_optimizer = torch.optim.Adam(model.style_encoder.parameters(), lr=config["lr"])
        supcon_spec = config["loss"]["supcon"]
        model.train()
        for p1_epoch in range(1, pretrain_style_epochs + 1):
            p1_pbar = tqdm(dataloader, total=len(dataloader), desc=f"[Pretrain style] Epoch {p1_epoch}")
            p1_loss_sum = 0.0
            p1_n = 0
            for batch in p1_pbar:
                p1_captions, p1_motions, p1_num_frames, p1_style_idxs, _p1_content_idxs = batch
                p1_motions = p1_motions.to(device)
                p1_num_frames = p1_num_frames.to(device)
                p1_style_idxs = p1_style_idxs.to(device)
                phase1_optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                    p1_latent, p1_len_mask = model._encode_motion_latent(p1_motions, p1_num_frames)
                    p1_style = model._extract_style_embedding(p1_latent, p1_len_mask)
                    p1_loss = LOSS_REGISTRY["supcon"](
                        supcon_spec, model, {"style": p1_style, "style_idx": p1_style_idxs}
                    )

                p1_loss.backward()
                phase1_optimizer.step()
                p1_loss_sum += float(p1_loss.item())
                p1_n += 1
                p1_pbar.set_postfix(supcon_loss=float(p1_loss.item()))

            p1_mean = p1_loss_sum / max(1, p1_n)
            print(f"[Pretrain style] Epoch {p1_epoch} | mean supcon loss: {p1_mean:.4f}")
            if writer is not None:
                writer.add_scalar("Pretrain/supcon_loss", p1_mean, p1_epoch)
        print("=== Phase 1 done, entering normal joint training ===")

    # Optional: freeze every parameter whose name doesn't contain a given
    # substring -- e.g. fine-tuning only the newly added TemporalGate
    # (models/transformer.py) so loss_tempo's gradient can only ever touch
    # the new mechanism, not perturb the rest of an already-good checkpoint.
    # Isolates what the new mechanism alone can do, cleanly separated from
    # "the rest of the network drifted during this fine-tune" as a
    # confound. The later optimizer-building code below already filters on
    # requires_grad, so this composes with it (and with content_adversary)
    # without further changes.
    train_only_matching = config.get("train_only_matching")
    if train_only_matching:
        n_trainable = n_frozen = 0
        for name, p in model.named_parameters():
            if train_only_matching in name:
                p.requires_grad = True
                n_trainable += 1
            else:
                p.requires_grad = False
                n_frozen += 1
        print(f"train_only_matching='{train_only_matching}': {n_trainable} trainable params, {n_frozen} frozen")

    # The content classifier is freshly initialized and has to learn a
    # 7-way task from scratch in ~9400 steps; sharing the main lr (tuned for
    # fine-tuning an already-pretrained backbone) left it stuck at the
    # random-guess loss the whole run even with a GRL warmup schedule (see
    # metrics_ours_train_stattn_advcontent_warmup.csv). Give it its own,
    # separately-configurable, higher lr instead.
    if model.content_adversary is not None:
        adv_params = list(model.content_adversary.net.parameters())
        adv_param_ids = {id(p) for p in adv_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in adv_param_ids]
        adv_lr = float(config.get("content_adversary_lr", config["lr"] * 10))
        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": config["lr"]},
            {"params": adv_params, "lr": adv_lr},
        ])
        print(f"content_adversary.net lr={adv_lr} (main lr={config['lr']})")
    else:
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=config['lr']
        )

    # EMA of trainable weights (opt-in, off by default so existing configs are
    # unaffected). Uses a warmup schedule so the shadow isn't dominated by the
    # noisy random-init weights early in the (short, ~9-10k step) training run:
    # decay ramps from 0 up to config.ema.decay over the first few hundred steps.
    ema_cfg = config.get("ema", {}) or {}
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.9995))
    ema_warmup = bool(ema_cfg.get("warmup", True))
    ema_shadow = None
    ema_step = 0
    if ema_enabled:
        ema_shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        print(f"EMA enabled: decay={ema_decay}, warmup={ema_warmup}")

    # Losses
    loss_cfg = config['loss']
    loss_fns = {name: LOSS_REGISTRY[name] for name in loss_cfg}
    normalize_flags = {
        name: loss_cfg[name].get("normalize", True)
        for name in loss_cfg
    }

    # Calibration
    scales = {}
    sum_sq = defaultdict(float)
    pbar = tqdm(dataloader, total=config['steps'])
    n = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            if i >= config['steps']: break
            captions, motions, num_frames, style_idxs, content_idxs = batch
            motions, num_frames, style_idxs, content_idxs = (
                motions.to(device), num_frames.to(device), style_idxs.to(device), content_idxs.to(device)
            )

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                out = model(captions, motions, num_frames, style_idxs, content_idx=content_idxs)
                losses = {}
                losses_denoiser = {}
                for name, fn in loss_fns.items():
                    spec = loss_cfg[name]
                    raw  = fn(spec, model, out)
                    if "cycle" in name:
                        losses_denoiser[name] = raw
                    else:
                        losses[name] = raw

                total_loss = torch.zeros((), device=device)
                for name, val in losses.items():
                    total_loss += loss_cfg[name]["weight"] * val

            all_losses = {}
            all_losses.update(losses)

            for name, val in all_losses.items():
                if normalize_flags.get(name, False):
                    v = val.detach().float().item()
                    sum_sq[name] += v * v

            n += 1
            pbar.set_postfix(loss=float(total_loss.item()))

        for name in loss_cfg.keys():
            if normalize_flags[name]:
                s2  = sum_sq[name]
                rms = (s2 / max(1, n)) ** 0.5
                scales[name] = max(rms, config['tau'])
            else:
                scales[name] = 1.0
        print("Frozen RMS denominators:", {k: round(v, 6) for k, v in scales.items()})

    # GRL lambda warmup (DANN-style, Ganin & Lempitsky 2015): ramp 0 -> the
    # configured lambda_ over training instead of using full strength from
    # step 1. A fixed lambda from the start left the content classifier
    # stuck at the random-guess cross-entropy the whole run (see
    # metrics_ours_train_stattn_advcontent.csv / tensorboard) -- the
    # reversed gradient erased any discriminative signal the classifier
    # found before it had a chance to learn it, so nothing ever adapted.
    adv_lambda_max = 0.0
    if model.content_adversary is not None:
        adv_lambda_max = model.content_adversary.grl.lambda_
    total_steps = config['epochs'] * len(dataloader)
    global_step = 0

    # Training
    model.train()
    for epoch in range(1, config['epochs'] + 1):
        losses_scaled_sum = defaultdict(float)
        losses_norm_sum   = defaultdict(float)
        losses_raw_sum    = defaultdict(float)

        pbar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"[Train] Epoch {epoch}"
        )

        batch_idx = -1
        for batch_idx, batch in enumerate(pbar):
            captions, motions, num_frames, style_idxs, content_idxs = batch
            motions = motions.to(device)
            num_frames = num_frames.to(device)
            style_idxs = style_idxs.to(device)
            content_idxs = content_idxs.to(device)
            optimizer.zero_grad(set_to_none=True)

            if model.content_adversary is not None:
                progress = global_step / max(1, total_steps - 1)
                model.content_adversary.grl.lambda_ = adv_lambda_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
            global_step += 1

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                out = model(captions, motions, num_frames, style_idxs, content_idx=content_idxs)
                losses = {}
                losses_denoiser = {}
                for name, fn in loss_fns.items():
                    spec = loss_cfg[name]
                    raw = fn(spec, model, out)
                    norm = raw / scales[name]
                    scaled = spec["weight"] * norm

                    if "cycle" in name:
                        losses_denoiser[name] = (scaled, norm, raw)
                    else:
                        losses[name] = (scaled, norm, raw)

                total_loss = torch.zeros((), device=device)

                for _, (scaled, _, _) in losses.items():
                    total_loss += scaled

            total_loss.backward()
            optimizer.step()

            if ema_enabled:
                ema_step += 1
                decay = min(ema_decay, (1 + ema_step) / (10 + ema_step)) if ema_warmup else ema_decay
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in ema_shadow:
                            ema_shadow[n].mul_(decay).add_(p.detach(), alpha=1.0 - decay)

            pbar.set_postfix(loss=float(total_loss.item()))

            all_losses = {}
            all_losses.update(losses)
            all_losses.update(losses_denoiser)

            for name, (scaled, norm, raw) in all_losses.items():
                losses_scaled_sum[name] += scaled.item()
                losses_norm_sum[name]   += norm.item()
                losses_raw_sum[name]    += raw.item()

        num_batches = max(batch_idx + 1, 1)
        train_total_scaled = sum(losses_scaled_sum.values()) / num_batches
        train_total_norm   = sum(losses_norm_sum.values()) / num_batches
        train_total_raw    = sum(losses_raw_sum.values()) / num_batches

        if writer is not None:
            writer.add_scalar("Train/Raw/Total", train_total_raw, epoch)
            writer.add_scalar("Train/Norm/Total", train_total_norm, epoch)
            writer.add_scalar("Train/Scaled/Total", train_total_scaled, epoch)

            for name in losses_scaled_sum.keys():
                writer.add_scalar(f"Train/Raw/{name}",    losses_raw_sum[name]    / num_batches, epoch)
                writer.add_scalar(f"Train/Norm/{name}",   losses_norm_sum[name]   / num_batches, epoch)
                writer.add_scalar(f"Train/Scaled/{name}", losses_scaled_sum[name] / num_batches, epoch)

            if model.content_adversary is not None:
                writer.add_scalar("Train/GRL_lambda", model.content_adversary.grl.lambda_, epoch)

        print(
            f"Epoch {epoch} | "
            f"Train scaled: {train_total_scaled:.4f} | "
            f"Train norm: {train_total_norm:.4f} | "
            f"Train raw: {train_total_raw:.4f}"
        )

        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        sd = model.state_dict()
        sd_trainable = {k: v for k, v in sd.items() if k in trainable}
        # If warm-started (init_checkpoint_path) with some parameters frozen
        # (train_only_matching), sd_trainable alone would only contain the
        # few params that were actually updated -- evaluate.py/
        # get_teaser_current.py build a fresh model and load ONE checkpoint
        # file directly, with no knowledge of init_checkpoint_path, so a
        # partial checkpoint would silently leave everything else at random
        # init. Merge onto the full initial state dict so every saved
        # checkpoint here is self-contained and loads correctly on its own.
        if init_sd is not None:
            sd_trainable = {**init_sd, **sd_trainable}

        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        torch.save(sd_trainable, os.path.join(config["checkpoint_dir"], "latest.ckpt"))

        save_every = config.get("save_every", 10)
        if save_every and (epoch % save_every == 0):
            torch.save(
                sd_trainable,
                os.path.join(config["checkpoint_dir"], f"epoch_{epoch:04d}.ckpt")
            )

        if ema_enabled:
            torch.save(ema_shadow, os.path.join(config["checkpoint_dir"], "latest_ema.ckpt"))
            if save_every and (epoch % save_every == 0):
                torch.save(
                    ema_shadow,
                    os.path.join(config["checkpoint_dir"], f"epoch_{epoch:04d}_ema.ckpt")
                )

        if tsne_every > 0 and epoch % tsne_every == 0:
            plot_tsne(
                model=model,
                loader=dataloader,
                device=device,
                epoch=epoch,
                title="train",
                result_dir=config["result_dir"],
                label_to_name_dict=label_to_name_dict,
                max_samples=tsne_max_samples,
                writer=writer,
            )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
