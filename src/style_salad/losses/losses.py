import torch
import torch.nn.functional as F


def loss_style(config, model, out):
    pred       = out["pred"]
    latent     = out["latent"]
    noise      = out["noise"]
    timesteps  = out["timesteps"]
    velocity   = model.scheduler.get_velocity(latent, noise, timesteps).detach()
    return F.mse_loss(pred, velocity)


def loss_supcon(config, model, out):
    temperature = config['temperature']
    style       = out['style']
    style_idx   = out['style_idx']
    len_mask  = out.get('len_mask', None)

    # Normalize so magnitude of z is not penalized
    style = F.normalize(style, dim=1)

    # Cosine similarity between samples
    sim = torch.matmul(style, style.T) / temperature
    N = sim.size(0)

    # Exclude self-comparison
    logits_mask = ~torch.eye(N, dtype=torch.bool, device=style.device)

    # Numerical stability for softmax
    sim_stable = sim - sim.max(dim=1, keepdim=True).values

    # Positive pair mask
    style_idx = style_idx.view(1, -1)
    pos_mask = (style_idx == style_idx.T) & logits_mask

    # Denominator over non-self entries
    exp_sim = torch.exp(sim_stable) * logits_mask
    denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8

    # Log probability
    log_prob = sim_stable - torch.log(denom)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
    return -mean_log_prob_pos.mean()


def loss_soft_supcon(config, model, out):
    temperature = config["temperature"]

    style = F.normalize(out["style"], dim=1)
    style_idx = out["style_idx"]

    sim = torch.matmul(style, style.T) / temperature
    N = sim.size(0)

    logits_mask = ~torch.eye(N, dtype=torch.bool, device=style.device)

    pred_logits = sim.masked_fill(~logits_mask, float("-inf"))
    log_p = F.log_softmax(pred_logits, dim=1)

    target_logits = model.style_affinity[style_idx][:, style_idx] / temperature
    target_logits = target_logits.masked_fill(~logits_mask, float("-inf"))
    target = F.softmax(target_logits, dim=1)

    # zero out invalid entries after softmax/log_softmax to avoid 0 * -inf
    log_p = torch.where(logits_mask, log_p, torch.zeros_like(log_p))
    target = torch.where(logits_mask, target, torch.zeros_like(target))

    return -(target * log_p).sum(dim=1).mean()


def loss_content_adversarial(config, model, out):
    """Cross-entropy of model.content_adversary's prediction of content_idx
    from the (un-detached) style embedding. self.net inside ContentAdversary
    learns normally to predict content well; the GradientReversal layer
    negates what flows back past it, so this same loss pushes the style
    encoder to make content unpredictable from `s` (Ganin & Lempitsky 2015 /
    Lample et al. 2017 mechanism -- see models/adversary.py)."""
    if model.content_adversary is None:
        raise ValueError(
            "loss_content_adversarial requires a content_adversary: block under "
            "model: in the config (see models/t2sm.py Text2StylizedMotion.__init__)."
        )
    content_idx = out.get("content_idx")
    if content_idx is None:
        raise ValueError(
            "loss_content_adversarial requires content_idx in the batch. "
            "Dataset100STYLE.__getitem__ returns it as the 5th element; make "
            "sure the training loop unpacks and forwards it."
        )
    logits = model.content_adversary(out["style"])
    return F.cross_entropy(logits, content_idx)


def loss_tempo(config, model, out):
    """Training-time tempo-matching loss (see memory:
    style-salad-tempo-not-transferred). Unlike the inference-only
    tempo_guidance sampling-time term (t2sm.py sampling_guidance), this
    trains the opt-in TemporalGate (models/transformer.py DenseFiLM) end to
    end, so the model itself learns to vary its style modulation over the
    frame axis -- instead of an external gradient forcing it there at
    sampling time, which the dose-response sweep showed has a real,
    monotonically-increasing FID/R-Prec/skate_ratio cost. Target: the swap
    partner's own real-world speed (out["tempo_target"], computed in
    Text2StylizedMotion.forward -- same definition as the inference-time
    target in _prepare_sampling_context)."""
    decoded = model._decode_motion_latent(out["pred_x0"], out["motion_len_mask"])
    return model._tempo_guidance_loss(decoded, out["motion_len_mask"], out["tempo_target"])


LOSS_REGISTRY = {
    "style"   : loss_style,
    "supcon"  : loss_supcon,
    "soft_supcon": loss_soft_supcon,
    "content_adversarial": loss_content_adversarial,
    "tempo": loss_tempo,
}
    
