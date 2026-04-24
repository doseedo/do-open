"""
Train the latent → BasicPitch student.

Loss:
    onset BCE (positive-weighted; onsets are sparse)
  + frame BCE (per-frame note activity)
  + velocity L1 (only where frames are active)
  all masked by the example's padding mask.

Usage:
    python -m latent_pitch.train \
        --pair-cache /scratch/latent_pitch_pairs.tsv \
        --out /scratch/latent_pitch_ckpts \
        --steps 80000 --batch 8 --workers 4
"""
from __future__ import annotations
import argparse, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from latent_pitch.dataset import (
    LatentMidiPairDataset, collate_pitch,
    DEFAULT_LATENTS_ROOT, DEFAULT_BASICPITCH_ROOT,
)
from latent_pitch.model import LatentBasicPitchStudent


def masked_bce(logits, target, mask, pos_weight=None):
    # logits/target: [B,T,128]; mask: [B,T]
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight,
    )
    m = mask.unsqueeze(-1)
    return (loss * m).sum() / (m.sum() * loss.shape[-1] + 1e-8)


def masked_focal_bce(logits, target, mask, gamma=2.0, alpha=0.75, pos_weight=None):
    """Focal BCE — fixes the recall plateau in standard weighted BCE.

    The modulating factor (1 - p_t)^gamma down-weights examples the model
    is already getting right and concentrates gradient on hard positives
    (true onsets the model is currently missing). alpha further biases
    toward positives. This is what gets us out of the "fire only when
    sure" local minimum that BCE+pos_weight settles into.
    """
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)        # prob assigned to true class
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight,
    )
    focal_w = (1.0 - p_t).clamp_min(1e-6).pow(gamma)
    alpha_w = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = focal_w * alpha_w * bce
    m = mask.unsqueeze(-1)
    return (loss * m).sum() / (m.sum() * loss.shape[-1] + 1e-8)


def note_level_onset_loss(onset_logits, onset_target, mask, tolerance: int = 2):
    """Note-level onset loss aligned with the F1 metric.

    For every TRUE onset (frame f, pitch p) in onset_target:
        positive_loss = -log(sigmoid(max_logit_in[f-r:f+r, p]))
    i.e. the model is rewarded if it fires at *any* frame within ±r frames
    of the true onset. This matches the mir_eval 50 ms-tolerance matching
    used by the eval, instead of demanding a frame-exact prediction the
    way per-frame BCE does.

    For every frame outside any true onset's tolerance window:
        negative_loss = -log(1 - sigmoid(logit))
    i.e. standard BCE with target 0, weighted normally.

    This is the right loss for "did the model detect the note as a note,
    irrespective of which exact frame it fired on", which is also what
    the F1 evaluator measures.
    """
    B, T, P = onset_logits.shape
    # mask of frames within ±tolerance of any true onset, per pitch
    # via 1D max-pool of onset_target along the time axis
    on_perm = onset_target.permute(0, 2, 1)  # [B, P, T]
    near_onset = F.max_pool1d(
        on_perm, kernel_size=2 * tolerance + 1, stride=1, padding=tolerance,
    )  # [B, P, T]
    near_onset = near_onset.permute(0, 2, 1)  # [B, T, P]
    near_mask = (near_onset > 0.5).float()
    far_mask  = (1.0 - near_mask) * mask.unsqueeze(-1)

    # POSITIVE: per-true-onset, max logit in the ±tol window must be high.
    log_perm = onset_logits.permute(0, 2, 1)
    pooled = F.max_pool1d(
        log_perm, kernel_size=2 * tolerance + 1, stride=1, padding=tolerance,
    ).permute(0, 2, 1)  # [B, T, P]
    pos_mask = (onset_target > 0.5).float()
    pos_logits = pooled[pos_mask.bool()]
    if pos_logits.numel() > 0:
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits), reduction="mean",
        )
    else:
        pos_loss = onset_logits.new_zeros(())

    # NEGATIVE: frames OUTSIDE any onset's tolerance window.
    # Hard-negative mining: instead of averaging over ALL far frames (which
    # dilutes the gradient and lets the model spurious-fire), pick the
    # k_hard_neg highest-logit far frames per batch and only train on those.
    # This concentrates negative gradient on the model's worst false
    # positives — same scale as the positive gradient.
    flat_far = (far_mask.bool() & (onset_logits.abs() < 1e9))  # all far cells
    if flat_far.any():
        neg_logits_all = onset_logits[flat_far]
        # take top-k (or all if fewer than k)
        k_neg = max(pos_logits.numel() * 4 if pos_logits.numel() > 0 else 256, 256)
        k_neg = min(k_neg, neg_logits_all.numel())
        topk_neg, _ = torch.topk(neg_logits_all, k_neg)
        neg_loss = F.binary_cross_entropy_with_logits(
            topk_neg, torch.zeros_like(topk_neg), reduction="mean",
        )
    else:
        neg_loss = onset_logits.new_zeros(())

    # weight neg_loss higher to compensate for the few-positives bias.
    # k_hard_neg = 4× n_positives keeps gradient ratios balanced toward
    # precision; tune via the 4× constant if F1 still tilts.
    return pos_loss + 2.0 * neg_loss


def onset_density_loss(onset_logits, onset_target, mask):
    """Per-pitch onset *rate* L1: forces predicted onset count over total
    frames to match true rate. Normalized to [0, ~0.1] units, comparable
    to BCE magnitudes. Asymmetric to penalize under-prediction harder."""
    p = torch.sigmoid(onset_logits)                       # [B,T,128]
    m = mask.unsqueeze(-1)
    n_frames = m.sum(dim=1).clamp_min(1.0)                # [B,1]
    pred_rate = (p * m).sum(dim=1) / n_frames             # [B,128]
    true_rate = (onset_target * m).sum(dim=1) / n_frames  # [B,128]
    diff = true_rate - pred_rate
    under = diff.clamp_min(0.0).mean()                    # missed-onset rate
    over  = (-diff).clamp_min(0.0).mean()                 # spurious-onset rate
    return 2.0 * under + over


def masked_l1_active(pred, target, mask, frame_target):
    # only score velocity where the frame is actually active
    active = (frame_target > 0.5).float()
    m = mask.unsqueeze(-1) * active
    diff = (pred - target).abs() * m
    return diff.sum() / (m.sum() + 1e-8)


def masked_l1_onset(pred, target, mask, onset_target):
    # only score sub-frame offset at the EXACT onset frame. With Gaussian
    # smoothing the onset roll peaks at 1.0 at the true frame and decays
    # outward, so onset_target >= 0.99 isolates the true onset frames.
    on = (onset_target >= 0.99).float()
    m = mask.unsqueeze(-1) * on
    diff = (pred - target).abs() * m
    return diff.sum() / (m.sum() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-root", default=DEFAULT_LATENTS_ROOT)
    ap.add_argument("--basicpitch-root", default=DEFAULT_BASICPITCH_ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None,
                    help="warm-start from a previous pitch checkpoint")
    ap.add_argument("--steps", type=int, default=80_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--win-frames", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--onset-pos-weight", type=float, default=40.0)
    ap.add_argument("--frame-pos-weight", type=float, default=5.0)
    ap.add_argument("--offset-weight",    type=float, default=1.0,
                    help="weight for sub-frame onset offset L1 loss")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="focal loss gamma for onset BCE (0=disable)")
    ap.add_argument("--density-weight", type=float, default=0.5,
                    help="weight for onset density loss")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    ds = LatentMidiPairDataset(
        latents_root=args.latents_root,
        basicpitch_root=args.basicpitch_root,
        win_frames=args.win_frames,
    )
    print(f"  date dirs: {len(ds.dates)}")
    dl = DataLoader(
        ds, batch_size=args.batch, num_workers=args.workers,
        collate_fn=collate_pitch, shuffle=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )

    model = LatentBasicPitchStudent(max_len=args.win_frames).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"warm-started from {args.init} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    onset_pw = torch.tensor([args.onset_pos_weight], device=device)
    frame_pw = torch.tensor([args.frame_pos_weight], device=device)

    step, t0 = 0, time.time()
    for batch in dl:
        if step >= args.steps:
            break
        L      = batch["latent"].to(device)
        onset  = batch["onset"].to(device)
        frame  = batch["frame"].to(device)
        vel    = batch["velocity"].to(device)
        offset = batch["offset"].to(device)
        mask   = batch["mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(L)
            # ONSET: per-frame BCE (v1's loss). Note-level loss made the
            # model too aggressive when paired with continuous frame targets.
            l_on = masked_bce(out["onset_logits"], onset, mask, onset_pw)
            # FRAME: when the target is a continuous multi-F0 posterior in
            # [0,1], use BCE-as-distillation (treats target as soft label
            # probability). This matches v1's loss formulation but with
            # richer supervision. Falls back to standard binary BCE when
            # the dataset returns a binary frame roll.
            l_fr = F.binary_cross_entropy_with_logits(
                out["frame_logits"], frame, reduction="none",
            )
            m = mask.unsqueeze(-1)
            l_fr = (l_fr * m).sum() / (m.sum() * l_fr.shape[-1] + 1e-8)
            l_vel = masked_l1_active(out["velocity"], vel, mask, frame)
            l_off = masked_l1_onset(out["onset_offset"], offset, mask, onset)
            loss = l_on + l_fr + 0.5 * l_vel + args.offset_weight * l_off

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(
                f"step {step:6d}  loss {loss.item():.4f}  "
                f"on {l_on.item():.4f}  fr {l_fr.item():.4f}  vel {l_vel.item():.4f}  "
                f"off {l_off.item():.4f}  "
                f"({dt/(step+1):.2f}s/it)"
            )
        if step > 0 and step % args.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step, "args": vars(args)},
                os.path.join(args.out, f"pitch_{step:06d}.pt"),
            )
            print(f"  saved pitch_{step:06d}.pt")
        step += 1

    torch.save(
        {"model": model.state_dict(), "step": step, "args": vars(args)},
        os.path.join(args.out, "pitch_final.pt"),
    )
    print("done.")


if __name__ == "__main__":
    main()
