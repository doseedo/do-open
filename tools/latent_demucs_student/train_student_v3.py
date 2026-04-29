#!/usr/bin/env python3
"""Train the latent-demucs student with frozen-VAE-decoder waveform loss.

Loss = λ_lat * L1(pred_lat, target_lat)         # latent space (mask-aware)
     + λ_wav * L1(decode(pred), decode(target))  # waveform space (mask-aware)

The waveform loss directly penalizes magnitude collapse, since a
near-silent latent prediction decodes to silence and has huge L1
distance from the real stem waveform.

Decoder is frozen and runs in bf16. We only decode the *supervised*
channels per item (looked up via mask) to avoid wasting compute.
"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/scratch/ACE-Step-1.5")
from student_model import LatentDemucsStudent
from dataset_combined import CombinedSeparationDataset, collate, CLASSES


def load_vae_decoder():
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(
        "/scratch/ACE-Step-1.5/checkpoints/vae"
    ).cuda().eval().to(torch.bfloat16)
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def decode_supervised(vae, latents_4ch, mask):
    """Decode only the supervised (mask=1) channels per item.

    latents_4ch: [B, 4, 64, T]   (pred or target, fp32)
    mask:        [B, 4]
    Returns dict {"audio": [N_sup, 2, samples], "items": list of (b, c)}
    """
    B = latents_4ch.shape[0]
    items = [(b, c) for b in range(B) for c in range(4) if mask[b, c] > 0]
    if not items:
        return None, items
    z = torch.stack([latents_4ch[b, c] for (b, c) in items])  # [N_sup, 64, T]
    z = z.to(torch.bfloat16)
    audio = vae.decode(z).sample                              # [N_sup, 2, N_samples]
    return audio, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/ckpts_v3")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=300)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_lat", type=float, default=0.3)
    ap.add_argument("--lambda_wav", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_teacher", action="store_true")
    ap.add_argument("--no_gt", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = CombinedSeparationDataset(
        crop_frames=args.crop,
        use_teacher=not args.no_teacher,
        use_gt=not args.no_gt,
    )
    if len(ds) == 0:
        print("ERROR: empty dataset"); return

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    print("[v3] loading frozen VAE decoder...")
    vae = load_vae_decoder()

    model = LatentDemucsStudent().cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[student] {n:.1f}M params")

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"[student] resumed from {args.resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    step = 0
    losses_total, losses_lat, losses_wav = [], [], []
    per_class = {c: [] for c in CLASSES}
    t0 = time.time()
    while step < args.steps:
        for mix, stems, mask in loader:
            mix = mix.cuda(non_blocking=True)
            stems = stems.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(mix)                                  # [B, 4, 64, T]

            # ── latent loss (masked L1) ────────────────────────────────
            diff_lat = (pred.float() - stems.float()).abs()        # [B, 4, 64, T]
            per_ch = diff_lat.mean(dim=(2, 3))                     # [B, 4]
            denom = mask.sum().clamp_min(1.0)
            loss_lat = (per_ch * mask).sum() / denom

            # ── waveform loss (decode supervised channels only) ────────
            with torch.no_grad():
                tgt_audio, _ = decode_supervised(vae, stems, mask)  # frozen branch
            pred_audio, items = decode_supervised(vae, pred, mask)  # autograd through pred
            if pred_audio is not None and tgt_audio is not None:
                # Both are bf16; cast to fp32 for stable loss
                loss_wav = F.l1_loss(pred_audio.float(), tgt_audio.float())
            else:
                loss_wav = torch.zeros((), device="cuda")

            loss = args.lambda_lat * loss_lat + args.lambda_wav * loss_wav
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            losses_total.append(loss.item())
            losses_lat.append(loss_lat.item())
            losses_wav.append(loss_wav.item() if isinstance(loss_wav, torch.Tensor) else 0.0)
            with torch.no_grad():
                for ci, c in enumerate(CLASSES):
                    m = mask[:, ci]
                    if m.sum() > 0:
                        per_class[c].append((per_ch[:, ci] * m).sum().item() / m.sum().item())
            step += 1

            if step % args.log_every == 0:
                avgT = sum(losses_total[-50:]) / max(1, len(losses_total[-50:]))
                avgL = sum(losses_lat[-50:])   / max(1, len(losses_lat[-50:]))
                avgW = sum(losses_wav[-50:])   / max(1, len(losses_wav[-50:]))
                el = time.time() - t0
                pc_str = " ".join(
                    f"{c}={sum(per_class[c][-30:])/max(1,len(per_class[c][-30:])):.3f}"
                    for c in CLASSES
                )
                print(f"[step {step:6d}] loss={avgT:.4f} lat={avgL:.4f} wav={avgW:.4f} "
                      f"{pc_str} lr={sched.get_last_lr()[0]:.2e} elapsed={el:.0f}s",
                      flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"student_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args),
                            "loss": sum(losses_total[-100:])/max(1,len(losses_total[-100:]))}, p)
                print(f"  → saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "student_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, final loss={sum(losses_total[-100:])/100:.4f}, saved {final}")


if __name__ == "__main__":
    main()
