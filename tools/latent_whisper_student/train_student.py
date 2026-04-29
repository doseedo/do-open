#!/usr/bin/env python3
"""Train the latent-whisper student.

Loss = λ_hid * L1(pred_hidden, target_hidden)        # encoder hidden-state loss
     + λ_ce  * CE(frozen_decoder(pred_hidden), tokens) # token-level distillation

The CE branch runs the *real* Whisper decoder (frozen, bf16) on the student's
predicted encoder output, so the student is punished for hidden states that
can't produce the right tokens — the analog of the VAE waveform loss used in
latent_demucs_student/train_student_v3.py.

Usage:
    python train_student.py --latent-root /scratch/stemphonic/data/ossl_latents \
        --whisper-model base --steps 20000
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student_model import LatentWhisperStudent, WHISPER_DIMS


def load_whisper_frozen(name: str, device: str = "cuda"):
    import whisper
    m = whisper.load_model(name, device=device,
                           download_root="/scratch/cache/whisper")
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", default="/scratch/latent_whisper_student/chunks",
                    help="directory produced by gen_vocal_teacher.py")
    ap.add_argument("--latent-root", default=None,
                    help="(legacy) old session+glob dataset root")
    ap.add_argument("--whisper-model", default="base",
                    choices=["tiny", "base", "small", "medium",
                             "large", "large-v2", "large-v3"])
    ap.add_argument("--out", default="/scratch/latent_whisper_student/ckpts")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_hid", type=float, default=1.0)
    ap.add_argument("--lambda_ce",  type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every",  type=int, default=20)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--max-tokens", type=int, default=224)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    # ── data ──────────────────────────────────────────────────────────
    if args.latent_root:
        from dataset import LatentWhisperDataset as _DS, collate as _col
        ds = _DS(latent_root=args.latent_root, model_name=args.whisper_model)
        collate = _col
    else:
        from chunk_dataset import VocalChunkDataset as _DS, collate as _col
        ds = _DS(chunks_dir=args.chunks_dir)
        collate = _col
    if len(ds) == 0:
        print("ERROR: empty dataset. Run gen_vocal_teacher.py first.")
        return

    import whisper.tokenizer as wt
    tok_helper = wt.get_tokenizer(multilingual=True)
    pad_id = tok_helper.eot  # use EOT as padding sentinel

    def _collate(b):
        return collate(b, max_tokens=args.max_tokens, pad_id=pad_id)

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=_collate,
                        persistent_workers=args.workers > 0)

    # ── frozen teacher (for CE through the decoder only) ─────────────
    # Keep teacher in fp32: whisper's internal LayerNorm casts to fp32 then
    # back, which collides with bf16-cast parameters. We call it with
    # pred.float() below — decoder compute is small relative to the student.
    print(f"[train] loading frozen whisper-{args.whisper_model}…")
    teacher = load_whisper_frozen(args.whisper_model, device="cuda")
    d_model = int(teacher.dims.n_audio_state)

    # ── student ───────────────────────────────────────────────────────
    model = LatentWhisperStudent(whisper_size=args.whisper_model).cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] student {n:.1f}M params  d_model={model.d_model} "
          f"layers={model.n_layers}")

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"[train] resumed from {args.resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # ── train loop ────────────────────────────────────────────────────
    model.train()
    step = 0
    hist_tot, hist_hid, hist_ce = [], [], []
    t0 = time.time()
    while step < args.steps:
        for batch in loader:
            mix = batch["mix_lat"].cuda(non_blocking=True)          # [B, 64, 750]
            tgt = batch["target_hidden"].cuda(non_blocking=True)    # [B, 1500, D]
            tok = batch["tokens"].cuda(non_blocking=True)           # [B, T_tok]
            tmask = batch["tok_mask"].cuda(non_blocking=True)       # [B, T_tok]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(mix)                                   # [B, 1500, D]

            # -- hidden-state L1 loss --
            loss_hid = F.l1_loss(pred.float(), tgt.float())

            # -- token-space CE through the frozen decoder --
            # teacher forcing: feed tokens[:, :-1], predict tokens[:, 1:]
            loss_ce = torch.zeros((), device="cuda")
            if args.lambda_ce > 0 and tok.shape[1] >= 2:
                inp = tok[:, :-1]
                lab = tok[:, 1:]
                lab_mask = tmask[:, 1:]
                # decoder() takes (tokens, encoder_hidden) → [B, T, V]
                # Call the frozen (fp32) decoder OUTSIDE autocast to avoid
                # whisper's LayerNorm cast issues.
                with torch.amp.autocast("cuda", enabled=False):
                    logits = teacher.decoder(inp, pred.float())
                V = logits.shape[-1]
                flat_logits = logits.reshape(-1, V)
                flat_lab = lab.reshape(-1)
                flat_mask = lab_mask.reshape(-1)
                if flat_mask.any():
                    ce = F.cross_entropy(
                        flat_logits[flat_mask], flat_lab[flat_mask],
                        reduction="mean")
                    loss_ce = ce

            loss = args.lambda_hid * loss_hid + args.lambda_ce * loss_ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            hist_tot.append(loss.item())
            hist_hid.append(loss_hid.item())
            hist_ce.append(loss_ce.item() if isinstance(loss_ce, torch.Tensor) else 0.0)
            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                el = time.time() - t0
                print(f"[step {step:6d}] loss={avg(hist_tot):.4f} "
                      f"hid={avg(hist_hid):.4f} ce={avg(hist_ce):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} elapsed={el:.0f}s",
                      flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"student_step{step}.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "args": vars(args),
                    "d_model": d_model,
                    "whisper_model": args.whisper_model,
                }, p)
                print(f"  → saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "student_final.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "args": vars(args),
        "d_model": d_model,
        "whisper_model": args.whisper_model,
    }, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
