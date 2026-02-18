# preview_working.py
# Minimal preview without stopping training. Mirrors training path:
# - PR patch + token adapter patch (with warmup-scaled gains)
# - scheduler.scale_model_input
# - optional near-GT start, optional EnCodec drop
# - env knobs: PR_STRONG, OTHER_MAX

import os
import argparse
import torch
import torchaudio

from trainer_performer import (
    Pipeline,
    PerformerAIDataset,
    collate_latent_cond,
    DCAE_HOP,
    DCAE_SR,
)

@torch.no_grad()
def build_preview_batch(manifest_json: str, window_slow: int, idx: int):
    ds_prev = PerformerAIDataset(
        json_path=manifest_json,
        conditioning_dropout={"piano_roll": 0.0, "amp": 0.0, "rbend": 0.0, "rframe": 0.0},
        use_trim=True,
        pre_roll_seconds=1.0,
        post_roll_seconds=0.0,
        keep_untrimmed_prob=0.0,
        amp_activity_thr=0.06,
        require_all_core=True,
        collapse_sparse_subgroups_to_any=False,
        static_window=True,
        window_slow=window_slow,
        seed=0,
    )
    item = ds_prev[int(idx) % len(ds_prev)]
    return collate_latent_cond([item])

@torch.no_grad()
def preview(
    model: Pipeline,
    batch: dict,
    steps: int = 40,
    sr_out: int = 48000,
    out_wav: str = "preview.wav",
    drop_encodec_preview: bool = False,
    near_gt_t0: float | None = None,
    gate_other_between_notes: bool = True,
):
    model.eval()
    device = model.device

    # --- target latent shape ---
    x0 = batch["latents"].to(device)
    B, _, _, T_slow = x0.shape

    # --- controls (optionally drop EnCodec) ---
    enc = batch["encodec_tokens"].to(device)
    if drop_encodec_preview:
        enc = torch.zeros_like(enc)

    tokens, mask = model.ctrl_enc(
        piano_roll=batch["conds"]["piano_roll"].to(device),
        amp=batch["conds"]["amp"].to(device),
        rframe=batch["conds"]["rframe"].to(device),
        rbend=batch["conds"]["rbend"].to(device),
        rbend_mask=batch["conds"]["rbend_mask"].to(device),
        encodec_tokens=enc,
        group_id=batch["instrument"]["group_id"].to(device),
        subgroup_id=batch["instrument"]["subgroup_id"].to(device),
    )

    # --- init x: pure noise or near-GT ---
    torch.manual_seed(0)
    z = torch.randn_like(x0)
    if near_gt_t0 is not None and 0.0 < float(near_gt_t0) < 1.0:
        t0 = float(near_gt_t0)
        x = (1.0 - t0) * x0 + t0 * z
        total_span = t0
    else:
        x = z
        total_span = 1.0

    # --- schedule ---
    T_train = int(getattr(model.scheduler.config, "num_train_timesteps", 1000))
    steps = max(1, int(steps))
    dt = float(total_span) / float(steps)

    # --- env scales (same knobs as training) ---
    pr_scale_env    = float(os.environ.get("PR_STRONG", "2.0"))
    other_scale_env = float(os.environ.get("OTHER_MAX", "1.0"))

    # cast tokens to adapter’s module dtype/device
    tokens_adapt = model._match_mod_dtype(tokens, model.cond_adapter)

    pr_slow = batch["conds"]["piano_roll"].to(device)
    voiced_slow = (pr_slow > 0).any(dim=1).float()  # [B, T_slow]

    for i in range(steps, 0, -1):
        t_cont = torch.full((B,), i * dt, device=device)
        t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

        T_out = x.shape[-1]

        # PR patch
        pr_patch = 0.0
        if hasattr(model, "pr_adapter"):
            pr_patch = model.pr_adapter(
                pr_slow,
                T_out=T_out,
                scale=(model._pr_gain_scale() * pr_scale_env),
            ).to(device=x.device, dtype=x.dtype)

        # token adapter (“other”)
        other_patch = model.cond_adapter(
            tokens_adapt, T_out=T_out, scale=(model._adapter_gain_scale() * other_scale_env)
        ).to(device=x.device, dtype=x.dtype)

        if gate_other_between_notes:
            # Gate shape [B,1,1,T]—broadcast across channels/heads
            gate = 1.0 - torch.nn.functional.interpolate(
                voiced_slow.unsqueeze(1), size=T_out, mode="nearest"
            ).unsqueeze(1)  # [B,1,1,T]
            relax = 0.4  # let some timbre through during notes
            other_patch = relax * other_patch + (1.0 - relax) * (other_patch * gate)

        x_in = x + pr_patch + other_patch
        x_scaled = model._scale_in(x_in, t_idx)  # scheduler scaling (mirrors train)

        v_pred = model._call_transformer_no_xattn(latents=x_scaled, t=t_idx)
        x = x - dt * v_pred  # RF/Euler

    # --- decode full span ---
    audio_len_out = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
    x_for_dcae = model._match_mod_dtype(x[:1], model.dcae)
    audio_lengths = torch.tensor([audio_len_out], device=x_for_dcae.device, dtype=torch.long)
    sr_pred, wav_pred = model.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    torchaudio.save(out_wav, wav_pred[0].float().cpu(), sr_pred)
    print(f"[preview] wrote {out_wav} (sr={sr_pred}, steps={steps})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--manifest_json", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_wav", default="./preview_out.wav")
    ap.add_argument("--preview_index", type=int, default=0)
    ap.add_argument("--window_slow", type=int, default=512)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--device", default=None, help="cuda:0 or cpu (auto if None)")
    ap.add_argument("--drop_encodec_preview", action="store_true")
    ap.add_argument("--near_gt_t0", type=float, default=None)
    ap.add_argument("--no_gate_other", action="store_true")
    args = ap.parse_args()

    # Lightweight pipeline (no training)
    model = Pipeline(
        checkpoint_dir=args.checkpoint_dir,
        manifest_json=args.manifest_json,
        batch_size=1,
        num_workers=0,
        window_slow=args.window_slow,
        preview_index=args.preview_index,
        preview_steps=args.steps,
    )

    # Load weights (strict=False for new heads)
    sd = torch.load(args.ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:   print(f"[load] missing: {len(missing)} (ok for new modules)")
    if unexpected: print(f"[load] unexpected: {len(unexpected)}")

    # Device
    dev = args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    # Deterministic preview batch
    batch = build_preview_batch(args.manifest_json, args.window_slow, args.preview_index)

    # Disable adapter warmups for preview so scales apply immediately
    model.pr_adapter_warmup_steps = 1
    model.other_adapter_warmup_steps = 1
    model.adapter_warmup_steps = 1

    preview(
        model,
        batch,
        steps=args.steps,
        sr_out=args.sr,
        out_wav=args.out_wav,
        drop_encodec_preview=args.drop_encodec_preview,
        near_gt_t0=args.near_gt_t0,
        gate_other_between_notes=not args.no_gate_other,
    )

if __name__ == "__main__":
    main()
