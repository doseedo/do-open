#!/usr/bin/env python3
"""
Generate synthetic-real latent pairs for clarifier training.

Uses the trained performer model (Pipeline) to generate synthetic latents
from conditioning, then pairs them with real latents.

Usage:
    python generate_pairs.py \
        --checkpoint /path/to/performer.ckpt \
        --manifest /home/arlo/Data.backup/final_training_manifest_final.json \
        --output_dir /mnt/msdd2/clarifier_pairs_brass \
        --group brass \
        --steps 30 \
        --max_samples 100
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Project imports
sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

try:
    from trainer_performer_backup import Pipeline
except ImportError:
    from trainer_performer import Pipeline

from dataloader import APPROVED_GROUPS, APPROVED_SUBGROUPS, PerformerAIDataset

# Custom collate that handles None encodec tokens
def collate_latent_cond_no_encodec(batch):
    """Collate function that handles missing encodec tokens."""
    from dataloader import _pad_dim, _pad_last

    maxT_slow = max(it["latents"].shape[2] for it in batch)

    lat_list = []
    cond_keys = list(batch[0]["conds"].keys())
    cond_lists = {k: [] for k in cond_keys}
    group_ids, subgroup_ids, metas = [], [], []

    for it in batch:
        lat_list.append(_pad_dim(it["latents"], maxT_slow, dim=2))
        for k in cond_keys:
            cond_lists[k].append(_pad_last(it["conds"][k], maxT_slow))
        group_ids.append(it["instrument"]["group_id"])
        subgroup_ids.append(it["instrument"]["subgroup_id"])
        metas.append(it["meta"])

    return {
        "latents": torch.stack(lat_list, 0),
        "encodec_tokens": None,  # Not used
        "conds": {k: torch.stack(v, 0) for k, v in cond_lists.items()},
        "instrument": {
            "group_id": torch.stack(group_ids, 0),
            "subgroup_id": torch.stack(subgroup_ids, 0),
        },
        "meta": metas,
    }


def load_model(ckpt_path: str, checkpoint_dir: str, manifest_json: str) -> Pipeline:
    """Load Pipeline model from checkpoint (same as genfromweb5.py)."""
    print(f"Loading checkpoint: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location="cpu")

    hp = blob.get("hyper_parameters", {})

    # Build ctor kwargs from hyperparameters
    ctor_kwargs = {}
    if "checkpoint_dir" in hp:
        ctor_kwargs["checkpoint_dir"] = hp["checkpoint_dir"]
    if "manifest_json" in hp:
        ctor_kwargs["manifest_json"] = hp["manifest_json"]
    if "d_text" in hp:
        ctor_kwargs["d_text"] = hp["d_text"]
    if "inst_emb_dim" in hp:
        ctor_kwargs["inst_emb_dim"] = hp["inst_emb_dim"]

    # Override with provided paths
    ctor_kwargs["checkpoint_dir"] = checkpoint_dir
    ctor_kwargs["manifest_json"] = manifest_json

    print("Instantiating Pipeline...")
    model = Pipeline(**ctor_kwargs).eval()

    sd = blob.get("state_dict", blob)

    # Patch mismatched embedding sizes
    def _resize_like(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if src.shape == tgt.shape:
            return src
        out = tgt.clone()
        slices = tuple(slice(0, min(s, t)) for s, t in zip(src.shape, tgt.shape))
        out[slices] = src[slices]
        return out

    patch_keys = [
        "ctrl_enc.subgroup_emb.weight",
        "ctrl_enc.group_emb.weight",
    ]
    for k in patch_keys:
        if k in sd:
            parts = k.split(".")
            target = model
            for p in parts[:-1]:
                target = getattr(target, p, None)
                if target is None:
                    break
            if target is not None:
                tgt_param = getattr(target, parts[-1], None)
                if tgt_param is not None and sd[k].shape != tgt_param.shape:
                    sd[k] = _resize_like(sd[k], tgt_param)

    print("Loading state dict...")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    return model.cuda().eval()


def _bank_softplus_resized(model, H: int, device, dtype):
    """Get pitch-to-height mapping matrix from model."""
    if hasattr(model, "_bank_softplus_resized"):
        return model._bank_softplus_resized(H, device, dtype)
    W = getattr(model, "pitch2h_bank", None)
    if W is None:
        W = torch.ones(H, 128, device=device, dtype=dtype) * 0.01
    else:
        W = W.to(device=device, dtype=dtype)
    if W.shape[0] != H:
        W = F.interpolate(W.T.unsqueeze(0), size=H, mode="linear", align_corners=False).squeeze(0).T
    return F.softplus(W)


def _adapter_gain_scale(model) -> float:
    """Get adapter gain scale from model."""
    if hasattr(model, "_adapter_gain_scale"):
        return model._adapter_gain_scale()
    steps = int(getattr(model, "adapter_warmup_steps", 2000))
    gstep = int(getattr(model, "global_step", 0))
    return float(min(1.0, (gstep + 1) / max(1, steps)))


@torch.no_grad()
def generate_synthetic(
    model: Pipeline,
    batch: Dict[str, Any],
    steps: int = 30,
    cfg_weight: float = 3.0,
    adapter_scale: float = 1.0,
    inst_boost: float = 2.5,
    pitch_fidelity_boost: float = 1.0,
    onset_guidance_boost: float = 2.0,
    noise_level: float = 1.0,  # 1.0 = pure noise, 0.1 = 90% GT + 10% noise
) -> torch.Tensor:
    """
    Generate synthetic latent from conditioning using the Pipeline.
    Matches the generate() function from genfrominterface.py.

    Returns: [B, 8, 16, T] synthetic latent
    """
    device = next(model.parameters()).device

    # Get target shape from real latent
    real_latent = batch["latents"].to(device)
    B, C, H, T_slow = real_latent.shape

    # Build conditioning dict for ctrl_enc
    # Handle missing encodec tokens - use zeros if not present
    encodec = batch.get("encodec_tokens")
    if encodec is None or (hasattr(encodec, 'numel') and encodec.numel() == 0):
        T_fast = T_slow * 8
        encodec = torch.zeros(B, 8, T_fast, dtype=torch.long, device=device)
    else:
        encodec = encodec.to(device)

    conds = {
        "piano_roll": batch["conds"]["piano_roll"].to(device),
        "amp": batch["conds"]["amp"].to(device),
        "rframe": batch["conds"]["rframe"].to(device),
        "rbend": batch["conds"]["rbend"].to(device),
        "rbend_mask": batch["conds"]["rbend_mask"].to(device),
        "encodec_tokens": encodec,
        "group_id": batch["instrument"]["group_id"].to(device),
        "subgroup_id": batch["instrument"]["subgroup_id"].to(device),
    }

    # Get conditioning tokens
    tokens, _ = model.ctrl_enc(**conds)
    tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype)

    # Adapter gain scale
    adapter_gain = _adapter_gain_scale(model)
    sample_patch = model.cond_adapter(tokens_adapt, T_out=T_slow, scale=adapter_gain)

    # Start from noise or GT+noise mixture
    if noise_level >= 1.0:
        x = torch.randn_like(sample_patch.to(device=device, dtype=tokens.dtype))
        t0 = 1.0
    else:
        # Mix GT latent with noise (img2img style)
        gt_latent = real_latent.to(device=device, dtype=tokens.dtype)
        noise = torch.randn_like(gt_latent)
        x = (1.0 - noise_level) * gt_latent + noise_level * noise
        t0 = noise_level  # Start from lower timestep since we have GT info

    # Scheduler
    T_train = int(getattr(model.scheduler.config, "num_train_timesteps", 1000))
    dt = float(t0) / float(steps)

    # Piano roll for height masking
    pr = conds["piano_roll"].to(device=x.device, dtype=x.dtype)

    # Pitch-to-height mapping matrix
    W_hp = _bank_softplus_resized(model, H, device=x.device, dtype=x.dtype)

    # Sampling loop (matches genfrominterface.py)
    for i in range(steps, 0, -1):
        t_cont = torch.full((B,), i * dt, device=device, dtype=torch.float32)
        t_idx = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

        # Instrument ON/OFF conditioning
        tokens_on = tokens_adapt.clone()
        tokens_on[:, 0, :] *= float(inst_boost)
        tokens_off = tokens_adapt.clone()
        tokens_off[:, 0, :].zero_()

        cond_on = model.cond_adapter(tokens_on, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)
        cond_off = model.cond_adapter(tokens_off, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)

        # Piano roll height masking (critical for quality)
        T_lat = x.shape[-1]
        pr_resized = pr
        if pr.shape[-1] != T_lat:
            pr_resized = F.interpolate(pr, size=T_lat, mode="nearest")

        # Normalize piano roll and apply pitch fidelity
        pr_norm = pr_resized / (pr_resized.amax(dim=1, keepdim=True) + 1e-6)
        pr_high = pr_norm * (1.0 + float(pitch_fidelity_boost))
        pr_low = pr_norm * (1.0 - float(pitch_fidelity_boost) * 0.5)

        # Map to height channels
        H_on = torch.einsum('bpt,hp->bht', pr_high, W_hp)
        H_off = torch.einsum('bpt,hp->bht', pr_low, W_hp)

        # Sharpen and normalize
        sharp = 1.0 + float(pitch_fidelity_boost) * 0.5
        H_on = (H_on + 1e-6).pow(sharp)
        H_off = (H_off + 1e-6).pow(sharp)
        H_on = H_on / (H_on.amax(dim=1, keepdim=True) + 1e-6)
        H_off = H_off / (H_off.amax(dim=1, keepdim=True) + 1e-6)

        # Apply height masking to conditioning
        cond_on = cond_on * H_on.unsqueeze(1)
        cond_off = cond_off * H_off.unsqueeze(1)

        # Onset-weighted guidance
        if pr_norm.shape[-1] > 1:
            onset = (pr_norm[:, :, 1:] > 0.1) & (pr_norm[:, :, :-1] <= 0.1)
            onset = F.pad(onset.float().amax(dim=1, keepdim=True), (1, 0))
        else:
            onset = torch.zeros_like(pr_norm[:, :1, :])
        if onset.shape[-1] != T_lat:
            onset = F.interpolate(onset, size=T_lat, mode="nearest")

        base_guid = max(1.0, float(cfg_weight))
        step_guid = base_guid * (1.0 + float(onset_guidance_boost) * onset)

        # Transformer calls
        v_un = model._call_transformer_no_xattn(latents=x + cond_off, t=t_idx)
        v_co = model._call_transformer_no_xattn(latents=x + cond_on, t=t_idx)

        # CFG with onset-weighted guidance
        v_pred = v_un + step_guid.unsqueeze(1) * (v_co - v_un)

        # Euler step
        x = x - dt * v_pred

    return x


def generate_pairs(
    checkpoint_path: str,
    manifest_path: str,
    output_dir: str,
    checkpoint_dir: str = "/home/arlo/Data/ACE-Step/checkpoints",
    group_filter: Optional[str] = None,
    steps: int = 30,
    cfg_weight: float = 3.0,
    max_samples: Optional[int] = None,
    window_slow: int = 512,
    skip_existing: bool = True,
    path_replace: str = "/mnt/msdd/,/mnt/msdd2/",
    noise_level: float = 1.0,
):
    """Generate synthetic-real pairs for clarifier training."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_model(checkpoint_path, checkpoint_dir, manifest_path)

    # Build group/subgroup lookups
    group2id = {g: i for i, g in enumerate(APPROVED_GROUPS)}
    all_subs = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})
    sub2id = {sg: i for i, sg in enumerate(all_subs)}
    id2sub = {i: sg for sg, i in sub2id.items()}

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Loaded {len(manifest)} entries from manifest")

    # Apply path replacement to manifest
    if path_replace:
        old_path, new_path = path_replace.split(",")
        def replace_paths(obj):
            if isinstance(obj, str):
                return obj.replace(old_path, new_path)
            elif isinstance(obj, dict):
                return {k: replace_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_paths(x) for x in obj]
            return obj
        manifest = replace_paths(manifest)
        print(f"Applied path replacement: {old_path} -> {new_path}")

    # Filter by group
    if group_filter:
        group_filter = group_filter.lower()
        manifest = [m for m in manifest if m.get("group", "").lower() == group_filter]
        print(f"Filtered to {len(manifest)} {group_filter} entries")

    # Skip ensemble-detected and session-flagged entries
    pre_ensemble = len(manifest)
    manifest = [m for m in manifest
                if not m.get("ensemble_detected", False)
                and not m.get("session_ensemble_flagged", False)]
    skipped_ensemble = pre_ensemble - len(manifest)
    if skipped_ensemble > 0:
        print(f"Skipped {skipped_ensemble} ensemble/session-flagged entries, {len(manifest)} remaining")

    # Store manifest for source path lookup
    manifest_lookup = {i: m for i, m in enumerate(manifest)}

    if max_samples:
        manifest = manifest[:max_samples]

    # Write temp manifest with replaced paths
    import tempfile
    temp_manifest = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(manifest, temp_manifest)
    temp_manifest.close()
    temp_manifest_path = temp_manifest.name

    # Create dataset for loading conditioning
    # require_all_core=False to allow missing encodec tokens
    dataset = PerformerAIDataset(
        json_path=temp_manifest_path,
        conditioning_dropout={"piano_roll": 0.0, "amp": 0.0, "rbend": 0.0, "rframe": 0.0},
        use_trim=True,
        pre_roll_seconds=1.0,
        post_roll_seconds=0.25,
        keep_untrimmed_prob=0.0,
        amp_activity_thr=0.06,
        require_all_core=False,
        static_window=True,
        window_slow=window_slow,
        seed=0,
    )

    # All dataset indices are valid (manifest already filtered)
    valid_indices = list(range(len(dataset)))
    print(f"Dataset has {len(valid_indices)} valid samples after filtering")

    saved = 0
    skipped = 0
    errors = 0

    for idx in tqdm(valid_indices, desc="Generating pairs"):
        out_path = os.path.join(output_dir, f"pair_{idx:06d}.pt")

        if skip_existing and os.path.exists(out_path):
            skipped += 1
            continue

        try:
            # Get item and batch it
            item = dataset[idx]
            batch = collate_latent_cond_no_encodec([item])

            # Generate synthetic latent
            synthetic = generate_synthetic(model, batch, steps=steps, cfg_weight=cfg_weight, noise_level=noise_level)

            # Get real latent
            real = batch["latents"].cuda()

            # Get IDs
            group_id = batch["instrument"]["group_id"].item()
            subgroup_id = batch["instrument"]["subgroup_id"].item()
            subgroup_name = id2sub.get(subgroup_id, "undefined")

            # Get source path from manifest lookup
            source_entry = manifest_lookup.get(idx, {})
            source_path = source_entry.get("audio_path", "")

            # Save pair
            torch.save({
                "synthetic": synthetic.squeeze(0).cpu(),  # [8, 16, T]
                "real": real.squeeze(0).cpu(),
                "group_id": group_id,
                "subgroup_id": subgroup_id,
                "meta": {
                    "subgroup": subgroup_name,
                    "index": idx,
                    "source_path": source_path,
                }
            }, out_path)

            saved += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"Error at idx {idx}: {e}")
            continue

        # Progress update
        if (saved + 1) % 50 == 0:
            print(f"  Saved {saved}, skipped {skipped}, errors {errors}")

    # Cleanup temp manifest
    os.unlink(temp_manifest_path)

    print(f"\nDone! Saved {saved} pairs to {output_dir}")
    print(f"Skipped {skipped} existing, {errors} errors")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic-real pairs for clarifier")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to performer model checkpoint (.ckpt)")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to training manifest JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for pairs")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/home/arlo/Data/ACE-Step/checkpoints",
                        help="ACE-Step checkpoints directory")
    parser.add_argument("--group", type=str, default=None,
                        help="Filter by instrument group (e.g., brass)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Diffusion steps for generation")
    parser.add_argument("--cfg", type=float, default=3.0,
                        help="CFG weight")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process")
    parser.add_argument("--window_slow", type=int, default=512,
                        help="Window size in slow frames")
    parser.add_argument("--path_replace", type=str, default="/mnt/msdd/,/mnt/msdd2/",
                        help="Path replacement 'old,new'")
    parser.add_argument("--no_skip", action="store_true",
                        help="Don't skip existing pairs")
    parser.add_argument("--noise_level", type=float, default=1.0,
                        help="Noise level (1.0=pure noise, 0.1=90%% GT + 10%% noise)")

    args = parser.parse_args()

    generate_pairs(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        group_filter=args.group,
        steps=args.steps,
        cfg_weight=args.cfg,
        max_samples=args.max_samples,
        window_slow=args.window_slow,
        skip_existing=not args.no_skip,
        path_replace=args.path_replace,
        noise_level=args.noise_level,
    )


if __name__ == "__main__":
    main()
