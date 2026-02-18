#!/usr/bin/env python3
"""Generate listening test audio files."""
import sys
import logging
import torch
import torchaudio
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)

OUT_DIR = Path('/tmp/listening_test')
MANIFEST = '/mnt/models/inverse_afx_data/manifest.json'
FX_TYPES = ['distortion', 'delay', 'reverb', 'chorus', 'compressor']


def find_latest_checkpoint():
    ckpt_dir = Path('/home/arlo/do-repo/home/arlo/Modulo/nablafx/checkpoints')
    unified_dirs = sorted(ckpt_dir.glob('unified_phase1_single-epoch=*'))
    if not unified_dirs:
        return None
    latest_dir = unified_dirs[-1]
    ckpt_files = list(latest_dir.glob('*.ckpt'))
    return ckpt_files[0] if ckpt_files else None


def compute_si_sdr(pred, target):
    L = min(pred.shape[-1], target.shape[-1])
    p, d = pred[:, :L], target[:, :L]
    dot = (d * p).sum()
    s_t = dot * d / (d * d).sum()
    e = p - s_t
    return 10 * torch.log10((s_t**2).sum() / (e**2).sum() + 1e-8)


def is_valid_sample(wet, dry, threshold=0.3):
    w_n = (wet - wet.mean()) / (wet.std() + 1e-8)
    d_n = (dry - dry.mean()) / (dry.std() + 1e-8)
    return (w_n * d_n).mean().item() > threshold


def main():
    OUT_DIR.mkdir(exist_ok=True)

    ckpt_path = find_latest_checkpoint()
    if not ckpt_path:
        log.error("No checkpoint found")
        return
    log.info(f"Checkpoint: {ckpt_path}")

    from inverse_afx.training.train_unified import UnifiedInverterSystem
    model = UnifiedInverterSystem.load_from_checkpoint(str(ckpt_path), map_location='cpu')
    model.eval()
    log.info("Model loaded")

    with open(MANIFEST) as f:
        samples = json.load(f)
    log.info(f"Manifest: {len(samples)} samples")

    test_effects = ['distortion', 'reverb']
    samples_per_effect = 2

    for fx in test_effects:
        log.info(f"Processing {fx}...")
        count = 0

        for s in samples:
            if s['chain_spec'][0][0] != fx:
                continue

            wet, sr = torchaudio.load(s['wet_path'])
            dry, _ = torchaudio.load(s['dry_path'])
            L = min(wet.shape[-1], dry.shape[-1])
            wet, dry = wet[:, :L], dry[:, :L]

            if not is_valid_sample(wet, dry):
                continue

            fx_idx = FX_TYPES.index(fx)
            effect_types = torch.tensor([[fx_idx]])
            params = list(s['chain_spec'][0][1].values())[:15]
            params += [0.0] * (15 - len(params))
            effect_params = torch.tensor([params]).unsqueeze(1).float()

            with torch.no_grad():
                pred = model(wet.unsqueeze(0), effect_types, effect_params).squeeze(0)

            # Normalize
            wet_out = wet / wet.abs().max() * 0.9
            dry_out = dry / dry.abs().max() * 0.9
            pred_out = pred / pred.abs().max() * 0.9

            # Save
            torchaudio.save(OUT_DIR / f'{fx}_{count}_1_wet.wav', wet_out, sr)
            torchaudio.save(OUT_DIR / f'{fx}_{count}_2_dry_target.wav', dry_out, sr)
            torchaudio.save(OUT_DIR / f'{fx}_{count}_3_pred.wav', pred_out, sr)

            si_sdr = compute_si_sdr(pred, dry)
            log.info(f"  {fx}_{count}: SI-SDR = {si_sdr.item():.2f} dB")

            count += 1
            if count >= samples_per_effect:
                break

    log.info("=" * 40)
    log.info(f"Saved to: {OUT_DIR}")
    for f in sorted(OUT_DIR.glob('*.wav')):
        log.info(f"  {f.name}")


if __name__ == '__main__':
    main()
