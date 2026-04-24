#!/usr/bin/env python3
"""Decode distill predictions: take a real mix wav, run student → N latents,
decode each through Oobleck VAE → N stem wavs.

Supports both 4-stem and 6-stem models (auto-detected from checkpoint)."""
import os, sys, argparse
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/scratch/ACE-Step-1.5")
from distill_model import WaveformToFourStemLatents
from distill_dataset import _load_audio_48k_stereo, _load_latent

STEMS_4 = ["drums", "bass", "vocals", "other"]
STEMS_6 = ["drums", "bass", "other", "vocals", "guitar", "piano"]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out_dir", default="/scratch/latent_demucs_student/distill_eval")
ap.add_argument("--secs", type=float, default=8.0)
ap.add_argument("--stems", type=int, default=0,
                help="4 or 6 (0 = auto-detect from checkpoint)")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# Detect n_stems from checkpoint
sd = torch.load(args.ckpt, map_location="cuda", weights_only=False)
if args.stems > 0:
    n_stems = args.stems
else:
    # conv2 output channels = n_stems * 64
    conv2_key = [k for k in sd["model"] if "conv2.weight" in k][0]
    out_ch = sd["model"][conv2_key].shape[0]
    n_stems = out_ch // 64
    print(f"[eval] auto-detected n_stems={n_stems} from conv2 out_ch={out_ch}")

if n_stems == 6:
    STEMS = STEMS_6
    teacher_prefix = "teacher6_"
    from distill_dataset_6 import DistillDataset6
    print("[eval] loading 6-stem dataset...")
    ds = DistillDataset6(crop_frames=200)
else:
    STEMS = STEMS_4
    teacher_prefix = "teacher_"
    from distill_dataset import DistillDataset
    print("[eval] loading 4-stem dataset...")
    ds = DistillDataset(crop_frames=200)

print(f"[eval] loading model (n_stems={n_stems})...")
model = WaveformToFourStemLatents(n_stems=n_stems).cuda().eval()
model.load_state_dict(sd["model"])
print(f"[eval] loaded ckpt step={sd.get('step','?')}")

# pick first item from each source
v7_items   = [it for it in ds.items if it["src"] == "v7"]
musdb_items = [it for it in ds.items if it["src"] == "musdb"]
picks = []
if v7_items:    picks.append(("v7", v7_items[0]))
if musdb_items: picks.append(("musdb", musdb_items[0]))

print("[eval] loading VAE decoder...")
from diffusers.models import AutoencoderOobleck
vae = AutoencoderOobleck.from_pretrained("/scratch/ACE-Step-1.5/checkpoints/vae")
vae = vae.cuda().eval().to(torch.bfloat16)

def decode_latent(z_64xT):
    z = z_64xT.unsqueeze(0).to(torch.bfloat16).cuda()
    with torch.no_grad():
        a = vae.decode(z).sample.float().cpu()
    return a.squeeze(0).T.numpy()

N = int(args.secs * 48000)
for tag, it in picks:
    print(f"\n=== {tag}: {it['audio']}")
    full_audio = _load_audio_48k_stereo(it["audio"])
    # Pick the loudest N-sample window (skips silent intros)
    win = N
    if full_audio.shape[1] > win:
        e = full_audio.abs().mean(dim=0)
        cums = torch.cumsum(torch.cat([torch.tensor([0.0]), e]), dim=0)
        seg_e = cums[win:] - cums[:-win]
        start = int(seg_e.argmax().item())
    else:
        start = 0
    audio = full_audio[:, start:start + N]
    print(f"  window: [{start/48000:.1f}s, {(start+N)/48000:.1f}s]")
    sf.write(f"{args.out_dir}/{tag}_input.wav", audio.T.numpy(), 48000)

    # student
    x = audio.unsqueeze(0).cuda().to(torch.bfloat16)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x).float()                       # [1, n_stems, 64, T]

    for ci, name in enumerate(STEMS):
        wav = decode_latent(pred[0, ci])
        sf.write(f"{args.out_dir}/{tag}_pred_{name}.wav", wav, 48000)
        print(f"  pred {name}: |x|={pred[0,ci].abs().mean():.3f}")

    # GT for reference (slice at same audio offset)
    frame_off = start // 1920
    stem_paths = it["stem_paths"]
    for ci, name in enumerate(STEMS):
        p = stem_paths[ci] if ci < len(stem_paths) else None
        if p is None:
            print(f"  tgt  {name}: (masked — no GT)")
            continue
        z = _load_latent(p).T                             # [64, T_total]
        T = pred.shape[-1]
        z = z[:, frame_off:frame_off + T]
        wav = decode_latent(z)
        sf.write(f"{args.out_dir}/{tag}_target_{name}.wav", wav, 48000)
        print(f"  tgt  {name}: |x|={z.abs().mean():.3f}")

print(f"\n[done] wavs in {args.out_dir}")
