#!/usr/bin/env python3
"""Print magnitudes of mix, targets, and predictions to find the bug."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student_model import LatentDemucsStudent
from dataset_combined import CombinedSeparationDataset, collate, _load, CLASSES
from torch.utils.data import DataLoader

ckpt = "/scratch/latent_demucs_student/ckpts_v2/student_step10000.pt"

ds = CombinedSeparationDataset(crop_frames=400, seed=7)
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate)

m = LatentDemucsStudent().cuda()
sd = torch.load(ckpt, map_location="cuda", weights_only=False)
m.load_state_dict(sd["model"]); m.eval()

mix, stems, mask = next(iter(loader))
mix, stems, mask = mix.cuda(), stems.cuda(), mask.cuda()
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = m(mix).float()

print(f"mix    : mean|x|={mix.abs().mean():.4f}  std={mix.std():.4f}  max|x|={mix.abs().max():.4f}")
print(f"stems  : mean|x|={stems.abs().mean():.4f}  std={stems.std():.4f}  max|x|={stems.abs().max():.4f}")
print(f"  (only-supervised, masked)")
masked = (stems.abs() * mask[:, :, None, None])
sup_mean = masked.sum() / mask.sum().clamp_min(1) / (stems.shape[2] * stems.shape[3])
print(f"  supervised stems mean|x| = {sup_mean:.4f}")
print(f"pred   : mean|x|={pred.abs().mean():.4f}  std={pred.std():.4f}  max|x|={pred.abs().max():.4f}")

print("\nper-channel pred mean|x|:")
for ci, c in enumerate(CLASSES):
    print(f"  {c:8s}: {pred[:, ci].abs().mean():.4f}")

# A teacher item has all 4 channels supervised — pull one
print("\nteacher item only:")
teacher_dirs = [it["dir"] for it in ds.items if it["kind"] == "teacher"]
if teacher_dirs:
    d = teacher_dirs[0]
    mix1 = _load(d / "full_mix.vae.pt")[:400].T.unsqueeze(0).cuda()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred1 = m(mix1).float()
    print(f"  mix    mean|x|={mix1.abs().mean():.4f}")
    for ci, c in enumerate(CLASSES):
        tgt = _load(d / f"teacher_{c}.vae.pt")[:400].T.cuda()
        print(f"  {c:8s}: pred mean|x|={pred1[0, ci].abs().mean():.4f}  "
              f"target mean|x|={tgt.abs().mean():.4f}  "
              f"l1={ (pred1[0, ci] - tgt).abs().mean():.4f}")
