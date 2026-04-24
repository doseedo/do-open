#!/usr/bin/env python3
"""Sample a diverse (audio, latent) pair list for PANNs distillation.

For each `group` in the v2.3 manifest we take up to `per_group` voice entries
whose Latents2 file exists (found via the bulk gsutil list at
/scratch/latent_whisper_student/latents2_list.txt — reused).

Writes /scratch/latent_panns_student/pairs.json with
    [{audio, latent, group}]
"""
import json
import os
import random
import sys

MANIFEST = "/home/arlo/gcs-bucket/Manifests/master_manifest_v2.3.json"
LATENTS_LIST = "/scratch/latent_whisper_student/latents2_list.txt"
OUT = "/scratch/latent_panns_student/pairs.json"
BUCKET_PREFIX = "/home/arlo/gcs-bucket/"

# groups to sample from (skip "undefined", "room" since they're not
# instrument-identifiable, but sample mix heavily).
TARGETS = {
    "drums":      2000,
    "voice":      2000,
    "guitar":     2000,
    "strings":    2000,
    "piano":      2000,
    "bass":       2000,
    "mix":        2000,
    "brass":      2000,
    "winds":      2000,
    "percussion": 2000,
    "synth":      1500,
    "organ":      1500,
    "dialogue":   1000,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("loading latent list…", flush=True)
latent_set: set[str] = set()
with open(LATENTS_LIST) as f:
    for line in f:
        line = line.strip()
        if line.endswith(".vae.pt") and line.startswith("gs://ptxsessiondata/"):
            latent_set.add(BUCKET_PREFIX + line[len("gs://ptxsessiondata/"):])
print(f"  {len(latent_set)} Latents2 files", flush=True)

print("loading manifest…", flush=True)
m = json.load(open(MANIFEST))
entries = m["entries"]
print(f"  {len(entries)} entries", flush=True)


def to_latent(p: str) -> str | None:
    if not p.startswith(BUCKET_PREFIX):
        return None
    stem, _ = os.path.splitext(p[len(BUCKET_PREFIX):])
    return BUCKET_PREFIX + "Latents2/" + stem + ".vae.pt"


random.seed(0)
by_group: dict[str, list] = {g: [] for g in TARGETS}
for k, v in entries.items():
    g = v.get("group")
    if g not in TARGETS:
        continue
    lp = to_latent(k)
    if not lp or lp not in latent_set:
        continue
    by_group[g].append({"audio": k, "latent": lp, "group": g})

picked: list = []
for g, lst in by_group.items():
    want = TARGETS[g]
    if len(lst) > want:
        random.shuffle(lst)
        lst = lst[:want]
    picked.extend(lst)
    print(f"  {g:12s}: picked {len(lst)} / {len(by_group[g])}")

random.shuffle(picked)
json.dump(picked, open(OUT, "w"))
print(f"wrote {OUT}  ({len(picked)} pairs, "
      f"{os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)
