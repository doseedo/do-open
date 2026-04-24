#!/usr/bin/env python3
"""Build (mix_latent, stem_latent, frame_offset, class) pair index from
timeline_master_index.json + local /scratch/Latents2 + /scratch/mixesV7_latents.

Each pair record:
  {
    "session": "<root>/<date>/<...>",
    "mix_path":  "/scratch/mixesV7_latents/<session>/full_mix.vae.pt",
    "stem_path": "/scratch/Latents2/<session>/Audio Files/<track>(_NN).vae.pt",
    "track_name": "Bass DI",
    "first_start_samples": 12345,        # offset into the *session timeline*
    "class":     "bass",                  # one of drums/bass/vocals/other
  }

Output: /scratch/cache/gt_pairs.json
"""
import json, os, re
from pathlib import Path

TIMELINE = "/scratch/cache/timeline_master_index.json"
MIX_ROOT  = "/scratch/mixesV7_latents"
STEM_ROOT = "/scratch/Latents2"
OUT       = "/scratch/cache/gt_pairs.json"

# Stem-name → demucs class. Order matters: first match wins.
CLASS_PATTERNS = [
    ("drums",  re.compile(r"\b(drum|kick|snare|hat|hihat|tom|cymb|ride|crash|oh\b|overhead|perc|conga|bongo|tabla|808|clap|click|drumkit)\b", re.I)),
    ("bass",   re.compile(r"\b(bass|sub\b|808 bass|upright|contrabass|bass\s*di)\b", re.I)),
    ("vocals", re.compile(r"\b(vocal|vox|voice|lead voc|bgv|backing|choir|sing|whisper|adlib|harmon)\b", re.I)),
    ("other",  re.compile(r"\b(guitar|gtr|piano|key|synth|organ|pad|string|brass|horn|sax|trumpet|trombone|flute|violin|cello|harp|mallet|wurli|rhodes|epiano|lead|fx)\b", re.I)),
]

def classify(name):
    for cls, pat in CLASS_PATTERNS:
        if pat.search(name):
            return cls
    return None  # unmatched → drop


def find_stem_file(audio_dir, track_name):
    """Find <track_name>.vae.pt or <track_name>_NN.vae.pt or <track_name>.NN.vae.pt"""
    if not os.path.isdir(audio_dir):
        return None
    target = f"{track_name}.vae.pt"
    direct = os.path.join(audio_dir, target)
    if os.path.exists(direct):
        return direct
    pat_us = re.compile(rf"^{re.escape(track_name)}_(\d+)\.vae\.pt$")
    pat_dot = re.compile(rf"^{re.escape(track_name)}\.(\d+)\.vae\.pt$")
    candidates = []
    for f in os.listdir(audio_dir):
        if pat_us.match(f) or pat_dot.match(f):
            candidates.append(f)
    if candidates:
        return os.path.join(audio_dir, sorted(candidates)[0])
    return None


def main():
    with open(TIMELINE) as f:
        d = json.load(f)
    sessions = d["sessions"]

    pairs = []
    n_sessions_with_mix = 0
    n_tracks_total = 0
    n_tracks_found = 0
    n_classified = 0
    unmatched_names = {}

    for skey, sv in sessions.items():
        mix_path = f"{MIX_ROOT}/{skey}/full_mix.vae.pt"
        if not os.path.exists(mix_path):
            continue
        n_sessions_with_mix += 1
        audio_dir = f"{STEM_ROOT}/{skey}/Audio Files"
        track_offsets = sv.get("track_offsets", {})
        for tname, info in track_offsets.items():
            n_tracks_total += 1
            stem_path = find_stem_file(audio_dir, tname)
            if stem_path is None:
                continue
            n_tracks_found += 1
            cls = classify(tname)
            if cls is None:
                unmatched_names[tname] = unmatched_names.get(tname, 0) + 1
                continue
            n_classified += 1
            pairs.append({
                "session": skey,
                "mix_path": mix_path,
                "stem_path": stem_path,
                "track_name": tname,
                "first_start_samples": int(info.get("first_start_samples", 0)),
                "class": cls,
            })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(pairs, f)

    by_class = {}
    for p in pairs:
        by_class[p["class"]] = by_class.get(p["class"], 0) + 1
    print(f"sessions w/ mix latent:        {n_sessions_with_mix}")
    print(f"tracks listed in timeline:     {n_tracks_total}")
    print(f"tracks w/ stem .vae.pt found:  {n_tracks_found}")
    print(f"tracks classified:             {n_classified}")
    print(f"by class: {by_class}")
    print(f"top unclassified: {sorted(unmatched_names.items(), key=lambda x:-x[1])[:15]}")
    print(f"wrote {len(pairs)} pairs → {OUT}")


if __name__ == "__main__":
    main()
