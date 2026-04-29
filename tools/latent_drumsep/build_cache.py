"""Offline teacher-data builder for latent drum sub-separation.

For each input drum-only audio file:
  1. Run MDX23C-DrumSep → 6 stem WAVs (kick/snare/toms/hh/ride/crash).
  2. VAE-encode the input mix → L_mix [T, 64].
  3. VAE-encode each stem      → L_stem [T, 64].
  4. Crop random windows of `--win-frames` frames and emit shards of
     {L_mix, L_stems, src} dicts.

We use the audio drumsep ONCE here, offline. The trained student needs
neither audio nor drumsep at inference.

Usage:
    python -m latent_drumsep.build_cache \
        --audio-roots /scratch/musdb18_wavs/train \
        --audio-glob "**/drums.wav" \
        --out /scratch/latent_drumsep_cache \
        --num 5000 --win-frames 64 --shard-size 500
"""
from __future__ import annotations
import argparse, glob, os, random, sys, time, traceback
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "/scratch/ACE-Step-1.5")
from acestep.handler import AceStepHandler  # noqa: E402
from latent_drumsep import STEMS  # noqa: E402

SR = 48000
SAMPLES_PER_FRAME = 1920


def get_drumsep(model_dir="/scratch/audio_separator_models",
                out_dir="/scratch/audio_separator_out",
                model="MDX23C-DrumSep-aufr33-jarredou.ckpt"):
    from audio_separator.separator import Separator
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    sep = Separator(model_file_dir=model_dir, output_dir=out_dir, log_level=30)
    sep.load_model(model)
    return sep, out_dir


def find_audio(roots, glob_pat):
    files = []
    for r in roots:
        files.extend(sorted(glob.glob(os.path.join(r, glob_pat), recursive=True)))
    return files


@torch.no_grad()
def encode(vae, wav_2S: np.ndarray) -> torch.Tensor:
    """[2, S] float -> [T, 64] cpu float."""
    x = torch.from_numpy(wav_2S).unsqueeze(0).to("cuda", torch.bfloat16)
    out = vae.encode(x)
    lat = out.latent_dist.sample() if hasattr(out, "latent_dist") else out.sample
    return lat[0].transpose(0, 1).float().cpu()  # [T, 64]


def load_stereo(path: str, sr: int = SR) -> np.ndarray | None:
    try:
        wav, src_sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        print(f"  read fail {path}: {e}")
        return None
    if wav.shape[1] == 1:
        wav = np.concatenate([wav, wav], axis=1)
    elif wav.shape[1] > 2:
        wav = wav[:, :2]
    if src_sr != sr:
        import librosa
        wav = librosa.resample(wav.T, orig_sr=src_sr, target_sr=sr).T
    return wav.T.astype(np.float32)  # [2, S]


def crop_aligned(L_mix, L_stems, win):
    """Random crop along time axis. L_mix [T,64], L_stems list of [T_i,64]."""
    T = min(L_mix.shape[0], min(s.shape[0] for s in L_stems))
    if T <= win:
        # zero-pad to win
        def pad(x):
            t = x.shape[0]
            if t >= win:
                return x[:win]
            return torch.cat([x, torch.zeros(win - t, 64)], 0)
        return pad(L_mix), [pad(s) for s in L_stems]
    s = random.randint(0, T - win)
    return L_mix[s:s+win], [st[s:s+win] for st in L_stems]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-roots", nargs="+", required=True)
    ap.add_argument("--audio-glob", default="**/drums.wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=5000)
    ap.add_argument("--win-frames", type=int, default=64)
    ap.add_argument("--crops-per-file", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("loading frozen ACE VAE...")
    h = AceStepHandler()
    h.initialize_service(project_root="/scratch/ACE-Step-1.5",
                         config_path="acestep-v15-sft", device="cuda")
    vae = h.vae
    for p in vae.parameters(): p.requires_grad = False
    vae.eval()

    print("loading MDX23C-DrumSep...")
    sep, sep_out_dir = get_drumsep()

    files = find_audio(args.audio_roots, args.audio_glob)
    print(f"  drum files: {len(files)}")
    if not files:
        raise SystemExit("no input files")
    random.shuffle(files)

    shard, sidx, written = [], 0, 0
    t0 = time.time()
    for fp in files:
        if written + len(shard) >= args.num:
            break
        wav_mix = load_stereo(fp)
        if wav_mix is None or wav_mix.shape[1] < SR:  # <1s
            continue
        # write a temp wav for audio_separator (it wants a path)
        tmp = os.path.join(sep_out_dir, f"_in_{os.getpid()}.wav")
        sf.write(tmp, wav_mix.T, SR)
        try:
            stem_files = sep.separate(tmp)
        except Exception as e:
            print(f"  drumsep fail {fp}: {e}")
            continue
        # map stem name -> wav path
        stem_paths = {s: None for s in STEMS}
        for fn in stem_files:
            full = fn if os.path.isabs(fn) else os.path.join(sep_out_dir, fn)
            base = os.path.basename(full).lower()
            for s in STEMS:
                if s in base:
                    stem_paths[s] = full
                    break
        if any(v is None for v in stem_paths.values()):
            print(f"  missing stems for {fp}: {stem_paths}")
            continue
        try:
            L_mix = encode(vae, wav_mix)
            L_stems = []
            for s in STEMS:
                w = load_stereo(stem_paths[s])
                if w is None: raise RuntimeError(f"reload fail {s}")
                L_stems.append(encode(vae, w))
        except Exception as e:
            print(f"  encode fail {fp}: {e}")
            continue
        finally:
            for p in [tmp] + [v for v in stem_paths.values() if v]:
                try: os.remove(p)
                except Exception: pass

        # multiple random crops per file
        for _ in range(args.crops_per_file):
            cm, cs = crop_aligned(L_mix, L_stems, args.win_frames)
            shard.append({
                "L_mix":   cm,                       # [win, 64]
                "L_stems": torch.stack(cs, dim=0),   # [n_stems, win, 64]
                "src":     os.path.basename(fp),
            })
            if len(shard) >= args.shard_size:
                p = os.path.join(args.out, f"shard_{sidx:05d}.pt")
                torch.save(shard, p); written += len(shard)
                rate = written / (time.time() - t0)
                print(f"  wrote {p}  ({written}/{args.num}, {rate:.1f} ex/s)")
                shard, sidx = [], sidx + 1
            if written + len(shard) >= args.num:
                break

    if shard:
        p = os.path.join(args.out, f"shard_{sidx:05d}.pt")
        torch.save(shard, p); written += len(shard)
        print(f"  wrote {p}  ({written}/{args.num})")
    print(f"done. {written} examples in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
