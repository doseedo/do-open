"""Build a latent soundfont: render every MIDI note (per soundfont, per
pitch, optionally per velocity) via FluidSynth and encode the resulting
wav to an Oobleck VAE latent. Saves one .pt per instrument containing
{pitch: [T, 64] latent tensor, sr, frames_per_note, ...}.

Usage:
  python -m latent_soundfont.build --vae /scratch/ACE-Step-1.5/checkpoints/vae \\
      --out /scratch/latent_soundfonts \\
      --pitch-range 21 108 --note-seconds 2.0 --velocity 100

Or call from a Python REPL:
  build_latent_soundfont("acoustic_piano", "/scratch/soundfonts/Piano.sf2", vae, ...)
"""
from __future__ import annotations
import argparse, os, subprocess, tempfile, time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch
import pretty_midi

SR = 48000
SAMPLES_PER_FRAME = 1920


def _render_single_note(sf_path: str, pitch: int, velocity: int,
                        duration_sec: float, is_drum: bool, gm_program: int,
                        out_wav: str) -> bool:
    """Render exactly one MIDI note via fluidsynth → WAV at 48k stereo.
    Returns True on success."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=gm_program, is_drum=is_drum)
    inst.notes.append(pretty_midi.Note(
        velocity=velocity, pitch=pitch, start=0.0, end=duration_sec,
    ))
    pm.instruments.append(inst)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
        midi_path = tf.name
    pm.write(midi_path)
    try:
        # Pad with 0.5 s tail to capture the release
        result = subprocess.run(
            ["fluidsynth", "-ni", "-T", "wav", "-g", "0.625", "-r", str(SR),
             "-F", out_wav, sf_path, midi_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0 or not os.path.exists(out_wav):
            return False
        data, _ = sf.read(out_wav, dtype="float32")
        return float(np.abs(data).max()) > 1e-4
    finally:
        try: os.unlink(midi_path)
        except OSError: pass


def _encode_wav_to_latent(wav_path: str, vae) -> torch.Tensor:
    """Read wav, ensure stereo @ 48k, encode via vae → [T, 64] cpu fp32."""
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim == 1:
        data = np.stack([data, data], axis=-1)
    if data.shape[1] == 1:
        data = np.concatenate([data, data], axis=1)
    if sr != SR:
        import librosa
        data = np.stack([
            librosa.resample(data[:, c], orig_sr=sr, target_sr=SR)
            for c in range(data.shape[1])
        ], axis=1)
    y = torch.from_numpy(data.T).float().unsqueeze(0).cuda().bfloat16()
    with torch.no_grad():
        L = vae.encode(y).latent_dist.sample().squeeze(0).transpose(0, 1).float().cpu()
    return L  # [T, 64]


def build_latent_soundfont(
    instrument: str,
    sf_path: str,
    vae,
    out_dir: str,
    *,
    pitch_lo: int = 21, pitch_hi: int = 108,
    velocity: int = 100,
    note_seconds: float = 2.0,
    is_drum: bool = False,
    gm_program: int = 0,
    drum_pitches: Optional[range] = None,
) -> Dict[int, torch.Tensor]:
    """Build a single instrument's latent soundfont. Returns a dict of
    {pitch: latent} and saves it to <out_dir>/<instrument>.pt"""
    os.makedirs(out_dir, exist_ok=True)
    pitches = list(drum_pitches) if drum_pitches is not None else list(range(pitch_lo, pitch_hi + 1))
    notes: Dict[int, torch.Tensor] = {}
    print(f"[{instrument}] sf={Path(sf_path).name} pitches={len(pitches)}")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        tmp_wav = os.path.join(td, "note.wav")
        for p in pitches:
            ok = _render_single_note(
                sf_path, pitch=p, velocity=velocity,
                duration_sec=note_seconds, is_drum=is_drum,
                gm_program=gm_program, out_wav=tmp_wav,
            )
            if not ok:
                continue
            L = _encode_wav_to_latent(tmp_wav, vae)
            notes[p] = L
    print(f"[{instrument}] rendered+encoded {len(notes)} notes in {time.perf_counter()-t0:.1f}s")

    out_path = os.path.join(out_dir, f"{instrument}.pt")
    torch.save({
        "instrument": instrument,
        "soundfont": str(sf_path),
        "sr": SR,
        "samples_per_frame": SAMPLES_PER_FRAME,
        "note_seconds": note_seconds,
        "velocity": velocity,
        "is_drum": is_drum,
        "gm_program": gm_program,
        "notes": notes,           # dict[pitch: int -> torch.Tensor [T, 64]]
    }, out_path)
    print(f"[{instrument}] → {out_path}")
    return notes


def build_all(
    instrument_soundfonts: Dict[str, str],
    gm_programs: Dict[str, int],
    drum_kit_programs: Dict[str, int],
    vae,
    out_dir: str,
    *,
    pitch_lo: int = 21, pitch_hi: int = 108,
    velocity: int = 100,
    note_seconds: float = 2.0,
):
    """Build latent soundfonts for every instrument in the dict.
    Drum subgroups (drum_kit/electronic/percussion) use the GM
    percussion key range 27..87 and is_drum=True."""
    default_sf = instrument_soundfonts.get("default")
    for instrument, sf_path in instrument_soundfonts.items():
        if not os.path.exists(sf_path):
            # Fall back to the default GM soundfont with the right
            # program — same fallback the deployed server uses via
            # get_soundfont(). FluidR3_GM has all the GM instruments.
            if default_sf and os.path.exists(default_sf):
                print(f"[{instrument}] sf not found at {sf_path}, falling back to default-GM")
                sf_path = default_sf
            else:
                print(f"[{instrument}] skip (sf not found: {sf_path})")
                continue
        ig = instrument.lower()
        is_drum = ig in drum_kit_programs or "drum" in ig or ig == "percussion"
        gm_program = drum_kit_programs.get(ig, gm_programs.get(ig, 0))
        if is_drum:
            build_latent_soundfont(
                instrument, sf_path, vae, out_dir,
                drum_pitches=range(27, 88),
                velocity=velocity, note_seconds=note_seconds,
                is_drum=True, gm_program=gm_program,
            )
        else:
            build_latent_soundfont(
                instrument, sf_path, vae, out_dir,
                pitch_lo=pitch_lo, pitch_hi=pitch_hi,
                velocity=velocity, note_seconds=note_seconds,
                is_drum=False, gm_program=gm_program,
            )


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="/scratch/ACE-Step-1.5/checkpoints/vae")
    ap.add_argument("--out", default="/scratch/latent_soundfonts")
    ap.add_argument("--instruments", nargs="*", default=None,
                    help="subset of instrument keys (default: all)")
    ap.add_argument("--pitch-range", nargs=2, type=int, default=[21, 108])
    ap.add_argument("--velocity", type=int, default=100)
    ap.add_argument("--note-seconds", type=float, default=2.0)
    args = ap.parse_args()

    from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck
    print(f"[init] loading vae from {args.vae}")
    vae = AutoencoderOobleck.from_pretrained(args.vae).to("cuda").to(torch.bfloat16).eval()

    # Extract the dicts from the deployed server source by exec'ing only
    # the relevant assignments in an isolated namespace — avoids the
    # full-module import (which would load 10GB of GPU models) and
    # handles f-strings that ast.literal_eval can't.
    import ast as _ast
    src = open("/scratch/stemphonic/stemphonic_server.py").read()
    tree = _ast.parse(src)
    wanted = {"SOUNDFONT_DIR", "DEFAULT_SOUNDFONT", "INSTRUMENT_SOUNDFONTS",
              "GM_PROGRAMS", "DRUM_KIT_PROGRAMS"}
    keep: list = []
    for node in tree.body:
        if isinstance(node, _ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, _ast.Name) and t.id in wanted:
                keep.append(node)
    mod = _ast.Module(body=keep, type_ignores=[])
    ns: dict = {}
    exec(compile(mod, "<server-constants>", "exec"), ns, ns)
    instrument_sfs = ns["INSTRUMENT_SOUNDFONTS"]
    gm_progs = ns["GM_PROGRAMS"]
    drum_kits = ns["DRUM_KIT_PROGRAMS"]

    if args.instruments:
        instrument_sfs = {k: v for k, v in instrument_sfs.items() if k in args.instruments}

    build_all(
        instrument_sfs, gm_progs, drum_kits, vae, args.out,
        pitch_lo=args.pitch_range[0], pitch_hi=args.pitch_range[1],
        velocity=args.velocity, note_seconds=args.note_seconds,
    )
