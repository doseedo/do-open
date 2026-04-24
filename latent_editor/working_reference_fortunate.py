"""CPU-only re-run of drum substem meter change using cached drumsep
output, then remix with existing non-drum repainted stems."""
import sys, os, glob, importlib.util, numpy as np, soundfile as sf, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/arlo/do2/time-sig-editor")

# Force-load updated tse_server (NOT from cache)
spec = importlib.util.spec_from_file_location("tse_server", "/home/arlo/do2/time-sig-editor/server.py")
tse = importlib.util.module_from_spec(spec)
sys.modules["tse_server"] = tse
sys.modules["server"] = tse
spec.loader.exec_module(tse)

# Bar starts from full mix
import beat_this.inference as bti
f2b = bti.File2Beats(checkpoint_path="final0", dbn=True)
src_full = "/home/arlo/do2/time-sig-editor/fortunate.wav"
beats, downbeats = f2b(src_full)
bpm = 60.0 / float(np.median(np.diff(beats)))
y, sr = sf.read(src_full, always_2d=True)
offset_samples = int(round(float(downbeats[0]) * sr))
bar_starts = [int(round(float(t) * sr)) - offset_samples for t in downbeats]
bar_starts.append(len(y) - offset_samples)
print(f"bpm={bpm:.2f}  bars={len(bar_starts)-1}")

sub_dir = "/scratch/stemphonic_outputs/fortunate_v2_drums/drum_stems"
out_dir = "/scratch/stemphonic_outputs/fortunate_v4_drums"
os.makedirs(out_dir, exist_ok=True)
sub_outs = {}
for sub_path in sorted(glob.glob(os.path.join(sub_dir, "*.wav"))):
    name = os.path.splitext(os.path.basename(sub_path))[0]
    wav_np, sr_s = sf.read(sub_path, always_2d=True)
    # Already trimmed by run_meter_change → don't trim again.
    wav_t = torch.from_numpy(wav_np.T.astype(np.float32))
    wav_t._bar_starts = bar_starts
    print(f"\n=== {name} ===")
    out = tse._process_stem_pattern_aware(
        wav_t, sr_s, bpm, 4, 4, 7, 8,
        stem_name=f"drum/{name}", job_id="v4",
    )
    op = os.path.join(out_dir, f"{name}.wav")
    sf.write(op, out.numpy().T, sr_s)
    sub_outs[name] = op

# Sum substems → drums_meter
mix = None
for p in sub_outs.values():
    a, sr_a = sf.read(p, always_2d=True)
    if mix is None: mix = a.astype(np.float32)
    else:
        n = min(mix.shape[0], a.shape[0])
        mix = mix[:n] + a[:n].astype(np.float32)
drums_out = os.path.join(out_dir, "drums_meter.wav")
sf.write(drums_out, mix, sr_a)
print(f"\ndrums → {drums_out}")

# Remix with v2 non-drum stems
others = [
    "/scratch/stemphonic_outputs/fortunate_v2_bass/bass_meter.wav",
    "/scratch/stemphonic_outputs/fortunate_v2_other/other_meter.wav",
    "/scratch/stemphonic_outputs/fortunate_v2_vocals/vocals_meter.wav",
    "/scratch/stemphonic_outputs/fortunate_v2_guitar/guitar_meter.wav",
    "/scratch/stemphonic_outputs/fortunate_v2_piano/piano_meter.wav",
    drums_out,
]
mix = None
for fp in others:
    a, sr_a = sf.read(fp, always_2d=True)
    if mix is None: mix = a.astype(np.float32)
    else:
        n = min(mix.shape[0], a.shape[0])
        mix = mix[:n] + a[:n].astype(np.float32)
peak = float(np.max(np.abs(mix)))
if peak > 0.99: mix *= 0.99/peak
final = "/scratch/stemphonic_outputs/fortunate_v2/fortunate_7_8.wav"
sf.write(final, mix, sr_a)
print("FINAL →", final)
