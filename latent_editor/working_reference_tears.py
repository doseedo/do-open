"""Run the EXACT compare-2667846b latent pipeline on tears using the
SAME drum_stems that compare-2667846b used. Bit-for-bit reproducible
modulo vae.encode .sample() noise."""
import sys, os, glob, time, numpy as np, soundfile as sf, torch, librosa
sys.path.insert(0, "/home/arlo/do2")
from latent_editor.infer import LatentEditorRuntime
from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck

print("loading vae (cpu encode, gpu decode) + editor (gpu)…")
vae_cpu = AutoencoderOobleck.from_pretrained("/scratch/ACE-Step-1.5/checkpoints/vae").to("cpu").eval()
vae_gpu = AutoencoderOobleck.from_pretrained("/scratch/ACE-Step-1.5/checkpoints/vae").to("cuda").to(torch.bfloat16).eval()
editor = LatentEditorRuntime("/scratch/latent_editor_ckpts/editor_final.pt", device="cuda")

SRC = "/home/arlo/do2/time-sig-editor/tearsforfearseverybodywantstoruletheworldofficia.mp3"
SR = 48000
SAMPLES_PER_FRAME = 1920
SRC_N, SRC_DEN = 4, 4
TGT_N, TGT_DEN = 7, 8
N_TGT_SLOTS = 14

# ── Bar starts (use original test_meter_compare path: ssrv._detect_bar_starts) ──
sys.path.insert(0, "/scratch/stemphonic")
import importlib.util as _iu
_sp = _iu.spec_from_file_location("ssrv_min", "/scratch/stemphonic/stemphonic_server.py")
# We don't init handler — only call _detect_bar_starts
import beat_this.inference as bti
f2b = bti.File2Beats(checkpoint_path="final0", dbn=True)
beats, downbeats = f2b(SRC)
y_full, sr_in = sf.read(SRC, always_2d=True)
offset = int(round(float(downbeats[0]) * sr_in))
bar_starts_src = [int(round(float(t) * sr_in)) - offset for t in downbeats]
bar_starts_src.append(len(y_full) - offset)
bs_48 = [int(round(b * SR / sr_in)) for b in bar_starts_src]
n_bars = len(bs_48) - 1
print(f"bars={n_bars}")

ACCENT  = {"kick", "snare", "tom"}
QUANT   = {"hh"}
MIXED   = {"ride", "crash"}

def encode_to_lat(audio_44k_2d, src_sr):
    a48 = np.stack([
        librosa.resample(audio_44k_2d[:, c].astype(np.float32), orig_sr=src_sr, target_sr=SR)
        for c in range(audio_44k_2d.shape[1])
    ], axis=1)
    if a48.shape[1] == 1:
        a48 = np.concatenate([a48, a48], axis=1)
    y_t = torch.from_numpy(a48.T).float().unsqueeze(0)
    with torch.no_grad():
        L = vae_cpu.encode(y_t).latent_dist.sample()
    return L.squeeze(0).transpose(0, 1).float().cpu(), a48

def decode_lat(L):
    """Chunk-decode in latent windows so a single forward pass fits in
    the small GPU headroom we have."""
    chunk_frames = 64  # ~2.5s at 25fps
    overlap = 0
    pieces = []
    T = L.shape[0]
    i = 0
    while i < T:
        end = min(T, i + chunk_frames)
        seg = L[i:end]
        with torch.no_grad():
            z = seg.transpose(0, 1).unsqueeze(0).cuda().bfloat16()
            a = vae_gpu.decode(z).sample.squeeze(0).cpu().float().numpy()
        pieces.append(a)
        torch.cuda.empty_cache()
        i = end
    return np.concatenate(pieces, axis=1).T

def canonical_tgt_bar_samples(s):
    return (s * TGT_N) // (SRC_N * 2)

def _interp_time(L_seg, target_frames):
    if L_seg.shape[0] == target_frames or L_seg.shape[0] == 0:
        return L_seg
    x = L_seg.transpose(0, 1).unsqueeze(0)
    y = torch.nn.functional.interpolate(x, size=target_frames, mode="linear", align_corners=False)
    return y.squeeze(0).transpose(0, 1)

def latent_remap_bar_4_4_to_7_8(L_bar, target_bar_frames):
    T = L_bar.shape[0]
    if T < 8:
        return _interp_time(L_bar, target_bar_frames)
    e = T / 8.0
    f4 = int(round(4 * e)); f6 = int(round(6 * e))
    first = L_bar[:f4]; mid_src = L_bar[f4:f6]; last_src = L_bar[f6:]
    f1 = int(round(target_bar_frames * 4.0 / 7.0))
    f2 = int(round(target_bar_frames * 5.5 / 7.0)) - f1
    f3 = target_bar_frames - f1 - f2
    out = torch.cat([_interp_time(first, max(1, f1)),
                     _interp_time(mid_src, max(1, f2)),
                     _interp_time(last_src, max(1, f3))], dim=0)
    if out.shape[0] != target_bar_frames:
        out = _interp_time(out, target_bar_frames)
    return out

def latent_change_meter_instrumental(L):
    chunks = []
    for k in range(n_bars):
        f_lo = int(round(bs_48[k] / SAMPLES_PER_FRAME))
        f_hi = min(int(round(bs_48[k+1] / SAMPLES_PER_FRAME)), L.shape[0])
        if f_hi - f_lo < 4: continue
        bar_lat = L[f_lo:f_hi]
        src_bar = bs_48[k+1] - bs_48[k]
        target_bar_frames = max(1, int(round(canonical_tgt_bar_samples(src_bar) / SAMPLES_PER_FRAME)))
        chunks.append(latent_remap_bar_4_4_to_7_8(bar_lat, target_bar_frames))
    return torch.cat(chunks, dim=0) if chunks else L

def compute_src_positions_for_bar(audio_48, bar_lo, bar_hi):
    src_bar = bar_hi - bar_lo
    n_cells = SRC_N * 4
    bar = audio_48[bar_lo:bar_hi]
    mono = bar.mean(axis=1).astype(np.float32) if bar.ndim == 2 else bar.astype(np.float32)
    env = np.abs(mono)
    sm = max(1, int(0.002 * SR))
    if sm > 1: env = np.convolve(env, np.ones(sm)/sm, mode='same')
    if float(env.max()) < 1e-4:
        return [bar_lo + int(round(s * src_bar / n_cells)) for s in range(n_cells)], n_cells
    half = max(1, src_bar // (n_cells * 2))
    out = []
    for s in range(n_cells):
        g = int(round(s * src_bar / n_cells))
        a = max(0, g - half); b = min(src_bar, g + half)
        if b <= a: out.append(bar_lo + g)
        else: out.append(bar_lo + a + int(np.argmax(env[a:b])))
    return out, n_cells

def latent_pluck_and_place(L, audio_48):
    chunks = []
    for k in range(n_bars):
        bar_lo = bs_48[k]
        bar_hi = min(bs_48[k+1], audio_48.shape[0])
        if bar_hi - bar_lo < SAMPLES_PER_FRAME * 2: continue
        src_bar_samples = bar_hi - bar_lo
        tgt_bar_samples = canonical_tgt_bar_samples(src_bar_samples)
        target_bar_frames = max(1, int(round(tgt_bar_samples / SAMPLES_PER_FRAME)))
        slot_samples = tgt_bar_samples // N_TGT_SLOTS
        slot_frames  = max(1, int(round(slot_samples / SAMPLES_PER_FRAME)))
        snapped, n_src_cells = compute_src_positions_for_bar(audio_48, bar_lo, bar_hi)
        L_bar = torch.zeros(target_bar_frames, 64, dtype=torch.float32)
        for s_idx in range(N_TGT_SLOTS):
            j = min(int(round(s_idx * n_src_cells / N_TGT_SLOTS)), n_src_cells - 1)
            src_pos = snapped[j]
            src_f_start = max(0, src_pos // SAMPLES_PER_FRAME)
            src_f_end   = min(L.shape[0], src_f_start + slot_frames + 4)
            chunk = L[src_f_start:src_f_end]
            if chunk.shape[0] == 0: continue
            cut_sample = s_idx * slot_samples
            cut_frame = cut_sample // SAMPLES_PER_FRAME
            if cut_frame >= L_bar.shape[0]: continue
            L_b = L_bar.clone()
            paste_end = min(L_b.shape[0], cut_frame + chunk.shape[0])
            L_b[cut_frame:paste_end] = chunk[:paste_end - cut_frame]
            L_bar = editor.edit(L_bar, L_b, cut_sample)
        chunks.append(L_bar)
    return torch.cat(chunks, dim=0) if chunks else L

# ── Use the SAME drum_stems compare-2667846b used ──
sub_dir = "/scratch/stemphonic_outputs/compare-2667846b/drum_stems"
out_dir = "/scratch/stemphonic_outputs/tears_exact"
os.makedirs(out_dir, exist_ok=True)
processed = {}
for p in sorted(glob.glob(os.path.join(sub_dir, "*.wav"))):
    name = os.path.splitext(os.path.basename(p))[0]
    a, sr = sf.read(p, always_2d=True)
    L, a48 = encode_to_lat(a, sr)
    t0 = time.perf_counter()
    if name in ACCENT:
        L_out = latent_change_meter_instrumental(L); path = "splice"
    elif name in QUANT:
        L_out = latent_pluck_and_place(L, a48); path = "pluck"
    elif name in MIXED:
        L_out = latent_change_meter_instrumental(L); path = "splice (mixed)"
    else:
        L_out = latent_change_meter_instrumental(L); path = "splice (default)"
    audio_out = decode_lat(L_out)
    processed[name] = audio_out
    print(f"  {name:7s}: {path:20s} {(time.perf_counter()-t0)*1000:.0f}ms shape={audio_out.shape}")

ml = max(s.shape[0] for s in processed.values())
mix = np.zeros((ml, 2), dtype=np.float32)
for name, s in processed.items():
    n = s.shape[0]
    if s.shape[1] == 1: s = np.concatenate([s, s], axis=1)
    mix[:n] += s[:n]
peak = np.abs(mix).max()
if peak > 0.95: mix *= 0.95 / peak
out = os.path.join(out_dir, "drums_meter.wav")
sf.write(out, mix, SR)
print(f"\nFINAL → {out}")
