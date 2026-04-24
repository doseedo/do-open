"""Universal drum-stem meter-change reference script.

ONE code path for all songs. Runs:
  - tears (compare-2667846b)  → must match tears_exact/drums_meter.wav
  - fortunate (fortunate_v4)  → must match fortunate_v4_drums/drums_meter.wav

Pipeline (latent-domain, copied verbatim from compare-2667846b):
  - Each drum substem (kick/snare/tom/hh/ride/crash) is encoded once via VAE
  - Routing dict (matches production after revert):
      kick / snare / tom            → latent_change_meter_instrumental
                                      (linear-interp 4 + 1.5 + 1.5 splice)
      hh / hat / hihat              → latent_pluck_and_place
                                      (preallocated L_bar = zeros,
                                       sample-accurate cut_sample = s*slot_samples,
                                       14-slot editor.edit per bar)
      ride / crash / cymbal         → latent_change_meter_instrumental (mixed)
  - Substems are decoded back to wav and summed.

The code is ENTIRELY identical between songs; only the input paths differ.
Logging emits the routing decision and edit count per substem so you can
verify the same path is taken in both runs.
"""
import sys, os, glob, time, hashlib
import numpy as np, soundfile as sf, torch, librosa
sys.path.insert(0, "/home/arlo/do2")
from latent_editor.infer import LatentEditorRuntime
from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck

# ── Constants (match compare-2667846b) ───────────────────────────────
SR = 48000
SAMPLES_PER_FRAME = 1920
SRC_N, SRC_DEN = 4, 4
TGT_N, TGT_DEN = 7, 8
# N_TGT_SLOTS is now derived per-substem from detected subdivision:
#   8th hh  → 7 slots (target eighths) on 8-cell src grid
#   16th hh → 14 slots (target 16ths)  on 16-cell src grid
def slots_for_subdiv(subdiv):
    cells_per_eighth = 2 if subdiv == "16th" else 1
    return TGT_N * cells_per_eighth if TGT_DEN == 8 else TGT_N * 2 * cells_per_eighth
ACCENT_SUBSTEMS  = {"kick", "snare", "tom", "toms", "bass"}
FORCE_QUANTIZE   = {"hh", "hat", "hihat"}
MIXED_SUBSTEMS   = {"ride", "crash", "cymbal"}

# ── Models (CPU encode + GPU decode + GPU editor) ────────────────────
print("[init] loading vae (gpu bf16) and editor (gpu)…")
vae_gpu = AutoencoderOobleck.from_pretrained("/scratch/ACE-Step-1.5/checkpoints/vae").to("cuda").to(torch.bfloat16).eval()
editor  = LatentEditorRuntime("/scratch/latent_editor_ckpts/editor_final.pt", device="cuda")

# Load the canonical silence latent (trained "silence" point in VAE
# space — NOT zeros, which would decode to garbage). We pull a single
# frame and tile it wherever we need silence in a latent buffer.
_sl_raw = torch.load("/scratch/ACE-Step-1.5/checkpoints/acestep-v15-sft/silence_latent.pt",
                     weights_only=True).transpose(1, 2).float()  # [1, T, 64]
SILENCE_FRAME = _sl_raw[0, 0:1, :].clone()                        # [1, 64]
print(f"[init] silence frame shape={tuple(SILENCE_FRAME.shape)} mean={SILENCE_FRAME.mean().item():.4f}")

def silence_latent(n_frames):
    """Return [n_frames, 64] of true latent-space silence (tiled from
    the trained silence frame). NEVER use torch.zeros in latent buffers
    — zeros decode to loud garbage because 0 is not silence in the VAE's
    learned distribution."""
    return SILENCE_FRAME.expand(n_frames, -1).clone()

# ── Helpers (verbatim from /tmp/test_meter_compare_2667846b.py) ─────

def encode_to_lat(audio_2d, src_sr):
    """Resample to 48k via librosa, encode via vae_cpu (matches the
    original test_meter_compare encode path bit-for-bit modulo
    .latent_dist.sample() noise)."""
    a48 = np.stack([
        librosa.resample(audio_2d[:, c].astype(np.float32), orig_sr=src_sr, target_sr=SR)
        for c in range(audio_2d.shape[1])
    ], axis=1)
    if a48.shape[1] == 1:
        a48 = np.concatenate([a48, a48], axis=1)
    # GPU chunk encode (small headroom). 128 frames @ 1920 samples ≈ 5.1s.
    chunk_samples = 128 * SAMPLES_PER_FRAME
    pieces = []
    total = a48.shape[0]
    i = 0
    while i < total:
        end = min(total, i + chunk_samples)
        seg = a48[i:end]
        y_t = torch.from_numpy(seg.T).float().unsqueeze(0).cuda().bfloat16()
        with torch.no_grad():
            L = vae_gpu.encode(y_t).latent_dist.sample().squeeze(0).transpose(0, 1).float().cpu()
        pieces.append(L)
        torch.cuda.empty_cache()
        i = end
    L_full = torch.cat(pieces, dim=0)
    return L_full, a48

def decode_lat(L):
    """Chunk-decode at 64-frame windows so a single forward pass fits
    in the GPU headroom."""
    chunk_frames = 64
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
    return (s * TGT_N) // (SRC_N * 2) if TGT_DEN == 8 else (s * TGT_N) // SRC_N

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

def _region_has_onset(audio_48, lo, hi, bar_peak):
    """Return True if the audio window has an onset above 25% of the
    bar's peak. Used to gate silent trailing cells (drop-not-keep) so
    bleed/decay from the previous beat doesn't sneak into a region that
    should be silent."""
    if hi <= lo or bar_peak < 1e-6:
        return False
    seg = audio_48[lo:hi]
    mono = seg.mean(axis=1).astype(np.float32) if seg.ndim == 2 else seg.astype(np.float32)
    env = np.abs(mono)
    sm = max(1, int(0.005 * SR))
    env = np.convolve(env, np.ones(sm)/sm, mode='same')
    return float(env.max()) >= bar_peak * 0.25


def latent_drum_accent_cut(L, audio_48, bs_48):
    """4 + 1.5 + 1.5 musical grouping via CELL-CUT (no time-stretch),
    with editor.edit smoothing at every region boundary.

    Per bar: preallocate L_out = zeros(target_bar_frames). Paste each
    of the three regions (first 4 eighths, first 1.5 of beat 3, first
    1.5 of beat 4) into L_out through editor.edit at sample-accurate
    cut positions. Regions whose source audio is below the onset
    threshold are SKIPPED (L_out stays zero there) — this prevents
    kick-decay bleed from producing a phantom hit when beat 4 is
    actually empty.

    Same target beat positions as the stretch path (snare beat 4 →
    tgt eighth 5.5) so no drift vs other stems. Frame-accurate source
    timing, editor-smoothed joins, no waveform interpolation."""
    out_chunks = []
    n_edits = 0
    n_bars = len(bs_48) - 1
    for k in range(n_bars):
        bar_lo_s = bs_48[k]
        bar_hi_s = min(bs_48[k+1], audio_48.shape[0])
        if bar_hi_s - bar_lo_s < SAMPLES_PER_FRAME * 8:
            continue
        f_lo = int(round(bar_lo_s / SAMPLES_PER_FRAME))
        f_hi = min(int(round(bar_hi_s / SAMPLES_PER_FRAME)), L.shape[0])
        if f_hi - f_lo < 8:
            continue
        L_bar = L[f_lo:f_hi]
        T = L_bar.shape[0]
        e = T / 8.0
        f4  = int(round(4 * e))
        f55 = int(round(5.5 * e))
        f6  = int(round(6 * e))
        f75 = int(round(7.5 * e))

        # Bar peak for onset gating (on the real audio)
        bar_audio = audio_48[bar_lo_s:bar_hi_s]
        mono_bar = bar_audio.mean(axis=1).astype(np.float32) if bar_audio.ndim == 2 else bar_audio.astype(np.float32)
        bar_peak = float(np.abs(mono_bar).max())

        # Audio sample ranges for each of the 3 regions (same proportions
        # as the latent-frame slices) so we can onset-gate them.
        src_bar_samples = bar_hi_s - bar_lo_s
        se = src_bar_samples / 8.0
        seg_first = (bar_lo_s,             bar_lo_s + int(round(4.0 * se)))
        seg_mid   = (bar_lo_s + int(round(4.0 * se)), bar_lo_s + int(round(5.5 * se)))
        seg_last  = (bar_lo_s + int(round(6.0 * se)), bar_lo_s + int(round(7.5 * se)))

        first = L_bar[:f4]
        mid   = L_bar[f4:f55]  if _region_has_onset(audio_48, *seg_mid,  bar_peak) else None
        last  = L_bar[f6:f75]  if _region_has_onset(audio_48, *seg_last, bar_peak) else None

        # Target layout: [first(4e) | mid(1.5e) | last(1.5e)] = 7 eighths
        src_bar_samples = bs_48[k+1] - bs_48[k]
        tgt_bar_samples = canonical_tgt_bar_samples(src_bar_samples)
        target_bar_frames = max(1, int(round(tgt_bar_samples / SAMPLES_PER_FRAME)))
        te = tgt_bar_samples / 7.0  # target frames per target eighth (samples)
        L_out = silence_latent(target_bar_frames)

        # Region 1: paste first 4 eighths at tgt offset 0
        paste_n = min(first.shape[0], target_bar_frames)
        L_out[:paste_n] = first[:paste_n]
        # (no editor call for region 1 — nothing to join yet)

        def paste_and_smooth(L_out, region, tgt_sample_offset):
            nonlocal n_edits
            if region is None or region.shape[0] == 0:
                return L_out
            cut_frame = tgt_sample_offset // SAMPLES_PER_FRAME
            if cut_frame >= L_out.shape[0]:
                return L_out
            L_b = L_out.clone()
            paste_end = min(L_b.shape[0], cut_frame + region.shape[0])
            L_b[cut_frame:paste_end] = region[:paste_end - cut_frame]
            L_out = editor.edit(L_out, L_b, tgt_sample_offset)
            n_edits += 1
            return L_out

        # Region 2: paste first 1.5 eighths of beat 3 starting at tgt eighth 4
        L_out = paste_and_smooth(L_out, mid,  int(round(4.0 * te)))
        # Region 3: paste first 1.5 eighths of beat 4 starting at tgt eighth 5.5
        L_out = paste_and_smooth(L_out, last, int(round(5.5 * te)))

        out_chunks.append(L_out)
    return (torch.cat(out_chunks, dim=0) if out_chunks else L), n_edits

def detect_hh_subdivision(audio_48, bs_48):
    """Vote 8th vs 16th across all bars. Method: per bar, take a smoothed
    envelope and count distinct onsets above an adaptive threshold. The
    expected onset count for 4/4 with steady hh is 8 (eighth) or 16
    (sixteenth). Pick whichever is closer per bar, then majority vote.

    The earlier "≥12 → 16th" cutoff was too lax: drumsep bleed from
    kick/snare added spurious peaks to 8th-pattern hh stems, pushing
    them past 12 and forcing the 14-slot 16th grid where 7 was correct."""
    votes = {"eighth": 0, "16th": 0}
    for k in range(len(bs_48) - 1):
        lo = bs_48[k]
        hi = min(bs_48[k+1], audio_48.shape[0])
        if hi - lo < SAMPLES_PER_FRAME * 2: continue
        bar = audio_48[lo:hi]
        mono = bar.mean(axis=1).astype(np.float32) if bar.ndim == 2 else bar.astype(np.float32)
        env = np.abs(mono)
        sm = max(1, int(0.005 * SR))
        env = np.convolve(env, np.ones(sm)/sm, mode='same')
        pk = float(env.max())
        if pk < 1e-4: continue
        # min_gap = 1/20 of bar so we can detect up to ~20 onsets max
        min_gap = max(1, len(env) // 20)
        # Count at LOW threshold to catch ghost notes between accents.
        thr = pk * 0.15
        n = 0; i = 0
        while i < len(env):
            if env[i] >= thr:
                n += 1; i += min_gap
            else:
                i += 1
        # Snap to nearest expected count (8 vs 16). Ties go to 8th
        # because over-detection from bleed is the common failure mode.
        votes["16th" if (n - 8) > (16 - n) else "eighth"] += 1
    if votes["16th"] == 0 and votes["eighth"] == 0:
        return "eighth"  # safer default
    print(f"  hh subdivision votes: {votes}")
    return "16th" if votes["16th"] > votes["eighth"] else "eighth"


def compute_src_positions_for_bar(audio_48, bar_lo, bar_hi, n_cells):
    src_bar = bar_hi - bar_lo
    bar_audio = audio_48[bar_lo:bar_hi]
    mono = bar_audio.mean(axis=1).astype(np.float32) if bar_audio.ndim == 2 else bar_audio.astype(np.float32)
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

def latent_hh_literal_cut(L, bs_48):
    """8th-grid hh path: per bar, keep the first 7 source eighths and
    drop the last. Per-bar output length is LOCKED to canonical_tgt_bar
    frames (same math the accent/splice paths use) so hh stays
    phase-aligned with kick/snare/tom across the whole song — a drift-
    free proportional cut would diverge by ±1 frame per bar and the
    accumulated slip becomes audibly off-beat by the last bar."""
    out_chunks = []
    n_bars = len(bs_48) - 1
    for k in range(n_bars):
        f_lo = int(round(bs_48[k] / SAMPLES_PER_FRAME))
        f_hi = min(int(round(bs_48[k+1] / SAMPLES_PER_FRAME)), L.shape[0])
        if f_hi - f_lo < 8:
            continue
        L_bar = L[f_lo:f_hi]
        # Canonical target bar length (matches accent path frame-for-frame)
        src_bar_samples = bs_48[k+1] - bs_48[k]
        tgt_bar_samples = canonical_tgt_bar_samples(src_bar_samples)
        target_bar_frames = max(1, int(round(tgt_bar_samples / SAMPLES_PER_FRAME)))
        if L_bar.shape[0] >= target_bar_frames:
            out_chunks.append(L_bar[:target_bar_frames])
        else:
            # Rare off-by-one: pad with silence latent so length locks.
            pad = target_bar_frames - L_bar.shape[0]
            out_chunks.append(torch.cat([L_bar, silence_latent(pad)], dim=0))
    return (torch.cat(out_chunks, dim=0) if out_chunks else L), 0


def latent_pluck_and_place(L, audio_48, bs_48, subdiv):
    """compare-2667846b version: preallocated L_bar = zeros, per-slot
    chunk = L[src:src+slot_frames+4], L_b = L_bar.clone() with paste,
    cut_sample = s_idx * slot_samples (sample-accurate).

    Slot count = `slots_for_subdiv(subdiv)` so 8th-note hh gets 7 slots
    (not 14), preventing the 14-slot path from doubling 8th hits into
    16ths on songs whose hh is already on the 8th grid."""
    n_tgt_slots = slots_for_subdiv(subdiv)
    n_src_cells_grid = SRC_N * 4 if subdiv == "16th" else SRC_N * 2  # 16 or 8
    chunks = []
    n_edits = 0
    n_bars = len(bs_48) - 1
    for k in range(n_bars):
        bar_lo = bs_48[k]
        bar_hi = min(bs_48[k+1], audio_48.shape[0])
        if bar_hi - bar_lo < SAMPLES_PER_FRAME * 2: continue
        src_bar_samples = bar_hi - bar_lo
        tgt_bar_samples = canonical_tgt_bar_samples(src_bar_samples)
        target_bar_frames = max(1, int(round(tgt_bar_samples / SAMPLES_PER_FRAME)))
        slot_samples = tgt_bar_samples // n_tgt_slots
        slot_frames  = max(1, int(round(slot_samples / SAMPLES_PER_FRAME)))
        snapped, n_src_cells = compute_src_positions_for_bar(audio_48, bar_lo, bar_hi, n_src_cells_grid)
        L_bar = silence_latent(target_bar_frames)
        for s_idx in range(n_tgt_slots):
            j = min(int(round(s_idx * n_src_cells / n_tgt_slots)), n_src_cells - 1)
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
            n_edits += 1
        chunks.append(L_bar)
    return (torch.cat(chunks, dim=0) if chunks else L), n_edits

# ── Bar starts via beat_this ─────────────────────────────────────────
import beat_this.inference as bti
f2b = bti.File2Beats(checkpoint_path="final0", dbn=True)

def detect_bs_48(src_path):
    beats, downbeats = f2b(src_path)
    bpm = 60.0 / float(np.median(np.diff(beats)))
    y_full, sr_in = sf.read(src_path, always_2d=True)
    offset = int(round(float(downbeats[0]) * sr_in))
    bar_starts_src = [int(round(float(t) * sr_in)) - offset for t in downbeats]
    bar_starts_src.append(len(y_full) - offset)
    bs_48 = [int(round(b * SR / sr_in)) for b in bar_starts_src]
    return bs_48, bpm

# ── Universal process function ───────────────────────────────────────
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 16)
            if not b: break
            h.update(b)
    return h.hexdigest()[:12]

def process(label, src, sub_dir, out_dir, reference=None):
    src_path = src
    reference_wav = reference
    print(f"\n========== {label} ==========")
    print(f"  src       : {src_path}")
    print(f"  drum_stems: {sub_dir}")
    bs_48, bpm = detect_bs_48(src_path)
    n_bars = len(bs_48) - 1
    print(f"  bpm={bpm:.2f}  bars={n_bars}")

    os.makedirs(out_dir, exist_ok=True)

    # Detect hh subdivision ONCE for the whole stem so we don't flip
    # 8th↔16th grids per bar. Vote across bars.
    hh_path_list = sorted(glob.glob(os.path.join(sub_dir, "hh.wav")))
    hh_subdiv = "16th"
    if hh_path_list:
        hh_audio_raw, hh_sr = sf.read(hh_path_list[0], always_2d=True)
        hh_a48 = np.stack([
            librosa.resample(hh_audio_raw[:, c].astype(np.float32), orig_sr=hh_sr, target_sr=SR)
            for c in range(hh_audio_raw.shape[1])
        ], axis=1)
        hh_subdiv = detect_hh_subdivision(hh_a48, bs_48)
    print(f"  hh_subdivision_lock = {hh_subdiv}  → pluck slots = {slots_for_subdiv(hh_subdiv)}")

    processed = {}
    for p in sorted(glob.glob(os.path.join(sub_dir, "*.wav"))):
        name = os.path.splitext(os.path.basename(p))[0]
        a, sr = sf.read(p, always_2d=True)
        L, a48 = encode_to_lat(a, sr)
        t0 = time.perf_counter()
        # Routing — IDENTICAL for both songs
        if name in ACCENT_SUBSTEMS:
            L_out, n_edits = latent_drum_accent_cut(L, a48, bs_48)
            path = "splice"
        elif name in FORCE_QUANTIZE:
            if hh_subdiv == "eighth":
                # 8th-grid hh: literal per-bar cell cut. Drop the last
                # eighth, keep the first 7. No editor calls needed —
                # the pattern is already on an 8th grid that fits 7/8.
                L_out, n_edits = latent_hh_literal_cut(L, bs_48)
                path = "cut/eighth"
            else:
                L_out, n_edits = latent_pluck_and_place(L, a48, bs_48, hh_subdiv)
                path = f"pluck/{hh_subdiv}"
        elif name in MIXED_SUBSTEMS:
            L_out, n_edits = latent_drum_accent_cut(L, a48, bs_48)
            path = "splice (mixed)"
        else:
            L_out, n_edits = latent_drum_accent_cut(L, a48, bs_48)
            path = "splice (default)"
        audio_out = decode_lat(L_out)
        processed[name] = audio_out
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"    [{label}] {name:7s}: route={path:18s} L_in={tuple(L.shape)} L_out={tuple(L_out.shape)} edits={n_edits:4d} time={elapsed:6.0f}ms")

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
    h = md5(out)
    print(f"  → {out}  md5={h}  shape={mix.shape}")
    if reference_wav and os.path.exists(reference_wav):
        rh = md5(reference_wav)
        print(f"  reference: {reference_wav}  md5={rh}")
        # Sonic similarity check via RMS over time alignment
        ref, _ = sf.read(reference_wav, always_2d=True)
        n = min(ref.shape[0], mix.shape[0])
        ref_chunk = ref[:n].astype(np.float32)
        mix_chunk = mix[:n].astype(np.float32)
        diff_rms = float(np.sqrt(np.mean((ref_chunk - mix_chunk) ** 2)))
        ref_rms = float(np.sqrt(np.mean(ref_chunk ** 2)))
        rel = diff_rms / max(ref_rms, 1e-9)
        print(f"  vs reference: diff_rms={diff_rms:.5f}  ref_rms={ref_rms:.5f}  rel_err={rel:.3f}")
    return out

# ── Run both songs through the SAME code ─────────────────────────────
SONGS = [
    {
        "label":     "tears",
        "src":       "/home/arlo/do2/time-sig-editor/tearsforfearseverybodywantstoruletheworldofficia.mp3",
        "sub_dir":   "/scratch/stemphonic_outputs/compare-2667846b/drum_stems",
        "out_dir":   "/scratch/stemphonic_outputs/universal_tears",
        "reference": "/scratch/stemphonic_outputs/tears_exact/drums_meter.wav",
    },
    {
        "label":     "fortunate",
        "src":       "/home/arlo/do2/time-sig-editor/fortunate.wav",
        "sub_dir":   "/scratch/stemphonic_outputs/fortunate_v2_drums/drum_stems",
        "out_dir":   "/scratch/stemphonic_outputs/universal_fortunate",
        "reference": "/scratch/stemphonic_outputs/fortunate_v4_drums/drums_meter.wav",
    },
]

for cfg in SONGS:
    process(**cfg)
