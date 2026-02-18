import os
import re
import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.onsets import RNNOnsetProcessor, OnsetPeakPickingProcessor, CNNOnsetProcessor
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter
from collections import deque
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

# === CONFIG ===
AUDIO_PATH_LIST = Path("/home/arlo/Data/all_audio_paths5.txt")
OUTPUT_DIR = Path("/mnt/msdd/audio_madmom_beats")
OUTPUT_JSON = Path("/mnt/msdd/drum_groups.json")
EVENT_JSON = Path("/mnt/msdd/drum_events")  # one JSON per group with merged events
PROTOOLS_ROOTS = [
    Path("/home/arlo/gcs-bucket/protools"),
    Path("/home/arlo/gcs-bucket/protoolsA")
]
SKIP_COUNT = 0
SAMPLE_RATE = 44100  # madmom prefers 44.1kHz
DURATION_TOLERANCE = 0.5  # seconds
MIN_DURATION = 20.0  # seconds - only process files longer than this
SILENCE_THRESHOLD = 0.001  # RMS threshold to detect if file is mostly silent
MAX_WORKERS = min(12, cpu_count())  # Use up to 12 CPU cores

# === EVENT CLASSIFICATION CONFIG ===
DRUM_CLASSES = ["kick", "snare", "hihat", "tom", "ride", "crash"]  # extend if you want
FRAME_HOP = 256  # for feature frames @ 44.1k (~5.8 ms)
MIN_CONF = 0.35   # tune
OH_COINC_MIN = 0.15
CLASS_BANDS = {
    "kick_low":  (50, 110),
    "kick_click":(2000, 4000),
    "sn_body":   (160, 260),
    "sn_crack":  (1800, 4000),
    "hat":       (6000, 12000),
    "cymbal":    (4000, 12000)
}
TOM_BAND_PAD = 0.25  # +/- 25%

# === DRUM BUS COMP CONFIG ===
BUS_DIR = Path("/home/arlo/gcs-bucket/drum_bus")          # where bus and stems are written
BUS_TARGET_LOUDNESS = -13.0                   # LUFS-ish target for drum bus
FS = SAMPLE_RATE

# Alignment / polarity
MAX_ALIGN_MS = 3.0       # search up to ±3 ms for close→OH alignment
ALIGN_REF_ROLE = "oh"    # use OH as timing reference
XCORR_WIN_MS = 30        # analyze +/- this window around candidate onsets

# Per-role gains (pre-bus)
ROLE_GAIN_DB = {
    "kick": 0.0, "snare": 0.0, "tom": -1.5, "hihat": -3.0,
    "ride": -3.0, "crash": -4.0, "oh": -3.0, "room": -6.0, "kit": -3.0
}

# Simple EQ shelves/peaks per role (Hz, Q, dB). Empty list = no EQ.
ROLE_EQ = {
    "kick":  [("lowshelf", 60, 0.7, +3.0), ("peak", 350, 1.0, -3.0), ("highcut", 9000, 0.7, 0.0)],
    "snare": [("highpass", 100, 0.7, 0.0), ("peak", 200, 1.0, +2.0), ("peak", 3500, 1.2, +2.0)],
    "tom":   [("highpass", 60, 0.7, 0.0), ("peak", 120, 1.0, +2.0), ("peak", 2500, 1.2, +1.5)],
    "hihat": [("highpass", 400, 0.7, 0.0), ("peak", 8000, 1.0, +2.0)],
    "ride":  [("highpass", 250, 0.7, 0.0), ("peak", 7000, 1.0, +1.5)],
    "crash": [("highpass", 250, 0.7, 0.0), ("peak", 9000, 1.0, +1.5)],
    "oh":    [("highpass", 120, 0.7, 0.0)],
    "room":  [("highpass", 80, 0.7, 0.0)],
    "kit":   [("highpass", 120, 0.7, 0.0)]
}

# Gentle per-role gating (threshold dBFS relative, ratio)
ROLE_GATE = {
    "kick":  {"thresh_db": -36, "ratio": 4.0, "att_ms": 3,  "rel_ms": 120},
    "snare": {"thresh_db": -38, "ratio": 3.0, "att_ms": 2,  "rel_ms": 150},
    "tom":   {"thresh_db": -35, "ratio": 3.0, "att_ms": 2,  "rel_ms": 200},
    "hihat": {"thresh_db": -40, "ratio": 2.0, "att_ms": 1,  "rel_ms": 120},
    "ride":  {"thresh_db": -40, "ratio": 2.0, "att_ms": 1,  "rel_ms": 180},
    "crash": {"thresh_db": -40, "ratio": 2.0, "att_ms": 1,  "rel_ms": 300}
}

# Bus compressor & limiter
BUS_COMP = {"ratio": 3.0, "att_ms": 25, "rel_ms": 120, "knee_db": 3.0, "makeup_db": 0.0, "thresh_db": -18}
BUS_LIMITER_CEIL_DBFS = -0.3

# === COMPREHENSIVE DRUM PATTERNS ===
DRUM_KEYWORDS = ["kick", "kik", "bd", "bdin", "bdout", "snare", "sn", "snr", "snrtop", "snrbottom", "hihat", "hh", "hat", "chh", "ohh", "closedhat", "openhat", "tom", "racktom", "floortom", "rtom", "ftom", "overhead", "oh", "ohl", "ohr", "cymbal", "cym", "crash", "ride", "china", "splash", "bell", "stack", "k in", "k out", "floor", "kkin", "kk in", "kkout", "kk out", "rack", "drums", "drum", "rimshot", "djembe"]

# === DRUM CLASSIFICATION HELPERS ===
def butter_band(f_lo, f_hi, sr, order=4):
    b, a = butter(order, [f_lo/(sr*0.5), f_hi/(sr*0.5)], btype='band')
    return b, a

def band_energy_envelope(x, sr, f_lo, f_hi, hop=FRAME_HOP):
    b, a = butter_band(f_lo, f_hi, sr)
    y = lfilter(b, a, x)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop, center=True)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    return t, rms

def transient_score(x, sr, band: Tuple[float, float]):
    t, env = band_energy_envelope(x, sr, *band)
    # simple transient = positive derivative clip
    d = np.maximum(np.diff(env, prepend=env[:1]), 0.0)
    return t, d / (env.mean() + 1e-8)

def estimate_tom_resonance_hz(x, sr, onset_times, min_hz=60, max_hz=320):
    peaks = []
    for t in onset_times[:10]:
        s = int((t + 0.04) * sr); e = int((t + 0.30) * sr)
        if e - s < 2048: continue
        seg = x[s:e]
        S = np.abs(librosa.stft(seg, n_fft=2048, hop_length=512))**2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mask = (freqs >= min_hz) & (freqs <= max_hz)
        mean_spec = S[mask].mean(axis=1)
        f0 = freqs[mask][np.argmax(mean_spec)]
        peaks.append(f0)
    return float(np.median(peaks)) if peaks else 150.0

def oh_coincidence_score(oh_mix, close, sr, t_hit, win_ms=12):
    # Compare transient slopes around t_hit (±win) to ensure physical coincidence
    win = int((win_ms/1000.0) * sr)
    c = int(t_hit * sr)
    a = max(0, c - win); b = min(len(close), c + win)
    if b - a < 256: return 0.0
    # High-band for cymbal presence
    _, d_close = transient_score(close[a:b], sr, (1500, 8000))
    _, d_oh    = transient_score(oh_mix[a:b],  sr, (1500, 8000))
    # Normalize and correlate
    if d_close.std() < 1e-6 or d_oh.std() < 1e-6:
        return 0.0
    cc = np.corrcoef((d_close - d_close.mean())/d_close.std(),
                     (d_oh - d_oh.mean())/d_oh.std())[0,1]
    return float(max(0.0, cc))

def project_out_local(target, others: np.ndarray, sr, t_hit, pre_ms=15, post_ms=60):
    """
    Local least-squares projection: remove energy explained by other mics around the hit.
    `others` shape: (M, T). Returns target_denoised (only inside the window).
    """
    a = int((t_hit - pre_ms/1000.0) * sr); a = max(0, a)
    b = int((t_hit + post_ms/1000.0) * sr); b = min(len(target), b)
    if b - a < 128: return target  # too small, skip
    y = target[a:b].astype(np.float32)
    X = others[:, a:b].astype(np.float32)
    if X.size == 0: return target
    # Solve min ||y - X^T w|| in window
    XT = X.T  # (L, M)
    try:
        w, _, _, _ = np.linalg.lstsq(XT, y, rcond=None)
        y_hat = XT @ w
        out = target.copy()
        out[a:b] = y - y_hat
        return out
    except np.linalg.LinAlgError:
        return target

def velocity_from_band(x, sr, t_hit, band, pre_ms=40, post_ms=30):
    a = int((t_hit - pre_ms/1000.0) * sr); a = max(0, a)
    b = int((t_hit + post_ms/1000.0) * sr); b = min(len(x), b)
    t, env = band_energy_envelope(x[a:b], sr, *band)
    if len(env) < 2: return 64
    peak = env.max()
    floor = np.percentile(env, 20)
    v = int(np.clip(127*(peak - floor)/(peak + 1e-6), 1, 127))
    return v

def dedup_events(evts, tol_ms=25):
    evts = sorted(evts, key=lambda e: e["t"])
    out = []
    last_t_by_cls = {}
    for e in evts:
        key = e["cls"]
        last = last_t_by_cls.get(key, -1e9)
        if (e["t"] - last) * 1000.0 >= tol_ms:
            out.append(e); last_t_by_cls[key] = e["t"]
        else:
            # keep higher confidence
            if out and e["conf"] > out[-1]["conf"] and abs(out[-1]["t"] - e["t"]) * 1000.0 < tol_ms:
                out[-1] = e
    return out

def role_from_name(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["bd", "kik", "kick", "k in", "kout", "kk"]): return "kick"
    if any(k in n for k in ["snare", "snr", "sntop", "snrtop", "snbot", "snrbottom", "rim"]): return "snare"
    if any(k in n for k in ["hihat", "hh", "hat", "chh", "ohh", "closedhat", "openhat"]): return "hihat"
    if any(k in n for k in ["tom", "rtom", "ftom", "racktom", "floortom"]): return "tom"
    if any(k in n for k in ["ride"]): return "ride"
    if any(k in n for k in ["crash", "china", "splash", "stack"]): return "crash"
    if any(k in n for k in ["overhead", "ohl", "ohr", "ovh"]): return "oh"
    if any(k in n for k in ["room"]): return "room"
    if any(k in n for k in ["kit", "drum", "drums"]): return "kit"
    return "other"

def build_role_index(group_files):
    idx = defaultdict(list)
    for f in group_files:
        r = role_from_name(f['filename'])
        idx[r].append(f)
    return idx

def load_mono(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1: audio = np.mean(audio, axis=1)
    return audio, sr

def compute_generic_onsets(audio, sr, onset_proc, onset_act_proc):
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    return onset_proc(onset_act_proc(audio)), sr

def class_confidence(close_sig, oh_mix, sr, t_hit, cls, tom_f0=None):
    # Reject cymbal classes without OH mix
    if oh_mix is None and cls in ("hihat","ride","crash"):
        return 0.0
    
    # compute band-based transient scores + OH coincidence
    if cls == "kick":
        _, d1 = transient_score(close_sig, sr, CLASS_BANDS["kick_low"])
        _, d2 = transient_score(close_sig, sr, CLASS_BANDS["kick_click"])
        sc = float(d1.max() * 0.7 + d2.max() * 0.3)
        ohc = oh_coincidence_score(oh_mix, close_sig, sr, t_hit)
        # kick can have lower OH coincidence
        return 0.7*sc + 0.3*max(0.0, 1.0 - 0.3 + ohc)
    elif cls == "snare":
        _, d1 = transient_score(close_sig, sr, CLASS_BANDS["sn_body"])
        _, d2 = transient_score(close_sig, sr, CLASS_BANDS["sn_crack"])
        sc = float(d1.max() * 0.4 + d2.max() * 0.6)
        ohc = oh_coincidence_score(oh_mix, close_sig, sr, t_hit)
        return 0.6*sc + 0.4*ohc
    elif cls == "hihat":
        _, d = transient_score(close_sig, sr, CLASS_BANDS["hat"])
        sc = float(d.max())
        ohc = oh_coincidence_score(oh_mix, close_sig, sr, t_hit)
        return 0.5*sc + 0.5*ohc
    elif cls == "tom" and tom_f0 is not None:
        band = (max(40, tom_f0*(1.0 - TOM_BAND_PAD)), min(400, tom_f0*(1.0 + TOM_BAND_PAD)))
        _, d = transient_score(close_sig, sr, band)
        sc = float(d.max())
        ohc = oh_coincidence_score(oh_mix, close_sig, sr, t_hit)
        return 0.5*sc + 0.5*ohc
    elif cls in ("ride","crash"):
        _, d = transient_score(close_sig, sr, CLASS_BANDS["cymbal"])
        sc = float(d.max())
        ohc = oh_coincidence_score(oh_mix, close_sig, sr, t_hit)
        return 0.5*sc + 0.5*ohc
    return 0.0

def choose_velocity(close_sig, sr, t_hit, cls, tom_f0=None):
    if cls == "kick":
        return velocity_from_band(close_sig, sr, t_hit, CLASS_BANDS["kick_low"])
    if cls == "snare":
        return velocity_from_band(close_sig, sr, t_hit, CLASS_BANDS["sn_crack"])
    if cls == "hihat":
        return velocity_from_band(close_sig, sr, t_hit, CLASS_BANDS["hat"])
    if cls == "tom" and tom_f0 is not None:
        band = (max(40, tom_f0*(1.0 - TOM_BAND_PAD)), min(400, tom_f0*(1.0 + TOM_BAND_PAD)))
        return velocity_from_band(close_sig, sr, t_hit, band)
    if cls in ("ride","crash"):
        return velocity_from_band(close_sig, sr, t_hit, CLASS_BANDS["cymbal"])
    return 64

# === DRUM BUS DSP HELPERS ===
def db_to_gain(db): 
    return 10.0 ** (db / 20.0)

def peak_normalize(x, ceil=0.999):
    peak = np.max(np.abs(x)) + 1e-12
    return (x / peak) * ceil

def apply_gain_db(x, db): 
    return x * db_to_gain(db)

# --- EQ ---
def biquad(type_, fs, f0, Q=0.707, gain_db=0.0):
    A = 10**(gain_db/40)
    w0 = 2*math.pi*f0/fs
    alpha = math.sin(w0)/(2*Q)
    c = math.cos(w0)

    if type_ == "highpass":
        b0 =  (1+c)/2; b1 = -(1+c); b2 = (1+c)/2; a0 = 1 + alpha; a1 = -2*c; a2 = 1 - alpha
    elif type_ == "lowshelf":
        # RBJ cookbook
        beta = math.sqrt(A)/Q
        b0 =    A*((A+1) - (A-1)*c + 2*math.sqrt(A)*alpha)
        b1 =  2*A*((A-1) - (A+1)*c)
        b2 =    A*((A+1) - (A-1)*c - 2*math.sqrt(A)*alpha)
        a0 =       (A+1) + (A-1)*c + 2*math.sqrt(A)*alpha
        a1 =   -2*((A-1) + (A+1)*c)
        a2 =       (A+1) + (A-1)*c - 2*math.sqrt(A)*alpha
    elif type_ == "highcut":
        # high shelf as cut with negative gain if needed; simpler: 2nd order LPF
        # fallback to butter(4) lowpass:
        b, a = butter(4, f0/(fs*0.5), btype='low')
        return b, a
    elif type_ == "peak":
        b0 = 1 + alpha*A
        b1 = -2*c
        b2 = 1 - alpha*A
        a0 = 1 + alpha/A
        a1 = -2*c
        a2 = 1 - alpha/A
    else:
        # default to highpass 20 Hz
        b0 =  (1+c)/2; b1 = -(1+c); b2 = (1+c)/2; a0 = 1 + alpha; a1 = -2*c; a2 = 1 - alpha

    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])

def apply_eq_chain(x, fs, eq_list):
    y = x.copy()
    for (typ, f0, Q, db) in eq_list:
        b, a = biquad(typ, fs, f0, Q, db)
        y = lfilter(b, a, y)
    return y

# --- Simple gate (downward expander) ---
def envelope_follow(x, fs, att_ms=5, rel_ms=100):
    att = math.exp(-1.0 / (fs * att_ms/1000.0))
    rel = math.exp(-1.0 / (fs * rel_ms/1000.0))
    env = np.zeros_like(x)
    for i, s in enumerate(np.abs(x)):
        if s > env[i-1] if i>0 else 0:
            env[i] = att * (env[i-1] if i>0 else 0) + (1-att) * s
        else:
            env[i] = rel * (env[i-1] if i>0 else 0) + (1-rel) * s
    return env + 1e-9

def apply_gate(x, fs, thresh_db=-40, ratio=3.0, att_ms=3, rel_ms=150):
    env = envelope_follow(x, fs, att_ms, rel_ms)
    thresh = db_to_gain(thresh_db)
    gain = np.ones_like(x)
    mask = env < thresh
    # gain below threshold -> (env/thresh)^(ratio-1)
    gain[mask] = (env[mask] / thresh) ** (ratio - 1.0)
    return x * gain

# --- Simple feed-forward compressor ---
def apply_compressor(x, fs, thresh_db=-18, ratio=3.0, att_ms=25, rel_ms=120, knee_db=3.0, makeup_db=0.0):
    thresh = db_to_gain(thresh_db)
    knee = db_to_gain(knee_db)
    att = math.exp(-1.0 / (fs * att_ms/1000.0))
    rel = math.exp(-1.0 / (fs * rel_ms/1000.0))
    env = envelope_follow(x, fs, att_ms, rel_ms)
    gain = np.ones_like(x)
    for i, e in enumerate(env):
        g = 1.0
        if e > thresh:
            # soft knee
            over = e / thresh
            if over < knee:
                # in-knee: ease into ratio
                r = 1.0 + (ratio-1.0) * ( (over-1.0)/(knee-1.0) )
            else:
                r = ratio
            g = (e / thresh) ** (1.0 - 1.0/r)
            g = 1.0 / g
        # smooth gain
        if i == 0:
            gain[i] = g
        else:
            tau = att if g < gain[i-1] else rel
            gain[i] = tau * gain[i-1] + (1 - tau) * g
    y = x * gain * db_to_gain(makeup_db)
    return y

# --- Limiter ---
def apply_limiter(x, ceil_db=-0.3):
    y = x.copy()
    y = peak_normalize(y, ceil=db_to_gain(ceil_db))
    return y

# --- Alignment & polarity ---
def estimate_delay_samples(x, y, fs, max_ms=3.0):
    # GCC-ish normalized cross-corr in ±max_ms
    max_lag = int(fs * max_ms / 1000.0)
    # use a small windowed segment around energetic region
    # fallback: full signal
    n = min(len(x), len(y))
    if n < 2048: return 0, 1
    # fast FFT cross-corr is overkill; do narrow window time-domain
    lags = range(-max_lag, max_lag+1)
    best, bestlag, bestsign = -1e9, 0, 1
    x_std = np.std(x) + 1e-9
    y_std = np.std(y) + 1e-9
    for lag in lags:
        if lag >= 0:
            a = x[:n-lag]; b = y[lag:n]
        else:
            a = x[-lag:n]; b = y[:n+lag]
        if len(a) < 1024: continue
        # check both polarities, pick better
        cpos = float(np.dot(a, b) / (np.std(a)+1e-9) / (np.std(b)+1e-9))
        cneg = float(np.dot(a, -b) / (np.std(a)+1e-9) / (np.std(b)+1e-9))
        if cpos > best:
            best, bestlag, bestsign = cpos, lag, +1
        if cneg > best:
            best, bestlag, bestsign = cneg, lag, -1
    return bestlag, bestsign

def apply_delay(x, samples):
    if samples == 0: return x
    if samples > 0:
        return np.concatenate([np.zeros(samples, dtype=x.dtype), x[:-samples]])
    else:
        s = -samples
        return np.concatenate([x[s:], np.zeros(s, dtype=x.dtype)])

# === DRUM BUS RENDERING FUNCTION ===
def render_drum_bus_for_group(session_name, group_idx, group_files, loaded_by_name, role_idx):
    # Skip rendering if no audio files were loaded
    if not loaded_by_name:
        print(f"   ⚠️  No audio files loaded for group {group_idx+1}, skipping drum bus rendering")
        return

    # 1) Build OH reference
    oh_list = []
    for key in ("oh", "room", "kit"):
        for f in role_idx.get(key, []):
            name = f['filename']
            if name in loaded_by_name:
                oh_list.append(loaded_by_name[name][0])
    if len(oh_list) == 0:
        # fallback: average everything
        oh_list = [loaded_by_name[n][0] for n in loaded_by_name]

    # Double check we have audio to work with
    if not oh_list:
        print(f"   ⚠️  No valid audio for OH reference in group {group_idx+1}, skipping drum bus rendering")
        return

    ref = np.mean(np.stack(oh_list, axis=0), axis=0)
    ref = peak_normalize(ref)

    # 2) Align & polarity-correct close mics to OH
    aligned = {}
    role_map = {}
    for role, files in role_idx.items():
        for f in files:
            name = f['filename']
            if name not in loaded_by_name: continue
            x, sr = loaded_by_name[name]
            # resample check already done
            lag, sign = estimate_delay_samples(x, ref, FS, MAX_ALIGN_MS)
            x = apply_delay(x * sign, lag)
            aligned[name] = x
            role_map[name] = role

    # 3) Per-role processing: gate -> EQ -> gain
    proc = {}
    for name, x in aligned.items():
        role = role_map.get(name, "other")
        y = x.copy()
        # gate if defined
        if role in ROLE_GATE:
            g = ROLE_GATE[role]
            y = apply_gate(y, FS, g["thresh_db"], g["ratio"], g["att_ms"], g["rel_ms"])
        # EQ
        eq_list = ROLE_EQ.get(role, [])
        if eq_list:
            y = apply_eq_chain(y, FS, eq_list)
        # gain
        y = apply_gain_db(y, ROLE_GAIN_DB.get(role, 0.0))
        # keep
        proc[name] = y

    # 4) Build semantic stems
    def sum_role(prefixes):
        arr = [proc[n] for n in proc if any(role_map.get(n,"").startswith(p) or p==role_map.get(n,"") for p in prefixes)]
        return np.sum(np.stack(arr, axis=0), axis=0) if arr else np.zeros_like(ref)

    kick_direct  = sum_role(["kick"])
    snare_direct = sum_role(["snare"])
    toms_sum     = sum_role(["tom"])
    hats_ride    = sum_role(["hihat","ride"])
    cymbals      = sum_role(["crash"])
    oh_room      = sum_role(["oh","room","kit"])

    # 5) Stereo image (simple: OH/room forms L/R if available; else dual-mono)
    # If you have OHL/OHR names, you could split; here we keep stereo as dual-mono with light widening
    def widen(stereo_like):
        # create fake stereo: M/S trick (subtle)
        M = stereo_like
        S = apply_eq_chain(M, FS, [("peak", 7000, 0.7, +1.5)])
        L = np.clip(M + 0.15*S, -1.0, 1.0)
        R = np.clip(M - 0.15*S, -1.0, 1.0)
        return np.stack([L, R], axis=1)

    # direct elements are mostly mono; ambience widened
    bus_L = (kick_direct + snare_direct + toms_sum + 0.5*hats_ride + 0.5*cymbals)
    bus_R = bus_L.copy()
    amb_st = widen(oh_room)

    # 6) Sum to stereo bus
    bus = np.stack([bus_L, bus_R], axis=1) + amb_st
    # gentle bus comp then limiter
    # process per channel
    for ch in (0,1):
        bus[:,ch] = apply_compressor(bus[:,ch],
                                     FS, BUS_COMP["thresh_db"], BUS_COMP["ratio"],
                                     BUS_COMP["att_ms"], BUS_COMP["rel_ms"],
                                     BUS_COMP["knee_db"], BUS_COMP["makeup_db"])
        bus[:,ch] = apply_limiter(bus[:,ch], BUS_LIMITER_CEIL_DBFS)

    # 7) Normalize to target-ish loudness (simple peak/approx; for exact LUFS use pyloudnorm later)
    bus = peak_normalize(bus, ceil=db_to_gain(BUS_LIMITER_CEIL_DBFS))

    # 8) Write files
    out_dir = BUS_DIR / session_name
    out_dir.mkdir(parents=True, exist_ok=True)
    bus_path = out_dir / f"group_{group_idx+1:02d}.bus.wav"
    sf.write(bus_path, bus, FS)

    # optional stems
    stem_dir = out_dir / f"group_{group_idx+1:02d}_stems"
    stem_dir.mkdir(parents=True, exist_ok=True)
    sf.write(stem_dir / "kick_direct.wav",  peak_normalize(kick_direct),  FS)
    sf.write(stem_dir / "snare_direct.wav", peak_normalize(snare_direct), FS)
    sf.write(stem_dir / "toms_sum.wav",     peak_normalize(toms_sum),     FS)
    sf.write(stem_dir / "hats_ride.wav",    peak_normalize(hats_ride),    FS)
    sf.write(stem_dir / "cymbals.wav",      peak_normalize(cymbals),      FS)
    sf.write(stem_dir / "oh_room.wav",      peak_normalize(oh_room),      FS)

    print(f"   🔊 wrote drum bus  -> {bus_path}")
    print(f"   🎚 wrote stems     -> {stem_dir}")

# === ORIGINAL HELPERS ===
def is_drum_file(filename: str) -> bool:
    fname = filename.lower()
    # Use word boundaries and common separators to avoid false positives
    import re
    # Create regex pattern for each keyword with word boundaries or common separators
    for kw in DRUM_KEYWORDS:
        # Skip very short keywords that are likely to cause false positives
        if len(kw) <= 2 and kw not in ["oh", "bd", "sn", "hh"]:
            continue
        # Use word boundaries or common audio file separators, also allow at start of compound words
        pattern = r'(?:^|[_\s\-\.])' + re.escape(kw) + r'(?:[_\s\-\.]|[a-z]*(?:[_\s\-\.]|$))'
        if re.search(pattern, fname):
            return True
    return False

def extract_track_suffix(filename: str) -> str:
    """Extract consistent suffix like _1 or 01_03 from filename"""
    # Look for patterns like _1, _01, 01_03, etc. at the end before file extension
    stem = Path(filename).stem.lower()
    patterns = [
        r'_(\d+)$',           # _1, _01, _12
        r'(\d{2}_\d{2})$',    # 01_03, 12_08
        r'_(\d{2})$',         # _01, _12
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return match.group(1)
    return ""

def get_duration(audio_file: Path) -> float:
    """Get audio file duration in seconds"""
    try:
        info = sf.info(audio_file)
        return info.duration
    except Exception:
        return 0.0

def is_mostly_silent(audio_file: Path, threshold: float = SILENCE_THRESHOLD) -> bool:
    """Check if audio file is mostly silent by analyzing RMS levels"""
    try:
        # Read a sample of the file (first 30 seconds max to avoid loading huge files)
        audio, sr = sf.read(audio_file, frames=int(30 * 44100))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono

        # Calculate RMS in chunks to detect if most of the file is silent
        chunk_size = int(sr * 2)  # 2-second chunks
        chunk_rms_values = []

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) > 0:
                rms = np.sqrt(np.mean(chunk**2))
                chunk_rms_values.append(rms)

        if not chunk_rms_values:
            return True  # Empty file

        # Check if more than 80% of chunks are below silence threshold
        silent_chunks = sum(1 for rms in chunk_rms_values if rms < threshold)
        silence_ratio = silent_chunks / len(chunk_rms_values)

        return silence_ratio > 0.8  # Mostly silent if >80% of chunks are quiet

    except Exception:
        return True  # If we can't read the file, consider it silent/invalid

def validate_drum_group(group: list) -> bool:
    """Validate that a drum group has at least 3 different mic types"""
    mic_types = set()

    for file_info in group:
        role = role_from_name(file_info['filename'])
        if role != "other":
            mic_types.add(role)

    # Require at least 3 different mic types (e.g., kick, snare, oh)
    return len(mic_types) >= 3

def extract_take_identifier(filename: str) -> str:
    """Extract take/version identifier from filename (the suffix, not base name)"""
    import re
    stem = Path(filename).stem.lower()

    # Look for common take patterns and return the SUFFIX (take number)
    patterns = [
        r'\.(\d{2}_\d{2})$',  # .01_02, .06_07 -> "01_02"
        r'_(\d{2}_\d{2})$',   # _01_02, _06_07 -> "01_02"
        r'\.(\d{2})$',        # .01, .06 -> "01"
        r'_(\d{2})$',         # _01, _06 -> "01"
        r'_(\d+)$',           # _1, _2 -> "1"
    ]

    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return match.group(1)

    # If no take pattern found, use a generic identifier based on similar duration
    return "no_take_suffix"

def group_by_duration_and_take(drum_files: list, tolerance: float = DURATION_TOLERANCE) -> list:
    """Group drum files by similar duration and same take, validate groups"""
    # First group by take identifier and duration
    take_groups = defaultdict(list)

    for file_info in drum_files:
        take_id = extract_take_identifier(file_info['filename'])
        duration_key = round(file_info['duration'] / tolerance) * tolerance  # Round to tolerance buckets
        key = (take_id, duration_key)
        take_groups[key].append(file_info)

    # Then validate each group
    valid_groups = []
    for (take_id, duration_key), files in take_groups.items():
        # Only include groups with multiple files AND at least 3 mic types
        if len(files) > 1 and validate_drum_group(files):
            # Additional check: ensure files have similar durations within tolerance
            durations = [f['duration'] for f in files]
            if max(durations) - min(durations) <= tolerance:
                valid_groups.append(files)

    return valid_groups

def find_common_suffix(group: list) -> str:
    """Find consistent suffix pattern across group files"""
    suffixes = [extract_track_suffix(f['path']) for f in group]
    suffixes = [s for s in suffixes if s]  # Remove empty strings
    
    if not suffixes or len(set(suffixes)) > 1:
        return ""  # No consistent suffix
    
    return suffixes[0] if len(set(suffixes)) == 1 else ""

def resolve_relative_to_roots(audio_file: Path):
    for root in PROTOOLS_ROOTS:
        try:
            return audio_file.relative_to(root)
        except ValueError:
            continue
    return None

def load_paths():
    with open(AUDIO_PATH_LIST, "r") as f:
        return [line.strip() for line in f if line.strip().endswith(".wav")][SKIP_COUNT:]

# === MAIN ===
def process_session_from_paths(all_paths, target_session_name, processors):
    """Process a specific session from the paths list"""
    beat_proc, beat_act_proc, onset_proc, onset_act_proc = processors
    drum_files = []
    seen_paths = set()  # Track unique paths to prevent duplicates

    for path_str in all_paths:
        path = Path(path_str)

        if not is_drum_file(path.name):
            continue

        # Skip duplicates using resolved path
        path_key = str(path.resolve()) if path.exists() else str(path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)

        # Extract session name from path parts (without checking if path exists)
        path_parts = path.parts
        session_name = None

        try:
            # Look for "New" or "Prev" in path parts
            if "New" in path_parts:
                session_name = path_parts[path_parts.index("New") + 1]
            elif "Prev" in path_parts:
                session_name = path_parts[path_parts.index("Prev") + 1]
            else:
                continue
        except (IndexError, ValueError):
            continue

        # Only process files from the target session
        if session_name != target_session_name:
            continue

        # Skip duration and silence checks if file doesn't exist, use placeholder
        if path.exists():
            duration = get_duration(path)
            if duration == 0.0 or duration < MIN_DURATION:
                continue

            # Check if file is mostly silent
            if is_mostly_silent(path):
                continue

        else:
            # Use a default duration for missing files - they'll be grouped separately
            duration = 60.0  # placeholder duration

        drum_files.append({
            'path': str(path),
            'filename': path.name,
            'duration': duration
        })

    # Final deduplication by filename within this session
    seen_filenames = set()
    unique_drum_files = []
    for file_info in drum_files:
        if file_info['filename'] not in seen_filenames:
            seen_filenames.add(file_info['filename'])
            unique_drum_files.append(file_info)

    return unique_drum_files

def get_all_session_names(all_paths):
    """Get unique session names from paths file"""
    sessions = set()

    for path_str in all_paths:
        path = Path(path_str)

        if not is_drum_file(path.name):
            continue

        # Extract session name from path parts
        path_parts = path.parts
        session_name = None

        try:
            # Look for "New" or "Prev" in path parts
            if "New" in path_parts:
                session_name = path_parts[path_parts.index("New") + 1]
            elif "Prev" in path_parts:
                session_name = path_parts[path_parts.index("Prev") + 1]
            else:
                continue
        except (IndexError, ValueError):
            continue

        sessions.add(session_name)

    return sorted(sessions)

def process_single_session(session_name, all_paths):
    """Process a single session - designed to be run in parallel"""
    # Initialize processors for this worker
    beat_proc = DBNBeatTrackingProcessor(fps=100)
    beat_act_proc = RNNBeatProcessor()
    onset_proc = OnsetPeakPickingProcessor(fps=100)
    onset_act_proc = CNNOnsetProcessor()
    processors = (beat_proc, beat_act_proc, onset_proc, onset_act_proc)

    # Get drum files for this specific session
    drum_files = process_session_from_paths(all_paths, session_name, processors)

    if not drum_files:
        return session_name, None, 0

    print(f"\n📂 Processing session: {session_name} ({len(drum_files)} drum files)")

    # Group files by similar duration and same take
    groups = group_by_duration_and_take(drum_files)

    if not groups:
        print(f"   No valid multi-mic groups found (need ≥3 mic types)")
        return session_name, None, 0

    session_groups = []
    session_total_groups = 0

    for group_idx, group in enumerate(groups):
            print(f"   🥁 Group {group_idx+1}: {len(group)} files, ~{group[0]['duration']:.1f}s")
            
            # Find common suffix if exists
            common_suffix = find_common_suffix(group)
            
            group_data = {
                'duration': group[0]['duration'],
                'common_suffix': common_suffix,
                'files': []
            }
            
            # Process each file in the group (deduplicate by filename)
            unique_files = []
            seen_names = set()
            for file_info in group:
                if file_info['filename'] not in seen_names:
                    unique_files.append(file_info)
                    seen_names.add(file_info['filename'])

            for file_info in unique_files:
                path = Path(file_info['path'])
                
                # Create output paths
                out_base = OUTPUT_DIR / session_name / path.stem
                beats_path = out_base.with_suffix(".beats.npy")
                onsets_path = out_base.with_suffix(".onsets.npy")
                out_base.parent.mkdir(parents=True, exist_ok=True)
                
                file_data = {
                    'path': file_info['path'],
                    'filename': file_info['filename'],
                    'beats_file': str(beats_path),
                    'onsets_file': str(onsets_path)
                }
                
                try:
                    if not path.exists():
                        print(f"      ⚠️  {path.name}: File not found (archived)")
                        file_data['error'] = "File not found (archived)"
                    else:
                        # Load and process audio
                        audio, sr = sf.read(path)
                        if audio.ndim > 1:
                            audio = np.mean(audio, axis=1)
                        if sr != SAMPLE_RATE:
                            import librosa
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                            sr = SAMPLE_RATE

                        # Extract beats
                        if not beats_path.exists():
                            beat_times = beat_proc(beat_act_proc(audio))
                            np.save(beats_path, np.array(beat_times, dtype=np.float32))

                        # Extract onsets
                        if not onsets_path.exists():
                            onset_times = onset_proc(onset_act_proc(audio))
                            np.save(onsets_path, np.array(onset_times, dtype=np.float32))

                        print(f"      ✅ {path.name}")

                except Exception as e:
                    print(f"      ❌ {path.name}: {str(e)}")
                    file_data['error'] = str(e)
                
                group_data['files'].append(file_data)
            
            # === NEW: build role index and event list for the group ===
            role_idx = build_role_index(group)
            loaded = {}  # filename -> (audio, sr)
            for f in group:
                p = Path(f['path'])
                try:
                    if not p.exists():
                        print(f"      ⚠️  Skipping {f['filename']}: File not found (archived)")
                        continue
                    audio, sr = load_mono(p)
                    if sr != SAMPLE_RATE:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                        sr = SAMPLE_RATE
                    loaded[f['filename']] = (audio, sr)
                except Exception as e:
                    print(f"      ❌ Error loading {f['filename']}: {str(e)}")
                    continue

            # Overhead/room mix:
            oh_list = []
            for key in ("oh", "room", "kit"):
                for f in role_idx.get(key, []):
                    if f['filename'] in loaded:
                        oh_list.append(loaded[f['filename']][0])
            oh_mix = np.mean(np.stack(oh_list, axis=0), axis=0) if oh_list else None

            # Precompute tom resonances per tom mic (optional)
            tom_f0_by_file = {}
            for f in role_idx.get("tom", []):
                if f['filename'] in loaded:
                    x, sr = loaded[f['filename']]
                    # Use that mic's own generic onsets as anchors for resonance
                    # Generate onsets path based on file path
                    path_obj = Path(f['path'])
                    out_base = OUTPUT_DIR / session_name / path_obj.stem
                    onsets_path = out_base.with_suffix(".onsets.npy")

                    if onsets_path.exists():
                        t_on = np.load(onsets_path)
                    else:
                        t_on, _sr = compute_generic_onsets(x, sr, onset_proc, onset_act_proc)
                    
                    # Fallback if too few onsets
                    if not t_on.size:
                        t_on = np.array([len(x)/sr/2])  # center
                    
                    tom_f0_by_file[f['filename']] = estimate_tom_resonance_hz(x, sr, t_on.tolist())

            events = []
            # process close mics likely to carry primary classes
            close_roles = ["kick", "snare", "hihat", "tom", "ride", "crash"]

            for role in close_roles:
                for f in role_idx.get(role, []):
                    name = f['filename']
                    if name not in loaded: continue
                    x, sr = loaded[name]
                    
                    # Build others_stack excluding current mic and OH/room
                    all_other_mics = [loaded[n][0] for n in loaded if (role_from_name(n) not in ("oh","room") and n != name)]
                    others_stack = np.stack(all_other_mics, axis=0) if len(all_other_mics) > 0 else None
                    # candidate onsets (generic)
                    # Generate onsets path based on file path
                    path_obj = Path(f['path'])
                    out_base = OUTPUT_DIR / session_name / path_obj.stem
                    onsets_path = out_base.with_suffix(".onsets.npy")

                    if onsets_path.exists():
                        cand_ts = np.load(onsets_path).tolist()
                    else:
                        cand_ts, _sr = compute_generic_onsets(x, sr, onset_proc, onset_act_proc)
                        cand_ts = cand_ts.tolist()

                    # classify each candidate
                    for t_hit in cand_ts:
                        x_use = x
                        if others_stack is not None and oh_mix is not None:
                            # bleed reduction around hit before scoring
                            x_use = project_out_local(x, others_stack, sr, t_hit)

                        # try classes by priority
                        candidates = []
                        if role in ("kick","snare","hihat"):
                            classes_try = [role]  # strong prior from file role
                        elif role == "tom":
                            classes_try = ["tom"]
                        else:
                            classes_try = [role, "hihat", "crash", "ride"]  # cymbal-ish

                        for cls in classes_try:
                            conf = class_confidence(x_use, oh_mix if oh_mix is not None else x_use, sr, t_hit,
                                                    cls, tom_f0_by_file.get(name))
                            
                            # Gate on OH coincidence for non-kick classes
                            if cls != "kick" and oh_mix is not None:
                                ohc = oh_coincidence_score(oh_mix, x_use, sr, t_hit)
                                if ohc < OH_COINC_MIN:
                                    continue
                            
                            if conf >= MIN_CONF:
                                vel = choose_velocity(x_use, sr, t_hit, cls, tom_f0_by_file.get(name))
                                events.append({"t": float(t_hit), "cls": cls, "vel": int(vel), "conf": float(conf)})

            # merge, de-dup by class (use 20ms for better hat handling)
            events = dedup_events(events, tol_ms=20)

            # save per-group events
            evt_path = EVENT_JSON / session_name / f"group_{group_idx+1:02d}.events.json"
            evt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(evt_path, "w") as jf:
                json.dump(events, jf, indent=2)
            print(f"   🎯 wrote {len(events)} merged drum events -> {evt_path}")
            
            # Add events info to group data
            group_data['events_file'] = str(evt_path)
            group_data['num_events'] = len(events)
            
            # === NEW: Render professional drum bus for this group ===
            render_drum_bus_for_group(session_name, group_idx, group, loaded, role_idx)
            
            session_groups.append(group_data)
            session_total_groups += 1

    return session_name, session_groups, session_total_groups

if __name__ == "__main__":
    all_paths = load_paths()

    # Get all session names first
    print("🔍 Discovering sessions with drum files...")
    session_names = get_all_session_names(all_paths)
    print(f"📁 Found {len(session_names)} sessions with drum files")

    # Process sessions in parallel using up to 12 CPU cores
    print(f"🚀 Processing sessions using {MAX_WORKERS} CPU cores...")

    # Create worker function with all_paths pre-bound
    worker_func = partial(process_single_session, all_paths=all_paths)

    result = {}
    total_groups = 0

    with Pool(processes=MAX_WORKERS) as pool:
        # Process sessions in parallel
        results = pool.map(worker_func, session_names)

        # Collect results
        for session_name, session_groups, session_group_count in results:
            if session_groups is not None:
                result[session_name] = session_groups
                total_groups += session_group_count

    # Save results to JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n🎉 Complete! Found {total_groups} multi-mic drum groups across {len(result)} sessions")
    print(f"📄 Results saved to: {OUTPUT_JSON}")
