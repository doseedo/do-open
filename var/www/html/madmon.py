import os
import re
import json
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

# === CONFIG ===
AUDIO_PATH_LIST = Path("/home/arlo/Data/all_audio_paths4.txt")
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

# === COMPREHENSIVE DRUM PATTERNS ===
DRUM_KEYWORDS = ["kick", "kik", "bd", "bdin", "bdout", "snare", "sn", "snr", "snrtop", "snrbottom", "hihat", "hh", "hat", "chh", "ohh", "closedhat", "openhat", "tom", "racktom", "floortom", "rtom", "ftom", "overhead", "oh", "ohl", "ohr", "cymbal", "cym", "crash", "ride", "china", "splash", "bell", "stack", "k in", "k out", "floor", "kkin", "kk in", "kkout", "kk out", "rack", "drums", "drum", "rt", "sb", "st", "ko_", "ki_", "flt", "ovh", "ovhl", "ovhr", "kit", "rimshot", "djembe"]

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

# === ORIGINAL HELPERS ===
def is_drum_file(filename: str) -> bool:
    fname = filename.lower()
    return any(kw in fname for kw in DRUM_KEYWORDS)

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

def group_by_duration(drum_files: list, tolerance: float = DURATION_TOLERANCE) -> list:
    """Group drum files by similar duration"""
    groups = []
    processed = set()
    
    for i, file_info in enumerate(drum_files):
        if i in processed:
            continue
            
        current_group = [file_info]
        processed.add(i)
        
        for j, other_info in enumerate(drum_files[i+1:], i+1):
            if j in processed:
                continue
                
            if abs(file_info['duration'] - other_info['duration']) <= tolerance:
                current_group.append(other_info)
                processed.add(j)
        
        if len(current_group) > 1:  # Only groups with multiple files
            groups.append(current_group)
    
    return groups

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
if __name__ == "__main__":
    all_paths = load_paths()
    
    # Initialize processors
    beat_proc = DBNBeatTrackingProcessor(fps=100)
    beat_act_proc = RNNBeatProcessor()
    onset_proc = OnsetPeakPickingProcessor(fps=100)
    onset_act_proc = CNNOnsetProcessor()  # Better transient detection
    
    # Collect drum files by session
    sessions = defaultdict(list)
    
    print("🔍 Scanning for drum files...")
    for path_str in all_paths:
        path = Path(path_str)
        
        if not is_drum_file(path.name):
            continue
            
        relative = resolve_relative_to_roots(path)
        if not relative:
            continue
            
        session_name = None
        if "New" in relative.parts:
            session_name = relative.parts[relative.parts.index("New") + 1]
        elif "Prev" in relative.parts:
            session_name = relative.parts[relative.parts.index("Prev") + 1]
        else:
            continue
            
        # Get duration for grouping
        duration = get_duration(path)
        if duration == 0.0:
            continue
            
        sessions[session_name].append({
            'path': str(path),
            'filename': path.name,
            'duration': duration
        })
    
    print(f"📁 Found drum files in {len(sessions)} sessions")
    
    # Process each session
    result = {}
    total_groups = 0
    
    for session_name, drum_files in sessions.items():
        print(f"\n📂 Processing session: {session_name} ({len(drum_files)} drum files)")
        
        # Group files by similar duration
        groups = group_by_duration(drum_files)
        
        if not groups:
            print(f"   No multi-mic groups found")
            continue
            
        session_groups = []
        
        for group_idx, group in enumerate(groups):
            print(f"   🥁 Group {group_idx+1}: {len(group)} files, ~{group[0]['duration']:.1f}s")
            
            # Find common suffix if exists
            common_suffix = find_common_suffix(group)
            
            group_data = {
                'duration': group[0]['duration'],
                'common_suffix': common_suffix,
                'files': []
            }
            
            # Process each file in the group
            for file_info in group:
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
                    audio, sr = load_mono(p)
                    if sr != SAMPLE_RATE:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                        sr = SAMPLE_RATE
                    loaded[f['filename']] = (audio, sr)
                except Exception:
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
                    onsets_path = Path(f['onsets_file'])
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
                    onsets_path = Path(f['onsets_file'])
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
            
            session_groups.append(group_data)
            total_groups += 1
        
        result[session_name] = session_groups
    
    # Save results to JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n🎉 Complete! Found {total_groups} multi-mic drum groups across {len(result)} sessions")
    print(f"📄 Results saved to: {OUTPUT_JSON}")
