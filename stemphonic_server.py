"""
Stemphonic inference server for DO2.
Uses the same inference path as probe_full_v3: StemphonicTrainingModule with
MIDI hooks + FSQ conditioning from soundfont-rendered MIDI.

Flow for MIDI input:
  1. MIDI → FluidSynth render (soundfont) → WAV
  2. WAV → VAE encode → latents → model.tokenize() → FSQ tokens
  3. MIDI → piano roll [1, 146, T] for MIDI adapter hooks
  4. generate_with_model_api(fsq + midi + text + optional cover_noise_strength)

API:
  POST /api/generate-stemphonic  → {task_id}
  GET  /api/generate-stemphonic/task/<id>  → {status, result}
  GET  /api/generate-stemphonic/download/<id>/<filename>  → WAV file
"""

import os, sys, uuid, threading, time, traceback, logging, subprocess
from pathlib import Path

# ── collections.abc shim for Python 3.10+ ──────────────────────────────
# Several legacy deps (beat_this, its transitive imports, some madmom
# code paths) still do `from collections import MutableSequence` or
# `collections.Mapping`, which were removed from the top-level
# `collections` module in Python 3.10+. Back-populate the names from
# `collections.abc` so these imports survive.
import collections as _coll
import collections.abc as _cabc
for _name in ("MutableSequence", "MutableMapping", "Mapping", "Sequence",
              "Iterable", "Iterator", "Container", "Hashable", "Sized",
              "Callable", "Set", "MutableSet", "ByteString"):
    if not hasattr(_coll, _name):
        setattr(_coll, _name, getattr(_cabc, _name))

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

sys.path.insert(0, "/scratch/ACE-Step-1.5")
sys.path.insert(0, "/mnt/data/system_home/arlo/do2/stemphonic_trainer")
sys.path.insert(0, "/mnt/data/system_home/arlo/do2")
# latent_demucs.runtime does `from distill_model import ...` (bareword
# import, not package-relative), so the student dir must be on sys.path.
sys.path.insert(0, "/mnt/data/system_home/arlo/do2/latent_demucs_student")
# latent_panns_student.runtime imports from student_model (bareword)
sys.path.insert(0, "/mnt/data/system_home/arlo/do2/latent_panns_student")
# latent_whisper_student.runtime imports from student_model (bareword)
sys.path.insert(0, "/mnt/data/system_home/arlo/do2/latent_whisper_student")

# Import train-matched inference
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "stemphonic_inference_trainmatch",
    "/mnt/data/system_home/arlo/do2/stemphonic_inference_trainmatch.py"
)
_tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)
generate_stemphonic_trainmatch = _tm.generate_stemphonic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ─── Sentry ──────────────────────────────────────────────────────────
# NO-OPs if SENTRY_DSN isn't in the environment. DSN is injected from
# Modal secret `doseedo-sentry` at deploy time. Scrubber strips auth
# headers + secret-ish body fields, drops bodies entirely for the two
# legacy 410'd encode routes (tagged legacy_route_hit=true so we still
# see stale-client traffic without logging the payload).
try:
    _SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
    if _SENTRY_DSN:
        import re as _re_sentry
        import sentry_sdk  # type: ignore
        from sentry_sdk.integrations.flask import FlaskIntegration  # type: ignore

        _SCRUB_HEADERS = {
            "Authorization", "Cookie", "X-Internal-Secret",
            "X-CSRF-Token", "X-API-Key", "Proxy-Authorization", "Set-Cookie",
        }
        _SCRUB_HEADERS_LC = {h.lower() for h in _SCRUB_HEADERS}
        _SCRUB_KEY_RE = _re_sentry.compile(r"token|jwt|secret|password|api[_-]?key", _re_sentry.I)
        _LEGACY_RE = _re_sentry.compile(r"/api/encode-(audio-latent|latents-bulk)\b")

        def _scrub_obj(o, depth=0):
            if depth > 6 or o is None: return o
            if isinstance(o, dict):
                return {k: ("[scrubbed]" if _SCRUB_KEY_RE.search(k) else _scrub_obj(v, depth + 1))
                        for k, v in o.items()}
            if isinstance(o, list):
                return [_scrub_obj(v, depth + 1) for v in o]
            return o

        def _before_send(event, hint):
            try:
                req = event.get("request") or {}
                hdrs = req.get("headers") or {}
                for k in list(hdrs.keys()):
                    if k.lower() in _SCRUB_HEADERS_LC:
                        hdrs[k] = "[scrubbed]"
                if "data" in req:
                    req["data"] = _scrub_obj(req["data"])
                if "cookies" in req:
                    req["cookies"] = "[scrubbed]"
                url = req.get("url") or ""
                if _LEGACY_RE.search(url):
                    event.setdefault("tags", {})["legacy_route_hit"] = "true"
                    req.pop("data", None)
                extra = event.get("extra") or {}
                event["extra"] = _scrub_obj(extra)
                for crumb in event.get("breadcrumbs", {}).get("values", []) or []:
                    if crumb.get("data"):
                        crumb["data"] = _scrub_obj(crumb["data"])
            except Exception:
                return {"message": "[sentry scrubber error — event redacted]",
                        "level": "error",
                        "tags": {"scrubber_failure": "true"}}
            return event

        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            environment=os.environ.get("SENTRY_ENV", "production"),
            release=os.environ.get("SENTRY_RELEASE") or None,
            traces_sample_rate=0.1,
            profiles_sample_rate=0.1,
            send_default_pii=False,
            integrations=[FlaskIntegration()],
            before_send=_before_send,
        )
        logger.info("[sentry] initialized (env=%s)", os.environ.get("SENTRY_ENV", "production"))
    else:
        logger.info("[sentry] SENTRY_DSN not set — skipping init (errors will NOT be captured)")
except Exception as _e:
    logger.warning("[sentry] init failed (non-fatal): %s", _e)

CKPT_DIR = "/scratch/stemphonic/checkpoints"
CKPT_PATH = "/scratch/stage2d_step130000.pt"  # legacy default symlink
OUTPUT_DIR = "/scratch/stemphonic_outputs"

# ---------------------------------------------------------------------------
# Checkpoint registry — id → metadata. has_midi=False means stage1 (no MIDI
# adapter / no per-layer hooks). Files are downloaded on demand from GCS.
# ---------------------------------------------------------------------------
CKPT_REGISTRY = {
    "stage2d-130k": {
        "label": "Stage 2d · step 130k (current)",
        "gcs": "gs://ptxsessiondata/DO2ckpts/stage2d_step130000.pt",
        "has_midi": True,
    },
    "stage2c-latest": {
        "label": "Stage 2c · latest",
        "gcs": "gs://ptxsessiondata/DO2ckpts/stage2c_latest.pt",
        "has_midi": True,
    },
    "stage2c-best": {
        "label": "Stage 2c · best",
        "gcs": "gs://ptxsessiondata/DO2ckpts/stage2c_best.pt",
        "has_midi": True,
    },
    "stage2b-30k": {
        "label": "Stage 2b · step 30k",
        "gcs": "gs://ptxsessiondata/DO2ckpts/stage2b_step30000.pt",
        "has_midi": True,
    },
    "stage2b-merged": {
        "label": "Stage 2b · merged",
        "gcs": "gs://ptxsessiondata/stemphonic_checkpoints/stage2b/merged.pt",
        "has_midi": True,
    },
    "v4-latest": {
        "label": "Stage 1 v4 · latest (no MIDI)",
        "gcs": "gs://ptxsessiondata/stemphonic_checkpoints/v4/latest.pt",
        # v4 ships an early MIDI adapter (hidden=64) that doesn't fit
        # the current StemphonicTrainingModule (hidden=256). Skip the
        # adapter load — decoder LoRA still applies.
        "has_midi": False,
    },
    "stage1v2-best": {
        "label": "Stage 1v2 · best (no MIDI)",
        "gcs": "gs://ptxsessiondata/stemphonic_checkpoints/stage1v2/best.pt",
        "has_midi": False,
    },
    "stage1v2-latest": {
        "label": "Stage 1v2 · latest (no MIDI)",
        "gcs": "gs://ptxsessiondata/stemphonic_checkpoints/stage1v2/latest.pt",
        "has_midi": False,
    },
    "stage1-best": {
        "label": "Stage 1 · best (no MIDI)",
        "gcs": "gs://ptxsessiondata/stemphonic_checkpoints/stage1/best.pt",
        "has_midi": False,
    },
    "stage1-latest": {
        "label": "Stage 1 · latest (no MIDI)",
        "gcs": "gs://ptxsessiondata/stemphonic_checkpoints/stage1/latest.pt",
        "has_midi": False,
    },
}
DEFAULT_CKPT_ID = "stage2d-130k"
SOUNDFONT_DIR = "/scratch/soundfonts"
DEFAULT_SOUNDFONT = "/usr/share/sounds/sf2/default-GM.sf2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INSTRUMENT_SOUNDFONTS = {
    "trombone": f"{SOUNDFONT_DIR}/trombone.sf2",
    "trumpet": f"{SOUNDFONT_DIR}/trumpet.sf2",
    "sax": f"{SOUNDFONT_DIR}/sax.sf2",
    "bassoon": f"{SOUNDFONT_DIR}/bassoon.sf2",
    "clarinet": f"{SOUNDFONT_DIR}/clarinet.sf2",
    "flute": f"{SOUNDFONT_DIR}/flute.sf2",
    "violin": f"{SOUNDFONT_DIR}/violin.sf2",
    "viola": f"{SOUNDFONT_DIR}/viola.sf2",
    "cello": f"{SOUNDFONT_DIR}/cello.sf2",
    "acoustic_piano": f"{SOUNDFONT_DIR}/Piano.sf2",
    "electric_piano": f"{SOUNDFONT_DIR}/Electric Piano.sf2",
    "acoustic_guitar": f"{SOUNDFONT_DIR}/acoustic guitar.sf2",
    "electric_guitar": f"{SOUNDFONT_DIR}/electric guitar.sf2",
    "electric_bass": f"{SOUNDFONT_DIR}/electric bass.sf2",
    "bass": f"{SOUNDFONT_DIR}/electric bass.sf2",
    "vocals": f"{SOUNDFONT_DIR}/vocals1.sf2",
    "voice": f"{SOUNDFONT_DIR}/vocals1.sf2",
    "drums": f"{SOUNDFONT_DIR}/drums.sf2",
    "default": DEFAULT_SOUNDFONT,
}

# Training-format captions — MUST match what the model saw during training
# From preprocess_v4.py SUBGROUP_DESCRIPTIONS / GROUP_DESCRIPTIONS
TRAINING_CAPTIONS = {
    "acoustic_piano": "acoustic piano, grand piano",
    "electric_piano": "electric piano, Rhodes, Wurlitzer",
    "electric_guitar": "electric guitar",
    "acoustic_guitar": "acoustic guitar, steel string",
    "electric_bass": "electric bass guitar",
    "bass": "electric bass guitar",
    "violin": "violin",
    "viola": "viola",
    "cello": "cello",
    "trumpet": "trumpet",
    "trombone": "trombone",
    "sax": "saxophone",
    "flute": "flute",
    "clarinet": "clarinet",
    "voice": "vocals, singing voice",
    "drums": "drum kit",
    # Stemphonic drum subgroups
    "drum_kit": "drum kit, acoustic drums",
    "electronic": "electronic drums, drum machine, 808",
    "percussion": "orchestral percussion, timpani, congas, hand percussion",
    # Stemphonic vocal subgroups
    "lead_vox": "lead vocals, singing voice, expressive",
    "bg_vox":   "background vocals, harmonies, oohs and aahs",
    "choir":    "choir, ensemble vocals, layered voices",
    "synth_vox": "synthesized vocals, vocoder, robotic voice",
}

# Instrument presets from real recordings
TIMBRE_PRESETS = {}   # {inst: [30, 64]} — first 30 frames for timbre ref
LATENT_PRESETS = {}   # {inst: [64, T]} — full latent for FSQ hints

_timbre_path = "/scratch/soundfonts/timbre_presets_v3.pt"
if os.path.exists(_timbre_path):
    TIMBRE_PRESETS = torch.load(_timbre_path, map_location="cpu", weights_only=True)
    logger.info("Loaded %d timbre presets", len(TIMBRE_PRESETS))

_full_path = "/scratch/soundfonts/instrument_presets_full.pt"
if os.path.exists(_full_path):
    _data = torch.load(_full_path, map_location="cpu", weights_only=True)
    LATENT_PRESETS = _data.get("latents", {})
    logger.info("Loaded %d full latent presets", len(LATENT_PRESETS))

FSQ_PRESETS = {}  # {inst: [T, 2048]} — tokenized via training-format path
for _p in ["/scratch/soundfonts/fsq_presets_v3.pt",
           "/scratch/soundfonts/fsq_presets_v2.pt",
           "/scratch/soundfonts/fsq_presets.pt"]:
    if os.path.exists(_p):
        FSQ_PRESETS = torch.load(_p, map_location="cpu", weights_only=True)
        logger.info("Loaded %d FSQ presets from %s", len(FSQ_PRESETS), _p)
        break

# Globals
tasks = {}
handler = None
module = None
current_ckpt_id = None
current_ckpt_meta = None  # {label, has_midi, step}
_base_decoder_state = None  # CPU snapshot of handler.model.decoder for clean swaps
model_lock = threading.Lock()
_timbre_ready = False  # flipped True by background _bg_timbre_init after precache completes


# ---------------------------------------------------------------------------
# Model loading (same as probe_full_v3.setup)
# ---------------------------------------------------------------------------

def _local_ckpt_path(ckpt_id):
    return os.path.join(CKPT_DIR, f"{ckpt_id}.pt")


def _ensure_ckpt_downloaded(ckpt_id):
    """Download ckpt from GCS if not already present locally."""
    meta = CKPT_REGISTRY.get(ckpt_id)
    if not meta:
        raise ValueError(f"Unknown checkpoint id: {ckpt_id}")
    local = _local_ckpt_path(ckpt_id)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    os.makedirs(CKPT_DIR, exist_ok=True)
    # Special-case: legacy default ckpt may already exist at /scratch/stage2d_step130000.pt
    if ckpt_id == DEFAULT_CKPT_ID and os.path.exists(CKPT_PATH):
        try:
            os.symlink(CKPT_PATH, local)
            logger.info("Linked default ckpt: %s -> %s", local, CKPT_PATH)
            return local
        except OSError:
            pass
    logger.info("Downloading ckpt %s from %s ...", ckpt_id, meta["gcs"])
    rc = subprocess.call([
        "gsutil", "-o", "GSUtil:parallel_thread_count=8",
        "-o", "GSUtil:sliced_object_download_max_components=8",
        "cp", meta["gcs"], local,
    ])
    if rc != 0 or not os.path.exists(local):
        raise RuntimeError(f"gsutil cp failed for {ckpt_id}")
    logger.info("Downloaded %s (%.1f MB)", local, os.path.getsize(local) / 1e6)
    return local


TIMBRE_CACHE_DIR = os.path.join(OUTPUT_DIR, "timbre_cache")
TIMBRE_VARIANTS_PER_INSTRUMENT = 10
TIMBRE_VARIANT_FRAMES = 750  # 30s @ 25Hz — full preview clip
TIMBRE_REF_FRAMES = 30       # what the model actually consumes (1.2s)

# Loaded at startup. {instrument: tensor[N_VARIANTS, FRAMES, 64]}
TIMBRE_VARIANT_LATENTS = {}


def _load_timbre_variants():
    """Build TIMBRE_VARIANT_LATENTS by slicing the long instrument
    presets file (each instrument is a [64, T] full latent track) into
    N variants of TIMBRE_VARIANT_FRAMES each. Falls back to the shorter
    v4 file (per-instrument 750 frames) if instrument_presets is missing."""
    global TIMBRE_VARIANT_LATENTS
    long_path = "/scratch/soundfonts/instrument_presets_v3.pt"
    long_data = None
    if os.path.exists(long_path):
        try:
            raw = torch.load(long_path, map_location="cpu", weights_only=True)
            long_data = raw.get("latents", raw)  # {inst: [64, T]}
            logger.info("Loaded long instrument latents: %d instruments", len(long_data))
        except Exception as e:
            logger.warning("Failed to load %s: %s", long_path, e)

    fallback_path = "/scratch/soundfonts/timbre_presets_v4_full.pt"
    fallback_data = None
    if os.path.exists(fallback_path):
        try:
            fallback_data = torch.load(fallback_path, map_location="cpu", weights_only=True)
        except Exception:
            pass

    instruments = sorted(set(
        list(TIMBRE_PRESETS.keys()) +
        (list(long_data.keys()) if long_data else []) +
        (list(fallback_data.keys()) if fallback_data else [])
    ))

    # Energy threshold: also serves as the floor for the model timbre window.
    # Empirically silent slices read RMS < 0.005; real audio reads > 0.05.
    SILENCE_RMS_THRESHOLD = 0.05

    out = {}
    for inst in instruments:
        variants = []
        # Prefer the long [64, T] track if it has enough frames for ≥1 clip
        if long_data is not None and inst in long_data:
            full = long_data[inst]
            if full.dim() == 2 and full.shape[0] in (64,) and full.shape[1] >= TIMBRE_VARIANT_FRAMES:
                full_tT = full.permute(1, 0)  # [T, 64]
                T = full_tT.shape[0]
                # Oversample candidate starts (4× the target N), score by RMS,
                # keep the top-N non-silent ones in their original order.
                # This avoids variants landing on silent intros/outros which
                # caused 'timbre token = zero ≈ CFG dropout' (REPORT issue #1).
                n_candidates = max(TIMBRE_VARIANTS_PER_INSTRUMENT * 4,
                                   TIMBRE_VARIANTS_PER_INSTRUMENT)
                if n_candidates > 1 and T > TIMBRE_VARIANT_FRAMES:
                    cand_stride = max(1, (T - TIMBRE_VARIANT_FRAMES) // (n_candidates - 1))
                else:
                    cand_stride = 0
                scored = []
                for i in range(n_candidates):
                    start = min(i * cand_stride, T - TIMBRE_VARIANT_FRAMES)
                    if start < 0:
                        start = 0
                    window = full_tT[start:start + TIMBRE_VARIANT_FRAMES]
                    # Score on the first TIMBRE_REF_FRAMES — that's the slice
                    # the model actually reads. A loud middle but silent head
                    # would still feed silence to the encoder.
                    head = window[:TIMBRE_REF_FRAMES]
                    rms = float((head.float() ** 2).mean().sqrt().item())
                    scored.append((start, rms, window))
                # Keep only candidates above the silence floor; if none clear
                # it (very quiet instrument), fall back to the loudest few.
                non_silent = [s for s in scored if s[1] >= SILENCE_RMS_THRESHOLD]
                if len(non_silent) >= TIMBRE_VARIANTS_PER_INSTRUMENT:
                    pick = non_silent
                else:
                    pick = sorted(scored, key=lambda s: -s[1])
                pick = sorted(pick, key=lambda s: s[0])  # back to time order
                # Spread N evenly across whatever survived
                if len(pick) > TIMBRE_VARIANTS_PER_INSTRUMENT:
                    step = len(pick) / TIMBRE_VARIANTS_PER_INSTRUMENT
                    pick = [pick[int(i * step)] for i in range(TIMBRE_VARIANTS_PER_INSTRUMENT)]
                variants = [w for _, _, w in pick]
                rmses = [round(s[1], 4) for s in scored]
                kept_rmses = [round(p[1], 4) for p in pick]
                logger.info("  %s: scored %d candidates rms=%s → kept %d rms=%s",
                            inst, len(scored), rmses, len(variants), kept_rmses)
        # Fallback: 30s file (single clip per instrument)
        if not variants and fallback_data is not None and inst in fallback_data:
            full_tT = fallback_data[inst]  # [T, 64]
            if full_tT.shape[0] >= TIMBRE_VARIANT_FRAMES:
                variants.append(full_tT[:TIMBRE_VARIANT_FRAMES])
        # Last resort: short v3 snippet (1.2s, but better than nothing)
        if not variants and inst in TIMBRE_PRESETS:
            variants.append(TIMBRE_PRESETS[inst])
        if variants:
            # Variants may differ in frame count when fallbacks are used,
            # so store as a list rather than stacking.
            out[inst] = variants
    TIMBRE_VARIANT_LATENTS = out
    logger.info("Built timbre variant latents: %s",
                {k: (len(v), tuple(v[0].shape)) for k, v in out.items()})


def precache_timbre_wavs():
    """Decode each timbre variant through the VAE once and cache the
    resulting WAVs. Variants are typically ~30s (750 latent frames).

    Also runs a *post-decode* silence filter: any variant whose decoded
    audio is below DECODED_SILENCE_RMS gets replaced with the loudest
    surviving variant for that instrument. Latent-domain RMS proved
    useless as a silence proxy (REPORT 2026-04-06-2228 §1 / §6.1)."""
    DECODED_SILENCE_RMS = 0.01
    if not TIMBRE_VARIANT_LATENTS:
        logger.info("No timbre variants to cache")
        return
    decoded_rms = {}  # {(inst, idx): rms}
    cached, skipped = 0, 0
    with torch.no_grad():
        for inst, variants in TIMBRE_VARIANT_LATENTS.items():
            inst_dir = os.path.join(TIMBRE_CACHE_DIR, inst)
            os.makedirs(inst_dir, exist_ok=True)
            for i, lat_tT in enumerate(variants):
                out = os.path.join(inst_dir, f"{i}.wav")
                try:
                    # lat_tT is [T, 64]; VAE expects [B, D, T]
                    lat = lat_tT.to("cuda", torch.bfloat16).permute(1, 0).unsqueeze(0)
                    audio = handler.vae.decode(lat).sample.squeeze(0).cpu().float().numpy()
                    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                    decoded_rms[(inst, i)] = rms
                    if not (os.path.exists(out) and os.path.getsize(out) > 0):
                        sf.write(out, audio.T if audio.ndim > 1 else audio, 48000)
                        cached += 1
                    else:
                        skipped += 1
                except Exception as e:
                    logger.warning("Failed to cache %s/%d: %s", inst, i, e)

    # Replace silent latent slices with the loudest surviving one per instrument
    for inst, variants in TIMBRE_VARIANT_LATENTS.items():
        rmses = [(i, decoded_rms.get((inst, i), 0.0)) for i in range(len(variants))]
        loud = sorted(rmses, key=lambda x: -x[1])
        loudest_idx = loud[0][0] if loud else 0
        loudest_rms = loud[0][1] if loud else 0.0
        # Whole-instrument fallback: when ALL slices are silent (e.g.
        # electric_bass DI signal decodes to ~0), try every other latent
        # source we have for this instrument until one decodes above the
        # silence floor.
        if loudest_rms < DECODED_SILENCE_RMS:
            fb_sources = []
            try:
                v4 = torch.load("/scratch/soundfonts/timbre_presets_v4_full.pt",
                                map_location="cpu", weights_only=True)
                if inst in v4:
                    fb_sources.append(("v4_full", v4[inst]))  # [T, 64]
            except Exception:
                pass
            try:
                full_pack = torch.load("/scratch/soundfonts/instrument_presets_full.pt",
                                       map_location="cpu", weights_only=True)
                full_lat = full_pack.get("latents", {})
                if inst in full_lat:
                    full_inst = full_lat[inst]  # [64, T]
                    if full_inst.dim() == 2 and full_inst.shape[0] == 64:
                        fb_sources.append(("instrument_full", full_inst.permute(1, 0)))
            except Exception:
                pass
            if inst in TIMBRE_PRESETS:
                fb_sources.append(("v3_short", TIMBRE_PRESETS[inst]))
            replaced_with = None
            for src_name, src_lat in fb_sources:
                if src_lat.shape[0] < 1:
                    continue
                # Take a TIMBRE_VARIANT_FRAMES window from the middle of
                # the source — middles tend to be the loudest sustain.
                T_src = src_lat.shape[0]
                if T_src >= TIMBRE_VARIANT_FRAMES:
                    start = (T_src - TIMBRE_VARIANT_FRAMES) // 2
                    window = src_lat[start:start + TIMBRE_VARIANT_FRAMES]
                else:
                    reps = (TIMBRE_VARIANT_FRAMES + T_src - 1) // T_src
                    window = src_lat.repeat(reps, 1)[:TIMBRE_VARIANT_FRAMES]
                with torch.no_grad():
                    lat = window.to("cuda", torch.bfloat16).permute(1, 0).unsqueeze(0)
                    fb_audio = handler.vae.decode(lat).sample.squeeze(0).cpu().float().numpy()
                fb_rms = float(np.sqrt(np.mean(fb_audio.astype(np.float64) ** 2)))
                logger.info("%s: trying fallback %s → decoded rms=%.4f",
                            inst, src_name, fb_rms)
                if fb_rms >= DECODED_SILENCE_RMS:
                    for i in range(len(variants)):
                        variants[i] = window.clone()
                        wav_path = os.path.join(TIMBRE_CACHE_DIR, inst, f"{i}.wav")
                        sf.write(wav_path, fb_audio.T if fb_audio.ndim > 1 else fb_audio, 48000)
                    replaced_with = src_name
                    break
            if replaced_with:
                logger.info("%s: ALL silent (%.4f) → fell back to %s (rms=%.4f)",
                            inst, loudest_rms, replaced_with, fb_rms)
                continue
        if loudest_rms < DECODED_SILENCE_RMS:
            logger.warning("%s: ALL variants below silence floor (%.4f) and no fallback — keeping as-is",
                           inst, loudest_rms)
            continue
        replaced = []
        for i, r in rmses:
            if r < DECODED_SILENCE_RMS:
                variants[i] = variants[loudest_idx].clone()
                replaced.append(i)
                # Also replace the cached WAV
                src_wav = os.path.join(TIMBRE_CACHE_DIR, inst, f"{loudest_idx}.wav")
                dst_wav = os.path.join(TIMBRE_CACHE_DIR, inst, f"{i}.wav")
                try:
                    if os.path.exists(src_wav):
                        import shutil
                        shutil.copyfile(src_wav, dst_wav)
                except Exception:
                    pass
        if replaced:
            logger.info("%s: replaced silent variants %s with #%d (rms=%.4f)",
                        inst, replaced, loudest_idx, loudest_rms)
    logger.info("Timbre cache: %d new, %d already cached, audio rms by inst: %s",
                cached, skipped,
                {inst: round(max((decoded_rms.get((inst, i), 0.0)
                                  for i in range(len(v))), default=0.0), 4)
                 for inst, v in TIMBRE_VARIANT_LATENTS.items()})


DRUM_TIMBRE_FALLBACK = "drums"  # All drum subgroups borrow the existing
                                 # 'drums' preset until per-subgroup banks
                                 # are extracted from the dataset.
VOCAL_TIMBRE_FALLBACK = "voice"  # Same idea for vocal subgroups.


def _resolve_timbre_preset(spec):
    """Resolve a timbre_preset string to a [1, TIMBRE_REF_FRAMES, 64]
    latent tensor for use as the model's timbre reference. Accepted
    forms: 'instrument', 'instrument:variant_index'. Even though the
    cached preview WAV may be 30s long, the model only consumes the
    first TIMBRE_REF_FRAMES frames as conditioning."""
    if not spec:
        return None
    inst, _, var = spec.partition(":")
    var_idx = int(var) if var.isdigit() else 0
    # Drum subgroups borrow the shared 'drums' preset until per-kit
    # samples are added to instrument_presets_v3.
    if inst in DRUM_KIT_PROGRAMS and inst not in TIMBRE_VARIANT_LATENTS:
        inst = DRUM_TIMBRE_FALLBACK
    # Vocal subgroups borrow the shared 'voice' preset.
    if inst in VOCAL_PROGRAMS and inst not in TIMBRE_VARIANT_LATENTS:
        inst = VOCAL_TIMBRE_FALLBACK
    if inst in TIMBRE_VARIANT_LATENTS:
        variants = TIMBRE_VARIANT_LATENTS[inst]
        var_idx = max(0, min(var_idx, len(variants) - 1))
        full = variants[var_idx]  # [T, 64]
        head = full[:TIMBRE_REF_FRAMES]
        rms = float((head.float() ** 2).mean().sqrt().item())
        # Defensive: if the requested slice is still effectively silent
        # (slipped past the loader filter), find any non-silent variant.
        if rms < 0.05:
            for j, alt in enumerate(variants):
                alt_head = alt[:TIMBRE_REF_FRAMES]
                alt_rms = float((alt_head.float() ** 2).mean().sqrt().item())
                if alt_rms >= 0.05:
                    logger.warning("Timbre %s:%d silent (rms=%.4f), using :%d (rms=%.4f)",
                                   inst, var_idx, rms, j, alt_rms)
                    head = alt_head
                    break
        return head.unsqueeze(0)
    if inst in TIMBRE_PRESETS:
        return TIMBRE_PRESETS[inst][:TIMBRE_REF_FRAMES].unsqueeze(0)
    return None


def init_handler():
    """One-time AceStepHandler init + StemphonicTrainingModule construction."""
    global handler, module, _base_decoder_state
    from acestep.handler import AceStepHandler
    from stemphonic_trainer.training_module import StemphonicTrainingModule

    logger.info("🎵 Initializing AceStepHandler...")
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root="/scratch/ACE-Step-1.5",
        config_path="acestep-v15-sft",
        device="cuda",
    )
    logger.info("Handler init: %s (ok=%s)", status, ok)

    # Snapshot base decoder weights on CPU so we can restore between
    # checkpoint swaps without contamination from a prior LoRA merge.
    _base_decoder_state = {
        k: v.detach().to("cpu", copy=True)
        for k, v in handler.model.decoder.state_dict().items()
    }
    logger.info("Captured base decoder snapshot: %d tensors", len(_base_decoder_state))

    # Construct module + install MIDI hooks once. Hooks no-op when
    # _midi_features_for_hook is None (set per generation).
    module = StemphonicTrainingModule(
        model=handler.model,
        freeze_lower_layers=0,
        enable_midi=True, enable_pr_loss=False,
        cross_attn_lora=False,
    )
    module = module.to(device="cuda", dtype=torch.bfloat16)
    module.eval()
    module.install_midi_layer_hooks()
    logger.info("🎹 MIDI layer hooks installed (gated by features)")


def _restore_base_decoder():
    """Restore handler.model.decoder to its initial pre-checkpoint state."""
    if _base_decoder_state is None:
        return
    handler.model.decoder.load_state_dict(_base_decoder_state, strict=True)


def _apply_lora_ckpt(sd, config):
    """Stage 1 / LoRA-wrapped ckpts: temporarily peft-wrap handler.model,
    load both LoRA and full-FT keys, then merge_and_unload so handler.model
    ends up bare again with the merged weights baked in."""
    from peft import LoraConfig, get_peft_model

    # Discover LoRA hyperparams from saved keys
    lora_keys = {k: v for k, v in sd.items() if "lora_" in k.lower()}
    if not lora_keys:
        return 0, 0

    # Strip the training-time wrapper prefix "model." → leaves
    # "base_model.model.decoder.X" which matches a peft-wrapped model.
    def _strip(k):
        return k[6:] if k.startswith("model.") else k

    lora_sample = next(iter(lora_keys.values()))
    rank = config.get("lora_rank") or lora_sample.shape[0]
    alpha = config.get("lora_alpha") or rank
    targets = sorted({
        k.split(".lora_A")[0].split(".")[-1]
        for k in lora_keys if ".lora_A" in k
    })
    logger.info("LoRA config: r=%d alpha=%d targets=%s", rank, alpha, targets)

    peft_cfg = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=0.0, bias="none",
    )
    handler.model = get_peft_model(handler.model, peft_cfg)

    # Build a single state dict with stripped prefixes; peft expects
    # base_model.model.* paths which is exactly what we get after _strip.
    full_sd = {}
    for k, v in sd.items():
        # Skip extra adapter keys that don't belong to the decoder
        if k.startswith(("activity_", "resonance_", "midi_", "pr_head", "pitch2h")):
            continue
        full_sd[_strip(k)] = v

    result = handler.model.load_state_dict(full_sd, strict=False)
    n_loaded = len(full_sd) - len(result.unexpected_keys)
    logger.info("LoRA+FT loaded: %d/%d keys (missing=%d unexpected=%d)",
                n_loaded, len(full_sd),
                len(result.missing_keys), len(result.unexpected_keys))

    handler.model = handler.model.merge_and_unload()
    handler.model.eval()
    return n_loaded, len(lora_keys)


def load_checkpoint(ckpt_id):
    """Load decoder + adapter weights from a checkpoint id. Idempotent
    when ckpt_id == current_ckpt_id."""
    global current_ckpt_id, current_ckpt_meta
    if ckpt_id == current_ckpt_id:
        return current_ckpt_meta

    meta = CKPT_REGISTRY.get(ckpt_id)
    if not meta:
        raise ValueError(f"Unknown checkpoint: {ckpt_id}")

    local = _ensure_ckpt_downloaded(ckpt_id)
    logger.info("Loading checkpoint %s (%s)", ckpt_id, local)
    ckpt = torch.load(local, map_location="cpu", weights_only=True)
    step = ckpt.get("step", 0)
    config = ckpt.get("config", {})
    sd = ckpt.get("trainable_state_dict", ckpt.get("model_state_dict", ckpt))

    # Always restore the base decoder snapshot first so we never carry
    # weights from a previous LoRA merge into a new ckpt.
    _restore_base_decoder()

    has_lora = any("lora_" in k.lower() for k in sd.keys())
    if has_lora:
        # Stage 1 / LoRA-wrapped path: peft wrap → load → merge_and_unload
        n_loaded, n_lora = _apply_lora_ckpt(sd, config)
        logger.info("Loaded LoRA-wrapped ckpt: %d weight tensors (%d lora)",
                    n_loaded, n_lora)
    else:
        # Stage 2 / direct path: strip "model." prefix and load decoder weights
        model_keys = {k.replace("model.", "", 1): v for k, v in sd.items()
                      if k.startswith("model.")}
        if model_keys:
            result = handler.model.load_state_dict(model_keys, strict=False)
            logger.info("Decoder loaded: step=%d keys=%d unexpected=%d",
                        step, len(model_keys), len(result.unexpected_keys))
    handler.model.eval()

    # Adapter weights (MIDI bits) → module, only if this ckpt has them.
    # Older ckpts (v4, stage2b-30k) shipped a smaller MIDI adapter
    # (hidden=64) that doesn't fit the current module (hidden=256). Drop
    # any tensor whose shape doesn't match — the rest still applies.
    if meta["has_midi"]:
        adapter_keys = {k: v for k, v in sd.items()
                        if not k.startswith("model.") and "lora" not in k.lower()}
        if adapter_keys:
            module_sd = module.state_dict()
            compatible, dropped = {}, []
            for k, v in adapter_keys.items():
                if k in module_sd and tuple(module_sd[k].shape) == tuple(v.shape):
                    compatible[k] = v
                else:
                    dropped.append((k, tuple(v.shape),
                                    tuple(module_sd[k].shape) if k in module_sd else None))
            if compatible:
                module.load_state_dict(compatible, strict=False)
            logger.info("MIDI adapter loaded: %d/%d keys (%d dropped for shape mismatch)",
                        len(compatible), len(adapter_keys), len(dropped))
            if dropped:
                for k, ckpt_shape, cur_shape in dropped[:5]:
                    logger.info("  dropped %s: ckpt=%s current=%s", k, ckpt_shape, cur_shape)
        else:
            logger.warning("Ckpt %s declared has_midi but no adapter keys", ckpt_id)

    current_ckpt_id = ckpt_id
    current_ckpt_meta = {**meta, "step": step, "id": ckpt_id}
    logger.info("✅ Active checkpoint: %s (step=%d, midi=%s)",
                ckpt_id, step, meta["has_midi"])
    return current_ckpt_meta


def load_model():
    """Boot path: init handler, load default ckpt, then defer timbre
    precache to a background thread so /health goes live immediately.

    The timbre precache (_load_timbre_variants + precache_timbre_wavs)
    takes ~12s on an A100 because it VAE-decodes 100 variant latents
    (10 instruments x 10 variants) and runs silent-variant detection
    with v4_full fallback. It is NOT on the generation critical path:
      - /api/generate-stemphonic (run_generation) only consults timbre
        variants when params['timbre_preset'] is set -- default prompt
        generation works without them.
      - /api/generate-stemphonic/timbres and /timbre/<inst>/<variant>
        are the only consumers, and both check _timbre_ready and return
        a structured {"status": "initializing", "retry_after_seconds": 5}
        response while the background thread is still running, so the
        frontend can show a "Loading timbres..." placeholder instead of
        treating an empty list as "no timbres available".
    """
    init_handler()
    try:
        load_checkpoint(DEFAULT_CKPT_ID)
    except Exception as e:
        logger.error("Failed to load default ckpt %s: %s", DEFAULT_CKPT_ID, e)
        traceback.print_exc()

    def _bg_timbre_init():
        global _timbre_ready
        try:
            _load_timbre_variants()
            precache_timbre_wavs()
            _timbre_ready = True
            logger.info("🎵 Background timbre precache done")
        except Exception as e:
            logger.warning("Background timbre precache failed: %s", e)

    threading.Thread(target=_bg_timbre_init, daemon=True).start()
    logger.info("🎵 Stemphonic server ready (timbre precache deferred to bg)")


# ---------------------------------------------------------------------------
# MIDI utilities
# ---------------------------------------------------------------------------

def get_soundfont(instrument_group):
    if not instrument_group:
        return INSTRUMENT_SOUNDFONTS["default"]
    for key, path in INSTRUMENT_SOUNDFONTS.items():
        if key != "default" and key.lower() in instrument_group.lower():
            if os.path.exists(path):
                return path
    return INSTRUMENT_SOUNDFONTS["default"]


GM_PROGRAMS = {
    'acoustic_piano': 0, 'electric_piano': 4, 'keys': 0,
    'acoustic_guitar': 25, 'electric_guitar': 27, 'plucked': 25,
    'electric_bass': 33, 'bass': 33,
    'violin': 40, 'viola': 41, 'cello': 42, 'strings': 48,
    'trumpet': 56, 'french_horn': 60, 'trombone': 57, 'tuba': 58, 'brass': 61,
    'sax': 65, 'flute': 73, 'clarinet': 71, 'oboe': 68, 'bassoon': 70,
    'voice': 52, 'vocals': 52,
    'drums': 0,  # channel 10
}

# Drum subgroup → GM drum kit number. FluidR3_GM ships kits at programs
# 0/8/16/24/25/32/40/48/56 — Standard / Room / Power / Electronic /
# TR-808 / Jazz / Brush / Orchestra / SFX. Pretty_midi sets this via
# inst.program on a drum track (is_drum=True).
DRUM_KIT_PROGRAMS = {
    'drum_kit':   0,    # Standard kit
    'electronic': 24,   # Electronic kit
    'percussion': 48,   # Orchestra kit (timpani, congas, woodblock, etc.)
}

# Vocal subgroups → GM vocal program for fluidsynth render. The actual
# inference path uses the lyric text + word onsets; the SF render is just
# for the cover-mode latent base.
VOCAL_PROGRAMS = {
    'voice':       52,  # Choir Aahs
    'lead_vox':    52,
    'choir':       52,
    'bg_vox':      53,  # Voice Oohs
    'synth_vox':   54,  # Synth Voice
}
VOCAL_WORD_ONSET_CH = 144  # see midi_utils PITCHED_BINS + DRUM_BINS

PIPER_VOICE_DIR = "/scratch/piper_voices"
PIPER_VOICES = {
    "lead_vox":   "en_US-amy-medium.onnx",
    "voice":      "en_US-amy-medium.onnx",
    "bg_vox":     "en_US-amy-medium.onnx",
    "choir":      "en_US-amy-medium.onnx",
    "synth_vox":  "en_US-amy-medium.onnx",
}
_piper_voice_cache = {}


def _get_piper_voice(name):
    """Lazy-load + cache a PiperVoice by file name."""
    if name in _piper_voice_cache:
        return _piper_voice_cache[name]
    try:
        from piper import PiperVoice
        path = os.path.join(PIPER_VOICE_DIR, name)
        if not os.path.exists(path):
            return None
        v = PiperVoice.load(path)
        _piper_voice_cache[name] = v
        return v
    except Exception as e:
        logger.warning("Failed to load piper voice %s: %s", name, e)
        return None


def render_tts_aligned(lyric_map, total_duration_sec, sample_rate=48000,
                        instrument_group=None, midi_path=None):
    """Render each lyric word via piper-tts and time-place it at its
    `time` slot. When midi_path is given, each TTS word is pitch-shifted
    via librosa (PSOLA) to match the MIDI note's pitch — so the TTS
    actually SINGS the melody instead of being spoken on top.

    Returns a mono float32 numpy array.
    """
    if not lyric_map:
        return None
    voice_name = PIPER_VOICES.get(
        (instrument_group or "voice").lower(),
        "en_US-amy-medium.onnx",
    )
    voice = _get_piper_voice(voice_name)
    if voice is None:
        logger.warning("No piper voice available, TTS skipped")
        return None

    import wave
    import io
    import librosa

    # Build a time-sorted list of (note_time, midi_pitch, duration)
    # so each lyric word can be retuned to the corresponding note.
    note_events = []
    if midi_path and os.path.exists(midi_path):
        try:
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(midi_path)
            for inst in pm.instruments:
                if getattr(inst, "is_drum", False):
                    continue
                for note in inst.notes:
                    note_events.append((float(note.start), int(note.pitch),
                                        float(note.end - note.start)))
            note_events.sort(key=lambda x: x[0])
        except Exception as e:
            logger.warning("pitch-shift: failed to parse MIDI: %s", e)

    # Piper en_US-amy-medium is roughly C5/E5 = ~330–520 Hz spoken
    # baseline. midi 65 ≈ 349 Hz; we use that as the unshifted target
    # pitch and bend each word relative to it.
    BASE_MIDI = 65

    def _nearest_note(t):
        if not note_events:
            return None
        best = None
        best_d = 1e9
        for nt, p, dur in note_events:
            if nt <= t + 0.05:  # word starts at/after note (within 50 ms)
                d = abs(nt - t)
                if d < best_d:
                    best_d, best = d, (nt, p, dur)
        return best

    total_samples = int(total_duration_sec * sample_rate)
    out = np.zeros(total_samples, dtype=np.float32)
    rendered = 0
    for entry in lyric_map:
        try:
            word = (entry.get("lyric") or "").strip()
            if not word:
                continue
            t = float(entry.get("time", 0.0))
            start_sample = int(t * sample_rate)
            if start_sample >= total_samples:
                continue
            # 1. Run piper
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                voice.synthesize_wav(word, wf)
            buf.seek(0)
            wav_data, sr = sf.read(buf, dtype="float32")
            if wav_data.ndim > 1:
                wav_data = wav_data.mean(axis=1)
            if sr != sample_rate:
                wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=sample_rate)
            # 2. Find target note + retune the word
            note = _nearest_note(t)
            if note is not None:
                _nt, target_pitch, note_dur = note
                semitones = target_pitch - BASE_MIDI
                if abs(semitones) > 0.1:
                    try:
                        wav_data = librosa.effects.pitch_shift(
                            wav_data, sr=sample_rate, n_steps=semitones,
                        )
                    except Exception as e:
                        logger.warning("pitch_shift failed for %r: %s", word, e)
                # 3. Time-stretch to fit the note duration
                target_len = int(note_dur * sample_rate)
                if len(wav_data) > 0 and target_len > 0:
                    stretch_rate = len(wav_data) / target_len
                    if 0.4 < stretch_rate < 2.5:  # avoid extreme stretches
                        try:
                            wav_data = librosa.effects.time_stretch(
                                wav_data, rate=stretch_rate,
                            )
                        except Exception as e:
                            logger.warning("time_stretch failed for %r: %s", word, e)
            # 4. Per-word peak normalize
            peak = float(np.abs(wav_data).max())
            if peak > 1e-6:
                wav_data = wav_data / peak * 0.85
            end = min(start_sample + len(wav_data), total_samples)
            out[start_sample:end] += wav_data[:end - start_sample].astype(np.float32)
            rendered += 1
        except Exception as e:
            logger.warning("Piper TTS failed for %r: %s", entry, e)
    logger.info("   Piper TTS aligned: %d/%d words rendered into %.1fs buffer (voice=%s, pitched=%s)",
                rendered, len(lyric_map), total_duration_sec, voice_name,
                bool(note_events))
    return out


def channel_vocoder(carrier, modulator, sr=48000, n_bands=40,
                    f_lo=80.0, f_hi=11000.0):
    """STFT-based channel vocoder with:
      - Mel-spaced band filterbank (matches human hearing)
      - Pre-emphasized carrier (high-shelf to boost the choir's
        otherwise-rolled-off highs so consonant bands have something
        to modulate)
      - Voiced/unvoiced split: a parallel white-noise excitation
        feeds the upper bands during unvoiced frames (sibilants,
        fricatives) so 's', 't', 'sh' come through naturally
      - Envelope smoothing on the modulator side
      - Soft-knee compression on the output

    carrier:   stereo or mono float32 [samples] or [channels, samples]
    modulator: mono float32 [samples]
    Returns: stereo float32 [2, samples]
    """
    import librosa

    if carrier.ndim == 1:
        carrier_mono = carrier
    else:
        carrier_mono = carrier.mean(axis=0)
    n = min(len(carrier_mono), len(modulator))
    carrier_mono = carrier_mono[:n].astype(np.float32)
    modulator = modulator[:n].astype(np.float32)

    # 1. Pre-emphasize the carrier so high-frequency bands have signal
    #    to modulate. GM Choir Aahs rolls off above ~3 kHz.
    #    First-order high-shelf: y[n] = x[n] - 0.97*x[n-1] then add back
    #    a gentler version. Simpler: librosa.effects.preemphasis.
    carrier_pre = np.concatenate([
        [carrier_mono[0]],
        carrier_mono[1:] - 0.97 * carrier_mono[:-1],
    ]).astype(np.float32)
    # Mix pre-emphasized with original (60/40 — keep some body)
    carrier_mono = 0.6 * carrier_pre + 0.4 * carrier_mono

    # 2. Generate a white noise excitation source the same length —
    #    used for unvoiced frames in the upper bands.
    noise = (np.random.randn(n) * 0.5).astype(np.float32)

    # 3. STFT all three signals
    n_fft = 2048
    hop = 512
    C = librosa.stft(carrier_mono, n_fft=n_fft, hop_length=hop)
    M = librosa.stft(modulator,    n_fft=n_fft, hop_length=hop)
    N = librosa.stft(noise,        n_fft=n_fft, hop_length=hop)
    mag_C, phase_C = np.abs(C), np.angle(C)
    mag_M = np.abs(M)
    mag_N, phase_N = np.abs(N), np.angle(N)

    # 4. Mel-spaced band edges over the band range
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mel_edges = librosa.mel_frequencies(n_mels=n_bands + 1, fmin=f_lo, fmax=f_hi)
    band_idx = []
    for i in range(n_bands):
        lo, hi = mel_edges[i], mel_edges[i + 1]
        idxs = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(idxs) > 0:
            band_idx.append(idxs)

    # 5. Voicing estimate per frame from the modulator: ratio of
    #    low-frequency energy (≤1 kHz) to total. Voiced segments have
    #    most of their energy in the lower bands.
    low_cut = np.searchsorted(freqs, 1000.0)
    low_e = (mag_M[:low_cut] ** 2).sum(axis=0)
    tot_e = (mag_M ** 2).sum(axis=0) + 1e-9
    voiced = low_e / tot_e  # [T_frames], ~0.7+ for voiced, ~0.2 for fricatives
    # Smooth the voicing curve so individual frames don't flicker
    if len(voiced) > 7:
        k = np.ones(7) / 7
        voiced = np.convolve(voiced, k, mode='same')
    voiced = np.clip(voiced, 0.0, 1.0)
    unvoiced = 1.0 - voiced

    # 6. Per-band envelope transfer
    new_mag = np.zeros_like(mag_C)
    new_phase = np.copy(phase_C)
    eps = 1e-6
    for bi, idxs in enumerate(band_idx):
        # Modulator envelope for this band, smoothed
        env_mod = mag_M[idxs].mean(axis=0)  # [T_frames]
        if len(env_mod) > 9:
            k = np.ones(9) / 9
            env_mod = np.convolve(env_mod, k, mode='same')

        # Carrier band — what we'd normally modulate
        car_band = mag_C[idxs]
        car_avg = car_band.mean(axis=0) + eps

        # Noise band — used in upper bands during unvoiced frames so
        # sibilants come through. Lower bands stay tonal.
        is_upper = bi >= n_bands // 2
        if is_upper:
            noise_band = mag_N[idxs]
            noise_avg = noise_band.mean(axis=0) + eps
            # Mix carrier and noise per frame based on voicing
            mix_car = voiced
            mix_noi = unvoiced
            scale_c = (env_mod * mix_car) / car_avg
            scale_n = (env_mod * mix_noi) / noise_avg
            new_mag[idxs] = car_band * scale_c + noise_band * scale_n
            # In heavily unvoiced frames, swap in noise phase too so
            # carrier harmonics don't bleed through.
            blend = unvoiced[None, :]
            new_phase[idxs] = phase_C[idxs] * (1 - blend) + phase_N[idxs] * blend
        else:
            scale = env_mod / car_avg
            new_mag[idxs] = car_band * scale

    voc = librosa.istft(new_mag * np.exp(1j * new_phase),
                        hop_length=hop, length=n)

    # 7. Soft compress to even out dynamics
    peak = float(np.abs(voc).max())
    if peak > 1e-6:
        voc = voc / peak * 0.95
    # tanh limiter
    voc = np.tanh(voc * 1.2)
    # Final normalize
    peak2 = float(np.abs(voc).max())
    if peak2 > 1e-6:
        voc = voc / peak2 * 0.9

    return np.stack([voc, voc]).astype(np.float32)


def render_midi_to_audio(midi_path, output_dir, instrument_group=None,
                          lyric_map=None):
    """Render MIDI → WAV via FluidSynth with appropriate soundfont.
    Injects correct GM program change for the selected instrument."""
    import pretty_midi
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = instrument_group or "default"
    audio_path = out / f"{Path(midi_path).stem}_{tag}_rendered.wav"
    sf_path = get_soundfont(instrument_group)

    # Inject program change into MIDI so FluidSynth uses the right sound.
    # Drum subgroups (drum_kit/electronic/percussion) take a GM drum kit
    # program AND force is_drum=True so FluidSynth routes to channel 10.
    patched_midi = out / f"{Path(midi_path).stem}_patched.mid"
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        prog = 0
        is_drum_kit = False
        ig = (instrument_group or "").lower()
        if ig in DRUM_KIT_PROGRAMS:
            prog = DRUM_KIT_PROGRAMS[ig]
            is_drum_kit = True
        elif ig in VOCAL_PROGRAMS:
            # Vocal subgroups must be looked up directly — the
            # GM_PROGRAMS substring loop below would never match them
            # ('lead_vox' doesn't contain 'voice' or 'vocals'), so the
            # fallback would silently render acoustic piano (prog=0).
            prog = VOCAL_PROGRAMS[ig]
        elif instrument_group:
            for key, val in GM_PROGRAMS.items():
                if key in ig:
                    prog = val
                    break
            is_drum_kit = ('drum' in ig)
        for inst in pm.instruments:
            inst.program = prog
            inst.is_drum = is_drum_kit
        pm.write(str(patched_midi))
        midi_to_render = str(patched_midi)
        logger.info("🎹 Patched MIDI with program=%d is_drum=%s for %s",
                    prog, is_drum_kit, instrument_group)
    except Exception as e:
        logger.warning("Could not patch MIDI: %s, using original", e)
        midi_to_render = str(midi_path)

    # Primary: FluidSynth with real instrument soundfonts
    logger.info("🎹 FluidSynth: %s → %s (sf: %s)", Path(midi_path).name, audio_path.name, Path(sf_path).name)
    try:
        result = subprocess.run(
            ["fluidsynth", "-ni", "-T", "wav", "-g", "0.625", "-r", "48000",
             "-F", str(audio_path), sf_path, midi_to_render],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and audio_path.exists():
            # Check if render has audio content
            data_check, sr_check = sf.read(str(audio_path), dtype="float32")
            if np.abs(data_check).max() > 0.001:
                logger.info("🎹 FluidSynth render OK: max=%.3f", np.abs(data_check).max())
                # Vocal mode: vocode the GM voice carrier with TTS modulator
                # to give the lyrics actual phonetic articulation. Digital
                # talk-box: GM voice provides the harmonic carrier (chord/
                # melody), TTS provides the spectral envelope per word.
                vocal_subgroups = {"voice", "lead_vox", "bg_vox", "choir", "synth_vox"}
                if (instrument_group or "").lower() in vocal_subgroups and lyric_map:
                    try:
                        # New vocal rendering: piper-tts per word, each word
                        # pitch-shifted via PSOLA to the matching MIDI note's
                        # pitch and time-stretched to its duration. The
                        # result IS the singing — no channel vocoder needed,
                        # which sounded harsh/distorted regardless of tuning.
                        # The GM choir carrier is now blended in only as a
                        # subtle pad for body (10/90 mix carrier/voice).
                        carrier = data_check.T if data_check.ndim > 1 else data_check
                        if carrier.ndim > 1:
                            carrier_mono = carrier.mean(axis=0)
                        else:
                            carrier_mono = carrier
                        n_samples = len(carrier_mono)
                        dur = n_samples / sr_check
                        sung = render_tts_aligned(
                            lyric_map, dur, sr_check,
                            instrument_group=instrument_group,
                            midi_path=midi_to_render,
                        )
                        if sung is not None and float(np.abs(sung).max()) > 1e-4:
                            # Pad/crop sung to carrier length
                            if len(sung) < n_samples:
                                sung = np.concatenate([sung, np.zeros(n_samples - len(sung), dtype=np.float32)])
                            else:
                                sung = sung[:n_samples]
                            # Mix: 90% pitched TTS, 10% choir for body. Both
                            # peak-normalized first.
                            sung_peak = float(np.abs(sung).max())
                            if sung_peak > 1e-6:
                                sung = sung / sung_peak * 0.95
                            car_peak = float(np.abs(carrier_mono).max())
                            if car_peak > 1e-6:
                                carrier_n = carrier_mono / car_peak * 0.95
                            else:
                                carrier_n = carrier_mono
                            mixed = 0.9 * sung + 0.1 * carrier_n
                            # Final peak limit
                            mp = float(np.abs(mixed).max())
                            if mp > 1e-6:
                                mixed = mixed / mp * 0.9
                            voc_stereo = np.stack([mixed, mixed]).astype(np.float32)
                            voc_path = out / f"{Path(midi_path).stem}_{tag}_vocoded.wav"
                            sf.write(str(voc_path), voc_stereo.T, sr_check)
                            logger.info("🎤 Sung vocal render: %s (voice=%s, sung_rms=%.3f)",
                                        voc_path.name, instrument_group,
                                        float(np.sqrt(np.mean(sung**2))))
                            return str(voc_path)
                    except Exception as e:
                        logger.warning("Vocal render failed (%s) — falling back to plain GM render", e)
                return str(audio_path)
            else:
                logger.warning("FluidSynth rendered silence, trying pretty_midi")
    except Exception as e:
        logger.warning("FluidSynth error: %s", e)

    # Fallback: pretty_midi.synthesize()
    import pretty_midi
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_to_render))
        audio_data = pm.synthesize(fs=48000)
        if np.abs(audio_data).max() > 0:
            audio_data = audio_data / np.abs(audio_data).max() * 0.9
        audio_stereo = np.stack([audio_data, audio_data])
        sf.write(str(audio_path), audio_stereo.T, 48000)
        logger.info("🎹 pretty_midi fallback: max=%.3f", np.abs(audio_data).max())
        return str(audio_path)
    except Exception as e:
        logger.warning("pretty_midi also failed: %s", e)
    raise RuntimeError(f"Could not render MIDI to audio: {midi_path}")


# GM percussion note → stemphonic drum-roll channel offset (0..15).
# Mirrors stemphonic_trainer.midi_utils.DRUM_KEY_TO_CHANNEL via the
# GM Standard Drum Kit mapping. Channels 128..143 in the [146, T]
# tensor — see midi_utils:38-65.
GM_DRUM_NOTE_TO_CHANNEL = {
    35: 0, 36: 0,                  # acoustic/electric kick
    38: 1, 40: 1, 37: 12,          # snare / electric snare; rim shot → rim
    42: 2, 44: 2,                  # closed hihat / pedal hihat
    46: 3,                         # open hihat
    48: 4, 50: 4,                  # high tom 1
    45: 5, 47: 5,                  # mid tom 2
    41: 6, 43: 6,                  # low tom 3
    49: 7, 57: 7,                  # crash
    51: 8, 59: 8,                  # ride / ride 2
    52: 9,                         # china
    55: 10,                        # splash
    53: 11,                        # ride bell
    39: 12,                        # hand clap → rim
    54: 4, 56: 7, 58: 9,           # tambourine→t1, cowbell→crash, vibraslap→china
    # any other GM percussion note falls into channel 15 ("other")
}
DRUM_OTHER_CHANNEL = 15
PITCHED_BINS = 128
DRUM_BINS = 16


# Latent-native drum sub-separation + onset detection.
#
# Pipeline:
#   [T, 64] full-mix latent
#       → LatentDemucs      (isolate drum stem latent)
#       → LatentDrumsep     (split drum latent into 6 stem latents)
#       → LatentVisual      (per-frame peak envelope, ~RMS)
#       → peak-pick the envelope's first difference → onset frames
#
# No VAE decode, no wav roundtrip, no librosa.onset. The drum roll at
# the latent frame rate (25 fps = 40 ms/frame) is written directly.
DRUMSEP_TO_CHANNEL = {
    "kick":  0,   # → channel 0
    "snare": 1,   # → channel 1
    "hh":    2,   # → channel 2 (closed hihat)
    "toms":  4,   # → channel 4 (lumped tom rows)
    "ride":  8,   # → channel 8
    "crash": 7,   # → channel 7
}
DRUMSEP_CACHE_DIR = "/scratch/cache/drumsep_onsets"
os.makedirs(DRUMSEP_CACHE_DIR, exist_ok=True)

_latent_drumsep_runtime = {"rt": None}
_latent_visual_runtime  = {"rt": None}


def _get_latent_drumsep():
    # Returns None if the drumsep checkpoint isn't baked (post-2026-04-11
    # wipe — ckpt lost, not recoverable from backups). Callers must treat
    # None as "drum sub-stem separation unavailable" and fall back to the
    # whole-drum stem.
    if _latent_drumsep_runtime["rt"] == "unavailable":
        return None
    if _latent_drumsep_runtime["rt"] is None:
        try:
            from latent_drumsep.infer import LatentDrumsepRuntime
            _latent_drumsep_runtime["rt"] = LatentDrumsepRuntime.get()
            logger.info("Loaded LatentDrumsep student (6 stems: kick/snare/toms/hh/ride/crash)")
        except FileNotFoundError as e:
            logger.warning("LatentDrumsep checkpoint missing — drum sub-separation disabled: %s", e)
            _latent_drumsep_runtime["rt"] = "unavailable"
            return None
    return _latent_drumsep_runtime["rt"]


def _get_latent_visual():
    if _latent_visual_runtime["rt"] is None:
        from latent_visual.infer import LatentVisualRuntime
        _latent_visual_runtime["rt"] = LatentVisualRuntime.get()
        logger.info("Loaded LatentVisual envelope model (62K-param peak predictor)")
    return _latent_visual_runtime["rt"]


def drumsep_latent_to_drum_roll(full_latent, T, fps=25):
    """Latent-native drum onset extraction.

    full_latent: [T_audio, 64] VAE latent of the full mix (torch.Tensor or np).
    T:           target drum-roll frame count (matches generation length).
    fps:         drum-roll frame rate — must equal the latent frame rate,
                 which is 48000/1920 = 25 fps.

    Returns: [16, T] float32 drum onset roll.
    """
    drum_roll = np.zeros((16, T), dtype=np.float32)
    try:
        # Stage 1: LatentDemucs → isolate drum stem latent
        ld = _get_latent_demucs()
        if ld is None:
            logger.warning("drumsep skipped: latent_demucs unavailable")
            return drum_roll
        L = full_latent if isinstance(full_latent, torch.Tensor) else torch.from_numpy(full_latent)
        if L.dim() == 2 and L.shape[0] == 64 and L.shape[1] != 64:
            L = L.transpose(0, 1)                        # → [T, 64]
        stems = ld.split(L) if hasattr(ld, "split") else None
        if stems is None:
            # latent_demucs runtime exposes .separate on waveform; for
            # latent input we need its .split if present, otherwise bail
            logger.warning("drumsep skipped: latent_demucs has no latent-input API")
            return drum_roll
        L_drum = stems.get("drums") if isinstance(stems, dict) else None
        if L_drum is None:
            logger.warning("drumsep skipped: no 'drums' stem from latent_demucs")
            return drum_roll

        # Stage 2: LatentDrumsep → split drum latent into 6 sub-stem latents.
        # Returns an empty drum roll when the ckpt is missing (post-
        # 2026-04-11 wipe) — callers only get the whole-drum stem.
        rt_drum = _get_latent_drumsep()
        if rt_drum is None:
            logger.warning("drumsep skipped: LatentDrumsep checkpoint unavailable")
            return drum_roll
        substems = rt_drum.split(L_drum)                  # {name: [T, 64]}

        # Stage 3: LatentVisual envelope → peak-pick onsets per stem
        rt_vis = _get_latent_visual()
        for name, L_stem in substems.items():
            ch = DRUMSEP_TO_CHANNEL.get(name)
            if ch is None:
                continue
            try:
                onset_frames = rt_vis.onsets(L_stem)
            except Exception as e:
                logger.warning("latent_visual onsets failed for %s: %s", name, e)
                continue
            for f in onset_frames:
                if 0 <= f < T:
                    drum_roll[ch, int(f)] = 1.0

        n_hits = int((drum_roll > 0).sum())
        logger.info("   drumsep (latent): %d hits across %d non-empty channels",
                    n_hits, int((drum_roll.sum(axis=1) > 0).sum()))
    except Exception as e:
        logger.warning("drumsep (latent) failed: %s", e)
    return drum_roll


def midi_tensor_to_pretty_midi(midi_tensor, fps=25, drum_mode=False):
    """Convert a [1, 146, T] midi_tensor (pitched roll on 0..127, drum
    onset roll on 128..143, vocal extras on 144..145) into a
    pretty_midi.PrettyMIDI object so it can be saved as a real .mid
    file the user can download/edit/reload."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    if midi_tensor is None:
        return pm
    arr = midi_tensor[0].cpu().numpy() if hasattr(midi_tensor, 'cpu') else midi_tensor[0]
    T = arr.shape[1]
    # Pitched roll → instrument 0
    pitched_roll = arr[:128]  # [128, T]
    nz_p = (pitched_roll > 0).any(axis=1)
    if nz_p.any():
        inst = pretty_midi.Instrument(program=0, is_drum=False, name="pitched")
        for pitch in range(128):
            row = pitched_roll[pitch]
            if not (row > 0).any():
                continue
            in_note = False
            note_start = 0
            for f in range(T):
                if row[f] > 0 and not in_note:
                    in_note = True
                    note_start = f
                    vel = max(1, min(127, int(row[f] * 127)))
                elif row[f] == 0 and in_note:
                    in_note = False
                    inst.notes.append(pretty_midi.Note(
                        velocity=vel,
                        pitch=pitch,
                        start=note_start / fps,
                        end=f / fps,
                    ))
            if in_note:
                inst.notes.append(pretty_midi.Note(
                    velocity=vel, pitch=pitch,
                    start=note_start / fps, end=T / fps,
                ))
        pm.instruments.append(inst)
    # Drum onset roll (channels 128..143) → drum instrument
    drum_roll = arr[PITCHED_BINS:PITCHED_BINS + DRUM_BINS]  # [16, T]
    nz_d = (drum_roll > 0).any(axis=1)
    if nz_d.any():
        # Inverse of GM_DRUM_NOTE_TO_CHANNEL — pick the most common
        # GM note for each drum class.
        CHANNEL_TO_GM = {
            0: 36, 1: 38, 2: 42, 3: 46, 4: 48, 5: 45,
            6: 41, 7: 49, 8: 51, 9: 52, 10: 55, 11: 53,
            12: 37, 13: 38, 14: 38, 15: 39,
        }
        dinst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
        for ch in range(DRUM_BINS):
            row = drum_roll[ch]
            note = CHANNEL_TO_GM.get(ch, 35)
            for f in range(T):
                v = row[f]
                if v > 0:
                    dinst.notes.append(pretty_midi.Note(
                        velocity=max(1, min(127, int(v * 127))),
                        pitch=note,
                        start=f / fps,
                        end=(f + 1) / fps,  # short hit
                    ))
        pm.instruments.append(dinst)
    return pm


def save_task_midi(midi_tensor, task_dir, task_id):
    """Persist the input midi_tensor as a .mid file alongside the task
    output. Returned URL is exposed in the generation result + a
    direct download endpoint."""
    if midi_tensor is None:
        return None
    try:
        pm = midi_tensor_to_pretty_midi(midi_tensor)
        if not pm.instruments:
            return None
        out_path = os.path.join(task_dir, "input.mid")
        pm.write(out_path)
        return f"/api/generate-stemphonic/download/{task_id}/input.mid"
    except Exception as e:
        logger.warning("save_task_midi failed: %s", e)
        return None


def add_lyric_word_onsets(midi_tensor, lyric_map, T, fps=25):
    """Populate channel 144 (word onsets) from a per-note lyric map.

    lyric_map: list of {'time': float_seconds, 'lyric': str} entries.
    Mirrors the training-time `_add_vocal_timing` channel-144 build —
    1.0 at each word-start frame, 0 elsewhere. Empty/whitespace-only
    lyrics are skipped (notes without an assigned syllable).
    """
    if not lyric_map:
        return midi_tensor
    n_set = 0
    for entry in lyric_map:
        try:
            lyric = (entry.get("lyric") or "").strip()
            if not lyric:
                continue
            t = float(entry.get("time", 0.0))
            f = int(round(t * fps))
            if 0 <= f < T:
                midi_tensor[0, VOCAL_WORD_ONSET_CH, f] = 1.0
                n_set += 1
        except Exception:
            continue
    logger.info("   Lyric word onsets: %d / %d entries set on ch144", n_set, len(lyric_map))
    return midi_tensor


def load_real_midi(midi_path, T=400, fps=25, drum_mode=False):
    """Load .mid → [1, 146, T] piano roll tensor.

    By default builds a pitched piano roll on channels 0..127 (the
    instrument flow). When drum_mode=True or any track has
    `inst.is_drum`, builds the drum onset roll on channels 128..143
    via GM_DRUM_NOTE_TO_CHANNEL — matching how training packed drums
    in stemphonic_trainer.midi_utils.get_stem_midi_representation.
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    midi_tensor = torch.zeros(1, 146, T)

    has_drum_track = drum_mode or any(getattr(inst, "is_drum", False)
                                      for inst in pm.instruments)

    if has_drum_track:
        # Drum onset roll: each hit lights one frame on the mapped channel
        # (training uses onset markers, not sustained notes — see
        # midi_utils.drum_onsets_to_roll). 0..15 → tensor channels 128..143.
        drum_roll = np.zeros((DRUM_BINS, T), dtype=np.float32)
        n_hits = 0
        for inst in pm.instruments:
            # When drum_mode is forced from the API, treat ALL tracks as
            # drum (the user explicitly chose drum mode). Otherwise honour
            # the per-track is_drum flag.
            if not (drum_mode or getattr(inst, "is_drum", False)):
                continue
            for note in inst.notes:
                f = int(note.start * fps)
                if 0 <= f < T:
                    ch = GM_DRUM_NOTE_TO_CHANNEL.get(note.pitch, DRUM_OTHER_CHANNEL)
                    drum_roll[ch, f] = max(drum_roll[ch, f], note.velocity / 127.0)
                    n_hits += 1
        midi_tensor[0, PITCHED_BINS:PITCHED_BINS + DRUM_BINS, :] = torch.from_numpy(drum_roll)
        logger.info("   Drum MIDI: %d hits across %d non-empty channels",
                    n_hits, int((drum_roll.sum(axis=1) > 0).sum()))
    else:
        # Pitched piano roll
        roll = np.zeros((128, T), dtype=np.float32)
        for inst in pm.instruments:
            if getattr(inst, "is_drum", False):
                continue
            for note in inst.notes:
                s = int(note.start * fps)
                e = min(int(note.end * fps), T)
                if s < T:
                    roll[note.pitch, s:e] = note.velocity / 127.0
        midi_tensor[0, :128, :] = torch.from_numpy(roll)
    return midi_tensor


def audio_to_fsq_tokens(audio_path, handler):
    """Render audio → VAE latents → FSQ tokens via handler pipeline."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim == 1:
        data = np.stack([data, data])  # mono → stereo [2, samples]
    else:
        data = data.T  # sf returns [samples, channels] → [channels, samples]
        if data.shape[0] == 1:
            data = np.concatenate([data, data], axis=0)  # mono → stereo
    if sr != 48000:
        import torchaudio
        data_t = torch.from_numpy(data)
        data_t = torchaudio.transforms.Resample(sr, 48000)(data_t)
        data = data_t.numpy()
    wav = torch.from_numpy(data)  # [2, samples]

    device = "cuda"
    dtype = torch.bfloat16

    with torch.inference_mode():
        # VAE encode → REAL VAE latents [T', 64]. These are what the
        # diffusion process operates on; renoise(), training noise sampling,
        # and the final decoder output all live in this space.
        latents = handler.vae.encode(
            wav.unsqueeze(0).to(device, dtype=dtype)
        ).latent_dist.sample().squeeze(0)  # [64, T']
        latents_seq = latents.permute(1, 0)  # [T', 64]

        # SEPARATELY produce the FSQ-roundtripped latents. These live in a
        # related but DIFFERENT subspace (the lm_hints space the model uses
        # for cover-mode semantic context). They go through context_latents
        # via precomputed_lm_hints_25Hz, NOT as the renoise base — using
        # them as src_latents produces silent diffusion output.
        attention_mask = torch.ones(latents_seq.shape[0], dtype=torch.bool, device=device)
        quantized, indices, _ = handler.model.tokenize(
            latents_seq.unsqueeze(0),
            handler.silence_latent,
            attention_mask.unsqueeze(0),
        )
        roundtripped = handler.model.detokenize(quantized).squeeze(0)  # [T'', 64]
        logger.info("FSQ tokenized: %d tokens from %s (raw latents %s, roundtrip lm_hints %s)",
                    indices.numel(), Path(audio_path).name,
                    tuple(latents_seq.shape), tuple(roundtripped.shape))
        # Return: (token_indices, raw_vae_latents, fsq_roundtripped_lm_hints)
        return indices, latents_seq, roundtripped


# ---------------------------------------------------------------------------
# Generation (same as probe_full_v3.generate_with_model_api)
# ---------------------------------------------------------------------------

def generate_training_style(prompt, midi_tensor=None, duration=16.0, steps=50, cfg=7.0, seed=42):
    """Replicates the training-time preview generation exactly.
    Uses simple from-scratch Euler denoising with encoder/decoder directly."""
    device = "cuda"
    dtype = torch.bfloat16
    T = int(duration * 25)
    D = 64

    from stemphonic_trainer.preprocess_v4 import format_sft_prompt, format_lyrics
    text_prompt = format_sft_prompt(prompt, duration_sec=int(duration))
    lyric_prompt = format_lyrics("[Instrumental]")

    with torch.inference_mode():
        # Text encoding
        text_tokens = handler.text_tokenizer(
            text_prompt, return_tensors="pt", padding="max_length",
            max_length=256, truncation=True,
        ).to(device)
        text_hs = handler.text_encoder(
            input_ids=text_tokens["input_ids"], lyric_attention_mask=None,
        ).last_hidden_state.to(dtype)
        text_mask = text_tokens["attention_mask"].to(dtype)

        lyric_tokens = handler.text_tokenizer(
            lyric_prompt, return_tensors="pt", padding="max_length",
            max_length=512, truncation=True,
        ).to(device)
        lyric_hs = handler.text_encoder.embed_tokens(lyric_tokens["input_ids"]).to(dtype)
        lyric_mask = lyric_tokens["attention_mask"].to(dtype)

        # Timbre ref: zeros (training preview default)
        refer = torch.zeros(1, 30, D, device=device, dtype=dtype)
        refer_order = torch.zeros(1, device=device, dtype=torch.long)

        # Build encoder state (what training preview does)
        enc_hs, enc_mask = handler.model.encoder(
            text_hidden_states=text_hs,
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hs,
            lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=refer,
            refer_audio_order_mask=refer_order,
        )

        # Context latents: [zeros | ones] = from-scratch mode
        context = torch.cat([
            torch.zeros(1, T, D, device=device, dtype=dtype),
            torch.ones(1, T, D, device=device, dtype=dtype),
        ], dim=-1)  # [1, T, 128]

        # Null conditioning for CFG
        null_enc = handler.model.null_condition_emb
        if null_enc is not None:
            null_enc_hs = null_enc.expand(1, -1, -1).to(device, dtype=dtype)
            null_enc_mask = torch.ones(1, null_enc_hs.shape[1], device=device, dtype=dtype)
        else:
            null_enc_hs = torch.zeros_like(enc_hs)
            null_enc_mask = torch.ones_like(enc_mask)

        # MIDI hooks (layer-wise injection)
        if midi_tensor is not None:
            midi_feat = module.build_midi_features(
                midi_tensor.to(device, dtype=dtype), torch.tensor([True]),
                B=1, T=T, device=device, dtype=dtype,
            )
            patch_size = getattr(handler.model.decoder, 'patch_size', 2)
            T_dec = T // patch_size
            mf = midi_feat
            if mf.shape[1] > T_dec:
                mf = mf[:, :T_dec * patch_size].reshape(1, T_dec, patch_size, -1).mean(dim=2)
            elif mf.shape[1] < T_dec:
                mf = F.pad(mf, (0, 0, 0, T_dec - mf.shape[1]))
            module._midi_features_for_hook = mf
        else:
            module._midi_features_for_hook = None

        # Euler denoising with CFG (matches training preview)
        torch.manual_seed(seed)
        xt = torch.randn(1, T, D, device=device, dtype=dtype)
        dt = 1.0 / steps
        attn_mask = torch.ones(1, T, device=device, dtype=dtype)

        for i in range(steps):
            t_val = 1.0 - i * dt
            t = torch.tensor([t_val], device=device, dtype=dtype)

            pred_cond = handler.model.decoder(
                hidden_states=xt, timestep=t, timestep_r=t,
                attention_mask=attn_mask,
                encoder_hidden_states=enc_hs, encoder_attention_mask=enc_mask,
                context_latents=context,
            )[0]
            pred_uncond = handler.model.decoder(
                hidden_states=xt, timestep=t, timestep_r=t,
                attention_mask=attn_mask,
                encoder_hidden_states=null_enc_hs, encoder_attention_mask=null_enc_mask,
                context_latents=context,
            )[0]
            pred = pred_uncond + cfg * (pred_cond - pred_uncond)
            xt = xt - dt * pred

        module._midi_features_for_hook = None
        latent = xt.squeeze(0).permute(1, 0)  # [D, T]
        audio = handler.vae.decode(latent.unsqueeze(0)).sample

    return audio.squeeze(0).cpu().float().numpy()


def generate_with_model_api(prompt, midi_tensor=None, fsq_tokens=None,
                            timbre_ref=None, cover_noise_strength=0.0,
                            lm_hints_25hz=None,
                            duration=16.0, steps=50, cfg=7.0, seed=42):
    """Generate using model.generate_audio() with FSQ + MIDI hooks."""
    device = "cuda"
    dtype = torch.bfloat16
    T = int(duration * 25)
    D = 64

    # Use training-exact prompt format — duration_sec=32.0 (training default)
    from stemphonic_trainer.preprocess_v4 import format_sft_prompt, format_lyrics
    text_prompt = format_sft_prompt(prompt, duration_sec=32.0)
    lyric_prompt = format_lyrics("[Instrumental]")

    with torch.inference_mode():
        text_tokens = handler.text_tokenizer(
            text_prompt, return_tensors="pt", padding="max_length",
            max_length=256, truncation=True,
        ).to(device)
        text_hs = handler.text_encoder(
            input_ids=text_tokens["input_ids"], lyric_attention_mask=None,
        ).last_hidden_state
        text_mask = text_tokens["attention_mask"].to(dtype)

        lyric_tokens = handler.text_tokenizer(
            lyric_prompt, return_tensors="pt", padding="max_length",
            max_length=512, truncation=True,
        ).to(device)
        lyric_hs = handler.text_encoder.embed_tokens(lyric_tokens["input_ids"])
        lyric_mask = lyric_tokens["attention_mask"].to(dtype)

        # Timbre ref: training uses REF_FRAMES=30 (1.2s @ 25Hz) — random non-silent window
        REF_FRAMES = 30
        if timbre_ref is not None:
            refer = timbre_ref.to(device, dtype=dtype)
            if refer.dim() == 2:
                refer = refer.unsqueeze(0)
            # Crop to 30 frames (training format)
            if refer.shape[1] > REF_FRAMES:
                # Pick a random-ish 30-frame window from middle of preset
                start = (refer.shape[1] - REF_FRAMES) // 2
                refer = refer[:, start:start + REF_FRAMES]
            elif refer.shape[1] < REF_FRAMES:
                refer = F.pad(refer, (0, 0, 0, REF_FRAMES - refer.shape[1]))
        else:
            refer = torch.zeros(1, REF_FRAMES, D, device=device, dtype=dtype)
        refer_order = torch.zeros(1, device=device, dtype=torch.long)

        src_latents = handler.silence_latent[:, :T, :].expand(1, -1, -1).to(device, dtype=dtype)
        if cover_noise_strength > 0 and lm_hints_25hz is not None:
            src_latents = lm_hints_25hz.to(device, dtype=dtype)
            if src_latents.dim() == 2:
                src_latents = src_latents.unsqueeze(0)
            if src_latents.shape[1] < T:
                src_latents = F.pad(src_latents, (0, 0, 0, T - src_latents.shape[1]))
            else:
                src_latents = src_latents[:, :T]

        # Build chunk_masks with MIDI-derived activity mask
        # Training zeros out chunk_masks[:, :, 64:128] where the target stem is silent.
        # In inference, we derive activity from the MIDI: frames with any pitch active = 1.
        # This tells the model "only generate audio in these frames" — matches training dist.
        chunk_masks = torch.ones(1, T, D, device=device, dtype=dtype)
        if midi_tensor is not None:
            # Derive activity from MIDI piano roll: any pitch/drum active per frame
            mt = midi_tensor.to(device, dtype=dtype)
            if mt.dim() == 3:  # [1, 146, T]
                activity = (mt[0, :, :T].sum(dim=0) > 0).to(dtype=dtype)  # [T]
            else:
                activity = torch.ones(T, device=device, dtype=dtype)
            # Pad if short
            if activity.shape[0] < T:
                activity = F.pad(activity, (0, T - activity.shape[0]))
            # Smooth: dilate activity by 4 frames on each side to preserve sustain
            # (MIDI note-off may be earlier than natural decay — especially bass)
            if activity.sum() > 0:
                kernel = torch.ones(1, 1, 9, device=device, dtype=dtype)
                smoothed = F.conv1d(activity.view(1, 1, -1), kernel, padding=4).squeeze()
                activity = (smoothed > 0).to(dtype=dtype)
            chunk_masks = chunk_masks * activity.unsqueeze(0).unsqueeze(-1)  # [1, T, 1] → [1, T, D]

        # FSQ → precomputed hints
        precomputed_hints = None
        is_covers = torch.zeros(1, device=device, dtype=torch.long)
        if fsq_tokens is not None:
            fsq_gpu = fsq_tokens.to(device, dtype=dtype)
            if fsq_gpu.dim() == 3:  # [1, T, 2048] raw tokens
                hints = handler.model.detokenize(fsq_gpu)
            elif fsq_gpu.shape[-1] == 2048:
                hints = handler.model.detokenize(fsq_gpu.unsqueeze(0))
            elif fsq_gpu.shape[-1] == 64:
                hints = fsq_gpu.unsqueeze(0) if fsq_gpu.dim() == 2 else fsq_gpu
            else:
                hints = handler.model.detokenize(fsq_gpu.unsqueeze(0))
            if hints.shape[1] < T:
                hints = F.pad(hints, (0, 0, 0, T - hints.shape[1]))
            else:
                hints = hints[:, :T]
            precomputed_hints = hints
            is_covers = torch.ones(1, device=device, dtype=torch.long)

        # MIDI hooks
        if midi_tensor is not None:
            midi_feat = module.build_midi_features(
                midi_tensor.to(device, dtype=dtype), torch.tensor([True]),
                B=1, T=T, device=device, dtype=dtype,
            )
            patch_size = getattr(handler.model.decoder, 'patch_size', 2)
            T_dec = T // patch_size
            mf = midi_feat
            if mf.shape[1] > T_dec:
                mf = mf[:, :T_dec * patch_size].reshape(1, T_dec, patch_size, -1).mean(dim=2)
            elif mf.shape[1] < T_dec:
                mf = F.pad(mf, (0, 0, 0, T_dec - mf.shape[1]))
            module._midi_features_for_hook = mf
        else:
            module._midi_features_for_hook = None

        outputs = handler.model.generate_audio(
            text_hidden_states=text_hs,
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hs,
            lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=refer,
            refer_audio_order_mask=refer_order,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            silence_latent=handler.silence_latent,
            seed=seed,
            infer_method="ode",
            infer_steps=steps,
            diffusion_guidance_scale=cfg,
            precomputed_lm_hints_25Hz=precomputed_hints,
            use_progress_bar=False,
            cover_noise_strength=cover_noise_strength,
        )

        module._midi_features_for_hook = None
        pred_latents = outputs["target_latents"]

    latent = pred_latents.squeeze(0).permute(1, 0)  # [D, T]
    with torch.no_grad():
        audio = handler.vae.decode(latent.unsqueeze(0)).sample
    return audio.squeeze(0).cpu().float().numpy()


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def run_generation(task_id, params):
    try:
        tasks[task_id]["status"] = "processing"
        task_dir = os.path.join(OUTPUT_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)

        raw_prompt = params.get("prompt", "a melodic beat with warm synths")
        instrument = params.get("instrument", "")
        # Use training-format caption if instrument is selected, else raw prompt
        prompt = TRAINING_CAPTIONS.get(instrument, raw_prompt)
        if instrument and instrument not in TRAINING_CAPTIONS:
            # Fuzzy match
            for key, cap in TRAINING_CAPTIONS.items():
                if instrument.lower() in key or key in instrument.lower():
                    prompt = cap
                    break
        duration = min(float(params.get("duration", 16)), 120.0)
        steps = int(params.get("steps", 50))
        cfg = float(params.get("cfg", params.get("cfg_scale", 7.0)))
        seed = int(params.get("seed", -1))
        noise_level = float(params.get("noise_level", 0.0))
        cover_strength = float(params.get("cover_noise_strength", 0.0))
        if seed < 0:
            seed = torch.randint(0, 2**31, (1,)).item()

        # Switch checkpoint if requested (under model_lock to serialize w/ generation)
        requested_ckpt = params.get("checkpoint") or params.get("ckpt")
        if requested_ckpt and requested_ckpt != current_ckpt_id:
            with model_lock:
                load_checkpoint(requested_ckpt)

        midi_path = params.get("midiFile_path") or params.get("midi_file_path")
        ref_path = params.get("refAudio_path") or params.get("ref_audio_path") or params.get("conditioningAudio_path")
        T = int(duration * 25)
        ckpt_has_midi = bool(current_ckpt_meta and current_ckpt_meta.get("has_midi"))
        # Drum subgroups (see DRUM_KIT_PROGRAMS) route MIDI to channels
        # 128..143 instead of the pitched piano roll. Frontend can also
        # send an explicit drum_mode flag (e.g. when a track was created
        # in drum-roll view but the active instrument tab is something
        # else) — that overrides the instrument-based check.
        is_drum_inst = (instrument or "").lower() in DRUM_KIT_PROGRAMS
        drum_mode_flag = str(params.get("drum_mode", "")).lower() in ("1", "true", "yes")
        if drum_mode_flag:
            is_drum_inst = True

        # Vocal mode: instrument is one of {voice, lead_vox, bg_vox, choir,
        # synth_vox} OR explicit vox_mode flag. In vocal mode the per-note
        # lyric map populates channel 144 (word onsets) and the lyrics
        # string is fed through embed_tokens via format_lyrics — see
        # midi_utils._add_vocal_timing for the training counterpart.
        is_vocal_inst = (instrument or "").lower() in VOCAL_PROGRAMS
        vox_mode_flag = str(params.get("vox_mode", "")).lower() in ("1", "true", "yes")
        if vox_mode_flag:
            is_vocal_inst = True
        # Parse the lyric map (sent as JSON string)
        lyric_map = []
        raw_lyric_map = params.get("lyric_map")
        if raw_lyric_map:
            try:
                import json as _json
                lyric_map = _json.loads(raw_lyric_map) if isinstance(raw_lyric_map, str) else raw_lyric_map
                if not isinstance(lyric_map, list):
                    lyric_map = []
            except Exception as e:
                logger.warning("Failed to parse lyric_map: %s", e)
                lyric_map = []
        # The lyric STRING for format_lyrics — either provided directly
        # in `lyrics`, or built from the lyric_map by joining all
        # non-empty entries in time order.
        lyrics_text = params.get("lyrics") or ""
        if not lyrics_text and lyric_map:
            try:
                ordered = sorted(lyric_map, key=lambda e: float(e.get("time", 0.0)))
                lyrics_text = " ".join(
                    str(e.get("lyric", "")).strip() for e in ordered if str(e.get("lyric", "")).strip()
                )
            except Exception:
                pass
        if not lyrics_text:
            lyrics_text = "[Instrumental]"

        midi_tensor = None
        fsq_tokens = None
        timbre_ref = None
        lm_hints = None
        render_src = None  # raw VAE latents of soundfont/ref render (renoise base)

        # ── Cover-src latents path: meter-repaint and similar tasks pass
        # an already-spliced latent .pt instead of asking the server to
        # re-encode an audio file. When set, we load the latent and use
        # it as the render_src + lm_hints. Skips the MIDI render path.
        cover_src_lat_path = params.get("cover_src_latents_path")
        if cover_src_lat_path and os.path.exists(cover_src_lat_path):
            try:
                blob = torch.load(cover_src_lat_path, map_location="cuda")
                lat = blob["latents"].to("cuda", dtype=torch.bfloat16)  # [T, 64]
                render_src = lat
                lm_hints = lat
                fsq_tokens = lat  # cover-mode fallback; model.tokenize would refine
                T = lat.shape[0]
                # Build timbre ref from first 30 frames of the latent itself
                timbre_ref = lat[:30, :].unsqueeze(0)
                logger.info("🎚️  Loaded cover_src_latents from %s, T=%d", cover_src_lat_path, T)
            except Exception as e:
                logger.warning("failed to load cover_src_latents_path %s: %s", cover_src_lat_path, e)

        # --- MIDI input: piano roll for hooks + optional FSQ from render ---
        if midi_path and Path(midi_path).suffix.lower() in ('.mid', '.midi'):
            logger.info("🎹 Processing MIDI input: %s", Path(midi_path).name)

            # 1. Try the latent-soundfont fast path first: precached
            #    per-note latents → MIDI walked into latent tracks → no
            #    fluidsynth render, no vae.encode at request time. Falls
            #    back to the legacy render+encode path on any failure
            #    (missing latent SF, unmapped pitches, vocal mode).
            raw_latents = None
            lm_hints_render = None
            rendered_path = None
            tried_latent_sf = False
            if not is_vocal_inst:
                try:
                    raw_latents = _latent_sf_synthesize_midi(midi_path, instrument)
                    tried_latent_sf = True
                except Exception as _lsf_err:
                    logger.warning("latent-sf synth failed (%s); using fluidsynth", _lsf_err)

            if raw_latents is None:
                rendered_path = render_midi_to_audio(
                    midi_path, task_dir, instrument,
                    lyric_map=lyric_map if is_vocal_inst else None,
                )
                # 2. VAE encode rendered audio → BOTH raw VAE latents (used as
                # the renoise base for cover_noise_strength) AND FSQ-roundtripped
                # latents (used as the lm_hints semantic context). These live
                # in different latent subspaces and must NOT be conflated —
                # using the FSQ-roundtripped tensor as the renoise base
                # produces silent diffusion output.
                with model_lock:
                    _, raw_latents, lm_hints_render = audio_to_fsq_tokens(rendered_path, handler)
            else:
                # latent-sf path: produce lm_hints by FSQ-roundtripping the
                # raw latents (cheap, no encode needed since we already
                # have the raw latents in VAE space).
                with model_lock:
                    attn = torch.ones(raw_latents.shape[0], dtype=torch.bool, device=raw_latents.device)
                    quantized, _, _ = handler.model.tokenize(
                        raw_latents.unsqueeze(0).to(handler.model.device, dtype=handler.model.dtype),
                        handler.silence_latent,
                        attn.unsqueeze(0),
                    )
                    lm_hints_render = handler.model.detokenize(quantized).squeeze(0).float().cpu()
                logger.info("🎹 latent-soundfont synth: %s frames=%d (no fluidsynth)",
                            instrument, raw_latents.shape[0])
            render_energy = raw_latents.abs().mean().item()
            if render_energy > 0.3:
                # fsq_tokens here is the lm-hints input for context_latents.
                # Pass the FSQ-roundtripped tensor to honour the training
                # distribution of the cover semantic context.
                fsq_tokens = lm_hints_render
                # render_src is the renoise base — must be raw VAE latents.
                render_src = raw_latents
                logger.info("   Render energy=%.3f → using as FSQ + cover src", render_energy)
            else:
                render_src = None
                logger.info("   Render energy=%.3f (likely sine) → skipping FSQ", render_energy)
            lm_hints = raw_latents  # legacy var for downstream gating

            # 3. MIDI piano roll for adapter hooks (pitch/rhythm conditioning).
            # Only built when the current checkpoint has MIDI hooks; for
            # stage1 (no_midi) we keep only the FSQ-from-render path above.
            if ckpt_has_midi:
                midi_tensor = load_real_midi(midi_path, T=T, drum_mode=is_drum_inst)
                # Vocal mode: stamp word onsets onto channel 144 from the
                # per-note lyric map. Pitched roll on channels 0..127 still
                # carries the melody.
                if is_vocal_inst and lyric_map:
                    midi_tensor = add_lyric_word_onsets(midi_tensor, lyric_map, T)
                if is_drum_inst:
                    # Check drum channels 128..143 instead of pitched 0..127
                    active = (midi_tensor[0, 128:144].sum(dim=1) > 0).sum().item()
                    logger.info("   Drum MIDI: %d active classes, render latents: %s",
                                active, raw_latents.shape)
                else:
                    active = (midi_tensor[0, :128].sum(dim=1) > 0).sum().item()
                    logger.info("   MIDI: %d active pitches, render latents: %s",
                                active, raw_latents.shape)
                if active == 0 and not is_vocal_inst:
                    # In vocal mode keep midi_tensor even with no pitched
                    # notes — the channel-144 word onsets are still useful.
                    midi_tensor = None
            else:
                logger.info("   Stage1 ckpt — skipping MIDI piano-roll, using FSQ render only")

            # Use real-recording presets for timbre ref AND FSQ hints
            # Explicit timbre_preset (e.g. "violin:3") wins for the timbre
            # reference; FSQ preset is still picked by fuzzy instrument match.
            explicit_timbre_latent = _resolve_timbre_preset(params.get("timbre_preset"))
            preset_key = None
            for key in FSQ_PRESETS:
                if key in (instrument or '').lower():
                    preset_key = key
                    break
            if preset_key is None and instrument:
                for key in FSQ_PRESETS:
                    if instrument.lower() in key or key in instrument.lower():
                        preset_key = key
                        break
            # Only fall back to the static FSQ preset when the user's MIDI
            # render didn't yield usable FSQ (e.g. silent/low-energy render).
            # Otherwise the rendered MIDI's actual content drives generation.
            if fsq_tokens is None and preset_key and preset_key in FSQ_PRESETS:
                fsq_tokens = FSQ_PRESETS[preset_key]
                logger.info("   Render too weak — falling back to static FSQ preset: %s [%s]",
                            preset_key, fsq_tokens.shape)
            elif fsq_tokens is not None:
                logger.info("   Using rendered MIDI FSQ (%s frames)", fsq_tokens.shape[0])
            if explicit_timbre_latent is not None:
                timbre_ref = explicit_timbre_latent
                logger.info("   Using explicit timbre: %s", params.get("timbre_preset"))
            elif preset_key and preset_key in TIMBRE_PRESETS:
                timbre_ref = TIMBRE_PRESETS[preset_key].unsqueeze(0)
                logger.info("   Using timbre preset: %s", preset_key)

        # --- Ref audio (non-MIDI): real audio → VAE encode → LatentPitch ---
        elif ref_path and os.path.exists(ref_path):
            logger.info("🎵 Processing ref audio: %s", ref_path)

            # VAE encode first (needed by LatentPitch AND drumsep below).
            # Serial here: LatentPitch is <1 ms after VAE finishes, and
            # both contend for the same GPU so parallelism doesn't help.
            with model_lock:
                _, raw_latents, lm_hints_ref = audio_to_fsq_tokens(ref_path, handler)

            bp_midi_path = None
            if ckpt_has_midi:
                try:
                    rt_pitch = _get_latent_pitch()
                    pm = rt_pitch.transcribe(raw_latents.float())
                    bp_midi_path = os.path.join(
                        task_dir, Path(ref_path).stem + "_latentpitch.mid"
                    )
                    pm.write(bp_midi_path)
                except Exception as e:
                    logger.warning("LatentPitch failed: %s", e)
                    bp_midi_path = None

            # Drum subgroups: latent-native drumsep (latent_demucs →
            # latent_drumsep → latent_visual peak-pick) writing directly
            # into channels 128..143 of the drum roll.
            drumsep_roll = None
            if is_drum_inst and ckpt_has_midi:
                T_local = raw_latents.shape[0]
                drumsep_roll = drumsep_latent_to_drum_roll(raw_latents, T_local)

            fsq_tokens = lm_hints_ref       # context goes through FSQ-roundtripped
            render_src = raw_latents        # renoise base = raw VAE latents
            timbre_ref = raw_latents[:30, :].unsqueeze(0)
            lm_hints = raw_latents
            logger.info("   Audio → VAE (raw=%s, lm_hints=%s), timbre_ref=first 30 frames",
                        tuple(raw_latents.shape), tuple(lm_hints_ref.shape))

            # Build MIDI piano-roll from BasicPitch output for the per-layer hook
            if bp_midi_path and ckpt_has_midi:
                try:
                    midi_tensor = load_real_midi(bp_midi_path, T=T)
                    active = (midi_tensor[0, :128].sum(dim=1) > 0).sum().item()
                    logger.info("   BasicPitch → %d active pitches over %d frames",
                                active, T)
                except Exception as e:
                    logger.warning("Failed to convert BasicPitch MIDI to roll: %s", e)
                    midi_tensor = None
            # Stamp drumsep onsets into channels 128..143 (drums) on the
            # same midi_tensor (creating one if needed). Crop/pad the
            # drumsep roll to the model's frame count (T) since the
            # input audio's natural length may differ from the
            # requested generation duration.
            if drumsep_roll is not None and drumsep_roll.sum() > 0:
                if midi_tensor is None:
                    midi_tensor = torch.zeros(1, 146, T)
                src = drumsep_roll  # [16, T_audio]
                if src.shape[1] >= T:
                    src = src[:, :T]
                else:
                    pad = np.zeros((16, T - src.shape[1]), dtype=src.dtype)
                    src = np.concatenate([src, pad], axis=1)
                midi_tensor[0, PITCHED_BINS:PITCHED_BINS + DRUM_BINS, :] = torch.from_numpy(src)
                logger.info("   drumsep stamped into ch128..143 (T=%d)", T)
            # If we still have nothing in either channel range, drop midi_tensor
            if midi_tensor is not None:
                pitched_active = (midi_tensor[0, :128].sum(dim=1) > 0).sum().item()
                drum_active = (midi_tensor[0, 128:144].sum(dim=1) > 0).sum().item()
                if pitched_active == 0 and drum_active == 0 and not is_vocal_inst:
                    midi_tensor = None

        # If user explicitly chose a timbre preset and nothing else set
        # timbre_ref (e.g. text-only or MIDI w/o matching FSQ preset), use it.
        if timbre_ref is None:
            t = _resolve_timbre_preset(params.get("timbre_preset"))
            if t is not None:
                timbre_ref = t
                logger.info("   Using explicit timbre preset: %s", params.get("timbre_preset"))

        # cover_noise_strength blends src_latents (rendered audio) with noise.
        # Only works when src_latents has real content (non-silent soundfont render).
        effective_noise = cover_strength  # use dedicated slider value
        if effective_noise > 0 and lm_hints is not None:
            hint_energy = lm_hints.abs().mean().item()
            if hint_energy < 0.01:  # soundfont render is silent — override to 0
                logger.info("   Soundfont render silent (energy=%.4f), forcing cover_noise=0", hint_energy)
                effective_noise = 0.0
            else:
                logger.info("   Soundfont render energy=%.4f, cover_noise=%.2f", hint_energy, effective_noise)
        elif effective_noise > 0:
            logger.info("   No src latents for cover_noise, forcing to 0")
            effective_noise = 0.0

        # audio_cover_strength<1 always crashes the decoder during the
        # cover→non-cover encoder swap because we don't pass the
        # non_cover_text_hidden_states arg to model.generate_audio.
        # Force 1.0 unconditionally — the slider's range is wrong for
        # this code path, the model effectively only supports cover=1.
        audio_cover = float(params.get("audio_cover_strength", 1.0))
        if audio_cover < 1.0:
            logger.warning("audio_cover_strength=%.2f forced to 1.0 (non-cover swap unsupported)", audio_cover)
            audio_cover = 1.0

        logger.info("🎵 Generating: prompt=%r dur=%.1f steps=%d cfg=%.1f seed=%d noise=%.2f(eff=%.2f) cov=%.2f midi=%s fsq=%s",
                     prompt[:60], duration, steps, cfg, seed, noise_level, effective_noise,
                     audio_cover, midi_tensor is not None, fsq_tokens is not None)

        with model_lock:
            # Use train-matched inference (direct decoder call, same as training_step)
            audio_np = generate_stemphonic_trainmatch(
                handler, module,
                prompt=prompt,
                lyrics=lyrics_text,
                midi_tensor=midi_tensor,
                timbre_ref=timbre_ref,
                fsq_raw=fsq_tokens,
                cover_src_latents=render_src,
                duration=duration, steps=steps, cfg=cfg, seed=seed,
                cover_noise_strength=effective_noise,
                audio_cover_strength=audio_cover,
            )

        if audio_np is None:
            raise RuntimeError("Generation returned None")

        filename = f"stemphonic_{seed}.wav"
        out_path = os.path.join(task_dir, filename)
        sf.write(out_path, audio_np.T, 48000)

        file_paths = [f"/api/generate-stemphonic/download/{task_id}/{filename}"]
        input_files = {}

        # Encode the generated audio back to a VAE latent and save it so
        # the studio can cache it per-track and reuse it for repaint /
        # cover-noise / meter-change tasks without re-encoding.
        try:
            _, gen_latents, _ = audio_to_fsq_tokens(out_path, handler)
            latent_id = uuid.uuid4().hex[:12]
            latent_path = os.path.join(LATENT_DIR, f"{latent_id}.pt")
            torch.save({
                "latents": gen_latents.cpu().float(),
                "shape": list(gen_latents.shape),
                "fps": 25,
                "source": filename,
                "task_id": task_id,
            }, latent_path)
            input_files["latent"] = f"/api/latents/download/{latent_id}.pt"
            input_files["latent_id"] = latent_id
        except Exception as e:
            logger.warning("post-gen latent encode failed: %s", e)

        # Include rendered MIDI audio for frontend preview
        rendered = Path(task_dir).glob("*_rendered.wav")
        for r in rendered:
            input_files["rendered_midi"] = f"/api/generate-stemphonic/download/{task_id}/{r.name}"

        # Persist the actual MIDI tensor used for generation as a .mid
        # file the user can download/edit/reload via the right sidebar.
        midi_url = save_task_midi(midi_tensor, task_dir, task_id)
        if midi_url:
            input_files["midi"] = midi_url

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = {"file_paths": file_paths, "input_files": input_files}
        logger.info("✅ Task %s completed", task_id)

    except Exception as e:
        traceback.print_exc()
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/api/generate-stemphonic", methods=["POST"])
def generate():
    task_id = str(uuid.uuid4())

    if request.content_type and "multipart" in request.content_type:
        params = {}
        for key in ["prompt", "lyrics", "duration", "steps", "cfg", "seed",
                     "instrument", "noise_level", "cover_noise_strength",
                     "audio_cover_strength", "drum_mode", "vox_mode",
                     "lyric_map",
                     "checkpoint", "ckpt", "timbre_preset"]:
            val = request.form.get(key)
            if val is not None:
                params[key] = val
        for field in ["midiFile", "midi_file", "refAudio", "ref_audio", "conditioningAudio"]:
            f = request.files.get(field)
            if f:
                task_dir = os.path.join(OUTPUT_DIR, task_id)
                os.makedirs(task_dir, exist_ok=True)
                fpath = os.path.join(task_dir, f.filename)
                f.save(fpath)
                params[field + "_path"] = fpath
    else:
        params = request.json or {}

    tasks[task_id] = {"status": "pending", "params": params, "created": time.time()}
    threading.Thread(target=run_generation, args=(task_id, params)).start()
    return jsonify({"task_id": task_id})


@app.route("/api/generate-stemphonic/task/<task_id>", methods=["GET"])
def task_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({
        "status": task["status"],
        "state": "SUCCESS" if task["status"] == "completed" else
                 "FAILURE" if task["status"] == "failed" else "PENDING",
        "result": task.get("result"),
        "error": task.get("error"),
    })


@app.route("/api/generate-stemphonic/download/<task_id>/<filename>", methods=["GET"])
def download(task_id, filename):
    filepath = os.path.join(OUTPUT_DIR, task_id, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, mimetype="audio/wav", as_attachment=True,
                     download_name=filename)


@app.route("/api/generate-stemphonic/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": handler is not None,
        "midi_hooks": module is not None,
        "current_checkpoint": current_ckpt_id,
        "current_checkpoint_meta": current_ckpt_meta,
    })


@app.route("/api/generate-stemphonic/checkpoints", methods=["GET"])
def list_checkpoints():
    """List available checkpoints + the currently-loaded one."""
    items = [
        {
            "id": cid,
            "label": meta["label"],
            "has_midi": meta["has_midi"],
            "downloaded": os.path.exists(_local_ckpt_path(cid)),
        }
        for cid, meta in CKPT_REGISTRY.items()
    ]
    return jsonify({
        "checkpoints": items,
        "current": current_ckpt_id,
        "default": DEFAULT_CKPT_ID,
    })


@app.route("/api/generate-stemphonic/timbres", methods=["GET"])
def list_timbres():
    """List timbre presets nested as instrument → variants. Drum
    subgroups (drum_kit/electronic/percussion) currently alias the
    shared 'drums' preset cache."""
    if not _timbre_ready:
        return jsonify({
            "status": "initializing",
            "retry_after_seconds": 5,
            "timbres": [],
        })
    items = []
    for inst, var_list in TIMBRE_VARIANT_LATENTS.items():
        variants = []
        for i in range(len(var_list)):
            wav = os.path.join(TIMBRE_CACHE_DIR, inst, f"{i}.wav")
            variants.append({
                "variant": i,
                "audio_url": f"/api/generate-stemphonic/timbre/{inst}/{i}" if os.path.exists(wav) else None,
            })
        items.append({
            "key": inst,
            "label": inst.replace("_", " ").title(),
            "variants": variants,
            "has_fsq_preset": inst in FSQ_PRESETS,
        })
    # Synthesize drum-subgroup entries that proxy the shared drums cache
    def _alias_to(fallback_key, sub_keys):
        if fallback_key not in TIMBRE_VARIANT_LATENTS:
            return
        var_list = TIMBRE_VARIANT_LATENTS[fallback_key]
        for sub in sub_keys:
            if sub == fallback_key:
                continue
            sub_variants = []
            for i in range(len(var_list)):
                wav = os.path.join(TIMBRE_CACHE_DIR, fallback_key, f"{i}.wav")
                sub_variants.append({
                    "variant": i,
                    "audio_url": f"/api/generate-stemphonic/timbre/{fallback_key}/{i}" if os.path.exists(wav) else None,
                })
            items.append({
                "key": sub,
                "label": sub.replace("_", " ").title(),
                "variants": sub_variants,
                "has_fsq_preset": False,
                "alias": fallback_key,
            })
    _alias_to(DRUM_TIMBRE_FALLBACK, list(DRUM_KIT_PROGRAMS.keys()))
    _alias_to(VOCAL_TIMBRE_FALLBACK, list(VOCAL_PROGRAMS.keys()))
    return jsonify({"timbres": items})


@app.route("/api/generate-stemphonic/timbre/<inst>/<int:variant>", methods=["GET"])
def get_timbre_wav(inst, variant):
    if not _timbre_ready:
        return jsonify({
            "status": "initializing",
            "retry_after_seconds": 5,
        }), 503
    wav = os.path.join(TIMBRE_CACHE_DIR, inst, f"{variant}.wav")
    if not os.path.exists(wav):
        return jsonify({"error": "not cached"}), 404
    return send_file(wav, mimetype="audio/wav")


@app.route("/api/generate-stemphonic/checkpoint", methods=["POST"])
def switch_checkpoint():
    """Eagerly switch the active checkpoint (downloads if missing)."""
    data = request.json or {}
    ckpt_id = data.get("checkpoint") or data.get("ckpt")
    if not ckpt_id or ckpt_id not in CKPT_REGISTRY:
        return jsonify({"error": f"Unknown checkpoint: {ckpt_id}"}), 400
    try:
        with model_lock:
            meta = load_checkpoint(ckpt_id)
        return jsonify({"ok": True, "current": current_ckpt_id, "meta": meta})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# Demucs 6-stem separation
# =============================================================================
# Mirrors the legacy /separate-stems endpoint signature so the existing
# studio TrackInfoSidebar code (handleSeparateIntoStems) keeps working
# unchanged. Async path: POST returns {status:'processing', task_id};
# the frontend polls /separate-stems/status/<id> until completed, then
# reads the {stems: {drums,bass,other,vocals,guitar,piano}} dict where
# each value is a download URL routed back through this same server.

DEMUCS_OUTPUT_DIR = "/scratch/stemphonic_outputs/separations"
os.makedirs(DEMUCS_OUTPUT_DIR, exist_ok=True)
DEMUCS_TASKS = {}  # {task_id: {status, stems, error, audio_path}}
demucs_lock = threading.Lock()
# htdemucs_6s removed — latent_demucs student is the only separation path.


# ── Latent editor (sample-accurate splice/concat for hh quantization) ─
_latent_editor_runtime = {"rt": None}

def _get_latent_editor():
    """Lazy-load the LatentEditorRuntime for latent meter-change paths."""
    if _latent_editor_runtime["rt"] is None:
        from latent_editor.infer import LatentEditorRuntime
        rt = LatentEditorRuntime("/scratch/latent_editor_ckpts/editor_final.pt", device="cuda")
        _latent_editor_runtime["rt"] = rt
        logger.info("Loaded LatentEditorRuntime")
    return _latent_editor_runtime["rt"]


# Silence latent frame: trained "silence" point in the VAE distribution.
# We use it to initialize per-bar L_out buffers in drum_meter_change so
# regions that should be silent decode to actual silence (NEVER zeros —
# zeros decode to garbage in the Oobleck VAE's learned space).
_silence_frame_cache = {"frame": None}

from typing import Optional, Dict as _Dict
_latent_sf_cache: _Dict[str, dict] = {}
LATENT_SOUNDFONT_DIR = "/scratch/latent_soundfonts"

def _load_latent_sf(instrument: str):
    """Lazy-load + cache a precached latent soundfont per instrument."""
    if instrument in _latent_sf_cache:
        return _latent_sf_cache[instrument]
    p = os.path.join(LATENT_SOUNDFONT_DIR, f"{instrument}.pt")
    if not os.path.exists(p):
        return None
    from latent_soundfont.synth import load_latent_soundfont
    blob = load_latent_soundfont(p)
    _latent_sf_cache[instrument] = blob
    logger.info("Loaded latent soundfont %s (%d notes)", instrument, len(blob.get("notes", {})))
    return blob


def _latent_sf_synthesize_midi(midi_path: str, instrument: str) -> Optional[torch.Tensor]:
    """Synthesize MIDI directly into a single [T, 64] raw VAE latent
    using the precached per-note latents. Returns None if the latent
    soundfont for this instrument isn't available. Polyphony is folded
    into one latent track via decode-of-each-voice → wav-sum →
    re-encode-once. (Latent-sum doesn't equal audio-sum so we go through
    the audio domain at the mix step only.)"""
    blob = _load_latent_sf(instrument)
    if blob is None:
        return None
    from latent_soundfont.synth import latent_synthesize_midi
    sframe = _get_silence_frame()
    editor_rt = _get_latent_editor()
    tracks = latent_synthesize_midi(midi_path, blob, sframe, editor=editor_rt)
    if not tracks:
        return None
    if len(tracks) == 1:
        return tracks[0]
    # Polyphony: decode each voice on GPU, sum waveforms, re-encode once.
    # This single re-encode is still cheaper than the legacy path's
    # render+encode and avoids the non-linear latent-sum problem.
    with torch.no_grad():
        audios = []
        for tr in tracks:
            z = tr.to("cuda").to(torch.bfloat16).permute(1, 0).unsqueeze(0)
            a = handler.vae.decode(z).sample.squeeze(0).cpu().float().numpy()  # [2, S]
            audios.append(a)
        ml = max(a.shape[1] for a in audios)
        mix = np.zeros((2, ml), dtype=np.float32)
        for a in audios:
            mix[:, :a.shape[1]] += a
        peak = float(np.abs(mix).max())
        if peak > 0.95:
            mix *= 0.95 / peak
        y = torch.from_numpy(mix).float().unsqueeze(0).to("cuda").to(torch.bfloat16)
        L = handler.vae.encode(y).latent_dist.sample().squeeze(0).transpose(0, 1).float().cpu()
    return L


# ── Latent Demucs student (waveform → 4 stem latents in one pass) ──
_latent_demucs_runtime = {"rt": None}

def _get_latent_demucs():
    if _latent_demucs_runtime["rt"] is None:
        from latent_demucs.runtime import LatentDemucsRuntime
        _latent_demucs_runtime["rt"] = LatentDemucsRuntime.get()
        logger.info("Loaded LatentDemucs student model")
    return _latent_demucs_runtime["rt"]


_latent_pitch_runtime = {"rt": None}

def _get_latent_pitch():
    """Lazy singleton for the latent → BasicPitch student (drop-in
    replacement for basic_pitch.inference.predict)."""
    if _latent_pitch_runtime["rt"] is None:
        from latent_pitch.infer import LatentPitchRuntime
        _latent_pitch_runtime["rt"] = LatentPitchRuntime(
            "/scratch/latent_pitch_ckpts/pitch_final.pt"
        )
        logger.info("Loaded LatentPitch student (latent → MIDI)")
    return _latent_pitch_runtime["rt"]


def _save_raw_latent(L_tensor, stem_type, source_path=None, parent=None):
    """Persist a [T, 64] latent to LATENT_DIR with the same metadata
    schema as `_encode_audio_path_to_latent` so the rest of the pipeline
    treats it identically. No vae.encode call — used by the latent
    demucs path which already returns latents directly."""
    latent_id = uuid.uuid4().hex[:12]
    latent_path = os.path.join(LATENT_DIR, f"{latent_id}.pt")
    blob = {
        "latents": L_tensor.cpu().float(),
        "shape": list(L_tensor.shape),
        "fps": 25,
        "source": "latent_demucs",
        "stem_type": stem_type,
        "vae_version": _vae_version_hash(),
    }
    if source_path:
        blob["source_path"] = source_path
    if parent:
        blob["parent"] = parent
    torch.save(blob, latent_path)
    return {
        "latent_id": latent_id,
        "latent_path": latent_path,
        "n_frames": int(L_tensor.shape[0]),
        "fps": 25,
        "source_path": source_path,
        "stem_type": stem_type,
    }


def _get_silence_frame():
    if _silence_frame_cache["frame"] is None:
        sl_raw = torch.load(
            "/scratch/ACE-Step-1.5/checkpoints/acestep-v15-sft/silence_latent.pt",
            weights_only=True,
        ).transpose(1, 2).float()  # [1, T, 64]
        _silence_frame_cache["frame"] = sl_raw[0, 0:1, :].clone()  # [1, 64]
        logger.info("Loaded silence latent frame")
    return _silence_frame_cache["frame"]


def _encode_audio_path_to_latent(audio_path, stem_type=None, source_path=None):
    """Encode an audio file via the loaded handler's VAE and persist
    the result to /scratch/cache/latents/<id>.pt. Returns the latent
    metadata dict {latent_id, latent_path, n_frames, fps, source_path}.
    """
    if handler is None:
        raise RuntimeError("VAE handler not loaded")
    _, raw_latents, _ = audio_to_fsq_tokens(audio_path, handler)
    latent_id = uuid.uuid4().hex[:12]
    latent_path = os.path.join(LATENT_DIR, f"{latent_id}.pt")
    torch.save({
        "latents": raw_latents.cpu().float(),
        "shape": list(raw_latents.shape),
        "fps": 25,
        "source": Path(audio_path).name,
        "source_path": source_path or audio_path,
        "stem_type": stem_type,
    }, latent_path)
    return {
        "latent_id": latent_id,
        "latent_path": latent_path,
        "n_frames": int(raw_latents.shape[0]),
        "fps": 25,
        "source_path": source_path or audio_path,
        "stem_type": stem_type,
    }


def _run_demucs_separation(task_id, audio_path):
    """Worker: separate audio into 6 stems via htdemucs_6s, encode each
    stem to a VAE latent, and persist both WAVs and latent_ids.

    The frontend downloads stem WAVs for playback/waveform display and
    uses latent_ids for downstream generation (repaint, regen, etc.).
    """
    try:
        DEMUCS_TASKS[task_id]["status"] = "processing"
        out_dir = os.path.join(DEMUCS_OUTPUT_DIR, task_id)
        os.makedirs(out_dir, exist_ok=True)
        stem_urls: dict = {}
        stem_latents: dict = {}

        # ── Real demucs 6-stem separation ──
        import demucs.api
        separator = demucs.api.Separator(model="htdemucs_6s", device="cuda")
        logger.info("[%s] htdemucs_6s separation starting", task_id)
        _, sources = separator.separate_audio_file(Path(audio_path))
        stem_names = list(sources.keys())
        logger.info("[%s] htdemucs_6s produced %d stems: %s",
                    task_id, len(stem_names), stem_names)

        for stem_name in stem_names:
            stem_audio = sources[stem_name]  # torch tensor [channels, samples]
            stem_path = os.path.join(out_dir, f"{stem_name}.wav")
            import torchaudio
            torchaudio.save(stem_path, stem_audio.cpu(), separator.samplerate)
            stem_urls[stem_name] = f"/separate-stems/download/{task_id}/{stem_name}.wav"
            logger.info("  %s → %s", stem_name, stem_path)

            # Encode to VAE latent
            try:
                meta = _encode_audio_path_to_latent(
                    stem_path, stem_type=stem_name, source_path=audio_path,
                )
                stem_latents[stem_name] = meta
                logger.info("  %s encoded → latent %s (%d frames)",
                            stem_name, meta["latent_id"], meta["n_frames"])
            except Exception as enc_err:
                logger.warning("  %s encode failed: %s", stem_name, enc_err)

        DEMUCS_TASKS[task_id]["stems"] = stem_urls
        DEMUCS_TASKS[task_id]["stem_latents"] = stem_latents
        DEMUCS_TASKS[task_id]["separator"] = "htdemucs_6s"
        DEMUCS_TASKS[task_id]["status"] = "completed"
        logger.info("✅ htdemucs_6s task %s done: %s", task_id, stem_names)
    except Exception as e:
        traceback.print_exc()
        DEMUCS_TASKS[task_id]["status"] = "failed"
        DEMUCS_TASKS[task_id]["error"] = str(e)


def _drumsep_and_encode_substems(drums_latent):
    """Split a drum-stem latent into 6 sub-stem latents via
    LatentDrumsep, persist each via _save_raw_latent. Returns
    dict[substem_name -> latent metadata]. Entirely in latent space —
    no VAE decode, no wav I/O.
    """
    rt = _get_latent_drumsep()
    if rt is None:
        # Checkpoint missing — caller (`_run_demucs_separation`) already
        # catches exceptions and continues with main 4 stems, but returning
        # an empty dict is cleaner than raising.
        logger.warning("drum sub-stem encoding skipped: LatentDrumsep unavailable")
        return {}
    L = drums_latent if isinstance(drums_latent, torch.Tensor) else torch.from_numpy(drums_latent)
    if L.dim() == 2 and L.shape[0] == 64 and L.shape[1] != 64:
        L = L.transpose(0, 1)                     # → [T, 64]
    substems = rt.split(L)                        # {name: [T, 64]}
    sub_latents = {}
    for name, L_stem in substems.items():
        try:
            meta = _save_raw_latent(L_stem, stem_type=f"drum/{name}", source_path=None)
            sub_latents[name] = meta
        except Exception as e:
            logger.warning("substem latent save failed for %s: %s", name, e)
    return sub_latents


@app.route("/separate-stems", methods=["POST"])
def separate_stems():
    """Accept a multipart audioFile upload, kick off htdemucs_6s in
    background, return a processing handle the frontend will poll."""
    f = request.files.get("audioFile") or request.files.get("file")
    if f is None:
        return jsonify({"error": "No audioFile in request"}), 400
    task_id = str(uuid.uuid4())
    out_dir = os.path.join(DEMUCS_OUTPUT_DIR, task_id)
    os.makedirs(out_dir, exist_ok=True)
    audio_path = os.path.join(out_dir, "input_" + (f.filename or "audio.wav"))
    f.save(audio_path)
    DEMUCS_TASKS[task_id] = {
        "status": "processing",
        "audio_path": audio_path,
        "created": time.time(),
    }
    threading.Thread(target=_run_demucs_separation,
                     args=(task_id, audio_path), daemon=True).start()
    return jsonify({"status": "processing", "task_id": task_id})


@app.route("/separate-stems/status/<task_id>", methods=["GET"])
def separate_stems_status(task_id):
    task = DEMUCS_TASKS.get(task_id)
    if not task:
        return jsonify({"status": "failed", "error": "task not found"}), 404
    # Return latent metadata so the studio can do meter changes purely
    # via latent IDs (no audio file fetch needed for the change itself).
    return jsonify({
        "status": task["status"],
        "stems": task.get("stems"),
        "stem_latents": task.get("stem_latents"),
        "drum_substem_latents": task.get("drum_substem_latents"),
        "error": task.get("error"),
    })


@app.route("/separate-stems/download/<task_id>/<filename>", methods=["GET"])
def separate_stems_download(task_id, filename):
    fpath = os.path.join(DEMUCS_OUTPUT_DIR, task_id, filename)
    if not os.path.exists(fpath):
        return jsonify({"error": "stem not found"}), 404
    return send_file(fpath, mimetype="audio/wav", as_attachment=False,
                     download_name=filename)


@app.route("/api/separate-stemphonic", methods=["POST"])
def separate_stemphonic():
    """Stemphonic separation pathway: feeds the input audio's VAE
    latents as `sub_mix_latents` (is_conditional=True training mode)
    so the decoder predicts the target stem given the mix + caption
    + optional MIDI piano-roll guidance + timbre reference.

    Form fields:
      audioFile          required — the mix to separate
      instrument         e.g. 'electric_guitar', 'lead_vox', 'drum_kit'
      midiFile           optional — pitched roll / drum roll to guide
                         the per-layer adapter hook
      checkpoint         which stemphonic ckpt to use (default current)
      timbre_preset      e.g. 'electric_guitar:0'
      cover_noise_strength  init blend (0 = full denoise, 1 = passthrough)
      cfg                CFG scale (default 7)
      steps              denoising steps (default 50)
      seed               int

    Returns multipart-style JSON with the predicted stem WAV.
    """
    f = request.files.get("audioFile") or request.files.get("file")
    if f is None:
        return jsonify({"error": "No audioFile in request"}), 400
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(OUTPUT_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    in_path = os.path.join(task_dir, "input_" + (f.filename or "mix.wav"))
    f.save(in_path)

    instrument   = request.form.get("instrument", "electric_guitar")
    ckpt_id      = request.form.get("checkpoint") or current_ckpt_id
    timbre       = request.form.get("timbre_preset") or f"{instrument}:0"
    cns          = float(request.form.get("cover_noise_strength", 0.0))
    cfg          = float(request.form.get("cfg", 7.0))
    steps        = int(request.form.get("steps", 50))
    seed         = int(request.form.get("seed", 42))
    midi_file    = request.files.get("midiFile")
    midi_path    = None
    if midi_file:
        midi_path = os.path.join(task_dir, "midi_" + (midi_file.filename or "in.mid"))
        midi_file.save(midi_path)

    try:
        # Switch ckpt if requested
        if ckpt_id and ckpt_id != current_ckpt_id and ckpt_id in CKPT_REGISTRY:
            with model_lock:
                load_checkpoint(ckpt_id)

        # 1. Mix → VAE latents (raw, not FSQ-roundtripped — separation
        #    expects real latents as the sub_mix conditioning)
        with model_lock:
            _, raw_latents, _ = audio_to_fsq_tokens(in_path, handler)
        T = raw_latents.shape[0]
        duration = T / 25.0

        # 2. Optional MIDI piano-roll for the layer hook
        midi_tensor = None
        ckpt_has_midi = bool(current_ckpt_meta and current_ckpt_meta.get("has_midi"))
        if midi_path and ckpt_has_midi:
            is_drum = (instrument or "").lower() in DRUM_KIT_PROGRAMS
            try:
                midi_tensor = load_real_midi(midi_path, T=T, drum_mode=is_drum)
            except Exception as e:
                logger.warning("separation: MIDI load failed: %s", e)

        # 3. Timbre ref
        timbre_ref = _resolve_timbre_preset(timbre)

        # 4. Caption from training table
        prompt = TRAINING_CAPTIONS.get(instrument, instrument.replace("_", " "))

        logger.info("🎯 Stemphonic separation: inst=%s ckpt=%s T=%d cns=%.2f midi=%s",
                    instrument, current_ckpt_id, T, cns, midi_tensor is not None)

        # 5. Run trainmatch with sub_mix_latents = mix latents
        with model_lock:
            audio_np = generate_stemphonic_trainmatch(
                handler, module,
                prompt=prompt,
                lyrics="[Instrumental]",
                midi_tensor=midi_tensor,
                timbre_ref=timbre_ref,
                fsq_raw=None,
                sub_mix_latents=raw_latents.unsqueeze(0),
                cover_src_latents=None,
                duration=duration, steps=steps, cfg=cfg, seed=seed,
                cover_noise_strength=cns,
                audio_cover_strength=1.0,
            )

        out_path = os.path.join(task_dir, f"sep_{instrument}_{seed}.wav")
        sf.write(out_path, audio_np.T, 48000)
        return jsonify({
            "status": "completed",
            "task_id": task_id,
            "stem": instrument,
            "audio_url": f"/api/generate-stemphonic/download/{task_id}/{Path(out_path).name}",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


_whisper_model_cache = {"model": None}


def _get_whisper_model():
    """Lazy-load whisper small (~470MB) on first transcription request."""
    if _whisper_model_cache["model"] is None:
        import whisper
        m = whisper.load_model("small")
        _whisper_model_cache["model"] = m
        logger.info("Loaded whisper small")
    return _whisper_model_cache["model"]


_latent_lyric_runtime = {"rt": None}

def _get_latent_lyric():
    # See _get_latent_drumsep for the unavailable-sentinel pattern.
    if _latent_lyric_runtime["rt"] == "unavailable":
        return None
    if _latent_lyric_runtime["rt"] is None:
        try:
            from latent_whisper_student.runtime import LatentLyricRuntime
            _latent_lyric_runtime["rt"] = LatentLyricRuntime.get(
                "/scratch/latent_whisper_student/ckpts_vocal/student_final.pt"
            )
            logger.info("Loaded LatentLyric student (latent → lyrics)")
        except FileNotFoundError as e:
            logger.warning("LatentLyric checkpoint missing — vocal transcription disabled: %s", e)
            _latent_lyric_runtime["rt"] = "unavailable"
            return None
    return _latent_lyric_runtime["rt"]


@app.route("/api/transcribe-vocals", methods=["POST"])
def transcribe_vocals():
    """Vocal transcription via latent-space lyric student.

    audio → VAE encode → latent → LatentLyricRuntime.transcribe → text.
    No Whisper, no audio-domain model. The student produces text per
    30s chunk; word-level timestamps are not yet available (returns
    empty words array).
    """
    f = request.files.get("audioFile") or request.files.get("file")
    if f is None:
        return jsonify({"error": "No audioFile in request"}), 400

    # Early-exit when the lyric checkpoint isn't baked (post-2026-04-11 wipe).
    # 200 + empty text so the caller can skip the transcription step.
    if _get_latent_lyric() is None:
        return jsonify({"text": "", "words": [], "unavailable": True})

    tmp_dir = "/scratch/cache/whisper_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    audio_path = os.path.join(tmp_dir, f"in_{uuid.uuid4().hex[:8]}_{f.filename or 'audio.wav'}")
    f.save(audio_path)
    try:
        with model_lock:
            _, raw_latents, _ = audio_to_fsq_tokens(audio_path, handler)
        rt = _get_latent_lyric()
        chunks = rt.transcribe(raw_latents.float())
        text = " ".join(c.strip() for c in chunks if c.strip())
        logger.info("LatentLyric %s: %d chunks, text=%r",
                    Path(audio_path).name, len(chunks), text[:80])
        return jsonify({"text": text, "words": []})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(audio_path)
        except OSError: pass
    # Legacy audio-domain implementation kept below (unreachable).
    f = request.files.get("audioFile") or request.files.get("file")
    if f is None:
        return jsonify({"error": "No audioFile in request"}), 400
    tmp_dir = "/scratch/cache/whisper_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    audio_path = os.path.join(tmp_dir, f"in_{uuid.uuid4().hex[:8]}_{f.filename or 'audio.wav'}")
    f.save(audio_path)
    try:
        model = _get_whisper_model()
        result = model.transcribe(
            audio_path,
            language="en",
            fp16=False,
            verbose=False,
            word_timestamps=True,
            no_speech_threshold=0.7,
        )
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []) or []:
                words.append({
                    "word": (w.get("word") or "").strip(),
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                })
        text = (result.get("text") or "").strip()
        logger.info("Whisper transcribed %s: %d words, text=%r",
                    Path(audio_path).name, len(words), text[:80])
        return jsonify({"text": text, "words": words})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(audio_path)
        except OSError: pass


def _normalize_chord_label(lab):
    """Convert Chordino's '<root>:<quality>[/<bass>]' or 'N' to a clean
    display label. Examples:
      'C:maj'    -> 'C'
      'C:maj7'   -> 'Cmaj7'
      'A:min7'   -> 'Am7'
      'G:7'      -> 'G7'
      'D:min9'   -> 'Dm9'
      'F:sus4'   -> 'Fsus4'
      'B:dim'    -> 'Bdim'
      'C:maj/E'  -> 'C/E'
      'A:min7/E' -> 'Am7/E'
      'N'        -> None  (no chord — silence)
    """
    if not lab or lab in ("N", "X"):
        return None
    bass = None
    if "/" in lab:
        lab, bass = lab.split("/", 1)
    if ":" in lab:
        root, qual = lab.split(":", 1)
    else:
        root, qual = lab, "maj"

    # Quality remap
    qmap = {
        "maj":   "",
        "min":   "m",
        "dim":   "dim",
        "aug":   "aug",
        "maj7":  "maj7",
        "min7":  "m7",
        "7":     "7",
        "minmaj7": "mMaj7",
        "maj6":  "6",
        "min6":  "m6",
        "9":     "9",
        "maj9":  "maj9",
        "min9":  "m9",
        "11":    "11",
        "min11": "m11",
        "13":    "13",
        "maj13": "maj13",
        "sus2":  "sus2",
        "sus4":  "sus4",
        "hdim7": "m7b5",
        "dim7":  "dim7",
        "aug7":  "aug7",
    }
    suffix = qmap.get(qual, qual)
    out = f"{root}{suffix}"
    if bass:
        out += f"/{bass}"
    return out


def _chord_root(label):
    """Extract pitch-class root from a normalized chord label like
    'Cmaj7', 'F#m', 'Bb7'. Returns 'C', 'F#', 'Bb', etc., or None."""
    if not label:
        return None
    if len(label) >= 2 and label[1] in ("#", "b"):
        return label[:2]
    return label[:1]


# Chordino + _get_chordino removed: chord detection is now purely
# symbolic via LatentPitch → _detect_chords_from_midi below.
# _separate_for_chords removed: chord detection is now purely symbolic
# (LatentPitch → MIDI → template match), no htdemucs, no Chordino.
# See _detect_chords_from_midi below.


# Chord templates — pitch-class sets relative to root (0). Ordered by
# specificity so the template match prefers fuller chords when multiple
# apply. Root is always bit 0; we rotate across all 12 roots at query
# time.
_CHORD_TEMPLATES = [
    # (suffix, pitch-class offsets from root)
    ("maj7",  (0, 4, 7, 11)),
    ("m7",    (0, 3, 7, 10)),
    ("7",     (0, 4, 7, 10)),
    ("dim7",  (0, 3, 6, 9)),
    ("m7b5",  (0, 3, 6, 10)),
    ("",      (0, 4, 7)),      # major triad
    ("m",     (0, 3, 7)),      # minor triad
    ("dim",   (0, 3, 6)),
    ("aug",   (0, 4, 8)),
    ("sus4",  (0, 5, 7)),
    ("sus2",  (0, 2, 7)),
    ("5",     (0, 7)),         # power chord (fallback)
]
_PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _chord_score(active_pcs, template_pcs):
    """Score how well an active pitch-class set matches a chord template.
    Returns (matched, missing, extra). Higher `matched - extra` wins."""
    tmpl = set(template_pcs)
    act  = set(active_pcs)
    return len(tmpl & act), len(tmpl - act), len(act - tmpl)


def _match_chord_from_pcs(active_pcs, bass_pc=None):
    """Given a set of active pitch classes (0..11), pick the best chord
    label. Tries all 12 roots × templates and scores by (matched - missing
    - 0.5*extra). Returns None if nothing reasonable fits.
    """
    if not active_pcs:
        return None
    best = None
    best_score = -99
    for root in range(12):
        for suffix, offsets in _CHORD_TEMPLATES:
            tmpl = tuple((root + o) % 12 for o in offsets)
            m, miss, extra = _chord_score(active_pcs, tmpl)
            # Require root to be present — no rootless voicings.
            if root not in active_pcs:
                continue
            score = m - miss - 0.5 * extra
            # Prefer fuller chords on ties
            score += 0.01 * len(offsets)
            if score > best_score:
                best_score = score
                best = (root, suffix)
    if best is None or best_score < 1.5:
        return None
    root, suffix = best
    label = f"{_PITCH_NAMES[root]}{suffix}"
    if bass_pc is not None and bass_pc != root:
        label = f"{label}/{_PITCH_NAMES[bass_pc]}"
    return label


def _detect_chords_from_midi(pm, beat_times):
    """Symbolic chord detection: given a PrettyMIDI (from LatentPitch)
    and a beat-time array, return {beat_idx: chord_label} for every
    beat whose chord differs from the prior one.

    Per-beat window = [beat_time, next_beat_time). Collect all notes
    whose [start, end) intersects the window, bin by pitch class,
    take the bass as the lowest active MIDI note in the window.
    """
    import numpy as _np
    if pm is None or not beat_times.size:
        return {}
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append((float(n.start), float(n.end), int(n.pitch), float(n.velocity)))
    if not notes:
        return {}
    notes.sort(key=lambda x: x[0])

    chords_out = {}
    last_label = None
    n_beats = len(beat_times)
    for i in range(n_beats):
        t0 = float(beat_times[i])
        t1 = float(beat_times[i + 1]) if i + 1 < n_beats else t0 + 0.5
        # Active notes: start < t1 AND end > t0
        active = [(s, e, p, v) for s, e, p, v in notes if s < t1 and e > t0]
        if not active:
            continue
        pcs = set(p % 12 for _, _, p, _ in active)
        bass_midi = min(p for _, _, p, _ in active)
        bass_pc = bass_midi % 12
        label = _match_chord_from_pcs(pcs, bass_pc=bass_pc)
        if label is None:
            continue
        if label != last_label:
            chords_out[str(i)] = label
            last_label = label
    return chords_out


# _bass_notes_per_beat removed: bass note per beat is now derived from
# the lowest active MIDI note in each beat window inside
# _detect_chords_from_midi (above).


@app.route("/api/detect-chords", methods=["POST"])
def detect_chords():
    """Robust chord + tempo + meter detection.

    Stack:
      - madmom DBN downbeat tracker  → BPM, downbeat times, meter (3/4/5/6/7)
      - Chordino (chord-extractor)   → chord labels with extensions + bass

    Form fields:
      audioFile  required
      mode       optional, "master" (default) or "stems"
                 - "master": run Chordino directly on the input
                 - "stems":  demucs first, run Chordino on harmony-only mix
                             (other+guitar+piano), then override the bass
                             note from the bass stem (pyin)

    Returns:
      {
        "bpm": 124.0,
        "beats_per_bar": 4,
        "downbeat_offset": 0.087,
        "mode": "stems",
        "chords": { "<beatIndex>": "Am7/E", ... }
      }

    beatIndex is the integer beat number from the start of the audio
    (NOT bar number). The chord row consumes integer beat keys.

    TEMPORARILY DISABLED — the old Chordino path required audio-domain
    packages (htdemucs + chordino .so + beat_this) that have broken
    transitive deps on Python 3.11 (beat_this hits `np.float` removed
    in numpy 1.20+, plus `collections.MutableSequence` removed in 3.10).
    The new symbolic path (LatentPitch → MIDI → template match) works
    on the server but still needs beat_this for downbeat detection.
    Until a latent-space beat tracker lands, return 503 and let the
    frontend gracefully degrade (no chord row drawn on upload).

    TODO: wire a latent-space beat tracker and re-enable.
    """
    return jsonify({
        "error": "Chord detection temporarily disabled — "
                 "latent-space beat tracker + chord detector pending.",
        "status": "disabled",
    }), 503
    # Legacy path below is unreachable.
    import numpy as np

    f = request.files.get("audioFile") or request.files.get("file")
    if f is None:
        return jsonify({"error": "No audioFile in request"}), 400
    mode = (request.form.get("mode") or "master").lower()
    if mode not in ("master", "stems"):
        mode = "master"
    tmp_dir = "/scratch/cache/chord_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    audio_path = os.path.join(
        tmp_dir, f"in_{uuid.uuid4().hex[:8]}_{f.filename or 'audio.wav'}"
    )
    f.save(audio_path)

    try:
        # ------------------------------------------------------------------
        # 1. beat_this (CPJKU transformer) → beats + downbeats
        # SOTA neural beat tracker. madmom's downbeat phase was unreliable
        # on shuffled / 12/8-feel songs; beat_this gets the phase right.
        # ------------------------------------------------------------------
        from beat_this.inference import File2Beats
        f2b = File2Beats(checkpoint_path="final0", dbn=True)
        beat_arr, downbeat_arr = f2b(audio_path)
        if len(beat_arr) < 2:
            return jsonify({"error": "Not enough beats detected"}), 400

        beat_times = np.asarray(beat_arr, dtype=float)
        downbeat_times = np.asarray(downbeat_arr, dtype=float)
        # Tag each beat with its position in the bar (1-indexed) by
        # measuring how many beats fall between consecutive downbeats.
        if len(downbeat_times) >= 2:
            db_idx = [int(np.argmin(np.abs(beat_times - t))) for t in downbeat_times]
            spacings = np.diff(db_idx)
            beats_per_bar = int(np.median(spacings)) if len(spacings) else 4
            if beats_per_bar < 2 or beats_per_bar > 12:
                beats_per_bar = 4
        else:
            beats_per_bar = 4
        # Build beat_pos: 1 on each downbeat, then 2,3,...,beats_per_bar
        beat_pos = np.zeros(len(beat_times), dtype=int)
        db_set = set(np.round(downbeat_times, 4).tolist())
        cur = 0
        for i, t in enumerate(beat_times):
            if round(float(t), 4) in db_set:
                cur = 1
            else:
                cur = (cur % beats_per_bar) + 1 if cur else 2
            beat_pos[i] = cur

        # BPM from median inter-beat interval (robust to outliers)
        ibis = np.diff(beat_times)
        bpm = float(60.0 / np.median(ibis)) if len(ibis) else 120.0
        # madmom's DBN can pick triplet/dotted interpretations (e.g. 166
        # for a 112 song). Cross-check with librosa.beat.beat_track which
        # uses a different algorithm (onset autocorrelation) and snap to
        # whichever lies in the 70–140 pop/rock range.
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60.0)
            lib_bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
            lib_bpm = float(lib_bpm)
            # Build candidate set: madmom * {1, 1/2, 2, 2/3, 3/2}, plus librosa
            cands = [bpm, bpm/2, bpm*2, bpm*2/3, bpm*3/2, lib_bpm]
            # Score: in-range first, then closeness to librosa
            in_range = [c for c in cands if 70 <= c <= 140]
            if in_range:
                bpm = min(in_range, key=lambda c: abs(c - lib_bpm))
            else:
                bpm = lib_bpm if 60 <= lib_bpm <= 180 else bpm
            logger.info("Tempo: madmom=%.1f librosa=%.1f → %.1f",
                        60.0/np.median(ibis) if len(ibis) else 0, lib_bpm, bpm)
        except Exception as e:
            logger.warning("librosa tempo crosscheck failed: %s", e)

        # beat_this already gives us the correct downbeat phase; no
        # extra phase-correction pass needed.

        # Find first downbeat for offset
        downbeat_idx = np.where(beat_pos == 1)[0]
        downbeat_offset = float(beat_times[downbeat_idx[0]]) if len(downbeat_idx) else float(beat_times[0])

        # ------------------------------------------------------------------
        # 2. LatentPitch → MIDI → symbolic chord detection
        #    The 'mode' query param (master/stems) is legacy — chord
        #    detection is now purely symbolic. We encode the audio to
        #    VAE latent, transcribe to MIDI with LatentPitch, then run
        #    a template-based chord matcher against the per-beat
        #    pitch-class histogram.
        #
        # TODO: train a dedicated latent-space chord detector student to
        # replace this template matcher (tracking with whisper + PANNs
        # latent-student efforts).
        # ------------------------------------------------------------------
        with model_lock:
            _, raw_latents, _ = audio_to_fsq_tokens(audio_path, handler)
        rt_pitch = _get_latent_pitch()
        pm = rt_pitch.transcribe(raw_latents.float())
        chords_out = _detect_chords_from_midi(pm, beat_times)

        logger.info(
            "Chord detect %s: mode=%s, bpm=%.1f, meter=%d, %d chord changes",
            Path(audio_path).name, mode, bpm, beats_per_bar, len(chords_out),
        )
        # Per-beat tempo map: list of (time_sec, beat_position_in_bar)
        # so the frontend can render bars at the actual detected beat
        # times rather than assuming a constant BPM. Crucial for songs
        # without a metronomic groove.
        beat_map = [
            {"t": float(t), "pos": int(p)}
            for t, p in zip(beat_times, beat_pos)
        ]
        return jsonify({
            "bpm": round(bpm, 2),
            "beats_per_bar": beats_per_bar,
            "downbeat_offset": downbeat_offset,
            "mode": mode,
            "chords": chords_out,
            "beat_map": beat_map,
        })
    except ImportError as e:
        traceback.print_exc()
        return jsonify({
            "error": f"Missing dependency: {e}. Install beat_this "
                     "and ensure latent_pitch checkpoint is present.",
        }), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(audio_path)
        except OSError: pass


# ======================================================================
# Phase A: per-upload analysis — basic-pitch + PANNs instrument classifier
# ======================================================================

EXTRACT_MIDI_DIR = "/scratch/cache/extract_midi"
os.makedirs(EXTRACT_MIDI_DIR, exist_ok=True)


@app.route("/api/extract-midi", methods=["POST"])
def extract_midi_endpoint():
    """Run BasicPitch on the uploaded audio file. Returns a download URL
    for the .mid file plus duration. Frontend calls this on every audio
    drop so every track has a MIDI representation cached server-side.

    Form fields:
      audioFile  required
    """
    f = request.files.get("audioFile") or request.files.get("file")
    if f is None:
        return jsonify({"error": "No audioFile in request"}), 400

    file_id = uuid.uuid4().hex[:12]
    audio_path = os.path.join(EXTRACT_MIDI_DIR, f"in_{file_id}_{f.filename or 'audio.wav'}")
    f.save(audio_path)
    try:
        # audio → VAE latent → LatentPitch student → PrettyMIDI
        with model_lock:
            _, raw_latents, _ = audio_to_fsq_tokens(audio_path, handler)
        rt_pitch = _get_latent_pitch()
        pm = rt_pitch.transcribe(raw_latents.float())
        midi_path = os.path.join(EXTRACT_MIDI_DIR, f"{file_id}.mid")
        pm.write(midi_path)
        n_notes = sum(len(inst.notes) for inst in pm.instruments)
        duration = float(max((n.end for inst in pm.instruments for n in inst.notes), default=0.0))
        logger.info("LatentPitch %s → %d notes, %.2fs", Path(audio_path).name, n_notes, duration)
        return jsonify({
            "midi_url": f"/api/extract-midi/download/{file_id}.mid",
            "n_notes": n_notes,
            "duration": duration,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(audio_path)
        except OSError: pass


@app.route("/api/extract-midi/download/<filename>", methods=["GET"])
def extract_midi_download(filename):
    fpath = os.path.join(EXTRACT_MIDI_DIR, filename)
    if not os.path.exists(fpath):
        return jsonify({"error": "Not found"}), 404
    return send_file(fpath, mimetype="audio/midi")


# ----------------------------------------------------------------------
# PANNs CNN14 instrument classifier (lazy-loaded)
# ----------------------------------------------------------------------
_PANNS = None
def _get_panns():
    global _PANNS
    if _PANNS is None:
        from panns_inference import AudioTagging
        _PANNS = AudioTagging(checkpoint_path=None, device="cuda")
        logger.info("Loaded PANNs CNN14 (AudioSet)")
    return _PANNS


# AudioSet label → our internal track-type registry
PANNS_LABEL_TO_TYPE = {
    "Acoustic guitar": "guitar",
    "Electric guitar": "guitar",
    "Guitar": "guitar",
    "Steel guitar, slide guitar": "guitar",
    "Tapping (guitar technique)": "guitar",
    "Strum": "guitar",
    "Bass guitar": "bass",
    "Double bass": "bass",
    "Synth bass": "bass",
    "Piano": "piano",
    "Electric piano": "piano",
    "Keyboard (musical)": "piano",
    "Organ": "piano",
    "Harpsichord": "piano",
    "Drum kit": "drums",
    "Drum machine": "drums",
    "Snare drum": "drums",
    "Bass drum": "drums",
    "Hi-hat": "drums",
    "Cymbal": "drums",
    "Tabla": "drums",
    "Drum": "drums",
    "Singing": "vocals",
    "Male singing": "vocals",
    "Female singing": "vocals",
    "Choir": "vocals",
    "Rapping": "vocals",
    "Speech": "vocals",
    "Synthesizer": "synth",
    "Electronic music": "synth",
    "Sampler": "synth",
    "Violin, fiddle": "strings",
    "Cello": "strings",
    "String section": "strings",
    "Bowed string instrument": "strings",
    "Brass instrument": "brass",
    "Trumpet": "brass",
    "Trombone": "brass",
    "Saxophone": "brass",
    "Flute": "winds",
    "Clarinet": "winds",
}


_latent_panns_runtime = {"rt": None}

def _get_latent_panns():
    # See _get_latent_drumsep for the unavailable-sentinel pattern.
    if _latent_panns_runtime["rt"] == "unavailable":
        return None
    if _latent_panns_runtime["rt"] is None:
        try:
            from latent_panns_student.runtime import LatentPANNsRuntime
            _latent_panns_runtime["rt"] = LatentPANNsRuntime.get(
                "/scratch/latent_panns_student/ckpts/panns_final.pt"
            )
            logger.info("Loaded LatentPANNs student (latent → instrument classification)")
        except FileNotFoundError as e:
            logger.warning("LatentPANNs checkpoint missing — instrument classification disabled: %s", e)
            _latent_panns_runtime["rt"] = "unavailable"
            return None
    return _latent_panns_runtime["rt"]


@app.route("/api/classify-instrument", methods=["POST"])
def classify_instrument():
    """Instrument classifier via latent-space PANNs student.

    audio → VAE encode → latent → LatentPANNsRuntime.top_k → AudioSet
    labels → mapped to internal track types via PANNS_LABEL_TO_TYPE.
    """
    f = request.files.get("audioFile") or request.files.get("file")
    if f is None:
        return jsonify({"error": "No audioFile in request"}), 400

    # Early-exit when the PANNs checkpoint isn't baked into the image.
    # Return a 200 with a degraded-but-valid payload so the frontend can
    # still proceed (it reads type/label/score and falls back to "other"
    # anyway when the classifier is uncertain).
    if _get_latent_panns() is None:
        return jsonify({
            "type": "other", "label": "unknown", "score": 0.0,
            "top5": [], "unavailable": True,
        })

    tmp = os.path.join("/scratch/cache/panns_tmp", f"in_{uuid.uuid4().hex[:8]}_{f.filename or 'audio.wav'}")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    f.save(tmp)
    try:
        # audio → VAE latent → LatentPANNs student
        with model_lock:
            _, raw_latents, _ = audio_to_fsq_tokens(tmp, handler)
        rt = _get_latent_panns()
        top5_raw = rt.top_k(raw_latents.float(), k=5)

        # Map AudioSet labels to internal track types
        best_type = None
        best_label = None
        best_score = 0.0
        top5 = []
        for r in top5_raw:
            tp = PANNS_LABEL_TO_TYPE.get(r["label"]) or "other"
            top5.append({"label": r["label"], "type": tp, "score": r["score"]})
            if tp != "other" and best_type is None:
                best_type = tp
                best_label = r["label"]
                best_score = r["score"]
        if best_type is None:
            best_type = "other"
            best_label = top5_raw[0]["label"] if top5_raw else "unknown"
            best_score = top5_raw[0]["score"] if top5_raw else 0.0

        logger.info("LatentPANNs %s → %s (%s, %.2f)",
                    Path(tmp).name, best_type, best_label, best_score)
        return jsonify({
            "type": best_type,
            "label": best_label,
            "score": best_score,
            "top5": top5,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(tmp)
        except OSError: pass


# ======================================================================
# Phase B+C: chord-aware per-stem regeneration
# ======================================================================

PITCH_NAMES_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
PITCH_TO_PC = {n: i for i, n in enumerate(PITCH_NAMES_SHARP)}
PITCH_TO_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})

CHORD_QUALITIES = {
    "":     [0, 4, 7],
    "maj":  [0, 4, 7],
    "m":    [0, 3, 7],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "5":    [0, 7],
    "6":    [0, 4, 7, 9],
    "m6":   [0, 3, 7, 9],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7":   [0, 3, 7, 10],
    "m7b5": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "9":    [0, 4, 7, 10, 14],
    "maj9": [0, 4, 7, 11, 14],
    "m9":   [0, 3, 7, 10, 14],
    "11":   [0, 4, 7, 10, 14, 17],
    "m11":  [0, 3, 7, 10, 14, 17],
    "13":   [0, 4, 7, 10, 14, 21],
    "maj13":[0, 4, 7, 11, 14, 21],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "add9": [0, 4, 7, 14],
}


def parse_chord(chord_str):
    """'G9/B' → {root_pc:7, intervals:[0,4,7,10,14], bass_pc:11, label:'G9/B'}.
    Returns None for empty/invalid."""
    if not chord_str:
        return None
    s = chord_str.strip()
    bass_pc = None
    if "/" in s:
        s, bass_str = s.split("/", 1)
        bass_pc = PITCH_TO_PC.get(bass_str.strip())
    # Root: 1 or 2 chars
    if len(s) >= 2 and s[1] in ("#", "b"):
        root_str, qual = s[:2], s[2:]
    else:
        root_str, qual = s[:1], s[1:]
    root_pc = PITCH_TO_PC.get(root_str)
    if root_pc is None:
        return None
    # Normalize quality aliases
    qual_clean = qual.replace("Δ", "maj").replace("-", "m")
    intervals = CHORD_QUALITIES.get(qual_clean) or CHORD_QUALITIES.get("")
    return {
        "root_pc": root_pc,
        "intervals": intervals,
        "bass_pc": bass_pc if bass_pc is not None else root_pc,
        "label": chord_str,
    }


def chord_pitch_classes(parsed):
    """Set of allowed pitch classes for a parsed chord."""
    return {(parsed["root_pc"] + iv) % 12 for iv in parsed["intervals"]}


def chord_diff(old_str, new_str):
    """Categorize what changed between two chord labels.
    Returns dict {root: bool, quality: bool, bass: bool, extension: bool}.
    """
    o, n = parse_chord(old_str), parse_chord(new_str)
    if o is None or n is None:
        return {"root": True, "quality": True, "bass": True, "extension": True}
    o_pcs = chord_pitch_classes(o)
    n_pcs = chord_pitch_classes(n)
    return {
        "root":      o["root_pc"] != n["root_pc"],
        "bass":      o["bass_pc"] != n["bass_pc"],
        "quality":   o_pcs != n_pcs,
        "extension": (max(n["intervals"]) > 7) != (max(o["intervals"]) > 7),
    }


def snap_pitch_to_chord(midi_pitch, allowed_pcs):
    """Move a single MIDI pitch to the nearest pitch class in allowed_pcs,
    keeping it in the same octave neighborhood."""
    if not allowed_pcs:
        return midi_pitch
    best = midi_pitch
    best_dist = 99
    for d in range(-6, 7):
        candidate = midi_pitch + d
        if (candidate % 12) in allowed_pcs and abs(d) < best_dist:
            best = candidate
            best_dist = abs(d)
    return best


def snap_midi_to_chord(midi_path, new_chord, role, region_start=0.0, region_end=None):
    """Edit notes in [region_start, region_end] to fit `new_chord`.
    `role` ∈ {bass, harmony, lead, drums, vocals, other}.
    Writes a new MIDI file and returns its path.
    """
    import pretty_midi
    parsed = parse_chord(new_chord)
    if parsed is None or role == "drums":
        return midi_path

    pm = pretty_midi.PrettyMIDI(midi_path)
    full_pcs = chord_pitch_classes(parsed)

    # Restrict pitch classes by role
    if role == "bass":
        # Bass plays root + occasionally 5th. Use bass_pc as primary.
        allowed = {parsed["bass_pc"], (parsed["root_pc"] + 7) % 12}
    elif role in ("harmony", "piano", "guitar", "keys", "synth"):
        allowed = full_pcs
    elif role in ("lead", "vocals"):
        # Lead snaps to chord tones (triad+7th, no extensions)
        triad = parsed["intervals"][:4]
        allowed = {(parsed["root_pc"] + iv) % 12 for iv in triad}
    else:
        allowed = full_pcs

    n_changed = 0
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if region_end is not None and (note.end < region_start or note.start > region_end):
                continue
            new_p = snap_pitch_to_chord(note.pitch, allowed)
            if new_p != note.pitch:
                note.pitch = new_p
                n_changed += 1

    out_path = midi_path.replace(".mid", f"_snap_{uuid.uuid4().hex[:6]}.mid")
    pm.write(out_path)
    logger.info("snap_midi_to_chord %s role=%s chord=%s → %d notes shifted",
                Path(midi_path).name, role, new_chord, n_changed)
    return out_path


@app.route("/api/regen-stem-for-chord", methods=["POST"])
def regen_stem_for_chord():
    """Regenerate a single stem to fit a chord change.

    Form fields:
      audioFile   required — original stem audio
      midiFile    optional — existing MIDI for that stem (if missing, run BasicPitch)
      role        required — bass | harmony | piano | guitar | keys | synth | lead | vocals | drums | other
      old_chord   required — e.g. 'Gm'
      new_chord   required — e.g. 'G9'
      region_start  optional float seconds (default 0)
      region_end    optional float seconds (default whole file)
      cover_noise   optional float 0..1 (default 0.7)
      prompt        optional text prompt (default derived from role)
      duration      optional seconds (default audio length)

    Returns task_id — poll /api/generate-stemphonic/task/<task_id> to fetch result.
    Reuses the existing async stemphonic task pipeline.
    """
    audio_f = request.files.get("audioFile")
    midi_f = request.files.get("midiFile")
    if audio_f is None:
        return jsonify({"error": "No audioFile"}), 400

    role = (request.form.get("role") or "harmony").lower()
    old_chord = request.form.get("old_chord") or ""
    new_chord = request.form.get("new_chord") or ""
    if not new_chord:
        return jsonify({"error": "Missing new_chord"}), 400

    # Diff: if nothing material changed, no-op
    diff = chord_diff(old_chord, new_chord)
    needs_regen = (
        (role == "bass" and (diff["root"] or diff["bass"])) or
        (role in ("harmony","piano","guitar","keys","synth") and (diff["quality"] or diff["extension"] or diff["root"])) or
        (role in ("lead","vocals") and (diff["root"] or diff["quality"]))
    )
    if not needs_regen:
        return jsonify({
            "skipped": True,
            "reason": f"role={role} unaffected by diff {diff}",
        })

    region_start = float(request.form.get("region_start") or 0.0)
    region_end = request.form.get("region_end")
    region_end = float(region_end) if region_end else None
    cover_noise = float(request.form.get("cover_noise") or 0.7)
    prompt = request.form.get("prompt") or f"{role}, in the style of the original"
    duration_s = request.form.get("duration")

    # Save audio + midi to the task dir that run_generation will create
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(OUTPUT_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    audio_path = os.path.join(task_dir, "stem_input.wav")
    audio_f.save(audio_path)

    if midi_f is not None:
        midi_path = os.path.join(task_dir, "input.mid")
        midi_f.save(midi_path)
    else:
        try:
            # audio → VAE latent → LatentPitch student → PrettyMIDI
            with model_lock:
                _, raw_latents, _ = audio_to_fsq_tokens(audio_path, handler)
            rt_pitch = _get_latent_pitch()
            pm = rt_pitch.transcribe(raw_latents.float())
            midi_path = os.path.join(task_dir, "input.mid")
            pm.write(midi_path)
        except Exception as e:
            return jsonify({"error": f"LatentPitch failed: {e}"}), 500

    snapped_midi = snap_midi_to_chord(
        midi_path, new_chord, role,
        region_start=region_start, region_end=region_end,
    )

    if duration_s:
        duration = float(duration_s)
    else:
        try:
            import soundfile as sf2
            info = sf2.info(audio_path)
            duration = float(info.duration)
        except Exception:
            duration = 16.0

    # Map our role → an instrument that the existing run_generation
    # pipeline understands (used for prompt + drum/vox routing).
    ROLE_TO_INST = {
        "bass": "bass", "guitar": "guitar", "piano": "piano",
        "keys": "piano", "synth": "synth", "harmony": "piano",
        "lead": "lead", "vocals": "lead_vox", "drums": "drum_kit",
    }
    payload = {
        "prompt": prompt,
        "instrument": ROLE_TO_INST.get(role, ""),
        "lyrics": "[Instrumental]" if role != "vocals" else "",
        "duration": duration,
        "steps": 50,
        "cfg": 7.0,
        "seed": 42,
        "cover_noise_strength": cover_noise,
        "audio_cover_strength": 1.0,
        # run_generation reads file inputs from these keys
        "midiFile_path": snapped_midi,
        "refAudio_path": audio_path,
    }
    tasks[task_id] = {"status": "queued", "params": payload}
    threading.Thread(target=run_generation, args=(task_id, payload), daemon=True).start()
    return jsonify({"task_id": task_id, "diff": diff})


# ======================================================================
# Video → Score: scene-aware chord generation + per-scene MIDI + concat
# Ported from genfrominterface.py (concatenate_midi_scenes, etc.)
# ======================================================================

# Make harmonymodule importable
sys.path.insert(0, "/scratch/Do/harmonymodule")
import mido as _mido_score
from mido import MidiFile, MidiTrack, MetaMessage, Message

SCORE_OUTPUT_DIR = "/scratch/stemphonic_outputs/scores"
os.makedirs(SCORE_OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Scene-aware chord progression generator
# ----------------------------------------------------------------------

# Genre → diatonic Roman-numeral templates (in major). Minor variants
# auto-derived. Each template is a sequence of degrees that loops.
GENRE_PROGRESSIONS = {
    "pop":         [["I","V","vi","IV"], ["vi","IV","I","V"], ["I","vi","IV","V"]],
    "rock":        [["I","IV","V","I"], ["I","bVII","IV","I"], ["I","V","IV","I"]],
    "jazz":        [["IIm7","V7","Imaj7","Imaj7"], ["Imaj7","VIm7","IIm7","V7"]],
    "ambient":     [["Imaj7","IVmaj7","VIm7","IIIm7"], ["I","iii","IV","I"]],
    "cinematic":   [["i","VI","III","VII"], ["i","iv","VII","III"], ["i","v","VI","iv"]],
    "lofi":        [["IIm7","V7","Imaj7","VIm7"], ["IVmaj7","Imaj7","IIm7","V7"]],
    "edm":         [["vi","IV","I","V"], ["i","VII","VI","VII"]],
    "default":     [["I","V","vi","IV"]],
}

# Roman → semitone offset from tonic (major key)
ROMAN_MAJOR = {
    "I":0,"i":0,"bII":1,"II":2,"ii":2,"iii":4,"III":4,"IV":5,"iv":5,
    "V":7,"v":7,"VI":9,"vi":9,"bVII":10,"VII":11,"vii":11,
}
# Roman → semitone offset from tonic (minor key)
ROMAN_MINOR = {
    "i":0,"I":0,"bII":1,"II":2,"iii":3,"III":3,"iv":5,"IV":5,
    "v":7,"V":7,"VI":8,"vi":8,"VII":10,"bVII":10,
}
PC_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


def _roman_to_chord_name(roman, tonic_pc, scale_type):
    """Convert e.g. 'IIm7' in C major → 'Dm7'."""
    import re as _re
    m = _re.match(r"^(b?[IiVv]+)(.*)$", roman)
    if not m:
        return None
    degree, qual = m.group(1), m.group(2)
    table = ROMAN_MAJOR if scale_type == "major" else ROMAN_MINOR
    semis = table.get(degree)
    if semis is None:
        return None
    pc = (tonic_pc + semis) % 12
    root = PC_NAMES[pc]
    # Capitalization implies quality if no explicit qual
    if not qual:
        qual = "" if degree[0].isupper() else "m"
    return root + qual


def generate_chord_progression_for_scenes(
    scene_durations, scene_tempos,
    key="C", scale_type="major", genre="cinematic",
    beats_per_bar=4, seed=None,
):
    """
    Build a beat-indexed chord map covering all scenes, sized so each
    scene gets a chord progression appropriate to its tempo + length.

    Returns:
        chord_map  : { beat_index: chord_name }
        scene_chord_specs : list of dicts (per scene) with start_beat, end_beat, chords
    """
    import random as _r
    if seed is not None:
        _r.seed(seed)
    tonic_pc = PC_NAMES.index(key) if key in PC_NAMES else 0
    templates = GENRE_PROGRESSIONS.get(genre.lower(), GENRE_PROGRESSIONS["default"])

    chord_map = {}
    specs = []
    cumulative_beat = 0
    for i, dur in enumerate(scene_durations):
        bpm = scene_tempos[i] if i < len(scene_tempos) else 120
        beats_in_scene = max(beats_per_bar, int(round(dur * bpm / 60.0)))
        bars_in_scene = max(1, beats_in_scene // beats_per_bar)

        template = list(_r.choice(templates))
        # Loop or trim template to fill bars_in_scene
        per_chord_bars = max(1, bars_in_scene // len(template))
        chords_for_scene = []
        b = cumulative_beat
        for ci, roman in enumerate(template):
            chord_name = _roman_to_chord_name(roman, tonic_pc, scale_type) or "C"
            chord_map[b] = chord_name
            chords_for_scene.append((b, chord_name))
            b += per_chord_bars * beats_per_bar
        # Pad rest of scene with last chord if any beats remain
        last_chord = chords_for_scene[-1][1] if chords_for_scene else "C"
        if b < cumulative_beat + beats_in_scene:
            # leave the last chord ringing
            pass

        specs.append({
            "scene_idx": i,
            "start_beat": cumulative_beat,
            "end_beat": cumulative_beat + beats_in_scene,
            "duration_sec": dur,
            "bpm": bpm,
            "chords": chords_for_scene,
        })
        cumulative_beat += beats_in_scene

    return chord_map, specs


# ----------------------------------------------------------------------
# Per-scene MIDI rendering using harmonymodule
# ----------------------------------------------------------------------

def render_scene_midi(scene_spec, voicing="open", rhythm="whole", style="block"):
    """Render one scene's chord progression to a MIDI file via harmonymodule."""
    from chord_progression_generator import generate_chord_progression_midi, ScaleContext

    # Build a beat map LOCAL to this scene (start at beat 0)
    local_chords = {}
    for (abs_beat, chord) in scene_spec["chords"]:
        local_beat = abs_beat - scene_spec["start_beat"]
        local_chords[local_beat] = chord

    # Pick a default scale context — let auto-detect figure it out
    out = os.path.join(SCORE_OUTPUT_DIR, f"scene_{uuid.uuid4().hex[:8]}.mid")
    return generate_chord_progression_midi(
        chord_beat_map=local_chords,
        bpm=int(scene_spec["bpm"]),
        voicing=voicing,
        rhythm=rhythm,
        style=style,
        output_path=out,
        auto_detect_scale=True,
    )


# ----------------------------------------------------------------------
# concatenate_midi_scenes — ported from genfrominterface.py:2029
# ----------------------------------------------------------------------

def concatenate_midi_scenes(scene_midi_paths, scene_durations, output_path):
    """
    Concat per-scene MIDIs into one file with proper tempo changes at
    each scene boundary. Each scene MIDI is trimmed/padded to its
    target duration. All tracks are merged into a single track.
    """
    combined = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    combined.tracks.append(track)

    cumulative_ticks = 0
    for scene_idx in sorted(scene_midi_paths.keys()):
        midi_path = scene_midi_paths[scene_idx]
        duration_sec = scene_durations[scene_idx]
        scene_midi = MidiFile(midi_path)

        # Get tempo from scene MIDI (default 120)
        tempo = 500000
        for msg in scene_midi.tracks[0]:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
        bpm = _mido_score.tempo2bpm(tempo)

        # Insert tempo change at scene start
        track.append(MetaMessage("set_tempo", tempo=tempo, time=0))

        ticks_per_second = bpm * combined.ticks_per_beat / 60.0
        max_ticks = int(duration_sec * ticks_per_second)
        current_ticks = 0
        first_msg = True

        for src_track in scene_midi.tracks:
            for msg in src_track:
                if msg.is_meta and msg.type == "set_tempo":
                    continue
                if msg.is_meta:
                    continue
                msg_time = msg.time
                if first_msg:
                    if scene_idx > 0:
                        msg_time = 0
                    first_msg = False
                if current_ticks + msg_time > max_ticks:
                    msg_time = max(0, max_ticks - current_ticks)
                msg_copy = msg.copy(time=msg_time)
                if hasattr(msg_copy, "channel"):
                    msg_copy.channel = 0
                # Avoid sub-bass mud
                if hasattr(msg_copy, "note") and msg_copy.type in ("note_on", "note_off"):
                    while msg_copy.note < 36 and msg_copy.note < 115:
                        msg_copy.note += 12
                track.append(msg_copy)
                current_ticks += msg_time
                if current_ticks >= max_ticks:
                    break
            if current_ticks >= max_ticks:
                break

        # Pad if scene ended early
        if current_ticks < max_ticks:
            pad = max_ticks - current_ticks
            track.append(Message("note_off", note=0, velocity=0, time=pad, channel=0))
            current_ticks = max_ticks

        # All-notes-off at scene boundary
        track.append(Message("control_change", control=123, value=0, time=0, channel=0))
        cumulative_ticks += current_ticks

    combined.save(output_path)
    logger.info("Concatenated %d scenes → %s (total ticks=%d)",
                len(scene_midi_paths), output_path, cumulative_ticks)
    return output_path


# ----------------------------------------------------------------------
# /api/generate-score-from-video — main orchestration endpoint
# ----------------------------------------------------------------------

@app.route("/api/generate-score-from-video", methods=["POST"])
def generate_score_from_video():
    """Generate a complete MIDI score from a video's scene structure.

    JSON body:
      {
        "scene_durations": [3.5, 2.8, 4.1, ...],  required
        "scene_tempos":    [120, 128, 120, ...],  required (one per scene)
        "key":             "C",                   default "C"
        "scale_type":      "major" | "minor",     default "major"
        "genre":           "cinematic" | "pop" | "jazz" | ... default "cinematic"
        "voicing":         "open"|"close"|"drop2"|... default "open"
        "rhythm":          "whole"|"half"|"quarter"|... default "whole"
        "beats_per_bar":   4 default
        "seed":            int or null
        "render_audio":    bool — also call stemphonic per-scene to produce
                                  audio (slower). default false.
        "instrument_prompt": text prompt for stemphonic rendering when
                             render_audio=true. default "ambient piano score"
      }

    Response (immediate, score MIDI is small + fast):
      {
        "score_id": "...",
        "midi_url": "/api/generate-score-from-video/download/<id>.mid",
        "chord_map": {beat_index: chord_name, ...},
        "scene_specs": [...],
        "audio_task_id": null | "..."   # if render_audio=true
      }
    """
    data = request.get_json(force=True) or {}
    scene_durations = data.get("scene_durations") or []
    scene_tempos = data.get("scene_tempos") or []
    if not scene_durations or not scene_tempos:
        return jsonify({"error": "scene_durations and scene_tempos required"}), 400
    if len(scene_durations) != len(scene_tempos):
        return jsonify({"error": "scene_durations and scene_tempos length mismatch"}), 400

    key = data.get("key", "C")
    scale_type = data.get("scale_type", "major")
    genre = data.get("genre", "cinematic")
    voicing = data.get("voicing", "open")
    rhythm = data.get("rhythm", "whole")
    beats_per_bar = int(data.get("beats_per_bar", 4))
    seed = data.get("seed")

    score_id = uuid.uuid4().hex[:12]
    score_dir = os.path.join(SCORE_OUTPUT_DIR, score_id)
    os.makedirs(score_dir, exist_ok=True)

    try:
        # 1. Build chord progression covering all scenes
        chord_map, scene_specs = generate_chord_progression_for_scenes(
            scene_durations, scene_tempos,
            key=key, scale_type=scale_type, genre=genre,
            beats_per_bar=beats_per_bar, seed=seed,
        )

        # 2. Render each scene to its own MIDI
        scene_midis = {}
        for i, spec in enumerate(scene_specs):
            try:
                scene_midis[i] = render_scene_midi(spec, voicing=voicing, rhythm=rhythm)
            except Exception as e:
                logger.warning("Scene %d render failed: %s", i, e)

        # 3. Concatenate into one master MIDI with tempo changes
        master_midi = os.path.join(score_dir, "score.mid")
        if scene_midis:
            concatenate_midi_scenes(scene_midis, scene_durations, master_midi)
        else:
            return jsonify({"error": "All scene renders failed"}), 500

        # 4. Optional: dispatch a stemphonic audio render in the background
        audio_task_id = None
        if data.get("render_audio"):
            total_dur = sum(scene_durations)
            instrument_prompt = data.get("instrument_prompt", "ambient piano score")
            payload = {
                "prompt": instrument_prompt,
                "instrument": "piano",
                "lyrics": "[Instrumental]",
                "duration": min(120.0, total_dur),
                "steps": 50,
                "cfg": 7.0,
                "seed": int(seed or 42),
                "cover_noise_strength": 0.0,
                "audio_cover_strength": 1.0,
                "midiFile_path": master_midi,
            }
            audio_task_id = str(uuid.uuid4())
            tasks[audio_task_id] = {"status": "queued", "params": payload}
            threading.Thread(
                target=run_generation, args=(audio_task_id, payload), daemon=True
            ).start()

        return jsonify({
            "score_id": score_id,
            "midi_url": f"/api/generate-score-from-video/download/{score_id}.mid",
            "chord_map": chord_map,
            "scene_specs": [
                {"scene_idx": s["scene_idx"], "start_beat": s["start_beat"],
                 "end_beat": s["end_beat"], "bpm": s["bpm"],
                 "duration_sec": s["duration_sec"],
                 "chords": [{"beat": b, "chord": c} for b, c in s["chords"]]}
                for s in scene_specs
            ],
            "audio_task_id": audio_task_id,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-score-from-video/download/<score_id>.mid", methods=["GET"])
def download_score(score_id):
    fpath = os.path.join(SCORE_OUTPUT_DIR, score_id, "score.mid")
    if not os.path.exists(fpath):
        return jsonify({"error": "Not found"}), 404
    return send_file(fpath, mimetype="audio/midi", as_attachment=False)


# ======================================================================
# Sprint 1+2: Latent encoding, caching, and meter repaint endpoints
# ======================================================================

LATENT_DIR = "/scratch/cache/latents"

# ── VAE version hash (latents are coupled to a specific VAE checkpoint;
# the browser-side decoder MUST match. We pin a 12-char SHA-256 hash of
# the VAE config + safetensors so any mismatch is detectable.) ─────────
_VAE_VERSION_CACHE = {"v": None}
def _vae_version_hash() -> str:
    if _VAE_VERSION_CACHE["v"] is not None:
        return _VAE_VERSION_CACHE["v"]
    import hashlib as _hash
    h = _hash.sha256()
    vae_dir = "/scratch/ACE-Step-1.5/checkpoints/vae"
    for fn in sorted(os.listdir(vae_dir)) if os.path.isdir(vae_dir) else []:
        p = os.path.join(vae_dir, fn)
        if not os.path.isfile(p):
            continue
        h.update(fn.encode())
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b: break
                h.update(b)
    _VAE_VERSION_CACHE["v"] = h.hexdigest()[:12]
    logger.info("VAE version hash: %s", _VAE_VERSION_CACHE["v"])
    return _VAE_VERSION_CACHE["v"]


# ── /api/latent/<id> binary streaming endpoint ───────────────────────
# Wire format (little-endian):
#   bytes 0..3   magic = b"DOAE"
#   bytes 4..5   uint16 version = 1
#   bytes 6..17  vae_version_hash (12 ascii chars)
#   bytes 18..19 uint16 fps (typically 25)
#   bytes 20..23 uint32 T (number of latent frames)
#   bytes 24..27 uint32 D (channels per frame, typically 64)
#   bytes 28..   T*D float32 little-endian
# Total overhead: 28 bytes. ~6.4 KB/s of latent at 25fps×64.

def _serialize_latent_doae(L_tensor: torch.Tensor, fps: int = 25) -> bytes:
    import struct
    if L_tensor.dim() != 2 or L_tensor.shape[1] != 64:
        raise ValueError(f"expected [T,64] latent, got {tuple(L_tensor.shape)}")
    L = L_tensor.detach().cpu().contiguous().to(torch.float32)
    T, D = L.shape
    vae_hash = _vae_version_hash().encode("ascii")[:12].ljust(12, b"\0")
    header = b"DOAE" + struct.pack("<H", 1) + vae_hash + struct.pack("<HII", fps, T, D)
    return header + L.numpy().tobytes()


@app.route("/api/latent/<latent_id>", methods=["GET"])
def get_latent_binary(latent_id):
    """Stream a cached latent in the .doae binary format. Frontend
    fetches this as ArrayBuffer, parses the 28-byte header, and feeds
    the float32 body straight to the WebGPU ONNX decoder."""
    safe_id = "".join(c for c in latent_id if c.isalnum() or c in "-_")
    if safe_id != latent_id or not safe_id:
        return jsonify({"error": "invalid latent_id"}), 400
    latent_path = os.path.join(LATENT_DIR, f"{safe_id}.pt")
    if not os.path.exists(latent_path):
        return jsonify({"error": "not found"}), 404
    try:
        blob = torch.load(latent_path, map_location="cpu", weights_only=False)
        L = blob.get("latents")
        fps = int(blob.get("fps", 25))
        if L is None:
            return jsonify({"error": "blob missing 'latents'"}), 500
        body = _serialize_latent_doae(L, fps=fps)
        return Response(
            body,
            mimetype="application/x-doae",
            headers={
                "Content-Length": str(len(body)),
                "Cache-Control": "public, max-age=3600, immutable",
                "X-Vae-Version": _vae_version_hash(),
                "X-Latent-Frames": str(L.shape[0]),
                "X-Latent-Fps": str(fps),
                # CORS so the frontend can fetch from any origin during dev
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "X-Vae-Version,X-Latent-Frames,X-Latent-Fps",
            },
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/vae-version", methods=["GET"])
def get_vae_version():
    """Browser checks this on load and refuses to play any latent whose
    .doae header doesn't match (= VAE checkpoint mismatch)."""
    return jsonify({"vae_version": _vae_version_hash()})


def _parse_doae(body: bytes):
    """Parse a .doae binary blob → (latent_tensor, fps, vae_hash)."""
    import struct
    if len(body) < 28 or body[:4] != b"DOAE":
        raise ValueError("not a .doae blob")
    version = struct.unpack("<H", body[4:6])[0]
    vae_hash = body[6:18].rstrip(b"\0").decode("ascii", errors="replace")
    fps, T, D = struct.unpack("<HII", body[18:28])
    if D != 64:
        raise ValueError(f"expected 64 channels, got {D}")
    expected = 28 + T * D * 4
    if len(body) != expected:
        raise ValueError(f"length mismatch: header says {expected}, body is {len(body)}")
    arr = np.frombuffer(body[28:], dtype="<f4").copy()
    L = torch.from_numpy(arr).reshape(T, D)
    return L, fps, vae_hash


@app.route("/api/upload-latent", methods=["POST"])
def upload_latent():
    """Browser POSTs a .doae binary (encoded locally via WebGPU encoder).
    Server validates the VAE version, writes to /scratch/cache/latents
    as a torch .pt for compatibility with the rest of the pipeline,
    returns the new latent_id. NO wav round-trip."""
    body = request.get_data() or b""
    try:
        L, fps, vae_hash = _parse_doae(body)
    except Exception as e:
        return jsonify({"error": f"invalid .doae body: {e}"}), 400
    if vae_hash != _vae_version_hash():
        return jsonify({
            "error": "vae_version mismatch — re-encode locally with the matching decoder",
            "expected": _vae_version_hash(),
            "got": vae_hash,
        }), 409
    latent_id = uuid.uuid4().hex[:12]
    latent_path = os.path.join(LATENT_DIR, f"{latent_id}.pt")
    torch.save({
        "latents": L.cpu().float(),
        "shape": list(L.shape),
        "fps": int(fps),
        "source": "browser-encode",
        "vae_version": vae_hash,
    }, latent_path)
    return jsonify({
        "latent_id": latent_id,
        "n_frames": int(L.shape[0]),
        "fps": int(fps),
        "vae_version": vae_hash,
    })


@app.route("/api/onnx/<fname>", methods=["GET"])
def get_onnx_bundle(fname):
    """Serve the packed Oobleck VAE encoder/decoder ONNX bundles for
    browser WebGPU inference. Long cache time + immutable since the
    files are pinned by VAE checkpoint."""
    safe = "".join(c for c in fname if c.isalnum() or c in "._-")
    if safe != fname:
        return jsonify({"error": "invalid filename"}), 400
    onnx_dir = "/scratch/onnx"
    p = os.path.join(onnx_dir, safe)
    if not os.path.exists(p):
        return jsonify({"error": "not found"}), 404
    resp = send_file(p, mimetype="application/octet-stream", conditional=True)
    resp.headers["Cache-Control"] = "public, max-age=86400, immutable"
    resp.headers["X-Vae-Version"] = _vae_version_hash()
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Expose-Headers"] = "X-Vae-Version"
    return resp
os.makedirs(LATENT_DIR, exist_ok=True)


def encode_audio_to_latent(audio_path):
    """Encode an audio file to VAE latents and save as .pt. Returns
    {latent_id, latent_path, n_frames, duration}."""
    handler = current_handler if 'current_handler' in globals() else None
    if handler is None:
        # Fall back: assume the global handler the run_generation flow uses
        global _global_handler
        handler = globals().get('_global_handler')
    if handler is None:
        # Last resort: look in module-level handler reference
        handler = globals().get('handler')

    if handler is None:
        raise RuntimeError("VAE handler not loaded")

    _, raw_latents, _ = audio_to_fsq_tokens(audio_path, handler)
    latent_id = uuid.uuid4().hex[:12]
    latent_path = os.path.join(LATENT_DIR, f"{latent_id}.pt")
    torch.save({
        "latents": raw_latents.cpu().float(),
        "shape": list(raw_latents.shape),
        "fps": 25,
        "source": Path(audio_path).name,
    }, latent_path)
    return {
        "latent_id": latent_id,
        "latent_path": latent_path,
        "n_frames": int(raw_latents.shape[0]),
        "duration": float(raw_latents.shape[0]) / 25.0,
    }


@app.route("/api/encode-audio-latent", methods=["POST"])
def encode_audio_latent_endpoint():
    """Encode uploaded audio to a VAE latent. Returns {latent_id, n_frames, duration}."""
    f = request.files.get("audioFile") or request.files.get("file")
    if f is None:
        return jsonify({"error": "No audioFile in request"}), 400
    tmp = os.path.join(LATENT_DIR, f"upload_{uuid.uuid4().hex[:8]}.wav")
    f.save(tmp)
    try:
        result = encode_audio_to_latent(tmp)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.route("/api/latents/download/<latent_id>.pt", methods=["GET"])
def download_latent(latent_id):
    fpath = os.path.join(LATENT_DIR, f"{latent_id}.pt")
    if not os.path.exists(fpath):
        return jsonify({"error": "Not found"}), 404
    return send_file(fpath, mimetype="application/octet-stream", as_attachment=False)


# ----------------------------------------------------------------------
# Meter repaint — port of time-sig-editor's splice logic, adapted to
# operate in the VAE LATENT space instead of waveform space.
# ----------------------------------------------------------------------

def latent_compute_splice_frames(n_frames, fps, bpm, src_n, src_den, tgt_n, tgt_den):
    """Compute frame-domain splice points for a meter change. Mirrors
    _compute_splice_times from time-sig-editor but in latent frames."""
    if src_n == tgt_n and src_den == tgt_den:
        return []

    sec_per_beat = 60.0 / bpm
    frames_per_beat = sec_per_beat * fps
    frames_per_eighth = frames_per_beat / 2

    if tgt_den == 8:
        tgt_bar_frames = frames_per_eighth * tgt_n
    else:
        tgt_bar_frames = frames_per_beat * tgt_n

    if tgt_bar_frames <= 0:
        return []

    splice_frames = []
    pos = 0.0
    while pos < n_frames:
        if pos > 1.0:
            splice_frames.append(int(round(pos)))
        # internal split for asymmetric meters (e.g. 4/4→7/8 has 4+3 split)
        if (src_n, src_den, tgt_n, tgt_den) == (4, 4, 7, 8):
            internal = pos + frames_per_eighth * 4
            if internal < n_frames:
                splice_frames.append(int(round(internal)))
        elif (src_n, src_den, tgt_n, tgt_den) == (4, 4, 3, 4):
            splice_frames.append(int(round(pos + tgt_bar_frames)))
        pos += tgt_bar_frames

    if len(splice_frames) > 24:
        step = max(1, len(splice_frames) // 20)
        splice_frames = splice_frames[::step]
    return splice_frames


def _silence_frames(n_frames, latents):
    """Get N frames of REAL latent silence (not zeros — VAE silence has
    a non-zero embedding). Pulls from handler.silence_latent and tiles
    to length n_frames, matching the dtype/device of `latents`."""
    h = globals().get('handler')
    if h is None or not hasattr(h, 'silence_latent') or h.silence_latent is None:
        # Last resort: zeros (will sound wrong but won't crash)
        return torch.zeros(n_frames, latents.shape[1], dtype=latents.dtype, device=latents.device)
    sl = h.silence_latent  # [1, T_silence, 64]
    if sl.dim() == 3:
        sl = sl.squeeze(0)  # [T_silence, 64]
    T_s = sl.shape[0]
    if T_s >= n_frames:
        out = sl[:n_frames]
    else:
        # Tile to fill n_frames
        reps = (n_frames // T_s) + 1
        out = sl.repeat(reps, 1)[:n_frames]
    return out.to(dtype=latents.dtype, device=latents.device)


def _stretch_latent(bar, target_len):
    """Time-stretch a latent bar [T, 64] to exactly target_len frames
    using linear interpolation. The latent-space equivalent of
    time-sig-editor's _stretch_to_len() / ffmpeg atempo."""
    if bar.shape[0] == target_len or target_len <= 0 or bar.shape[0] == 0:
        return bar
    bar_t = bar.t().unsqueeze(0).float()  # [1, 64, T]
    bar_t = F.interpolate(bar_t, size=target_len, mode="linear", align_corners=False)
    return bar_t.squeeze(0).t().to(bar.dtype)


def _remap_latent_bar(bar, src_n, src_den, tgt_n, tgt_den, frames_per_beat, frames_per_eighth):
    """Remap one bar of latents from src to tgt meter using the same
    musically-aware splice strategies as time-sig-editor:_remap_bar.
    Operates in latent frame space instead of waveform sample space.
    Downbeat is always preserved.
    """
    src_sig = f"{src_n}/{src_den}"
    tgt_sig = f"{tgt_n}/{tgt_den}"
    fpb = int(round(frames_per_beat))
    fpe = int(round(frames_per_eighth))

    # 4/4 → 7/8: first half (4 eighths) unchanged, second half (4 eighths) compressed to 3 eighths
    if src_sig == "4/4" and tgt_sig == "7/8":
        half = bar.shape[0] // 2
        first = bar[:half]
        second = bar[half:]
        compressed = _stretch_latent(second, fpe * 3)
        return torch.cat([first, compressed], dim=0)

    # 7/8 → 4/4: first 4 eighths unchanged, last 3 eighths stretched to 4
    if src_sig == "7/8" and tgt_sig == "4/4":
        split = fpe * 4
        first = bar[:split]
        last = bar[split:]
        stretched = _stretch_latent(last, fpe * 4)
        return torch.cat([first, stretched], dim=0)

    # 4/4 → 6/8: 8 eighths → keep first 6 (drop last 2)
    if src_sig == "4/4" and tgt_sig == "6/8":
        return bar[:fpe * 6]

    # 6/8 → 4/4: 6 eighths → first 4 unchanged, last 2 stretched to 4
    if src_sig == "6/8" and tgt_sig == "4/4":
        split = fpe * 4
        first = bar[:split]
        last = bar[split:]
        stretched = _stretch_latent(last, fpe * 4)
        return torch.cat([first, stretched], dim=0)

    # 4/4 → 3/4: drop last beat (beat 4)
    if src_sig == "4/4" and tgt_sig == "3/4":
        return bar[:fpb * 3]

    # 3/4 → 4/4: duplicate last beat
    if src_sig == "3/4" and tgt_sig == "4/4":
        last_beat = bar[-fpb:]
        return torch.cat([bar, last_beat], dim=0)

    # 4/4 → 5/4: duplicate last beat
    if src_sig == "4/4" and tgt_sig == "5/4":
        last_beat = bar[-fpb:]
        return torch.cat([bar, last_beat], dim=0)

    # 5/4 → 4/4: drop last beat
    if src_sig == "5/4" and tgt_sig == "4/4":
        return bar[:fpb * 4]

    # 3/4 → 7/8: 6 eighths → first 4 unchanged, last 2 compressed to 3 eighths
    if src_sig == "3/4" and tgt_sig == "7/8":
        split = fpe * 4
        first = bar[:split]
        last = bar[split:]
        return torch.cat([first, _stretch_latent(last, fpe * 3)], dim=0)

    # Generic fallback: proportional time stretch
    src_eighths = src_n * (2 if src_den == 4 else 1)
    tgt_eighths = tgt_n * (2 if tgt_den == 4 else 1)
    return _stretch_latent(bar, fpe * tgt_eighths)


def latent_remap_meter(latents, fps, bpm, src_n, src_den, tgt_n, tgt_den, stem_type="other", downbeat_offset_sec=0.0):
    """Rearrange a latent tensor [T, 64] from src meter to tgt meter
    using the per-meter splice strategies from time-sig-editor (port
    of _remap_bar / _remap_vocal_bar but in latent frame space).

    All stem types share the same SPLICE STRATEGY (preserve downbeat,
    cut/stretch second half) — what differs is the cover_noise applied
    by the caller during the stemphonic repaint pass:

    - drums: lighter cover_noise (~0.40), preserves transients
    - vocals: heavier cover_noise (~0.65), cleans up edit artefacts
    - instrumental: medium (~0.55)
    """
    if src_n == tgt_n and src_den == tgt_den:
        return latents

    n_frames = latents.shape[0]
    sec_per_beat = 60.0 / bpm
    frames_per_beat = sec_per_beat * fps
    frames_per_eighth = frames_per_beat / 2

    src_bar = frames_per_beat * src_n if src_den == 4 else frames_per_eighth * src_n
    src_bar_int = int(round(src_bar))

    # Skip pre-roll: start slicing on the first detected downbeat so
    # bar 1 of the output corresponds to musical bar 1 of the source.
    out_chunks = []
    pos = max(0, int(round(downbeat_offset_sec * fps)))
    while pos + src_bar_int <= n_frames:
        bar = latents[pos:pos + src_bar_int]
        remapped = _remap_latent_bar(bar, src_n, src_den, tgt_n, tgt_den,
                                     frames_per_beat, frames_per_eighth)
        out_chunks.append(remapped)
        pos += src_bar_int

    if not out_chunks:
        return latents
    return torch.cat(out_chunks, dim=0)


@app.route("/api/encode-latents-bulk", methods=["POST"])
def encode_latents_bulk():
    """Removed — bulk encoding is now done client-side via latentEncoder.js (WebGPU/WASM)."""
    return jsonify({"error": "Endpoint removed. Use client-side latentEncoder.js."}), 410


_BAR_STARTS_CACHE = {}  # source audio path → (bar_starts_samples, sr)

def _detect_bar_starts(src_wav_path):
    """Run beat_this once on a source audio file and return
    (bar_starts_samples_list, sr) for the TRIMMED-to-first-downbeat
    audio. Result is cached by path so multiple stems share the same
    boundaries within a song. Returns None on failure."""
    if src_wav_path in _BAR_STARTS_CACHE:
        return _BAR_STARTS_CACHE[src_wav_path]
    try:
        from beat_this.inference import File2Beats
        import soundfile as _sf
        f2b = File2Beats(checkpoint_path="final0", dbn=True)
        beats, downbeats = f2b(src_wav_path)
        if len(downbeats) < 2:
            return None
        y_full, sr = _sf.read(src_wav_path, always_2d=True)
        offset_sec = float(downbeats[0])
        offset_samples = int(round(offset_sec * sr))
        starts = [int(round(float(t) * sr)) - offset_samples for t in downbeats]
        starts.append(len(y_full) - offset_samples)
        _BAR_STARTS_CACHE[src_wav_path] = (starts, sr, offset_samples)
        return _BAR_STARTS_CACHE[src_wav_path]
    except Exception as e:
        print(f"[bar_starts] detection failed for {src_wav_path}: {e}")
        return None


def run_meter_change(task_id, src_wav_path, stem_type, src_bpm, tgt_bpm,
                     src_n, src_den, tgt_n, tgt_den, bar_starts_override=None):
    """Generic meter-change for ANY stem type and ANY meter combo.

    - drums  → drumsep into 6 sub-stems → per-substem _process_stem_pattern_aware
               (per-bar pattern detection: triplet bars get pluck-and-place
               on a perfect 16th grid, others use the splice / 4+1.5+1.5)
    - other  → _process_stem_pattern_aware directly on the trimmed audio

    All stems detect downbeats with beat_this and trim pre-roll, then use the
    REAL bar boundaries (not constant src_bar) for slicing — works correctly
    on songs with tempo drift. The user can pass `bar_starts_override` to
    share the same downbeats across multiple stems of the same song.
    """
    try:
        # Force-load time-sig-editor as the `server` module
        import importlib.util as _ilu
        if "tse_server" not in sys.modules:
            _spec = _ilu.spec_from_file_location("tse_server", "/home/arlo/do2/time-sig-editor/server.py")
            _tse = _ilu.module_from_spec(_spec)
            sys.modules["tse_server"] = _tse
            sys.modules["server"] = _tse
            _spec.loader.exec_module(_tse)
        TSE = sys.modules["tse_server"]
        import shutil, soundfile as _sf, torch as _torch

        out_dir = os.path.join("/scratch/stemphonic_outputs", task_id)
        os.makedirs(out_dir, exist_ok=True)
        is_drums = stem_type in ("drums", "drum_kit", "percussion")

        tasks[task_id] = {"status": "processing", "stage": "downbeats"}

        # ── 1. Detect bar boundaries (or use override from caller) ─────
        if bar_starts_override is not None:
            bar_starts, sr_ref, offset_samples = bar_starts_override
        else:
            db = _detect_bar_starts(src_wav_path)
            if db is None:
                tasks[task_id] = {"status": "failed", "error": "downbeat detection failed"}
                return
            bar_starts, sr_ref, offset_samples = db

        # ── 2. Trim source to first downbeat ───────────────────────────
        y_full, sr = _sf.read(src_wav_path, always_2d=True)
        if sr != sr_ref:
            tasks[task_id] = {"status": "failed", "error": f"sr mismatch {sr}!={sr_ref}"}
            return
        trimmed_path = os.path.join(out_dir, f"{stem_type}_trim.wav")
        _sf.write(trimmed_path, y_full[offset_samples:], sr)

        if is_drums:
            # ── 3a. Drum meter-change — TEMPORARILY DISABLED ─────────────
            # The latent drum meter-change path lives in
            # latent_editor.meter_change.latent_meter_change_drum_kit
            # and needs wiring: audio → VAE encode → latent_demucs
            # (drums) → latent_drumsep (6 substems) → latent_meter_change_
            # drum_kit → VAE decode → wav. Non-trivial rewire — deferring
            # until after the launch ship. The MDX23C-DrumSep + TSE
            # pluck-and-place path this used to call has been removed
            # along with audio-separator.
            #
            # TODO: wire latent_editor.meter_change.latent_meter_change_drum_kit.
            tasks[task_id] = {
                "status": "failed",
                "state": "FAILURE",
                "error": "drum meter-change temporarily disabled — "
                         "latent_editor integration pending",
            }
            return
        else:
            # ── 3b. Non-drum stem: pattern-aware directly ──────────────
            tasks[task_id]["stage"] = "process_stem"
            y, sr = _sf.read(trimmed_path, always_2d=True)
            wav = _torch.from_numpy(y.T).float()
            if sr != sr_ref:
                scale = sr / sr_ref
                wav._bar_starts = [int(round(b * scale)) for b in bar_starts]
            else:
                wav._bar_starts = bar_starts
            out = TSE._process_stem_pattern_aware(
                wav, sr, src_bpm, src_n, src_den, tgt_n, tgt_den,
                stem_type, task_id, "auto",
            )
            out_wav = os.path.join(out_dir, f"{stem_type}_meter.wav")
            _sf.write(out_wav, out.numpy().T, sr)
            file_path = f"/api/generate-stemphonic/download/{task_id}/{stem_type}_meter.wav"

        tasks[task_id] = {
            "status": "completed",
            "state": "SUCCESS",
            "result": {"file_paths": [file_path]},
        }
    except Exception as e:
        traceback.print_exc()
        tasks[task_id] = {"status": "failed", "state": "FAILURE", "error": str(e)}


# Backwards-compat alias for older callers
def run_drum_meter_change(task_id, src_wav_path, src_bpm, tgt_bpm,
                          src_n, src_den, tgt_n, tgt_den):
    return run_meter_change(task_id, src_wav_path, "drums", src_bpm, tgt_bpm,
                            src_n, src_den, tgt_n, tgt_den)


@app.route("/api/repaint-meter", methods=["POST"])
def repaint_meter_endpoint():
    """Repaint per-stem audio for a meter change.

    JSON body:
      {
        "stems": [
          {"latent_id": "abcd1234", "stem_type": "drums"},
          {"latent_id": "efgh5678", "stem_type": "bass"},
          ...
        ],
        "src_meter": [4, 4],
        "tgt_meter": [7, 8],
        "src_bpm": 120,
        "tgt_bpm": 120,
        "cover_noise": 0.55,
        "prompt": "preserve original style"
      }

    Returns task_ids for each stem repaint, plus splice frame info.
    """
    data = request.get_json(force=True) or {}
    stems = data.get("stems") or []
    if not stems:
        return jsonify({"error": "no stems provided"}), 400
    src_n, src_den = data.get("src_meter", [4, 4])
    tgt_n, tgt_den = data.get("tgt_meter", [4, 4])
    src_bpm = float(data.get("src_bpm", 120))
    tgt_bpm = float(data.get("tgt_bpm", src_bpm))
    cover_noise = float(data.get("cover_noise", 0.55))
    downbeat_offset_sec = float(data.get("downbeat_offset", 0.0) or 0.0)

    # Sanity: meter unchanged AND bpm unchanged → no-op
    if (src_n, src_den) == (tgt_n, tgt_den) and abs(src_bpm - tgt_bpm) < 0.1:
        return jsonify({"skipped": True, "reason": "no change"})

    # ── Detect downbeats ONCE for the whole song ──────────────────────
    # Find a "reference" stem to detect on (drums preferred — its kicks
    # give beat_this the cleanest signal). All other stems will reuse
    # the same bar boundaries so they stay phase-locked.
    ref_src_path = None
    drum_stem = next((s for s in stems if s.get("stem_type") in ("drums","drum_kit","percussion")), None)
    ref_stem = drum_stem or stems[0]
    try:
        ref_blob = torch.load(os.path.join(LATENT_DIR, f"{ref_stem.get('latent_id')}.pt"), map_location="cpu")
        ref_src_path = ref_blob.get("source_path")
    except Exception:
        pass
    shared_bar_starts = _detect_bar_starts(ref_src_path) if ref_src_path else None
    if shared_bar_starts:
        print(f"[repaint] detected {len(shared_bar_starts[0])-1} bars from {ref_src_path}")

    results = []
    for stem in stems:
        latent_id = stem.get("latent_id")
        stem_type = stem.get("stem_type", "other")
        latent_path = os.path.join(LATENT_DIR, f"{latent_id}.pt")
        if not os.path.exists(latent_path):
            results.append({"latent_id": latent_id, "error": "latent not found"})
            continue

        try:
            blob = torch.load(latent_path, map_location="cpu")
            lat = blob["latents"]  # [T, 64]
            n_frames = lat.shape[0]

            # ── Latent fast path for drums when substem latents are cached ──
            if stem_type in ("drums", "drum_kit"):
                sub_latents_meta = None
                for _t in DEMUCS_TASKS.values():
                    sl = _t.get("drum_substem_latents") or {}
                    parent_sl = _t.get("stem_latents") or {}
                    drums_meta = parent_sl.get("drums")
                    if drums_meta and drums_meta.get("latent_id") == latent_id and sl:
                        sub_latents_meta = sl
                        break
                if sub_latents_meta and shared_bar_starts:
                    try:
                        # Generalized latent-only drum meter-change.
                        # Returns processed latents per substem; the
                        # server NEVER decodes — the browser does that
                        # locally via WebGPU after fetching /api/latent.
                        from latent_editor.drum_meter_change import (
                            drum_meter_change_substems_to_latents,
                        )
                        import soundfile as _sf
                        bs_48 = list(shared_bar_starts[0])
                        substems_in = {}
                        for sname, smeta in sub_latents_meta.items():
                            sp = smeta.get("source_path")
                            if sp and os.path.exists(sp):
                                wav, sr = _sf.read(sp, always_2d=True)
                                substems_in[sname] = (wav, sr)
                        if not substems_in:
                            raise RuntimeError("no substem source wavs available for re-encode")
                        editor_rt = _get_latent_editor()
                        sframe = _get_silence_frame()
                        out_lats = drum_meter_change_substems_to_latents(
                            substems=substems_in,
                            bs_48=bs_48,
                            src_meter=(src_n, src_den),
                            tgt_meter=(tgt_n, tgt_den),
                            vae_gpu=handler.vae,
                            editor=editor_rt,
                            silence_frame=sframe,
                            verbose=True,
                        )
                        # Save each substem latent → returns latent IDs
                        # for the browser to fetch via /api/latent/<id>.
                        new_substem_latents = {}
                        for sname, L_out in out_lats.items():
                            new_sid = uuid.uuid4().hex[:12]
                            sp = os.path.join(LATENT_DIR, f"{new_sid}.pt")
                            torch.save({
                                "latents": L_out.cpu().float(),
                                "shape": list(L_out.shape),
                                "fps": 25,
                                "source": f"repaint_drum/{sname}",
                                "parent": sub_latents_meta[sname]["latent_id"],
                                "vae_version": _vae_version_hash(),
                            }, sp)
                            new_substem_latents[sname] = {
                                "latent_id": new_sid,
                                "latent_url": f"/api/latent/{new_sid}",
                                "n_frames": int(L_out.shape[0]),
                            }
                        results.append({
                            "latent_id": latent_id,
                            "stem_type": stem_type,
                            "drum_substem_latents": new_substem_latents,
                            "latent_path": True,
                            "vae_version": _vae_version_hash(),
                        })
                        continue
                    except Exception as lat_err:
                        traceback.print_exc()
                        print(f"[repaint] latent drum path failed, falling back: {lat_err}")

            # ── ALL stems: raw-audio path with per-bar pattern detection ──
            # Bypass the latent splice entirely. The new path supports any
            # source/target meter combo via _process_stem_pattern_aware:
            # triplet bars get pluck-and-place to a perfect 16th grid,
            # other bars get the splice (4+1.5+1.5 for 4/4→7/8, etc).
            src_wav = blob.get("source_path")
            if src_wav and os.path.exists(src_wav):
                task_id = str(uuid.uuid4())
                tasks[task_id] = {"status": "queued"}
                threading.Thread(
                    target=run_meter_change,
                    args=(task_id, src_wav, stem_type, src_bpm, tgt_bpm,
                          src_n, src_den, tgt_n, tgt_den, shared_bar_starts),
                    daemon=True,
                ).start()
                results.append({
                    "latent_id": latent_id,
                    "stem_type": stem_type,
                    "task_id": task_id,
                    "new_latent_id": latent_id,
                    "raw_audio_path": True,
                })
                continue
            else:
                print(f"[repaint] {stem_type} latent {latent_id} has no source_path; "
                      f"falling back to latent splice")

            # 1. Compute splice points + remap latent
            splice_frames = latent_compute_splice_frames(
                n_frames, blob.get("fps", 25), src_bpm, src_n, src_den, tgt_n, tgt_den
            )
            remapped = latent_remap_meter(lat, blob.get("fps", 25), src_bpm,
                                          src_n, src_den, tgt_n, tgt_den,
                                          stem_type=stem_type,
                                          downbeat_offset_sec=downbeat_offset_sec)
            # Per-stem-type cover noise: drums need lighter touch to
            # preserve transients, vocals need more to clean up edits.
            stem_noise = cover_noise
            if stem_type in ("drums", "drum_kit", "percussion", "electronic"):
                stem_noise = min(cover_noise, 0.40)
            elif stem_type in ("vocals", "lead_vox", "bg_vox", "choir", "synth_vox"):
                stem_noise = max(cover_noise, 0.65)

            # 2. Save remapped latent as a new id
            new_id = uuid.uuid4().hex[:12]
            new_path = os.path.join(LATENT_DIR, f"{new_id}.pt")
            torch.save({
                "latents": remapped.cpu().float(),
                "shape": list(remapped.shape),
                "fps": blob.get("fps", 25),
                "source": f"repaint_{stem_type}",
                "parent": latent_id,
                "splice_frames": splice_frames,
            }, new_path)

            # 3. Dispatch a stemphonic generation task using the remapped
            #    latent as cover_src_latents and the same audio as semantic
            #    context. Cover_noise applied at splice regions repaints
            #    transitions cleanly.
            task_id = str(uuid.uuid4())
            duration = float(remapped.shape[0]) / blob.get("fps", 25)
            payload = {
                "prompt": data.get("prompt") or f"{stem_type}, preserve original style",
                "instrument": stem_type if stem_type in ("piano", "guitar", "bass", "drums") else "",
                "lyrics": "[Instrumental]" if stem_type != "vocals" else "",
                "duration": min(120.0, duration),
                "steps": 50,
                "cfg": 7.0,
                "seed": 42,
                "cover_noise_strength": stem_noise,
                "audio_cover_strength": 1.0,
                # Custom: cover_src_latents_path lets run_generation skip
                # re-encoding and use the spliced latent directly.
                "cover_src_latents_path": new_path,
            }
            tasks[task_id] = {"status": "queued", "params": payload}
            threading.Thread(target=run_generation, args=(task_id, payload), daemon=True).start()
            results.append({
                "latent_id": latent_id,
                "stem_type": stem_type,
                "task_id": task_id,
                "new_latent_id": new_id,
                "splice_count": len(splice_frames),
                "src_frames": n_frames,
                "tgt_frames": remapped.shape[0],
            })
        except Exception as e:
            traceback.print_exc()
            results.append({"latent_id": latent_id, "error": str(e)})

    return jsonify({"results": results, "splice_meter": [src_n, src_den, tgt_n, tgt_den]})


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8765, threaded=True)
