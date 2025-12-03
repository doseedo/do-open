#!/usr/bin/env python3
"""
Unified Preprocessing Script for trainer_performerCN2.py (Instrumental Mode)

Extracts all required formats with toggleable options:
  - dcae_latents: DCAE latents [8, 16, T_slow] @ 10.77 Hz
  - encodec: EnCodec tokens [8, T_fast] @ 75 Hz
  - midi: BasicPitch MIDI extraction
  - piano_roll: Piano roll [128, T_slow] from MIDI
  - conditioning: amp, rframe, rbend, f0, f0_masked, onsets

Usage:
  # Extract all formats
  python unified_preprocess.py --input /path/to/audio_list.txt --output /mnt/msdd/extracts

  # Extract specific formats only
  python unified_preprocess.py --input audio_list.txt --output /mnt/msdd/extracts \
      --formats dcae_latents,encodec,piano_roll

  # Multi-GPU processing
  python unified_preprocess.py --input audio_list.txt --output /mnt/msdd/extracts --gpus 0,1,2,3

  # Skip already processed files
  python unified_preprocess.py --input audio_list.txt --output /mnt/msdd/extracts --skip-existing

  # Generate manifest JSON
  python unified_preprocess.py --input audio_list.txt --output /mnt/msdd/extracts --manifest manifest.json

Outputs per audio file:
  <output>/<session>/<stem>.dcae.pt        - DCAE latents
  <output>/<session>/<stem>.encodec.pt     - EnCodec tokens
  <output>/<session>/<stem>.mid            - MIDI file
  <output>/<session>/<stem>.pianoroll.npy  - Piano roll
  <output>/<session>/<stem>.amp.npy        - RMS envelope
  <output>/<session>/<stem>.rframe.npy     - Voiced mask
  <output>/<session>/<stem>.rbend.npy      - Pitch bend
  <output>/<session>/<stem>.f0.npy         - F0 contour
  <output>/<session>/<stem>.f0_masked.npy  - F0 masked by voicing
  <output>/<session>/<stem>.onsets.npy     - Onset detection
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import gc
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from multiprocessing import Process, Queue, Lock
from collections import defaultdict
import re

import numpy as np
import torch
import torchaudio


# ===================== CONFIGURATION =====================
@dataclass
class Config:
    # Audio parameters
    sample_rate: int = 44100
    hop_length: int = 4096
    n_fft: int = 8192
    fmin: float = 65.41   # C2
    fmax: float = 2093.00  # C7

    # EnCodec parameters
    encodec_sr: int = 24000
    encodec_bandwidth: float = 6.0

    # DCAE parameters
    dcae_sr: int = 44100
    dcae_hop: int = 4096
    latent_shape: Tuple[int, int] = (8, 16)

    # Validation thresholds
    silence_threshold: float = 1e-6
    min_duration_sec: float = 0.5
    max_duration_sec: float = 720.0  # 12 minutes
    std_range: Tuple[float, float] = (0.4, 2.2)
    mean_range: Tuple[float, float] = (-1.3, 1.3)

    # BasicPitch parameters
    onset_threshold: float = 0.60
    frame_threshold: float = 0.20
    min_note_length_ms: int = 6

    # Exclude patterns (drums, etc.)
    exclude_keywords: List[str] = field(default_factory=lambda: [
        "Kick", "KICK", "KickIn", "KickOut", "kick",
        "Snare", "Snr", "SNR", "SNARE", "snare",
        "HiHat", "HH", "Hat", "HAT", "hihat",
        "Tom", "RackTom", "FloorTom", "TOM",
        "Cymbal", "Cym", "Crash", "Ride", "CYM",
        "OH", "Overhead", "OHL", "OHR",
        "Perc", "Tamb", "Cowbell", "Clap", "Shaker",
        "Drum", "Drums", "Drumkit", "Kit", "DRUM"
    ])

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length  # ~10.77 Hz

    @property
    def encodec_frame_rate(self) -> float:
        return self.encodec_sr / 320  # 75 Hz


# All available extraction formats
ALL_FORMATS = {
    "dcae_latents",
    "encodec",
    "midi",
    "piano_roll",
    "amp",
    "rframe",
    "rbend",
    "f0",
    "f0_masked",
    "onsets"
}

# Format groups for convenience
FORMAT_GROUPS = {
    "all": ALL_FORMATS,
    "conditioning": {"amp", "rframe", "rbend", "f0", "f0_masked", "onsets"},
    "audio_tokens": {"dcae_latents", "encodec"},
    "midi_related": {"midi", "piano_roll"},
    "minimal": {"dcae_latents", "encodec", "piano_roll", "amp", "rframe", "rbend"}
}


# ===================== LAZY MODEL LOADING =====================
class ModelCache:
    """Lazy-loaded model cache for GPU efficiency"""

    def __init__(self, checkpoint_dir: str, device: torch.device):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self._dcae_model = None
        self._encodec_model = None

    @property
    def dcae(self):
        if self._dcae_model is None:
            print(f"Loading DCAE model on {self.device}...")
            from acestep.pipeline_ace_step import ACEStepPipeline
            pipeline = ACEStepPipeline(checkpoint_dir=self.checkpoint_dir)
            pipeline.load_checkpoint(self.checkpoint_dir)
            self._dcae_model = pipeline.music_dcae.eval().to(self.device)
            print(f"DCAE model loaded on {self.device}")
        return self._dcae_model

    @property
    def encodec(self):
        if self._encodec_model is None:
            print(f"Loading EnCodec model on {self.device}...")
            from encodec import EncodecModel
            self._encodec_model = EncodecModel.encodec_model_24khz()
            self._encodec_model.set_target_bandwidth(6.0)
            self._encodec_model.to(self.device).eval()
            print(f"EnCodec model loaded on {self.device}")
        return self._encodec_model

    def clear(self):
        """Free GPU memory"""
        self._dcae_model = None
        self._encodec_model = None
        torch.cuda.empty_cache()
        gc.collect()


# ===================== EXTRACTION FUNCTIONS =====================

def load_audio(audio_path: Path, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
    """Load and resample audio to target sample rate"""
    waveform, sr = torchaudio.load(str(audio_path))

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr


def extract_dcae_latents(
    waveform: torch.Tensor,
    sr: int,
    model_cache: ModelCache,
    config: Config
) -> torch.Tensor:
    """Extract DCAE latents [8, 16, T_slow]"""

    # Ensure stereo for DCAE
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # Normalize
    waveform = waveform / (waveform.abs().max() + 1e-8)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        waveform = waveform.to(model_cache.device)
        audio_batch = waveform.unsqueeze(0).float()
        audio_lengths = torch.tensor([waveform.shape[-1]], device=model_cache.device)

        latents, _ = model_cache.dcae.encode(
            audios=audio_batch,
            audio_lengths=audio_lengths,
            sr=sr
        )

    latents = latents.float().squeeze(0).cpu()

    # Validate
    if not torch.isfinite(latents).all():
        raise ValueError("DCAE latents contain NaN/inf")
    if latents.shape[:2] != config.latent_shape:
        raise ValueError(f"DCAE shape {latents.shape[:2]} != {config.latent_shape}")

    return latents


def extract_encodec_tokens(
    waveform: torch.Tensor,
    sr: int,
    model_cache: ModelCache,
    config: Config
) -> torch.Tensor:
    """Extract EnCodec tokens [8, T_fast]"""
    from encodec.utils import convert_audio

    # Convert to 24kHz mono
    wf_24k = convert_audio(waveform, sr, config.encodec_sr, 1)

    with torch.no_grad():
        tokens = model_cache.encodec.encode(wf_24k.unsqueeze(0).to(model_cache.device))

    # tokens is a list of EncodedFrame, extract codes
    # Shape: [1, num_codebooks, time]
    codes = tokens[0][0]  # First (and only) frame's codes
    codes = codes.squeeze(0).cpu()  # [8, T_fast]

    return codes


def extract_midi_and_piano_roll(
    audio_path: Path,
    out_midi: Path,
    out_pr: Path,
    config: Config
) -> Dict[str, Any]:
    """Extract MIDI and piano roll using BasicPitch (subprocess for isolation)"""
    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile(prefix="bp_", suffix=".json", delete=False) as tf:
        result_json = tf.name

    code = f'''
import os, sys, json, traceback
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["BASIC_PITCH_DISABLE_TFLITE"] = "1"

from pathlib import Path
import numpy as np
import pretty_midi

audio = Path(sys.argv[1])
out_mid = Path(sys.argv[2])
out_pr = Path(sys.argv[3])
json_out = Path(sys.argv[4])

try:
    import basic_pitch
    from basic_pitch.inference import predict as basicpitch_predict

    onnx_model = Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx"

    _, midi_pm, _ = basicpitch_predict(
        str(audio),
        model_or_model_path=str(onnx_model),
        onset_threshold={config.onset_threshold},
        frame_threshold={config.frame_threshold},
        minimum_note_length={config.min_note_length_ms}
    )
    midi_pm.write(str(out_mid))

    pm = pretty_midi.PrettyMIDI(str(out_mid))
    pr = pm.get_piano_roll(fs={config.frame_rate})
    pr[pr > 0] = 1
    np.save(out_pr, pr.astype(np.uint8))

    with open(json_out, "w") as f:
        json.dump({{
            "midi_path": str(out_mid),
            "piano_roll_path": str(out_pr),
            "piano_roll_shape": [int(pr.shape[0]), int(pr.shape[1])],
            "frame_rate_hz": {config.frame_rate}
        }}, f)

except Exception as e:
    with open(json_out, "w") as f:
        json.dump({{"error": str(e), "trace": traceback.format_exc()}}, f)
    sys.exit(1)
'''

    proc = subprocess.run(
        [sys.executable, "-c", code, str(audio_path), str(out_midi), str(out_pr), result_json],
        capture_output=True, text=True
    )

    try:
        with open(result_json, "r") as f:
            data = json.load(f)
    except:
        data = {"error": "No JSON output", "stdout": proc.stdout, "stderr": proc.stderr}

    os.unlink(result_json)

    if "error" in data:
        raise RuntimeError(f"BasicPitch failed: {data['error']}")

    return data


def extract_conditioning_features(
    waveform: torch.Tensor,
    sr: int,
    config: Config
) -> Dict[str, np.ndarray]:
    """Extract conditioning features: amp, rframe, rbend, f0, f0_masked, onsets"""
    import torchcrepe

    device = "cpu"  # Use CPU for torchcrepe to avoid GPU memory issues

    y = waveform[0]  # Mono

    # Ensure minimum length
    win = config.n_fft
    hop = config.hop_length
    if y.numel() < win:
        y = torch.nn.functional.pad(y, (0, win - y.numel()))

    # RMS amplitude
    frames = y.unfold(0, win, hop)
    amp = torch.sqrt((frames ** 2).mean(dim=1) + 1e-12)
    if amp.max() > 0:
        amp = amp / amp.max()
    amp = amp.numpy().astype(np.float32)

    # F0 with torchcrepe
    y_crepe = y.unsqueeze(0).to(device)

    with torch.inference_mode():
        f0, periodicity = torchcrepe.predict(
            y_crepe,
            sample_rate=sr,
            hop_length=hop,
            fmin=config.fmin,
            fmax=config.fmax,
            pad=True,
            model="tiny",
            batch_size=128,
            device=device,
            return_periodicity=True
        )
        f0 = f0[0].cpu()
        periodicity = periodicity[0].cpu()

    # Align lengths
    T = min(len(amp), len(f0))
    amp = amp[:T]
    f0 = f0[:T]
    periodicity = periodicity[:T]

    # Voiced mask (rframe)
    vmask = ((periodicity > 0.5) & (torch.from_numpy(amp) > 0.02)).float().numpy()

    # Pitch bend (rbend) - semitones relative to A4
    f0_safe = np.where(f0.numpy() > 0, f0.numpy(), 1.0)
    rbend = 12.0 * np.log2(f0_safe / 440.0)
    rbend = np.where(np.isfinite(rbend), rbend, 0.0).astype(np.float32)
    rbend = rbend * vmask

    # F0 and masked F0
    f0_np = f0.numpy().astype(np.float32)
    f0_masked = f0_np * vmask

    # Onsets from amp derivative
    amp_diff = np.diff(amp, prepend=amp[0])
    onsets = (amp_diff > 0.05).astype(np.float32)

    return {
        "amp": amp.astype(np.float32),
        "rframe": vmask.astype(np.float32),
        "rbend": rbend.astype(np.float32),
        "f0": f0_np,
        "f0_masked": f0_masked.astype(np.float32),
        "onsets": onsets.astype(np.float32)
    }


# ===================== MAIN PROCESSOR =====================

class UnifiedProcessor:
    def __init__(
        self,
        output_dir: Path,
        checkpoint_dir: str,
        formats: Set[str],
        config: Config,
        device: torch.device,
        skip_existing: bool = False
    ):
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.formats = formats
        self.config = config
        self.device = device
        self.skip_existing = skip_existing

        self.model_cache = None
        self.exclude_pattern = re.compile(
            "|".join(re.escape(w) for w in config.exclude_keywords),
            re.IGNORECASE
        )

        self.stats = defaultdict(int)

    def _init_models(self):
        """Lazy initialize models"""
        if self.model_cache is None:
            needs_gpu = bool(self.formats & {"dcae_latents", "encodec"})
            if needs_gpu:
                self.model_cache = ModelCache(self.checkpoint_dir, self.device)

    def _get_output_paths(self, audio_path: Path) -> Dict[str, Path]:
        """Get output paths for all formats"""
        # Extract session folder structure
        parts = audio_path.parts
        try:
            # Try to find gcs-bucket or protools in path
            for marker in ["gcs-bucket", "protools", "protoolsA"]:
                if marker in parts:
                    idx = parts.index(marker)
                    rel_parts = parts[idx + 1:-1]  # Skip marker and filename
                    break
            else:
                rel_parts = [audio_path.parent.name]
        except:
            rel_parts = [audio_path.parent.name]

        session_dir = self.output_dir.joinpath(*rel_parts)
        session_dir.mkdir(parents=True, exist_ok=True)

        stem = audio_path.stem
        return {
            "dcae_latents": session_dir / f"{stem}.dcae.pt",
            "encodec": session_dir / f"{stem}.encodec.pt",
            "midi": session_dir / f"{stem}.mid",
            "piano_roll": session_dir / f"{stem}.pianoroll.npy",
            "amp": session_dir / f"{stem}.amp.npy",
            "rframe": session_dir / f"{stem}.rframe.npy",
            "rbend": session_dir / f"{stem}.rbend.npy",
            "f0": session_dir / f"{stem}.f0.npy",
            "f0_masked": session_dir / f"{stem}.f0_masked.npy",
            "onsets": session_dir / f"{stem}.onsets.npy"
        }

    def _should_skip(self, audio_path: Path, out_paths: Dict[str, Path]) -> Optional[str]:
        """Check if file should be skipped"""
        # Check exclude pattern
        if self.exclude_pattern.search(str(audio_path)):
            return "excluded_pattern"

        # Check if all requested outputs exist
        if self.skip_existing:
            all_exist = all(
                out_paths[fmt].exists()
                for fmt in self.formats
                if fmt in out_paths
            )
            if all_exist:
                return "already_exists"

        return None

    def process_file(self, audio_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Process a single audio file"""
        self._init_models()

        out_paths = self._get_output_paths(audio_path)

        # Check skip conditions
        skip_reason = self._should_skip(audio_path, out_paths)
        if skip_reason:
            self.stats[f"skipped_{skip_reason}"] += 1
            return "skipped", {"reason": skip_reason}

        results = {"audio_path": str(audio_path)}

        try:
            # Load audio once
            waveform, sr = load_audio(audio_path, self.config.sample_rate)

            # Validate duration
            duration = waveform.shape[-1] / sr
            if duration < self.config.min_duration_sec:
                raise ValueError(f"Too short: {duration:.2f}s < {self.config.min_duration_sec}s")
            if duration > self.config.max_duration_sec:
                raise ValueError(f"Too long: {duration:.2f}s > {self.config.max_duration_sec}s")

            # Validate audio
            if waveform.abs().max() < self.config.silence_threshold:
                raise ValueError("Silent audio")
            if not torch.isfinite(waveform).all():
                raise ValueError("Audio contains NaN/inf")

            # Extract DCAE latents
            if "dcae_latents" in self.formats:
                latents = extract_dcae_latents(waveform, sr, self.model_cache, self.config)
                torch.save({
                    "latents": latents,
                    "length": latents.shape[-1],
                    "original_path": str(audio_path),
                    "original_duration": duration
                }, out_paths["dcae_latents"])
                results["latent_path"] = str(out_paths["dcae_latents"])

            # Extract EnCodec tokens
            if "encodec" in self.formats:
                tokens = extract_encodec_tokens(waveform, sr, self.model_cache, self.config)
                torch.save(tokens, out_paths["encodec"])
                results["encodec_path"] = str(out_paths["encodec"])

            # Extract MIDI and piano roll
            if "midi" in self.formats or "piano_roll" in self.formats:
                midi_result = extract_midi_and_piano_roll(
                    audio_path, out_paths["midi"], out_paths["piano_roll"], self.config
                )
                if "midi" in self.formats:
                    results["midi_path"] = str(out_paths["midi"])
                if "piano_roll" in self.formats:
                    results["piano_roll_path"] = str(out_paths["piano_roll"])

            # Extract conditioning features
            cond_formats = self.formats & {"amp", "rframe", "rbend", "f0", "f0_masked", "onsets"}
            if cond_formats:
                cond_features = extract_conditioning_features(waveform, sr, self.config)
                results["conditioning_paths"] = {}
                for fmt in cond_formats:
                    np.save(out_paths[fmt], cond_features[fmt])
                    results["conditioning_paths"][fmt] = str(out_paths[fmt])

            self.stats["success"] += 1
            return "success", results

        except Exception as e:
            self.stats["failed"] += 1
            return "failed", {
                "error": str(e),
                "error_type": type(e).__name__,
                "trace": traceback.format_exc()
            }

        finally:
            torch.cuda.empty_cache()
            gc.collect()


def gpu_worker(
    gpu_id: int,
    audio_paths: List[str],
    output_dir: Path,
    checkpoint_dir: str,
    formats: Set[str],
    config: Config,
    skip_existing: bool,
    progress_queue: Queue,
    result_queue: Queue
):
    """Worker process for a single GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    processor = UnifiedProcessor(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        formats=formats,
        config=config,
        device=device,
        skip_existing=skip_existing
    )

    results = []
    for i, audio_path in enumerate(audio_paths):
        status, result = processor.process_file(Path(audio_path))
        results.append({"path": audio_path, "status": status, **result})
        progress_queue.put((gpu_id, i + 1, len(audio_paths), status))

    result_queue.put((gpu_id, results, dict(processor.stats)))


def parse_formats(format_str: str) -> Set[str]:
    """Parse format string into set of formats"""
    if format_str.lower() == "all":
        return ALL_FORMATS.copy()

    formats = set()
    for part in format_str.split(","):
        part = part.strip().lower()
        if part in FORMAT_GROUPS:
            formats |= FORMAT_GROUPS[part]
        elif part in ALL_FORMATS:
            formats.add(part)
        else:
            raise ValueError(f"Unknown format: {part}. Available: {ALL_FORMATS | set(FORMAT_GROUPS.keys())}")

    return formats


def main():
    parser = argparse.ArgumentParser(
        description="Unified preprocessing for trainer_performerCN2.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available formats:
  dcae_latents  - DCAE latents [8, 16, T_slow]
  encodec       - EnCodec tokens [8, T_fast]
  midi          - MIDI file from BasicPitch
  piano_roll    - Piano roll [128, T_slow]
  amp           - RMS amplitude envelope
  rframe        - Voiced/unvoiced mask
  rbend         - Pitch bend (semitones from A4)
  f0            - F0 contour (Hz)
  f0_masked     - F0 masked by voicing
  onsets        - Onset detection

Format groups:
  all           - All formats
  conditioning  - amp, rframe, rbend, f0, f0_masked, onsets
  audio_tokens  - dcae_latents, encodec
  midi_related  - midi, piano_roll
  minimal       - dcae_latents, encodec, piano_roll, amp, rframe, rbend
"""
    )

    parser.add_argument("--input", required=True, help="Path to audio list file or single audio file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--checkpoint-dir", default="/home/arlo/Data/ACE-Step/checkpoints",
                        help="DCAE checkpoint directory")
    parser.add_argument("--formats", default="all",
                        help="Comma-separated formats or groups (default: all)")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs (default: 0)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files where all outputs already exist")
    parser.add_argument("--manifest", type=str, help="Output manifest JSON path")
    parser.add_argument("--group", type=str, help="Instrument group for manifest")
    parser.add_argument("--subgroup", type=str, help="Instrument subgroup for manifest")

    args = parser.parse_args()

    # Parse formats
    formats = parse_formats(args.formats)
    print(f"Extracting formats: {', '.join(sorted(formats))}")

    # Parse GPUs
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    print(f"Using GPUs: {gpu_ids}")

    # Load audio paths
    input_path = Path(args.input)
    if input_path.is_file() and input_path.suffix == ".txt":
        with open(input_path) as f:
            audio_paths = [line.strip() for line in f if line.strip().endswith(".wav")]
    elif input_path.is_file():
        audio_paths = [str(input_path)]
    else:
        raise ValueError(f"Input must be a .txt file or audio file: {input_path}")

    print(f"Found {len(audio_paths)} audio files")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config()

    # Split work across GPUs
    chunks = [audio_paths[i::len(gpu_ids)] for i in range(len(gpu_ids))]

    # Create queues
    progress_queue = Queue()
    result_queue = Queue()

    # Start workers
    workers = []
    for gpu_id, chunk in zip(gpu_ids, chunks):
        if not chunk:
            continue
        p = Process(target=gpu_worker, args=(
            gpu_id, chunk, output_dir, args.checkpoint_dir,
            formats, config, args.skip_existing,
            progress_queue, result_queue
        ))
        p.start()
        workers.append(p)

    # Monitor progress
    from tqdm import tqdm
    pbar = tqdm(total=len(audio_paths), desc="Processing")

    completed = 0
    while completed < len(audio_paths):
        try:
            gpu_id, current, total, status = progress_queue.get(timeout=1)
            pbar.update(1)
            pbar.set_postfix({"gpu": gpu_id, "status": status})
            completed += 1
        except:
            # Check if workers are still alive
            if not any(p.is_alive() for p in workers):
                break

    pbar.close()

    # Collect results
    all_results = []
    all_stats = defaultdict(int)

    for _ in workers:
        try:
            gpu_id, results, stats = result_queue.get(timeout=5)
            all_results.extend(results)
            for k, v in stats.items():
                all_stats[k] += v
        except:
            pass

    for p in workers:
        p.join()

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for k, v in sorted(all_stats.items()):
        print(f"  {k}: {v}")

    # Generate manifest if requested
    if args.manifest:
        manifest = []
        for r in all_results:
            if r["status"] != "success":
                continue

            entry = {
                "audio_path": r["audio_path"],
                "group": args.group or "guitar",
                "sub_group": args.subgroup or "undefined"
            }

            if "latent_path" in r:
                entry["latent_path"] = r["latent_path"]
            if "encodec_path" in r:
                entry["encodec_path"] = r["encodec_path"]
            if "piano_roll_path" in r:
                entry["piano_roll_path"] = r["piano_roll_path"]
            if "conditioning_paths" in r:
                entry["conditioning_paths"] = r["conditioning_paths"]

            manifest.append(entry)

        with open(args.manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest written: {args.manifest} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
