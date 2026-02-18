import os
import sys
import time
import gc
import unicodedata
from pathlib import Path
from multiprocessing import Process, Lock

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

import tensorflow as tf
from basic_pitch.inference import predict
import pretty_midi

# === CONFIG ===
ALL_AUDIO_PATHS = Path("/home/arlo/Data/all_audio_paths3.txt")
PROTOOLS_ROOT = Path("/home/arlo/gcs-bucket/protools")
ENCODEC_OUT = Path("/home/arlo/Data/encodec_tokens")
BASICPITCH_OUT = Path("/home/arlo/gcs-bucket/BasicPitch/protools")
DONE_LOG = Path("/home/arlo/Data/encodec_done.txt")
SKIP_COUNT = 6000
SKIP_DONE = True
GPU_IDS = [1, 2, 3, 4]  # Use all 4 GPUs (e.g., Encodec on 3,4; BasicPitch on 1,2)

ENCODEC_OUT.mkdir(parents=True, exist_ok=True)
BASICPITCH_OUT.mkdir(parents=True, exist_ok=True)

# === Load previously completed paths ===
done_files = set()
if SKIP_DONE and DONE_LOG.exists():
    with open(DONE_LOG, "r") as f:
        done_files = set(line.strip() for line in f if line.strip())

# === Helpers ===
def normalize_path(p):
    return unicodedata.normalize("NFKC", p)

def load_paths():
    with open(ALL_AUDIO_PATHS, "r") as f:
        all_paths = [normalize_path(line.strip()) for line in f if line.strip().endswith(".wav") and "/prev/" not in line.lower()]
    return sorted(all_paths[SKIP_COUNT:])

def save_midi(note_events, midi_path):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.extend(pretty_midi.Note(
        velocity=int(note['velocity']),
        pitch=int(note['pitch']),
        start=float(note['start_time']),
        end=float(note['end_time'])
    ) for note in note_events)
    midi.instruments.append(instrument)
    midi.write(str(midi_path))

def select_model(sr, model_24, model_48):
    if sr == 48000:
        return model_48, 48000, 2
    elif sr == 24000:
        return model_24, 24000, 1
    else:
        chosen = model_48 if abs(sr - 48000) < abs(sr - 24000) else model_24
        target_sr = 48000 if chosen == model_48 else 24000
        channels = 2 if target_sr == 48000 else 1
        print(f"⚠️ Resampling {sr}Hz → {target_sr}Hz")
        return chosen, target_sr, channels

def worker(paths, gpu_id, lock):
    device_type = "encodec" if gpu_id in [3, 4] else "basicpitch"
    print(f"🚀 [GPU {gpu_id}] Starting {device_type} worker on {len(paths)} files")

    if device_type == "encodec":
        model_24 = EncodecModel.encodec_model_24khz()
        model_24.set_target_bandwidth(6.0)
        model_24.to(torch.device(f"cuda:{gpu_id}")).eval()
        model_48 = EncodecModel.encodec_model_48khz()
        model_48.set_target_bandwidth(6.0)
        model_48.to(torch.device(f"cuda:{gpu_id}")).eval()
    else:
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    new_done = []

    for idx, raw_path in enumerate(paths, 1):
        audio_file = Path(raw_path)
        try:
            real_path = str(audio_file.resolve())
        except FileNotFoundError:
            continue

        if SKIP_DONE and real_path in done_files:
            continue

        try:
            relative_path = audio_file.relative_to(PROTOOLS_ROOT)
            session_parts = relative_path.parts[:3]
            relative_session = Path(*session_parts)
        except ValueError:
            continue

        try:
            if device_type == "encodec":
                out_path = ENCODEC_OUT / relative_session / (audio_file.stem + ".pt")
                if out_path.exists():
                    new_done.append(real_path)
                    continue

                waveform, sr = torchaudio.load(audio_file)
                model, target_sr, channels = select_model(sr, model_24, model_48)
                if sr != target_sr or waveform.shape[0] != channels:
                    waveform = convert_audio(waveform, sr, target_sr, channels)
                if waveform.shape[0] != channels:
                    if channels == 2 and waveform.shape[0] == 1:
                        waveform = waveform.repeat(2, 1)
                    elif channels == 1 and waveform.shape[0] == 2:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    else:
                        raise ValueError(f"Channel mismatch: {waveform.shape[0]} ≠ {channels}")
                with torch.no_grad():
                    tokens = model.encode(waveform.unsqueeze(0).to(torch.device(f"cuda:{gpu_id}")))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(tokens, out_path)
                print(f"✅ [GPU {gpu_id}] Encodec saved: {out_path.name}")

            else:
                midi_path = BASICPITCH_OUT / relative_path.with_suffix(".mid")
                if midi_path.exists():
                    continue
                _, midi_data, _ = predict(str(audio_file))
                midi_path.parent.mkdir(parents=True, exist_ok=True)
                midi_data.write(str(midi_path))
                print(f"✅ [GPU {gpu_id}] MIDI saved: {midi_path.name}")

            new_done.append(real_path)
        except Exception as e:
            print(f"❌ [GPU {gpu_id}] {audio_file.name}: {e}")
        finally:
            if device_type == "encodec":
                torch.cuda.empty_cache()
            else:
                tf.keras.backend.clear_session()
            gc.collect()

    with lock:
        with open(DONE_LOG, "a") as f:
            for path in new_done:
                f.write(path + "\n")

# === MAIN ===
if __name__ == "__main__":
    all_paths = load_paths()
    print(f"🎧 Total audio files to process: {len(all_paths)}")

    chunks = [all_paths[i::len(GPU_IDS)] for i in range(len(GPU_IDS))]
    lock = Lock()
    workers = []
    for gpu_id, chunk in zip(GPU_IDS, chunks):
        p = Process(target=worker, args=(chunk, gpu_id, lock))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    print("\n🏁 All processing complete.")
