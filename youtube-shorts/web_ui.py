#!/usr/bin/env python3
"""Video Grid Music — interactive web UI.

Edit all parameters, generate videos with multi-stem ACE-Step audio,
and preview results in-browser.

Usage:
    eval "$(conda shell.bash hook)" && conda activate ace_step
    python web_ui.py                  # default port 8099
    python web_ui.py --port 8085      # custom port
"""

import argparse
import io
import os
import sys
import time
import threading
import uuid
import glob as globmod
import contextlib

# Setup paths before any generator imports
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

from generator import config
from generator.renderer import render_video
from generator.scenes import video_grid_music
from generator.music.engine import MusicEngine

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
app = FastAPI(title="Video Grid Music UI")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "test_gen")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_tasks = {}  # task_id -> {status, log, result_video, result_audio, params, ...}
_gen_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_videos():
    """Find all .mp4 files that can be used as input."""
    vids = []
    for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
        vids.extend(globmod.glob(os.path.join(PROJECT_DIR, ext)))
    return sorted(vids)


def _check_acestep():
    try:
        from generator.music.acestep_bridge import is_model_available
        return is_model_available()
    except Exception:
        return False


class LogCapture:
    """Capture stdout to a buffer while also printing to real stdout."""
    def __init__(self):
        self.buffer = io.StringIO()
        self._real = sys.stdout

    def write(self, s):
        self.buffer.write(s)
        self._real.write(s)

    def flush(self):
        self.buffer.flush()
        self._real.flush()

    def getvalue(self):
        return self.buffer.getvalue()


# ---------------------------------------------------------------------------
# Generation worker
# ---------------------------------------------------------------------------

def _run_generation(task_id, params):
    """Run in background thread."""
    import tempfile
    import shutil

    task = _tasks[task_id]
    log = LogCapture()
    old_stdout = sys.stdout
    sys.stdout = log
    task["log_capture"] = log

    try:
        task["status"] = "running"
        task["started_at"] = time.time()

        mode = params.pop("generation_mode", "acestep")
        skip_zsync = params.pop("skip_zsync", False)
        seed = int(params.get("seed", 42))
        input_video = params.get("input_video")
        if not input_video or not os.path.isfile(input_video):
            task["status"] = "error"
            task["error"] = f"Video not found: {input_video}"
            return

        scene_params = {
            "input_video": input_video,
            "grid_rows": int(params.get("grid_rows", 12)),
            "grid_cols": int(params.get("grid_cols", 16)),
            "target_duration": int(params.get("target_duration", 30)),
            "motion_threshold": float(params.get("motion_threshold", 6.0)),
            "smoothing_frames": int(params.get("smoothing_frames", 3)),
            "min_note_interval": float(params.get("min_note_interval", 0.05)),
            "brightness_note_scale": float(params.get("brightness_note_scale", 1.0)),
            "num_stems": int(params.get("num_stems", 3)),
            "stem_split_mode": params.get("stem_split_mode", "bands"),
            "show_grid": bool(params.get("show_grid", True)),
            "show_motion_bars": bool(params.get("show_motion_bars", True)),
            "show_z_overlay": bool(params.get("show_z_overlay", True)),
            "show_stem_bands": bool(params.get("show_stem_bands", True)),
            "palette": params.get("palette", "neon"),
            "retrigger_mode": params.get("retrigger_mode", "reactive"),
            "acestep_noise_level": float(params.get("acestep_noise_level", 0.7)),
            "acestep_steps": int(params.get("acestep_steps", 30)),
        }
        cfg_weight = float(params.get("cfg_weight", 3.0))

        out_prefix = f"webui_{task_id[:8]}"
        out_mp4 = os.path.join(OUTPUT_DIR, f"{out_prefix}.mp4")

        # ---- DRY RUN ----
        if mode == "dry-run":
            print(f"[dry-run] Collecting grid timeseries...")
            t0 = time.time()
            for _ in video_grid_music.run(
                seed=seed, params=scene_params,
                collect_events_only=True,
                collect_grid_timeseries=True,
            ):
                pass
            events = video_grid_music.get_collision_events()
            grid_ts = video_grid_music.get_grid_timeseries()
            elapsed = time.time() - t0
            print(f"[dry-run] {len(events)} events in {elapsed:.1f}s")
            if grid_ts:
                import numpy as np
                motion = grid_ts["motion"]
                print(f"[dry-run] Grid: {motion.shape}")
                print(f"[dry-run] Motion: mean={motion.mean():.2f}, max={motion.max():.2f}")
                active = int((motion > scene_params["motion_threshold"]).any(axis=-1).sum())
                print(f"[dry-run] Active cells: {active}/{motion.shape[0]*motion.shape[1]}")
                instruments = set(e.get("instrument", "?") for e in events)
                print(f"[dry-run] Instruments: {instruments}")

            # Still render video (no audio) so user can see the grid
            print(f"[dry-run] Rendering video (no audio)...")
            frame_gen = video_grid_music.run(seed=seed, params=scene_params)
            render_video(out_mp4, frame_gen)
            task["result_video"] = f"{out_prefix}.mp4"
            task["status"] = "done"
            return

        # ---- MIDI ONLY ----
        if mode == "midi-only":
            temp_dir = tempfile.mkdtemp(prefix="webui_midi_")
            engine = MusicEngine(temp_dir=temp_dir, seed=seed)

            print("[midi] Pass 1: Collecting events...")
            for _ in video_grid_music.run(
                seed=seed, params=scene_params, collect_events_only=True,
            ):
                pass
            events = video_grid_music.get_collision_events()
            print(f"[midi] {len(events)} events")

            audio_path = None
            if events:
                print("[midi] Generating MIDI audio...")
                audio_path = engine.generate_from_video_events(
                    events=events,
                    key=params.get("key", "C major"),
                    tempo=int(params.get("tempo", 120)),
                )

            print("[midi] Pass 2: Rendering video...")
            frame_gen = video_grid_music.run(seed=seed, params=scene_params)
            render_video(out_mp4, frame_gen, audio_path=audio_path)
            task["result_video"] = f"{out_prefix}.mp4"
            if audio_path:
                # Copy audio to output dir
                import shutil
                out_wav = os.path.join(OUTPUT_DIR, f"{out_prefix}.wav")
                shutil.copy2(audio_path, out_wav)
                task["result_audio"] = f"{out_prefix}.wav"
            shutil.rmtree(temp_dir, ignore_errors=True)
            task["status"] = "done"
            return

        # ---- ACE-STEP ----
        from generator.music.acestep_bridge import (
            generate_from_grid_analysis, generate_multi_stem_from_grid,
            post_process_z_for_sync, decode_z_to_audio, load_model,
            _split_grid_into_bands,
        )
        from generator.music.mixer import mix_stems_weighted, add_reverb

        num_stems = scene_params["num_stems"]
        target_duration = scene_params["target_duration"]
        grid_rows = scene_params["grid_rows"]
        grid_cols = scene_params["grid_cols"]

        # Pass 1
        print(f"[acestep] Pass 1: Collecting grid timeseries...")
        t0 = time.time()
        for _ in video_grid_music.run(
            seed=seed, params=scene_params,
            collect_events_only=True,
            collect_grid_timeseries=True,
        ):
            pass
        events = video_grid_music.get_collision_events()
        grid_ts = video_grid_music.get_grid_timeseries()
        print(f"[acestep] {len(events)} events, collected in {time.time()-t0:.1f}s")

        if grid_ts is None:
            task["status"] = "error"
            task["error"] = "No grid timeseries collected"
            return

        audio_path = None
        z_overlay_data = None
        stem_bands = None

        # Pass 2 — generation
        if num_stems > 1:
            print(f"[acestep] Pass 2: Multi-stem ({num_stems} stems)...")
            t0 = time.time()
            stem_results = generate_multi_stem_from_grid(
                motion_ts=grid_ts["motion"],
                hue_ts=grid_ts["hue"],
                sat_ts=grid_ts["saturation"],
                bright_ts=grid_ts["brightness"],
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                num_stems=num_stems,
                motion_threshold=scene_params["motion_threshold"],
                duration_sec=target_duration,
                seed=seed,
                noise_level=scene_params["acestep_noise_level"],
                steps=scene_params["acestep_steps"],
                cfg_weight=cfg_weight,
                temp_dir=OUTPUT_DIR,
                retrigger_mode=scene_params["retrigger_mode"],
            )
            print(f"[acestep] Generated {len(stem_results)} stems in {time.time()-t0:.1f}s")

            if stem_results:
                import torch
                # Offload transformer for decode
                try:
                    model = load_model()
                    if hasattr(model, 'transformers'):
                        model.transformers.cpu()
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                stem_audio_paths = []
                z_overlay_data = []
                stem_bands = []

                for sr in stem_results:
                    rs, re = sr["row_start"], sr["row_end"]
                    stem_bands.append((rs, re, sr["band_name"]))
                    final_audio = sr["audio_path"]
                    final_z = sr["z_flat"]

                    if not skip_zsync:
                        band_motion = grid_ts["motion"][rs:re, :, :]
                        z_ed, n_ed = post_process_z_for_sync(
                            sr["z_latents"], band_motion,
                            re - rs, grid_cols, target_duration,
                        )
                        if n_ed > 0:
                            try:
                                print(f"[zsync] Re-decoding {sr['band_name']} ({n_ed} corrections)...")
                                final_audio = decode_z_to_audio(
                                    z_ed, duration_sec=target_duration, temp_dir=OUTPUT_DIR,
                                )
                                T_lat = z_ed.shape[-1]
                                final_z = z_ed.reshape(1, 128, T_lat).permute(0, 2, 1).squeeze(0)
                            except Exception as e:
                                print(f"[zsync] Re-decode failed: {e}")

                    stem_audio_paths.append(final_audio)
                    z_overlay_data.append((final_z, rs, re))

                # Mix
                if len(stem_audio_paths) > 1:
                    gains, eq_bands = [], []
                    for sr in stem_results:
                        bn = sr["band_name"]
                        if bn == "bass":
                            gains.append(0.35); eq_bands.append({"high_cut_hz": 4000})
                        elif bn == "high":
                            gains.append(0.25); eq_bands.append({"low_cut_hz": 200})
                        else:
                            gains.append(0.30); eq_bands.append({"low_cut_hz": 80, "high_cut_hz": 10000})

                    mixed_path = os.path.join(OUTPUT_DIR, f"{out_prefix}_mixed.wav")
                    mix_stems_weighted(stem_audio_paths, mixed_path, gains=gains, eq_bands=eq_bands)
                    final_wav = os.path.join(OUTPUT_DIR, f"{out_prefix}.wav")
                    add_reverb(mixed_path, final_wav, reverberance=18, room_scale=45)
                    try:
                        os.remove(mixed_path)
                    except OSError:
                        pass
                    audio_path = final_wav
                    task["result_audio"] = f"{out_prefix}.wav"
                elif stem_audio_paths:
                    audio_path = stem_audio_paths[0]

        else:
            # Single stem
            print("[acestep] Pass 2: Single stem generation...")
            t0 = time.time()
            result = generate_from_grid_analysis(
                motion_ts=grid_ts["motion"],
                hue_ts=grid_ts["hue"],
                sat_ts=grid_ts["saturation"],
                bright_ts=grid_ts["brightness"],
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                motion_threshold=scene_params["motion_threshold"],
                duration_sec=target_duration,
                seed=seed,
                noise_level=scene_params["acestep_noise_level"],
                steps=scene_params["acestep_steps"],
                cfg_weight=cfg_weight,
                temp_dir=OUTPUT_DIR,
                retrigger_mode=scene_params["retrigger_mode"],
            )
            print(f"[acestep] Generated {result['group']}/{result['subgroup']} in {time.time()-t0:.1f}s")
            audio_path = result["audio_path"]
            z_overlay_data = result["z_flat"]

            if not skip_zsync:
                z_ed, n_ed = post_process_z_for_sync(
                    result["z_latents"], grid_ts["motion"],
                    grid_rows, grid_cols, target_duration,
                )
                if n_ed > 0:
                    try:
                        import torch
                        model = load_model()
                        if hasattr(model, 'transformers'):
                            model.transformers.cpu()
                            torch.cuda.empty_cache()
                        print(f"[zsync] Re-decoding ({n_ed} corrections)...")
                        audio_path = decode_z_to_audio(
                            z_ed, duration_sec=target_duration, temp_dir=OUTPUT_DIR,
                        )
                        T_lat = z_ed.shape[-1]
                        z_overlay_data = z_ed.reshape(1, 128, T_lat).permute(0, 2, 1).squeeze(0)
                    except Exception as e:
                        print(f"[zsync] Re-decode failed: {e}")

        # Fallback
        if audio_path is None and events:
            print("[fallback] Using MIDI audio...")
            temp_dir = tempfile.mkdtemp(prefix="webui_fb_")
            engine = MusicEngine(temp_dir=temp_dir, seed=seed)
            audio_path = engine.generate_from_video_events(
                events=events, key="C major", tempo=120,
            )

        # Pass 3 — render
        print(f"[acestep] Pass 3: Rendering video...")
        frame_gen = video_grid_music.run(
            seed=seed, params=scene_params,
            z_overlay_data=z_overlay_data,
            stem_bands=stem_bands,
        )
        render_video(out_mp4, frame_gen, audio_path=audio_path)
        task["result_video"] = f"{out_prefix}.mp4"
        task["status"] = "done"
        print(f"[done] Output: {out_mp4}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        task["status"] = "error"
        task["error"] = str(e)
    finally:
        sys.stdout = old_stdout
        task["finished_at"] = time.time()
        _gen_lock.release()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/videos")
def list_videos():
    return {"videos": _list_videos()}


@app.get("/api/status")
def get_status():
    return {
        "acestep_available": _check_acestep(),
        "output_dir": OUTPUT_DIR,
        "running_task": next(
            (tid for tid, t in _tasks.items() if t["status"] == "running"), None
        ),
    }


@app.post("/api/generate")
def start_generation(params: dict):
    if not _gen_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409,
            content={"error": "Generation already running"},
        )

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "queued",
        "log_capture": None,
        "result_video": None,
        "result_audio": None,
        "error": None,
        "params": dict(params),
        "created_at": time.time(),
    }

    t = threading.Thread(target=_run_generation, args=(task_id, dict(params)), daemon=True)
    t.start()
    return {"task_id": task_id}


@app.get("/api/task/{task_id}")
def get_task(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    log_text = ""
    if task.get("log_capture"):
        log_text = task["log_capture"].getvalue()

    elapsed = None
    if task.get("started_at"):
        end = task.get("finished_at") or time.time()
        elapsed = round(end - task["started_at"], 1)

    return {
        "status": task["status"],
        "log": log_text,
        "result_video": task["result_video"],
        "result_audio": task["result_audio"],
        "error": task["error"],
        "elapsed": elapsed,
    }


@app.get("/api/output/{filename}")
def serve_output(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    media_type = "video/mp4" if filename.endswith(".mp4") else "audio/wav"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    dest = os.path.join(PROJECT_DIR, file.filename)
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"path": dest, "filename": file.filename}


# ---------------------------------------------------------------------------
# Inline HTML UI
# ---------------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<base href="/videos/">
<title>Video Grid Music</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Segoe UI', Roboto, monospace; background: #0d1117; color: #c9d1d9; }
.layout { display: flex; height: 100vh; }
.panel-left { width: 380px; overflow-y: auto; border-right: 1px solid #21262d; padding: 16px; flex-shrink: 0; }
.panel-right { flex: 1; display: flex; flex-direction: column; padding: 16px; overflow: hidden; }

h1 { font-size: 18px; color: #58a6ff; margin-bottom: 12px; }
h2 { font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px;
     margin: 16px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #21262d; }

.field { margin-bottom: 10px; }
.field label { display: flex; justify-content: space-between; align-items: center;
               font-size: 12px; color: #8b949e; margin-bottom: 3px; }
.field label .val { color: #58a6ff; font-weight: 600; min-width: 40px; text-align: right; }
.field input[type=range] { width: 100%; accent-color: #58a6ff; }
.field input[type=number] { width: 100%; background: #161b22; border: 1px solid #30363d;
    color: #c9d1d9; padding: 6px 8px; border-radius: 4px; font-size: 13px; }
.field select { width: 100%; background: #161b22; border: 1px solid #30363d;
    color: #c9d1d9; padding: 6px 8px; border-radius: 4px; font-size: 13px; }

.check-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.check-row input { accent-color: #58a6ff; }
.check-row label { font-size: 12px; color: #c9d1d9; cursor: pointer; }

.radio-group { display: flex; gap: 12px; margin-bottom: 10px; }
.radio-group label { font-size: 12px; display: flex; align-items: center; gap: 4px; cursor: pointer; }
.radio-group input { accent-color: #58a6ff; }

.presets { display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }
.presets button { background: #21262d; border: 1px solid #30363d; color: #8b949e;
    padding: 4px 10px; border-radius: 12px; font-size: 11px; cursor: pointer; }
.presets button:hover { background: #30363d; color: #c9d1d9; }

.btn-generate { width: 100%; padding: 12px; background: #238636; border: none; color: #fff;
    font-size: 14px; font-weight: 600; border-radius: 6px; cursor: pointer; margin-top: 12px; }
.btn-generate:hover { background: #2ea043; }
.btn-generate:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }

.btn-stop { width: 100%; padding: 8px; background: #da3633; border: none; color: #fff;
    font-size: 12px; border-radius: 6px; cursor: pointer; margin-top: 6px; display: none; }

.status-bar { font-size: 12px; padding: 8px; background: #161b22; border-radius: 4px;
    margin-top: 8px; color: #8b949e; min-height: 32px; }
.status-bar.running { color: #d29922; }
.status-bar.done { color: #3fb950; }
.status-bar.error { color: #f85149; }

.video-container { flex: 1; display: flex; align-items: center; justify-content: center;
    background: #010409; border-radius: 8px; overflow: hidden; min-height: 0; margin-bottom: 12px; position: relative; }
.video-container video { max-width: 100%; max-height: 100%; border-radius: 8px; }
.video-placeholder { color: #30363d; font-size: 14px; }

.log-container { height: 200px; flex-shrink: 0; }
.log-container textarea { width: 100%; height: 100%; background: #0d1117; border: 1px solid #21262d;
    color: #7ee787; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 11px;
    padding: 8px; resize: none; border-radius: 6px; }

.download-links { display: flex; gap: 8px; margin-bottom: 8px; }
.download-links a { color: #58a6ff; font-size: 12px; text-decoration: none; }
.download-links a:hover { text-decoration: underline; }

.upload-row { display: flex; gap: 6px; margin-top: 6px; }
.upload-row input[type=file] { font-size: 11px; color: #8b949e; flex: 1; }
.upload-row button { background: #21262d; border: 1px solid #30363d; color: #8b949e;
    padding: 3px 10px; border-radius: 4px; font-size: 11px; cursor: pointer; }
</style>
</head>
<body>
<div class="layout">

<!-- LEFT PANEL: PARAMETERS -->
<div class="panel-left">
    <h1>Video Grid Music</h1>

    <div class="presets">
        <button onclick="applyPreset('fast')">Fast Preview</button>
        <button onclick="applyPreset('midi')">MIDI Quick</button>
        <button onclick="applyPreset('full')">Full 3-Stem</button>
        <button onclick="applyPreset('quick_ace')">Quick ACE 1-Stem</button>
    </div>

    <!-- VIDEO -->
    <h2>Video</h2>
    <div class="field">
        <label>Input Video</label>
        <select id="p_input_video"></select>
        <div class="upload-row">
            <input type="file" id="upload_file" accept="video/*">
            <button onclick="uploadVideo()">Upload</button>
        </div>
    </div>
    <div class="field">
        <label>Duration (s) <span class="val" id="v_target_duration">30</span></label>
        <input type="range" id="p_target_duration" min="5" max="60" step="1" value="30"
               oninput="updateVal(this)">
    </div>

    <!-- GRID -->
    <h2>Grid</h2>
    <div class="field">
        <label>Rows <span class="val" id="v_grid_rows">12</span></label>
        <input type="range" id="p_grid_rows" min="4" max="24" step="1" value="12"
               oninput="updateVal(this)">
    </div>
    <div class="field">
        <label>Columns <span class="val" id="v_grid_cols">16</span></label>
        <input type="range" id="p_grid_cols" min="4" max="32" step="1" value="16"
               oninput="updateVal(this)">
    </div>
    <div class="field">
        <label>Motion Threshold <span class="val" id="v_motion_threshold">6.0</span></label>
        <input type="range" id="p_motion_threshold" min="1" max="20" step="0.5" value="6"
               oninput="updateVal(this)">
    </div>
    <div class="field">
        <label>Smoothing Frames <span class="val" id="v_smoothing_frames">3</span></label>
        <input type="range" id="p_smoothing_frames" min="1" max="10" step="1" value="3"
               oninput="updateVal(this)">
    </div>
    <div class="field">
        <label>Min Note Interval <span class="val" id="v_min_note_interval">0.05</span></label>
        <input type="range" id="p_min_note_interval" min="0.01" max="0.2" step="0.01" value="0.05"
               oninput="updateVal(this)">
    </div>
    <div class="field">
        <label>Brightness Scale <span class="val" id="v_brightness_note_scale">1.0</span></label>
        <input type="range" id="p_brightness_note_scale" min="0.1" max="3" step="0.1" value="1.0"
               oninput="updateVal(this)">
    </div>

    <!-- STEMS -->
    <h2>Stems</h2>
    <div class="field">
        <label>Num Stems <span class="val" id="v_num_stems">3</span></label>
        <input type="range" id="p_num_stems" min="1" max="4" step="1" value="3"
               oninput="updateVal(this)">
    </div>
    <div class="field">
        <label>Split Mode</label>
        <select id="p_stem_split_mode">
            <option value="bands" selected>Horizontal Bands</option>
        </select>
    </div>

    <!-- ACE-STEP -->
    <h2>ACE-Step</h2>
    <div class="field">
        <label>Seed</label>
        <input type="number" id="p_seed" value="42" min="0" max="999999">
    </div>
    <div class="field">
        <label>Retrigger Mode</label>
        <select id="p_retrigger_mode">
            <option value="reactive" selected>Reactive</option>
            <option value="steady">Steady</option>
        </select>
    </div>
    <div class="field">
        <label>Noise Level <span class="val" id="v_acestep_noise_level">0.70</span></label>
        <input type="range" id="p_acestep_noise_level" min="0" max="1.5" step="0.05" value="0.7"
               oninput="updateVal(this)">
    </div>
    <div class="field">
        <label>Diffusion Steps <span class="val" id="v_acestep_steps">30</span></label>
        <input type="range" id="p_acestep_steps" min="5" max="60" step="1" value="30"
               oninput="updateVal(this)">
    </div>
    <div class="field">
        <label>CFG Weight <span class="val" id="v_cfg_weight">3.0</span></label>
        <input type="range" id="p_cfg_weight" min="1" max="7" step="0.5" value="3"
               oninput="updateVal(this)">
    </div>
    <div class="check-row">
        <input type="checkbox" id="p_skip_zsync">
        <label for="p_skip_zsync">Skip Z-Sync (faster)</label>
    </div>

    <!-- DISPLAY -->
    <h2>Display</h2>
    <div class="check-row">
        <input type="checkbox" id="p_show_grid" checked>
        <label for="p_show_grid">Show Grid</label>
    </div>
    <div class="check-row">
        <input type="checkbox" id="p_show_motion_bars" checked>
        <label for="p_show_motion_bars">Show Motion Bars</label>
    </div>
    <div class="check-row">
        <input type="checkbox" id="p_show_z_overlay" checked>
        <label for="p_show_z_overlay">Show Z-Overlay</label>
    </div>
    <div class="check-row">
        <input type="checkbox" id="p_show_stem_bands" checked>
        <label for="p_show_stem_bands">Show Stem Bands</label>
    </div>
    <div class="field">
        <label>Palette</label>
        <select id="p_palette">
            <option value="neon" selected>Neon</option>
            <option value="classic">Classic</option>
            <option value="sunset">Sunset</option>
            <option value="ocean">Ocean</option>
            <option value="candy">Candy</option>
        </select>
    </div>

    <!-- MODE -->
    <h2>Generation Mode</h2>
    <div class="radio-group">
        <label><input type="radio" name="mode" value="acestep" checked> ACE-Step</label>
        <label><input type="radio" name="mode" value="midi-only"> MIDI Only</label>
        <label><input type="radio" name="mode" value="dry-run"> Dry Run</label>
    </div>

    <button class="btn-generate" id="btn_gen" onclick="startGeneration()">Generate</button>
    <div class="status-bar" id="status_bar">Ready</div>
</div>

<!-- RIGHT PANEL: OUTPUT -->
<div class="panel-right">
    <div class="download-links" id="download_links" style="display:none">
        <a id="dl_video" href="#" download>Download .mp4</a>
        <a id="dl_audio" href="#" download style="display:none">Download .wav</a>
    </div>
    <div class="video-container" id="video_container">
        <video id="player" controls style="display:none"></video>
        <div class="video-placeholder" id="placeholder">Generate a video to preview</div>
    </div>
    <div class="log-container">
        <textarea id="log_area" readonly placeholder="Generation log will appear here..."></textarea>
    </div>
</div>

</div>

<script>
let currentTaskId = null;
let pollInterval = null;

function updateVal(el) {
    const id = el.id.replace('p_', 'v_');
    const span = document.getElementById(id);
    if (span) span.textContent = el.value;
}

// Load available videos
async function loadVideos() {
    const resp = await fetch('api/videos');
    const data = await resp.json();
    const sel = document.getElementById('p_input_video');
    sel.innerHTML = '';
    data.videos.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v.split('/').pop();
        sel.appendChild(opt);
    });
}

async function uploadVideo() {
    const input = document.getElementById('upload_file');
    if (!input.files.length) return;
    const form = new FormData();
    form.append('file', input.files[0]);
    const resp = await fetch('api/upload-video', { method: 'POST', body: form });
    const data = await resp.json();
    if (data.path) {
        await loadVideos();
        // Select the uploaded file
        document.getElementById('p_input_video').value = data.path;
    }
}

function getParams() {
    const mode = document.querySelector('input[name=mode]:checked').value;
    return {
        generation_mode: mode,
        input_video: document.getElementById('p_input_video').value,
        target_duration: +document.getElementById('p_target_duration').value,
        grid_rows: +document.getElementById('p_grid_rows').value,
        grid_cols: +document.getElementById('p_grid_cols').value,
        motion_threshold: +document.getElementById('p_motion_threshold').value,
        smoothing_frames: +document.getElementById('p_smoothing_frames').value,
        min_note_interval: +document.getElementById('p_min_note_interval').value,
        brightness_note_scale: +document.getElementById('p_brightness_note_scale').value,
        num_stems: +document.getElementById('p_num_stems').value,
        stem_split_mode: document.getElementById('p_stem_split_mode').value,
        seed: +document.getElementById('p_seed').value,
        retrigger_mode: document.getElementById('p_retrigger_mode').value,
        acestep_noise_level: +document.getElementById('p_acestep_noise_level').value,
        acestep_steps: +document.getElementById('p_acestep_steps').value,
        cfg_weight: +document.getElementById('p_cfg_weight').value,
        skip_zsync: document.getElementById('p_skip_zsync').checked,
        show_grid: document.getElementById('p_show_grid').checked,
        show_motion_bars: document.getElementById('p_show_motion_bars').checked,
        show_z_overlay: document.getElementById('p_show_z_overlay').checked,
        show_stem_bands: document.getElementById('p_show_stem_bands').checked,
        palette: document.getElementById('p_palette').value,
    };
}

function setParam(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.type === 'checkbox') el.checked = value;
    else if (el.type === 'radio') {
        document.querySelectorAll(`input[name="${el.name}"]`).forEach(r => r.checked = r.value === value);
    } else el.value = value;
    updateVal(el);
}

function applyPreset(name) {
    if (name === 'fast') {
        setParam('p_target_duration', 10);
        setParam('p_num_stems', 1);
        document.querySelector('input[name=mode][value="dry-run"]').checked = true;
    } else if (name === 'midi') {
        setParam('p_target_duration', 10);
        setParam('p_num_stems', 1);
        setParam('p_acestep_steps', 15);
        document.querySelector('input[name=mode][value="midi-only"]').checked = true;
    } else if (name === 'full') {
        setParam('p_target_duration', 30);
        setParam('p_num_stems', 3);
        setParam('p_acestep_steps', 30);
        setParam('p_acestep_noise_level', 0.7);
        setParam('p_cfg_weight', 3);
        document.querySelector('input[name=mode][value="acestep"]').checked = true;
    } else if (name === 'quick_ace') {
        setParam('p_target_duration', 10);
        setParam('p_num_stems', 1);
        setParam('p_acestep_steps', 15);
        setParam('p_skip_zsync', true);
        document.querySelector('input[name=mode][value="acestep"]').checked = true;
    }
}

async function startGeneration() {
    const btn = document.getElementById('btn_gen');
    const bar = document.getElementById('status_bar');
    const logEl = document.getElementById('log_area');

    btn.disabled = true;
    bar.className = 'status-bar running';
    bar.textContent = 'Starting...';
    logEl.value = '';
    document.getElementById('download_links').style.display = 'none';

    const params = getParams();

    try {
        const resp = await fetch('api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        const data = await resp.json();

        if (data.error) {
            bar.className = 'status-bar error';
            bar.textContent = data.error;
            btn.disabled = false;
            return;
        }

        currentTaskId = data.task_id;
        pollInterval = setInterval(pollTask, 1500);
    } catch (e) {
        bar.className = 'status-bar error';
        bar.textContent = 'Request failed: ' + e.message;
        btn.disabled = false;
    }
}

async function pollTask() {
    if (!currentTaskId) return;
    try {
        const resp = await fetch(`api/task/${currentTaskId}`);
        const data = await resp.json();
        const bar = document.getElementById('status_bar');
        const logEl = document.getElementById('log_area');

        logEl.value = data.log || '';
        logEl.scrollTop = logEl.scrollHeight;

        const elapsed = data.elapsed ? `${data.elapsed}s` : '';

        if (data.status === 'running') {
            bar.className = 'status-bar running';
            bar.textContent = `Running... ${elapsed}`;
        } else if (data.status === 'done') {
            clearInterval(pollInterval);
            bar.className = 'status-bar done';
            bar.textContent = `Done in ${elapsed}`;
            document.getElementById('btn_gen').disabled = false;

            if (data.result_video) {
                const url = `api/output/${data.result_video}?t=${Date.now()}`;
                const player = document.getElementById('player');
                player.src = url;
                player.style.display = 'block';
                document.getElementById('placeholder').style.display = 'none';
                player.load();
                player.play().catch(() => {});

                document.getElementById('download_links').style.display = 'flex';
                document.getElementById('dl_video').href = url;
                document.getElementById('dl_video').download = data.result_video;
            }
            if (data.result_audio) {
                const aurl = `api/output/${data.result_audio}`;
                const dla = document.getElementById('dl_audio');
                dla.href = aurl;
                dla.download = data.result_audio;
                dla.style.display = 'inline';
            }
        } else if (data.status === 'error') {
            clearInterval(pollInterval);
            bar.className = 'status-bar error';
            bar.textContent = `Error: ${data.error || 'unknown'} (${elapsed})`;
            document.getElementById('btn_gen').disabled = false;
        }
    } catch (e) {
        // Ignore transient fetch errors
    }
}

// Init
loadVideos();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Video Grid Music Web UI")
    parser.add_argument("--port", type=int, default=8099, help="Port (default: 8099)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"Starting Video Grid Music UI on http://{args.host}:{args.port}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"ACE-Step available: {_check_acestep()}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
