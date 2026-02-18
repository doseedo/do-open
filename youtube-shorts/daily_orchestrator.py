#!/usr/bin/env python3
"""Daily YouTube Shorts generator.

Selects scene types and music modes from master_ideas.json,
randomizes parameters, generates physics animations with music,
and outputs them to the upload queue.

Usage:
    python daily_orchestrator.py               # generate 5 random videos
    python daily_orchestrator.py --count 3     # generate 3 videos
    python daily_orchestrator.py --scene plinko # force a specific scene type
    python daily_orchestrator.py --no-music    # generate without music (silent)
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
import tempfile

# Ensure parent directory is on path so 'generator' package resolves
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set headless before importing pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from generator import config
from generator.renderer import render_video
from generator.scenes import get_scene, list_scenes, SCENE_REGISTRY
from generator.music.engine import MusicEngine

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_IDEAS_PATH = os.path.join(PROJECT_DIR, "master_ideas.json")


def load_master_ideas():
    with open(MASTER_IDEAS_PATH) as f:
        return json.load(f)


def select_daily_lineup(ideas, count, force_scene=None):
    """Select scene types for today. Ensures variety (max 2 of same type)."""
    available_scenes = [s for s in ideas["scene_types"] if get_scene(s) is not None]
    if not available_scenes:
        print("ERROR: No scenes registered!")
        sys.exit(1)

    lineup = []
    for _ in range(count):
        if force_scene:
            if force_scene not in available_scenes:
                print(f"ERROR: Scene '{force_scene}' not available. Available: {available_scenes}")
                sys.exit(1)
            chosen = force_scene
        else:
            # Avoid too many repeats
            candidates = [s for s in available_scenes if lineup.count(s) < 2]
            if not candidates:
                candidates = available_scenes
            chosen = random.choice(candidates)
        lineup.append(chosen)
    return lineup


def randomize_params(param_spec, rng):
    """Given a param spec from master_ideas.json, randomize within ranges."""
    params = {}
    for key, spec in param_spec.items():
        if isinstance(spec, dict) and "min" in spec and "max" in spec:
            if isinstance(spec["min"], float) or isinstance(spec["max"], float):
                params[key] = round(rng.uniform(spec["min"], spec["max"]), 3)
            else:
                params[key] = rng.randint(spec["min"], spec["max"])
        elif isinstance(spec, list):
            params[key] = rng.choice(spec)
        else:
            params[key] = spec
    return params


def pick_music_params(scene_spec, rng):
    """Randomly select music parameters."""
    mp = scene_spec["music_params"]
    return {
        "instrument": rng.choice(mp["instrument"]),
        "key": rng.choice(mp["key"]),
        "tempo": rng.randint(*mp["tempo_range"]),
        "pitch_mapping": mp.get("pitch_mapping", "y_position"),
    }


def generate_one_video(scene_name, ideas, video_num, no_music=False, input_video=None):
    """Generate a single video with music."""
    scene_spec = ideas["scene_types"][scene_name]
    seed = random.randint(0, 2**32 - 1)
    rng = random.Random(seed)
    timestamp = int(time.time())

    # Randomize parameters
    scene_params = randomize_params(scene_spec["params"], rng)

    # Pick palette
    palette_choices = scene_spec.get("palette_choices", ["classic"])
    scene_params["palette"] = rng.choice(palette_choices)

    # Pass input_video for video_grid_music scenes
    if scene_name == "video_grid_music" and input_video:
        scene_params["input_video"] = input_video

    # Pick music settings
    music_params = pick_music_params(scene_spec, rng)

    # Pick music mode
    if no_music:
        music_mode = "none"
    else:
        music_mode = rng.choice(scene_spec["compatible_music_modes"])

    # Temp directory for intermediate files
    temp_dir = tempfile.mkdtemp(prefix=f"yt_{scene_name}_")

    # Output paths
    filename = f"{scene_name}_{timestamp}_{video_num}"
    video_path = os.path.join(config.OUTPUT_DIR, f"{filename}.mp4")
    json_path = os.path.join(config.OUTPUT_DIR, f"{filename}.json")

    print(f"\n{'='*60}")
    print(f"  Scene: {scene_spec['display_name']}")
    print(f"  Seed: {seed}")
    print(f"  Music: {music_mode} | {music_params['instrument']} | {music_params['key']} | {music_params['tempo']} BPM")
    print(f"  Palette: {scene_params['palette']}")
    print(f"  Params: {scene_params}")
    print(f"{'='*60}")

    scene_module = get_scene(scene_name)
    music_engine = MusicEngine(temp_dir=temp_dir, seed=seed)
    audio_path = None

    try:
        if music_mode == "animation_drives_music":
            audio_path = _generate_animation_drives_music(
                scene_module, scene_params, seed, music_engine, music_params, video_path
            )
        elif music_mode == "music_drives_animation":
            audio_path = _generate_music_drives_animation(
                scene_module, scene_params, seed, music_engine, music_params, video_path
            )
        elif music_mode == "parallel_sync":
            audio_path = _generate_parallel_sync(
                scene_module, scene_params, seed, music_engine, music_params, video_path
            )
        else:
            # No music
            frame_gen = scene_module.run(seed=seed, params=scene_params)
            render_video(video_path, frame_gen)

    except Exception as e:
        print(f"  ERROR generating video: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: generate without music
        print("  Falling back to silent video...")
        try:
            frame_gen = scene_module.run(seed=seed, params=scene_params)
            render_video(video_path, frame_gen)
        except Exception as e2:
            print(f"  FATAL: Could not generate video at all: {e2}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    # Generate metadata
    title_template = rng.choice(scene_spec["title_templates"])
    try:
        title = title_template.format(num=video_num, **scene_params)
    except KeyError:
        title = title_template.format(num=video_num)

    tags = list(scene_spec["tags_base"]) + [music_params["instrument"], "music"]
    instrument_display = music_params["instrument"].replace("_", " ")

    metadata = {
        "title": title,
        "description": f"Satisfying {scene_spec['display_name']} with {instrument_display}! #Shorts #satisfying #physics",
        "tags": tags,
        "category": "24",
        "scene_type": scene_name,
        "music_mode": music_mode,
        "seed": seed,
        "params": scene_params,
        "music_params": music_params,
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Clean up temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"  Output: {video_path}")
    return video_path


def _generate_animation_drives_music(scene_module, params, seed, engine, music_params, video_path):
    """Two-pass (or three-pass for video_grid_music with ACE-Step)."""
    # Check if this is video_grid_music with ACE-Step available
    has_grid_ts = hasattr(scene_module, 'get_grid_timeseries')
    use_acestep = False
    if has_grid_ts:
        try:
            from generator.music.acestep_bridge import is_model_available
            use_acestep = is_model_available()
        except ImportError:
            pass

    if use_acestep and has_grid_ts:
        return _generate_acestep_video_grid(scene_module, params, seed, engine, music_params, video_path)

    # Standard two-pass for all other scenes
    print("  Pass 1: Collecting collision events...")

    for _ in scene_module.run(seed=seed, params=params, collect_events_only=True):
        pass
    events = scene_module.get_collision_events()
    print(f"  Collected {len(events)} collision events")

    audio_path = None
    if events:
        print("  Generating audio from events...")
        has_instruments = any("instrument" in ev for ev in events)
        if has_instruments:
            audio_path = engine.generate_from_video_events(
                events=events,
                key=music_params["key"],
                tempo=music_params["tempo"],
            )
        else:
            audio_path = engine.generate_from_events(
                events=events,
                key=music_params["key"],
                instrument=music_params["instrument"],
                tempo=music_params["tempo"],
                pitch_mapping=music_params.get("pitch_mapping", "y_position"),
            )

    print("  Pass 2: Rendering video...")
    frame_gen = scene_module.run(seed=seed, params=params)
    render_video(video_path, frame_gen, audio_path=audio_path)
    return audio_path


def _generate_acestep_video_grid(scene_module, params, seed, engine, music_params, video_path):
    """Three-pass ACE-Step generation for video_grid_music.

    Supports multi-stem generation: splits grid into horizontal bands,
    generates one ACE-Step stem per band, mixes them together.

    Pass 1: Collect grid timeseries + collision events (for fallback)
    Pass 2: Generate audio via ACE-Step (single or multi-stem) + z-space sync
    Pass 3: Render video with z-space overlay and stem band indicators
    """
    from generator.music.acestep_bridge import (
        generate_from_grid_analysis, generate_multi_stem_from_grid,
        post_process_z_for_sync, decode_z_to_audio,
        _split_grid_into_bands,
    )
    from generator.music.mixer import mix_stems_weighted, add_reverb

    target_duration = params.get("target_duration", 30)
    grid_rows = params.get("grid_rows", 12)
    grid_cols = params.get("grid_cols", 16)
    motion_threshold = params.get("motion_threshold", 6.0)
    num_stems = params.get("num_stems", 3)

    # Pass 1: collect grid timeseries
    print("  Pass 1: Collecting grid timeseries...")
    for _ in scene_module.run(seed=seed, params=params,
                              collect_events_only=True,
                              collect_grid_timeseries=True):
        pass
    events = scene_module.get_collision_events()
    grid_ts = scene_module.get_grid_timeseries()
    print(f"  Collected {len(events)} events, grid timeseries: "
          f"{grid_ts['motion'].shape if grid_ts else 'None'}")

    audio_path = None
    z_overlay_data = None
    stem_bands = None

    if grid_ts is not None:
        if num_stems > 1:
            # ---- Multi-stem ACE-Step generation ----
            audio_path, z_overlay_data, stem_bands = _generate_multi_stem_audio(
                grid_ts, grid_rows, grid_cols, num_stems,
                motion_threshold, target_duration, seed, params, video_path,
            )
        else:
            # ---- Single-stem ACE-Step generation (original path) ----
            try:
                print("  Pass 2: ACE-Step generation (single stem)...")
                result = generate_from_grid_analysis(
                    motion_ts=grid_ts["motion"],
                    hue_ts=grid_ts["hue"],
                    sat_ts=grid_ts["saturation"],
                    bright_ts=grid_ts["brightness"],
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    motion_threshold=motion_threshold,
                    duration_sec=target_duration,
                    seed=seed,
                    noise_level=params.get("acestep_noise_level", 0.7),
                    steps=params.get("acestep_steps", 30),
                    temp_dir=os.path.dirname(video_path),
                    retrigger_mode=params.get("retrigger_mode", "reactive"),
                )
                audio_path = result["audio_path"]
                z_latents = result["z_latents"]
                z_overlay_data = result["z_flat"]
                print(f"  Generated: {result['group']}/{result['subgroup']}")

                # Z-space sync correction
                z_edited, n_edits = post_process_z_for_sync(
                    z_latents, grid_ts["motion"], grid_rows, grid_cols, target_duration
                )
                if n_edits > 0:
                    try:
                        import torch
                        from generator.music.acestep_bridge import load_model
                        model = load_model()
                        if hasattr(model, 'transformers'):
                            model.transformers.cpu()
                            torch.cuda.empty_cache()
                            print(f"  Offloaded transformer to CPU, freeing VRAM for decode")

                        print(f"  Re-decoding after {n_edits} z-space sync corrections...")
                        audio_path = decode_z_to_audio(
                            z_edited, duration_sec=target_duration,
                            temp_dir=os.path.dirname(video_path),
                        )
                        T_lat = z_edited.shape[-1]
                        z_overlay_data = z_edited.reshape(1, 128, T_lat).permute(0, 2, 1).squeeze(0)
                    except Exception as re_err:
                        print(f"  Re-decode failed ({re_err}), using initial ACE-Step audio")

            except Exception as e:
                print(f"  ACE-Step generation failed: {e}")
                import traceback
                traceback.print_exc()
                audio_path = None
                z_overlay_data = None

    # Fallback to fluidsynth if ACE-Step failed
    if audio_path is None and events:
        print("  Falling back to fluidsynth...")
        audio_path = engine.generate_from_video_events(
            events=events,
            key=music_params["key"],
            tempo=music_params["tempo"],
        )

    # Pass 3: Render video with z-overlay and stem bands
    print("  Pass 3: Rendering video with z-overlay...")
    frame_gen = scene_module.run(
        seed=seed, params=params,
        z_overlay_data=z_overlay_data,
        stem_bands=stem_bands,
    )
    render_video(video_path, frame_gen, audio_path=audio_path)
    return audio_path


def _generate_multi_stem_audio(grid_ts, grid_rows, grid_cols, num_stems,
                                motion_threshold, target_duration, seed,
                                params, video_path):
    """Generate multiple ACE-Step stems from grid bands, mix them, apply z-sync.

    Returns:
        (audio_path, z_overlay_data, stem_bands) where:
            audio_path: path to mixed WAV
            z_overlay_data: list of (z_flat, row_start, row_end) for visualization
            stem_bands: list of (row_start, row_end, band_name) for band indicators
    """
    import torch
    from generator.music.acestep_bridge import (
        generate_multi_stem_from_grid, post_process_z_for_sync,
        decode_z_to_audio, _split_grid_into_bands, load_model,
    )
    from generator.music.mixer import mix_stems_weighted, add_reverb

    try:
        print(f"  Pass 2: Multi-stem ACE-Step generation ({num_stems} stems)...")

        stem_results = generate_multi_stem_from_grid(
            motion_ts=grid_ts["motion"],
            hue_ts=grid_ts["hue"],
            sat_ts=grid_ts["saturation"],
            bright_ts=grid_ts["brightness"],
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            num_stems=num_stems,
            motion_threshold=motion_threshold,
            duration_sec=target_duration,
            seed=seed,
            noise_level=params.get("acestep_noise_level", 0.7),
            steps=params.get("acestep_steps", 30),
            temp_dir=os.path.dirname(video_path),
            retrigger_mode=params.get("retrigger_mode", "reactive"),
        )

        if not stem_results:
            return None, None, None

        # Per-stem z-sync correction
        stem_audio_paths = []
        z_overlay_data = []
        stem_bands = []

        # Offload transformer once before all decodes
        try:
            model = load_model()
            if hasattr(model, 'transformers'):
                model.transformers.cpu()
                torch.cuda.empty_cache()
                print(f"  Offloaded transformer to CPU for stem decodes")
        except Exception:
            pass

        for sr in stem_results:
            band_name = sr["band_name"]
            row_start = sr["row_start"]
            row_end = sr["row_end"]
            stem_bands.append((row_start, row_end, band_name))

            # Z-sync correction per stem
            band_motion = grid_ts["motion"][row_start:row_end, :, :]
            z_edited, n_edits = post_process_z_for_sync(
                sr["z_latents"], band_motion,
                row_end - row_start, grid_cols, target_duration,
            )

            final_audio = sr["audio_path"]
            final_z_flat = sr["z_flat"]

            if n_edits > 0:
                try:
                    print(f"  Re-decoding {band_name} stem after {n_edits} sync corrections...")
                    final_audio = decode_z_to_audio(
                        z_edited, duration_sec=target_duration,
                        temp_dir=os.path.dirname(video_path),
                    )
                    T_lat = z_edited.shape[-1]
                    final_z_flat = z_edited.reshape(1, 128, T_lat).permute(0, 2, 1).squeeze(0)
                except Exception as e:
                    print(f"  Re-decode failed for {band_name} ({e}), using initial audio")

            stem_audio_paths.append(final_audio)
            z_overlay_data.append((final_z_flat, row_start, row_end))

        # Mix stems with register-appropriate EQ
        if len(stem_audio_paths) > 1:
            # Gains: bass slightly louder, high slightly quieter
            n = len(stem_audio_paths)
            gains = []
            eq_bands = []
            for i, sr in enumerate(stem_results):
                bname = sr["band_name"]
                if bname == "bass":
                    gains.append(0.35)
                    eq_bands.append({"high_cut_hz": 4000})
                elif bname == "high":
                    gains.append(0.25)
                    eq_bands.append({"low_cut_hz": 200})
                else:
                    gains.append(0.30)
                    eq_bands.append({"low_cut_hz": 80, "high_cut_hz": 10000})

            mixed_path = video_path + ".stems_mixed.wav"
            mix_stems_weighted(
                stem_audio_paths, mixed_path,
                gains=gains, eq_bands=eq_bands,
            )

            # Add final reverb
            final_path = video_path + ".final_audio.wav"
            add_reverb(mixed_path, final_path, reverberance=18, room_scale=45)

            # Clean up intermediate
            try:
                os.remove(mixed_path)
            except OSError:
                pass

            print(f"  Mixed {len(stem_audio_paths)} stems → {final_path}")
            return final_path, z_overlay_data, stem_bands
        elif stem_audio_paths:
            return stem_audio_paths[0], z_overlay_data, stem_bands
        else:
            return None, None, None

    except Exception as e:
        print(f"  Multi-stem generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def _generate_music_drives_animation(scene_module, params, seed, engine, music_params, video_path):
    """Generate note schedule first, then run animation driven by it."""
    print("  Generating note schedule...")

    num_notes = params.get("num_balls", params.get("num_marbles", 30))
    schedule = engine.generate_note_schedule(
        duration_sec=30.0,
        key=music_params["key"],
        tempo=music_params["tempo"],
        num_notes=num_notes,
    )
    print(f"  Generated {len(schedule)} notes")

    # Render audio from schedule
    print("  Rendering audio...")
    audio_path = engine.render_note_schedule_to_audio(
        schedule=schedule,
        instrument=music_params["instrument"],
        key=music_params["key"],
        tempo=music_params["tempo"],
    )

    # Run animation with note schedule
    print("  Rendering video with note schedule...")
    frame_gen = scene_module.run(seed=seed, params=params, note_schedule=schedule)
    render_video(video_path, frame_gen, audio_path=audio_path)
    return audio_path


def _generate_parallel_sync(scene_module, params, seed, engine, music_params, video_path):
    """Generate music and animation independently, mux together."""
    print("  Rendering video frames...")

    # Run animation and count frames
    frames = list(scene_module.run(seed=seed, params=params))
    duration_sec = len(frames) / config.VIDEO_FPS
    print(f"  {len(frames)} frames ({duration_sec:.1f}s)")

    # Generate background music to match duration
    print("  Generating background music...")
    audio_path = engine.generate_background(
        duration_sec=duration_sec,
        key=music_params["key"],
        tempo=music_params["tempo"],
        instrument=music_params["instrument"],
    )

    # Render video from cached frames
    render_video(video_path, iter(frames), audio_path=audio_path)
    return audio_path


def main():
    parser = argparse.ArgumentParser(description="Daily YouTube Shorts generator")
    parser.add_argument("--count", type=int, default=5,
                        help="Number of videos to generate (default: 5)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Force a specific scene type")
    parser.add_argument("--no-music", action="store_true",
                        help="Generate without music (silent videos)")
    parser.add_argument("--list-scenes", action="store_true",
                        help="List available scenes and exit")
    parser.add_argument("--video", type=str, default=None,
                        help="Input video path (for video_grid_music scene)")
    args = parser.parse_args()

    ideas = load_master_ideas()

    if args.list_scenes:
        registered = list_scenes()
        print("Registered scenes:")
        for name in sorted(ideas["scene_types"].keys()):
            status = "OK" if name in registered else "NOT REGISTERED"
            print(f"  {name}: {status}")
        return

    lineup = select_daily_lineup(ideas, args.count, force_scene=args.scene)
    print(f"Daily lineup ({args.count} videos): {lineup}")
    print(f"Output: {config.OUTPUT_DIR}")

    generated = 0
    for i, scene_name in enumerate(lineup):
        video_num = random.randint(1, 9999)
        result = generate_one_video(scene_name, ideas, video_num,
                                    no_music=args.no_music,
                                    input_video=args.video)
        if result:
            generated += 1

    print(f"\nDone! Generated {generated}/{args.count} video(s) in {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
