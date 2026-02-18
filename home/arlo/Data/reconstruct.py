import os
import json
import shutil
import subprocess
import re
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import timedelta

FFMPEG_PATH = "/usr/local/bin/ffmpeg"
SORTED_DIR = Path("/home/arlo/Data/sessionmetadata/Test")
OUTPUT_DIR = Path("/home/arlo/gcs-bucket/reconstructed")
SCRATCH_DISK = Path("/mnt/renderdisk")
KEEP_TEMP = os.environ.get("KEEP_TEMP") == "1"
AVERAGE_RENDER_TIME = 3.5

SCRATCH_DISK.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name):
    return re.sub(r"[^\w\-_.]", "_", name)

def should_mute_track(track):
    """Determine if track should be rendered to muted folder"""
    track_state = track.get("track_state", "").lower()
    return track_state in ("muted", "hidden inactive", "hidden")

def should_process_clip(clip):
    """Determine if clip should be processed (only unmuted clips)"""
    return clip.get("clip_state", "Unmuted") != "Muted"

def find_audio_file(clip_name, audio_file_path):
    """Find the appropriate audio file, handling L/R suffixes and variations"""
    original_path = Path(audio_file_path)
    
    # First try the exact path
    if original_path.exists():
        return original_path
    
    # Handle cases where clip has L/R suffix but file doesn't
    base_name = original_path.stem
    if base_name.endswith(('.L', '.R')):
        # Try without the channel suffix
        no_suffix_path = original_path.with_stem(base_name[:-2])
        if no_suffix_path.exists():
            return no_suffix_path
    
    # Try removing version numbers (like _2, _3) from the end
    if '_' in base_name:
        base_part = base_name.rsplit('_', 1)[0]
        parent_dir = original_path.parent
        for f in parent_dir.iterdir():
            if f.stem.startswith(base_part) and f.suffix == original_path.suffix:
                return f
    
    # Try more flexible matching for complex cases
    simplified_name = re.sub(r'[-_]\d+$', '', base_name)  # Remove trailing numbers
    simplified_name = re.sub(r'[-_](L|R)$', '', simplified_name)  # Remove L/R
    parent_dir = original_path.parent
    for f in parent_dir.iterdir():
        if re.sub(r'[-_]\d+$', '', f.stem) == simplified_name and f.suffix == original_path.suffix:
            return f
    
    return None

def validate_clip(clip):
    """Validate clip parameters before processing"""
    if not all(k in clip for k in ("start_sec", "end_sec", "audio_file")):
        return False
        
    if clip["end_sec"] <= clip["start_sec"]:
        print(f"⚠️ Invalid clip duration: start={clip['start_sec']}, end={clip['end_sec']}")
        return False
        
    return True

def build_ffmpeg_command(inputs, filters, output_path):
    """Build and validate FFmpeg command"""
    cmd = [
        FFMPEG_PATH,
        *inputs,
        "-filter_complex", filters,
        "-map", "[aout]",
        "-c:a", "pcm_s16le",
        "-y", str(output_path)
    ]
    
    # Validate the command doesn't contain None values
    if any(arg is None for arg in cmd):
        raise ValueError("FFmpeg command contains None values")
    
    return cmd

def process_track(args):
    track_name, clips, output_dir, is_stereo = args
    start = time.time()
    print(f"\n🎧 Rendering track: {track_name} ...")

    # Filter out muted clips
    active_clips = [clip for clip in clips if should_process_clip(clip) and validate_clip(clip)]
    if not active_clips:
        print(f"⏭️ Skipping track {track_name} (no valid unmuted clips)")
        return (track_name, None, "No valid unmuted clips")

    # Calculate full track duration from JSON
    try:
        track_duration = max(clip["end_sec"] for clip in active_clips)
        global_start = min(clip["start_sec"] for clip in active_clips)
    except ValueError:
        print(f"❌ Error calculating duration for {track_name}")
        return (track_name, None, "Error calculating duration")

    input_cmds, filter_cmds, tags = [], [], []
    for i, clip in enumerate(active_clips):
        try:
            clip_start = clip["start_sec"] - global_start
            clip_duration = clip["end_sec"] - clip["start_sec"]
            
            if clip_duration <= 0:
                print(f"⚠️ Skipping invalid clip duration: {clip_duration}s")
                continue

            input_cmds.extend(["-i", str(clip["local_path"])])
            delay_ms = int(clip_start * 1000)
            tag = f"[a{i}]"
            
            # Trim to exact duration and position in timeline
            filters = [
                f"atrim=0:{clip_duration:.3f}",
                f"adelay={delay_ms}|{delay_ms}"
            ]
            filter_cmds.append(f"[{i}:a]{','.join(filters)}{tag}")
            tags.append(tag)
        except Exception as e:
            print(f"⚠️ Error processing clip {i} in {track_name}: {str(e)}")
            continue

    if not tags:
        print(f"❌ No valid clips for track: {track_name}")
        return (track_name, None, "No valid clips")

    # Handle stereo tracks
    try:
        if is_stereo and len(tags) >= 2:
            # For stereo tracks, join first two inputs as left/right channels
            filter_cmds.append(f"{tags[0]}{tags[1]}join=inputs=2:channel_layout=stereo[aout]")
        else:
            # Mono track or fallback for stereo with single clip
            filter_cmds.append(
                f"{''.join(tags)}amix=inputs={len(tags)}:duration=first,"
                f"apad=whole_duration={track_duration:.3f}[aout]"
            )

        track_out = output_dir / f"{sanitize_filename(track_name)}.wav"
        cmd = build_ffmpeg_command(
            input_cmds,
            ";".join(filter_cmds),
            track_out
        )

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            duration = time.time() - start
            print(f"✅ Rendered {track_name} ({track_duration:.2f}s) in {timedelta(seconds=int(duration))}")
            return (track_name, track_out, None)
        else:
            error_msg = result.stderr.splitlines()[-1] if result.stderr else "Unknown error"
            print(f"❌ FFmpeg error in {track_name}: {error_msg}")
            print(f"🔧 Command was: {' '.join(cmd)}")
            return (track_name, None, error_msg)
    except Exception as e:
        print(f"❌ Unexpected error processing {track_name}: {str(e)}")
        return (track_name, None, str(e))

def reconstruct_session(json_path: Path, session_dir: Path):
    with open(json_path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error loading JSON {json_path}: {str(e)}")
            return

    session_id = data.get("session_id", session_dir.name)
    rel = json_path.relative_to(SORTED_DIR)
    outdir = OUTPUT_DIR / rel.parent
    
    # Create output directories
    main_out = outdir / "session_mix.wav"
    muted_dir = outdir / "muted_tracks"
    muted_dir.mkdir(parents=True, exist_ok=True)
    
    if main_out.exists():
        print(f"⏭️ Skipping {session_id} (already reconstructed)")
        return

    print(f"🔧 Processing session: {session_id}")
    
    # Create working directories
    temp_dir = SCRATCH_DISK / f"session_{sanitize_filename(session_id)}_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"📥 Temp dir: {temp_dir}")

    main_mix_inputs = []
    muted_mix_inputs = []
    track_args = []
    session_end = 0.0

    # Prepare all clips first
    for track_name, track in data.get("tracks", {}).items():
        is_stereo = track.get("is_stereo", False)
        is_muted = should_mute_track(track)
        valid_clips = []
        
        for clip in track.get("clips", []):
            if not validate_clip(clip):
                continue
                
            src = Path(clip["audio_file"])
            # Try to find the actual audio file, handling L/R suffixes
            actual_src = find_audio_file(clip.get("clip_name", ""), src)
            if not actual_src:
                print(f"⚠️ Skipped clip {src} - file not found")
                continue
                
            local_path = temp_dir / actual_src.name
            try:
                if not local_path.exists():
                    shutil.copy(actual_src, local_path)
                valid_clips.append({
                    "local_path": local_path,
                    "start_sec": clip["start_sec"],
                    "end_sec": clip["end_sec"],
                    "clip_state": clip.get("clip_state", "Unmuted")
                })
                session_end = max(session_end, clip["end_sec"])
            except Exception as e:
                print(f"⚠️ Skipped clip {actual_src} - {e}")
        
        if valid_clips:
            output_dir = muted_dir if is_muted else outdir
            track_args.append((track_name, valid_clips, output_dir, is_stereo))

    # Process tracks in parallel with error handling
    if track_args:
        with Pool(processes=min(cpu_count(), len(track_args))) as pool:
            results = pool.map(process_track, track_args)
        
        for track_name, track_out, error in results:
            if track_out:
                if should_mute_track(data["tracks"][track_name]):
                    muted_mix_inputs.append(track_out)
                else:
                    main_mix_inputs.append(track_out)
            elif error:
                print(f"❌ Error in {track_name}: {error}")

    # Create final mixes
    def create_mix(inputs, output_path):
        if not inputs:
            return
            
        print(f"🎛️ Mixing {len(inputs)} tracks to {output_path}")
        temp_mix = SCRATCH_DISK / f"session_{sanitize_filename(session_id)}_mix_temp.wav"
        
        try:
            # Build mix command
            cmd = [FFMPEG_PATH]
            for f in inputs:
                cmd.extend(["-i", str(f)])
            cmd.extend([
                "-filter_complex", f"amix=inputs={len(inputs)}:duration=longest[outa]",
                "-map", "[outa]",
                "-c:a", "pcm_s16le",
                "-y", str(temp_mix)
            ])
            
            subprocess.run(cmd, check=True)
            shutil.move(temp_mix, output_path)
            print(f"💽 Mix saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create mix: {e.stderr}")
            print(f"🔧 Command was: {' '.join(cmd)}")
        except Exception as e:
            print(f"❌ Unexpected error creating mix: {str(e)}")

    # Create main mix and muted mix
    create_mix(main_mix_inputs, main_out)
    create_mix(muted_mix_inputs, muted_dir / "muted_mix.wav")

    # Cleanup
    if not KEEP_TEMP:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"🧹 Deleted temp dir: {temp_dir}")
    else:
        print(f"🔍 Preserved temp: {temp_dir}")

def batch():
    if not SORTED_DIR.exists():
        print(f"❌ Directory not found: {SORTED_DIR}")
        return

    sessions = [s for s in SORTED_DIR.iterdir() if s.is_dir() and next(s.glob("*.json"), None)]
    total = len(sessions)
    start_time = time.time()

    for idx, session_dir in enumerate(sessions, 1):
        json_path = next(session_dir.glob("*.json"))
        print(f"\n🚧 {idx}/{total} | Starting: {session_dir.name}")
        reconstruct_session(json_path, session_dir)

        elapsed = time.time() - start_time
        remaining = (total - idx) * AVERAGE_RENDER_TIME * 3
        print(f"⏱️ Elapsed: {timedelta(seconds=int(elapsed))} | ETA: {timedelta(seconds=int(remaining))}")

if __name__ == "__main__":
    batch()