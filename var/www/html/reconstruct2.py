import os
import json
import shutil
import subprocess
import re
import time
import math
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import timedelta
from collections import defaultdict

FFMPEG_PATH = "/usr/local/bin/ffmpeg"
FFPROBE_PATH = "/usr/local/bin/ffprobe"
SORTED_DIR = Path("/home/arlo/Data/sessionmetadata/Sorted")
OUTPUT_DIR = Path("/home/arlo/gcs-bucket/reconstructed")
SCRATCH_DISK = Path("/mnt/renderdisk")
KEEP_TEMP = os.environ.get("KEEP_TEMP") == "1"
AVERAGE_RENDER_TIME = 3.5
BAR_DURATION = 1.5

SCRATCH_DISK.mkdir(parents=True, exist_ok=True)

# --------------------- APPROXIMATION FUNCTIONS ---------------------
def get_audio_duration(file_path):
    """Get duration of audio file using ffprobe"""
    try:
        cmd = [
            FFPROBE_PATH,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]  
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"⚠️ Couldn't get duration for {file_path}: {str(e)}")
        return None


def analyze_source_files(clips):
    """Group source files by duration and find timeline positions"""
    duration_groups = defaultdict(list)
    file_positions = {}
    
    for clip in clips:
        src = Path(clip["audio_file"])
        if not src.exists():
            continue
            
        # Get or calculate duration
        if "duration" not in clip or clip["duration"] <= 0:
            duration = get_audio_duration(src)
            if duration is None:
                continue
        else:
            duration = clip["duration"]
            
        # Group by rounded duration
        rounded_duration = round(duration, 1)
        duration_groups[rounded_duration].append({
            "path": src,
            "start_sec": clip["start_sec"],
            "clip_duration": clip["end_sec"] - clip["start_sec"]
        })
        
        # Track earliest timeline position for each file
        if src not in file_positions or clip["start_sec"] < file_positions[src]["start_sec"]:
            file_positions[src] = {
                "start_sec": clip["start_sec"],
                "duration": duration
            }
            
    return duration_groups, file_positions

def approximate_source_start(clip, file_positions):
    """Approximate source start with detailed logging"""
    src = Path(clip["audio_file"])
    clip_duration = clip["end_sec"] - clip["start_sec"]
    
    # Case 1: File not in position data
    if src not in file_positions:
        print(f"📝 Clip '{clip['clip_name']}': File not in position data, using source_start=0.0")
        return 0.0
    
    file_data = file_positions[src]
    file_duration = file_data["duration"]
    file_start = file_data["start_sec"]
    
    # Case 2: Full file usage
    if abs(clip_duration - file_duration) < 0.1:
        print(f"📝 Clip '{clip['clip_name']}': Using full file, source_start=0.0")
        return 0.0
    
    # Case 3: Punch-in scenario
    source_start = clip["start_sec"]
    
    # Prevent reading beyond file duration
    if source_start + clip_duration > file_duration:
        new_start = max(0, file_duration - clip_duration)
        print(f"⚠️ Clip '{clip['clip_name']}': Adjusted source_start from {source_start:.2f} to {new_start:.2f} (file_duration={file_duration:.2f})")
        source_start = new_start
    else:
        print(f"📝 Clip '{clip['clip_name']}': Using timeline start as source_start: {source_start:.2f}s")
    
    return source_start
# --------------------- END APPROXIMATION FUNCTIONS ---------------------

def sanitize_filename(name):
    return re.sub(r"[^\w\-_.]", "_", name)

def get_base_filename(name):
    return re.sub(r'\.(L|R)(?=\.[^.]+$|$)', '', str(name))

def should_mute_track(track):
    track_state = track.get("track_state", "").lower()
    return track_state in ("muted", "hidden inactive")

def should_process_clip(clip):
    return clip.get("clip_state", "Unmuted") != "Muted"

def validate_clip(clip):
    """Validate clip with detailed duration checks"""
    if not Path(clip["audio_file"]).exists():
        print(f"⚠️ Missing file: {clip['audio_file']}")
        return False

    if not all(k in clip for k in ("start_sec", "end_sec", "audio_file")):
        return False
        
    duration = clip["end_sec"] - clip["start_sec"]
    if duration <= 0.001:
        print(f"⚠️ Skipping invalid clip: {clip.get('clip_name','')} (duration: {duration:.3f}s)")
        return False
        
    # Additional validation: Source file duration check
    try:
        src_duration = get_audio_duration(clip["audio_file"])
        if src_duration is None:
            print(f"⚠️ Couldn't verify duration for {clip['audio_file']}")
        else:
            source_start = clip.get("source_start", 0.0)
            if source_start >= src_duration:
                print(f"❌ Invalid source_start {source_start:.2f} >= file duration {src_duration:.2f} for {clip['clip_name']}")
                return False
            if source_start + duration > src_duration:
                print(f"⚠️ Clip exceeds source file: {clip['clip_name']} (start: {source_start:.2f}, dur: {duration:.2f}, file: {src_duration:.2f})")
    except Exception as e:
        print(f"⚠️ Duration validation error for {clip['clip_name']}: {str(e)}")
        
    return True

def process_track(args):
    track_name, clips, output_dir, is_stereo = args
    start = time.time()
    print(f"\n🎧 Rendering track: {track_name} ...")

    # Filter and validate clips
    active_clips = [clip for clip in clips if should_process_clip(clip) and validate_clip(clip)]
    
    if not active_clips:
        print(f"⏭️ Skipping track {track_name} (no valid unmuted clips)")
        return (track_name, None, "No valid unmuted clips")

    # Calculate global start and track duration
    global_start = 0.0  # Always
    track_duration = max(clip["end_sec"] for clip in active_clips)  # ✅ Full timeline length


    if len(active_clips) > 20:
        print(f"⚠️ Using fallback for {track_name} (too many clips: {len(active_clips)})")
        return fallback_render(track_name, active_clips, output_dir, is_stereo, 
                             global_start, track_duration)

    input_cmds = []
    filter_cmds = []
    tags = []
    
    for i, clip in enumerate(active_clips):
        clip_offset = clip["start_sec"] - global_start
        clip_duration = clip["end_sec"] - clip["start_sec"]
        delay_ms = int(clip_offset * 1000)
        source_start = clip.get("source_start", 0.0)
        
        # Log clip details for validation
        print(f"  ➡️ Clip {i+1}: '{clip['clip_name']}'")
        print(f"     Timeline: {clip['start_sec']:.2f}-{clip['end_sec']:.2f}s "
              f"(duration: {clip_duration:.3f}s)")
        print(f"     Source: {source_start:.2f}s in {Path(clip['audio_file']).name}")
        
        input_cmds.extend([
            "-ss", str(source_start),
            "-t", str(clip_duration),
            "-i", str(clip["local_path"])
        ])
        
        if clip["is_stereo"]:
            filter_cmds.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]")
            tags.append(f"[a{i}]")
        else:
            filter_cmds.append(f"[{i}:a]adelay={delay_ms}[a{i}]")
            filter_cmds.append(f"[a{i}]asplit[L{i}][R{i}]")
            filter_cmds.append(f"[L{i}]adelay={delay_ms}[Ld{i}]")
            filter_cmds.append(f"[R{i}]adelay={delay_ms}[Rd{i}]")
            tags.extend([f"[Ld{i}]", f"[Rd{i}]"])
    
    filter_cmds.append(
        f"{''.join(tags)}amix=inputs={len(tags)}:duration=longest:dropout_transition=0,"
        f"apad=whole_dur={track_duration}[aout]"
    )
     

    track_out = output_dir / f"{sanitize_filename(track_name)}.wav"
    cmd = [
        FFMPEG_PATH,
        *input_cmds,
        "-filter_complex", ";".join(filter_cmds),
        "-map", "[aout]",
        "-ac", "2" if is_stereo else "1",
        "-c:a", "pcm_s16le",
        "-y", str(track_out)
    ]
    
    # Print simplified command for debugging
    print(f"🔧 FFmpeg command: ffmpeg {' '.join([c if len(c)<50 else '...' for c in cmd])}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            duration = time.time() - start
            print(f"✅ Rendered {track_name} ({track_duration:.2f}s) in {timedelta(seconds=int(duration))}")
            
            # Verify output duration
            output_duration = get_audio_duration(track_out)
            if output_duration:
                diff = abs(output_duration - track_duration)
                if diff > 0.1:
                    print(f"⚠️ Duration mismatch: Expected {track_duration:.2f}s, got {output_duration:.2f}s (diff: {diff:.2f}s)")
                else:
                    print(f"📏 Output duration verified: {output_duration:.2f}s")
            return (track_name, track_out, None)
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            print(f"❌ FFmpeg error in {track_name}:")
            print(error_msg[:1000])  # Print first 1000 chars of error
            print(f"🔧 Attempting fallback rendering...")
            return fallback_render(track_name, active_clips, output_dir, is_stereo, 
                                 global_start, track_duration)
    except Exception as e:
        print(f"❌ Unexpected error processing {track_name}: {str(e)}")
        return (track_name, None, str(e))

def fallback_render(track_name, clips, output_dir, is_stereo, global_start, track_duration):
    print(f"🔄 Using fallback rendering for {track_name}")
    try:
        temp_dir = output_dir / f"temp_fallback_{sanitize_filename(track_name)}"
        temp_dir.mkdir(exist_ok=True)
        temp_files = []
        
        for i, clip in enumerate(clips):
            clip_start = clip["start_sec"] 
            clip_duration = clip["end_sec"] - clip["start_sec"]
            
            if clip_duration <= 0.001:
                continue
                
            temp_file = temp_dir / f"temp_{i}.wav"
            source_start = clip.get("source_start", 0.0)
            
            print(f"  🎬 Rendering clip {i+1}: {clip['clip_name']} "
                  f"(source: {source_start:.2f}s, duration: {clip_duration:.3f}s)")
            
            cmd = [
                FFMPEG_PATH,
                "-i", str(clip["local_path"]),
                "-ss", str(source_start),
                "-t", str(clip_duration),
                "-af", f"adelay={clip_start*1000}|{clip_start*1000}",
                "-ac", "2" if is_stereo else "1",
                "-c:a", "pcm_s16le",
                "-y", str(temp_file)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                if temp_file.exists():
                    temp_files.append(temp_file)
                else:
                    print(f"⚠️ Temp file not created: {temp_file}")
            except Exception as e:
                print(f"⚠️ Failed to process clip {i}: {str(e)}")
                continue

        if not temp_files:
            raise Exception("No valid clips to render")

        concat_file = temp_dir / "concat_list.txt"
        with open(concat_file, "w") as f:
            for tf in temp_files:
                f.write(f"file {str(tf)}\n")

        final_out = output_dir / f"{sanitize_filename(track_name)}.wav"
        cmd = [
            FFMPEG_PATH,
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:a", "pcm_s16le",
            "-y", str(final_out)
        ]
        
        print(f"🔗 Concatenating {len(temp_files)} clips...")
        subprocess.run(cmd, check=True)
        
        # Verify output
        if final_out.exists():
            output_duration = get_audio_duration(final_out)
            print(f"📏 Fallback output duration: {output_duration:.2f}s")
            return (track_name, final_out, None)
        else:
            raise Exception("Fallback output file not created")
        
    except Exception as e:
        print(f"❌ Fallback rendering failed for {track_name}: {str(e)}")
        return (track_name, None, str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def reconstruct_session(json_path: Path, session_dir: Path):
    with open(json_path) as f:
        data = json.load(f)

    session_id = data.get("session_id", session_dir.name)
    rel = json_path.relative_to(SORTED_DIR)
    outdir = OUTPUT_DIR / rel.parent
    
    main_out = outdir / "session_mix.wav"
    muted_dir = outdir / "muted_tracks"
    muted_dir.mkdir(parents=True, exist_ok=True)
    
    if main_out.exists():
        print(f"⏭️ Skipping {session_id} (already reconstructed)")
        return

    print(f"\n{'='*80}")
    print(f"🔧 Processing session: {session_id}")
    print(f"{'='*80}")
    
    temp_dir = SCRATCH_DISK / f"session_{sanitize_filename(session_id)}_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"📥 Temp dir: {temp_dir}")

    # Collect all clips for source file analysis
    all_clips = []
    for track in data.get("tracks", {}).values():
        all_clips.extend(track.get("clips", []))
    
    # Analyze source files across entire session
    duration_groups, file_positions = analyze_source_files(all_clips)
    print(f"📊 Found {len(duration_groups)} source duration groups")
    for dur, files in duration_groups.items():
        print(f"  - {dur:.1f}s: {len(files)} files")

    main_mix_inputs = []
    muted_mix_inputs = []
    track_args = []
    session_end = 0.0

    # Prepare all clips with approximated source_start
    for track_name, track in data.get("tracks", {}).items():
        is_stereo = track.get("is_stereo", False)
        is_muted = should_mute_track(track)
        valid_clips = []
        
        print(f"\n🎛️ Preparing track: {track_name} "
              f"{'(STEREO)' if is_stereo else '(MONO)}'} "
              f"{'(MUTED)' if is_muted else ''}")
        
        for clip in track.get("clips", []):
            if not all(k in clip for k in ("start_sec", "end_sec", "audio_file")):
                continue
                
            src = Path(clip["audio_file"])
            
            # Add approximated source_start to clip data
            clip["source_start"] = approximate_source_start(clip, file_positions)
            
            clip_data = {
                "audio_file": str(src),
                "local_path": src,
                "start_sec": clip["start_sec"],
                "end_sec": clip["end_sec"],
                "clip_state": clip.get("clip_state", "Unmuted"),
                "clip_name": clip.get("clip_name", ""),
                "is_stereo": is_stereo,
                "source_start": clip["source_start"]
            }
            
            # Skip zero-duration clips immediately
            if clip_data["end_sec"] - clip_data["start_sec"] <= 0.001:
                print(f"  ⏩ Skipping zero-duration clip: {clip_data['clip_name']}")
                continue
                
            valid_clips.append(clip_data)
            session_end = max(session_end, clip["end_sec"])
        
        print(f"  📋 {len(valid_clips)} valid clips in track")
        
        if valid_clips:
            output_dir = muted_dir if is_muted else outdir
            track_args.append((track_name, valid_clips, output_dir, is_stereo))

    # Process tracks in parallel
    if track_args:
        print(f"\n🚀 Rendering {len(track_args)} tracks...")
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
            print(f"⏭️ No tracks to mix for {output_path.name}")
            return
            
        print(f"\n🎚️ Mixing {len(inputs)} tracks to {output_path}")
        temp_mix = SCRATCH_DISK / f"session_{sanitize_filename(session_id)}_mix_temp.wav"
        
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
        
        print(f"🔧 Mix command: ffmpeg {' '.join(cmd[1:5])} ... [{len(cmd)-5} more args]")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                shutil.move(temp_mix, output_path)
                mix_duration = get_audio_duration(output_path)
                print(f"💽 Mix saved to {output_path} (duration: {mix_duration:.2f}s)")
            else:
                print(f"❌ Failed to create mix: {result.stderr[:500]}")
        except Exception as e:
            print(f"❌ Mix creation error: {str(e)}")

    # Create main mix and muted mix
    create_mix(main_mix_inputs, main_out)
    create_mix(muted_mix_inputs, muted_dir / "muted_mix.wav")

    # Cleanup
    if not KEEP_TEMP:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"🧹 Deleted temp dir: {temp_dir}")
    else:
        print(f"🔍 Preserved temp: {temp_dir}")
    
    print(f"\n✅ Finished processing session: {session_id}")
    print(f"{'='*80}\n")

def batch():
    if not SORTED_DIR.exists():
        print(f"❌ Directory not found: {SORTED_DIR}")
        return

    sessions = [s for s in SORTED_DIR.iterdir() if s.is_dir() and next(s.glob("*.json"), None)]
    total = len(sessions)
    start_time = time.time()

    for idx, session_dir in enumerate(sessions, 1):
        json_path = next(session_dir.glob("*.json"))
        print(f"\n{'#'*50}")
        print(f"🚧 {idx}/{total} | Starting: {session_dir.name}")
        print(f"{'#'*50}")
        reconstruct_session(json_path, session_dir)

        elapsed = time.time() - start_time
        avg_per_session = elapsed / idx if idx > 0 else 0
        remaining = (total - idx) * avg_per_session
        print(f"⏱️ Elapsed: {timedelta(seconds=int(elapsed))} | "
              f"ETA: {timedelta(seconds=int(remaining))}")

if __name__ == "__main__":
    batch()