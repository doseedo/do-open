import os
import subprocess
from celery import shared_task
from celery_config import celery_app

# --- Build Filter Complex ---
def build_filter_complex(tracks):
    filters = []
    input_labels = []

    for idx, track in enumerate(tracks):
        delay_ms = int(round(track.get("start", 0) * 1000))

        filters.append(f"[{idx+1}:a]adelay={delay_ms}|{delay_ms}[a{idx}]")
        input_labels.append(f"[a{idx}]")

    filter_complex = ";".join(filters) + ";" + "".join(input_labels) + f"amix=inputs={len(tracks)}:duration=longest:dropout_transition=0[aout]"
    return filter_complex

# --- Export Audio to Video ---
@shared_task(name='audio_tasks.export_audio_to_video')
def export_audio_to_video(audio_file_path, video_file_path):
    try:
        # Generate output path in the same directory as the video
        import uuid
        output_dir = os.path.dirname(video_file_path)
        output_filename = f"output_{uuid.uuid4()}.mp4"
        output_file_path = os.path.join(output_dir, output_filename)

        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', video_file_path,
            '-i', audio_file_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',  # Audio bitrate
            '-shortest',  # Match shortest input duration
            output_file_path
        ]

        print(f"🎬 Running ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ Video export successful: {output_file_path}")

        return {"output_video": output_file_path, "status": "success"}
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e.stderr}")
        return {'error': str(e), 'stderr': e.stderr}
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return {'error': str(e)}

# --- Export Final Video with Mixed Audio ---
@shared_task(bind=True, max_retries=3)
def export_video_task(self, video_path, audio_tracks, output_path):
    try:
        print("\n🛠️ Starting export_video_task...")

        ffmpeg_inputs = ['-i', video_path]

        for idx, track in enumerate(audio_tracks):
            audio_file_path = os.path.join(os.path.dirname(video_path), track["filename"])

            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Missing audio file: {audio_file_path}")

            ffmpeg_inputs.extend(['-i', audio_file_path])

        filter_complex = build_filter_complex(audio_tracks)

        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            *ffmpeg_inputs,
            '-filter_complex', filter_complex,
            '-map', '0:v',
            '-map', '[aout]',
            '-c:v', 'copy',
            '-c:a', 'aac',
            output_path
        ]

        print("\n🔵 Running ffmpeg command:", " ".join(ffmpeg_cmd))

        subprocess.run(ffmpeg_cmd, check=True)

        print("\n✅ Export successful!")

    except subprocess.CalledProcessError as e:
        print("\n🔴 ffmpeg error:", e.stderr)
        raise self.retry(exc=e, countdown=30)

    except Exception as e:
        print("\n🔴 Unexpected error:", str(e))
        raise self.retry(exc=e, countdown=30)
