"""Headless renderer: Pygame surface → ffmpeg → MP4 video."""

import os
import subprocess
import pygame
from . import config


def create_surface():
    """Create a headless pygame surface."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.display.set_mode((1, 1))
    return pygame.Surface((config.WIDTH, config.HEIGHT))


def _render_video_only(output_path: str, frame_generator):
    """Pipe raw frames to ffmpeg (video-only)."""
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{config.WIDTH}x{config.HEIGHT}",
        "-r", str(config.VIDEO_FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame_count = 0
    for surface in frame_generator:
        raw = pygame.image.tobytes(surface, "RGB")
        proc.stdin.write(raw)
        frame_count += 1

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    return frame_count


def _mux_audio_video(video_path: str, audio_path: str, output_path: str):
    """Combine a video file and audio file into final MP4."""
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]
    proc = subprocess.run(ffmpeg_cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {proc.stderr.decode()}")


def render_video(output_path: str, frame_generator, audio_path: str = None):
    """Render frames + optional audio into final MP4.

    Args:
        output_path: Path to write the .mp4 file.
        frame_generator: A generator that yields pygame.Surface objects.
        audio_path: Optional path to a WAV file to mux with the video.
    """
    if audio_path and os.path.exists(audio_path):
        temp_video = output_path + ".tmp_video.mp4"
        frame_count = _render_video_only(temp_video, frame_generator)
        _mux_audio_video(temp_video, audio_path, output_path)
        os.remove(temp_video)
    else:
        frame_count = _render_video_only(output_path, frame_generator)

    duration = frame_count / config.VIDEO_FPS
    print(f"Rendered {frame_count} frames ({duration:.1f}s) → {output_path}")
    return output_path
