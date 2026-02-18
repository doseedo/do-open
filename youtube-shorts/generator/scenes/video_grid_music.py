"""Video Grid Music scene v2.

Takes an input video, overlays a high-resolution grid, analyzes pixel color
and motion per cell, and generates reactive music via ACE-Step diffusion.
Grid cells are tinted with z-space latent colors after generation.

Audio generation: grid metrics → 5 ACE-Step conditioning signals → diffusion
→ z-space post-processing for sync → DCAE decode.
Fallback: MIDI → fluidsynth if ACE-Step model unavailable.
"""

import random
import math
import numpy as np
import cv2
import pygame
from .. import config
from ..palettes import get_palette

DEFAULT_PARAMS = {
    "input_video": None,
    "grid_cols": 16,
    "grid_rows": 12,
    "motion_threshold": 6.0,
    "smoothing_frames": 3,
    "min_note_interval": 0.05,
    "brightness_note_scale": 1.0,
    "target_duration": 30,
    "show_grid": True,
    "show_motion_bars": True,
    "show_z_overlay": True,
    "show_stem_bands": True,
    "num_stems": 3,
    "stem_split_mode": "bands",
    "palette": "neon",
}

# Per-stem band colors for visualization (matching acestep_bridge.STEM_BAND_COLORS)
STEM_BAND_COLORS = [
    (80, 140, 255),   # bass — blue
    (100, 220, 120),  # mid — green
    (255, 180, 80),   # high — orange
    (220, 100, 220),  # extra — purple
]

# Hue → instrument mapping (OpenCV hue is 0-179) — used for MIDI fallback
HUE_INSTRUMENTS = [
    (0, 15, "marimba"),
    (15, 45, "xylophone"),
    (45, 75, "vibraphone"),
    (75, 105, "acoustic_piano"),
    (105, 135, "celesta"),
    (135, 179, "electric_piano"),
]

_collision_events = []
_grid_timeseries = None


def get_collision_events():
    """Return collected events and clear the list."""
    global _collision_events
    events = list(_collision_events)
    _collision_events = []
    return events


def get_grid_timeseries():
    """Return collected grid timeseries and clear."""
    global _grid_timeseries
    ts = _grid_timeseries
    _grid_timeseries = None
    return ts


def _p(params, key):
    return params.get(key, DEFAULT_PARAMS[key])


def _hue_to_instrument(hue):
    """Map OpenCV hue (0-179) to instrument name."""
    for low, high, inst in HUE_INSTRUMENTS:
        if low <= hue < high:
            return inst
    return "xylophone"


def _hue_to_rgb(hue_cv):
    """Convert OpenCV hue (0-179) to RGB tuple for visualization."""
    hsv = np.array([[[int(hue_cv), 200, 220]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return int(rgb[0, 0, 0]), int(rgb[0, 0, 1]), int(rgb[0, 0, 2])


def _load_video_frames(path, target_fps=30, target_duration=30):
    """Load video, loop to target duration, resample to target FPS.

    Returns list of RGB numpy arrays (H, W, 3) uint8.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(rgb)
    cap.release()

    if not raw_frames:
        raise RuntimeError(f"No frames read from: {path}")

    src_duration = len(raw_frames) / src_fps
    target_total_frames = int(target_duration * target_fps)

    output_frames = []
    for out_idx in range(target_total_frames):
        out_time = out_idx / target_fps
        src_time = out_time % src_duration
        src_idx = int(src_time * src_fps)
        src_idx = min(src_idx, len(raw_frames) - 1)

        frame = raw_frames[src_idx]

        # Crossfade at loop boundaries (last 0.3s of each loop iteration)
        crossfade_dur = 0.3
        loop_pos = out_time % src_duration
        if src_duration > crossfade_dur * 2 and loop_pos > (src_duration - crossfade_dur):
            blend_progress = (loop_pos - (src_duration - crossfade_dur)) / crossfade_dur
            blend_idx = int(blend_progress * crossfade_dur * src_fps)
            blend_idx = min(blend_idx, len(raw_frames) - 1)
            alpha = blend_progress
            frame = cv2.addWeighted(frame, 1.0 - alpha, raw_frames[blend_idx], alpha, 0)

        output_frames.append(frame)

    return output_frames


def _fit_to_output(frame, out_w, out_h):
    """Scale and letterbox frame to fit output dimensions."""
    src_h, src_w = frame.shape[:2]
    scale_w = out_w / src_w
    scale_h = out_h / src_h
    scale = min(scale_w, scale_h)

    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    output = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y_off = (out_h - new_h) // 2
    x_off = (out_w - new_w) // 2
    output[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return output, x_off, y_off, new_w, new_h


def _compute_cell_metrics(prev_gray_cell, curr_gray_cell, curr_hsv_cell):
    """Compute motion and color metrics for a single grid cell."""
    if prev_gray_cell is not None:
        diff = cv2.absdiff(prev_gray_cell, curr_gray_cell)
        motion = float(diff.mean())
    else:
        motion = 0.0

    hue = float(curr_hsv_cell[:, :, 0].mean())
    sat = float(curr_hsv_cell[:, :, 1].mean())
    brightness = float(curr_hsv_cell[:, :, 2].mean())

    return {
        "motion": motion,
        "hue": hue,
        "saturation": sat,
        "brightness": brightness,
    }


def run(seed=None, params=None, note_schedule=None, collect_events_only=False,
        collect_grid_timeseries=False, z_overlay_data=None, stem_bands=None):
    """Generator yielding pygame surfaces for each frame.

    Args:
        seed: random seed
        params: scene parameters dict
        note_schedule: unused (kept for API compat)
        collect_events_only: if True, skip rendering, just collect events
        collect_grid_timeseries: if True, store per-cell [rows, cols, T] arrays
        z_overlay_data: if provided, [T_lat, 128] tensor for z-space visualization
            For multi-stem: list of (z_flat, row_start, row_end) tuples
        stem_bands: if provided, list of (row_start, row_end, band_name) for
            per-stem band visualization
    """
    global _collision_events, _grid_timeseries
    _collision_events = []
    _grid_timeseries = None

    if seed is not None:
        random.seed(seed)

    params = params or {}

    input_video = _p(params, "input_video")
    if not input_video:
        raise ValueError("video_grid_music requires 'input_video' param")

    grid_cols = _p(params, "grid_cols")
    grid_rows = _p(params, "grid_rows")
    motion_threshold = _p(params, "motion_threshold")
    smoothing_frames = _p(params, "smoothing_frames")
    min_note_interval = _p(params, "min_note_interval")
    brightness_scale = _p(params, "brightness_note_scale")
    target_duration = _p(params, "target_duration")
    show_grid = _p(params, "show_grid")
    show_motion_bars = _p(params, "show_motion_bars")
    show_z_overlay = _p(params, "show_z_overlay")
    show_stem_bands = _p(params, "show_stem_bands")
    num_stems = _p(params, "num_stems")
    pal = get_palette(_p(params, "palette"))

    # Build row→stem band mapping for visualization
    row_to_band = {}
    if stem_bands and show_stem_bands:
        for band_idx, (rs, re, bname) in enumerate(stem_bands):
            color = STEM_BAND_COLORS[band_idx % len(STEM_BAND_COLORS)]
            for r in range(rs, re):
                row_to_band[r] = (band_idx, bname, color)

    out_w, out_h = config.WIDTH, config.HEIGHT

    # Load and prepare video frames
    video_frames = _load_video_frames(input_video, config.VIDEO_FPS, target_duration)
    total_frames = len(video_frames)

    if collect_events_only:
        surface = pygame.Surface((1, 1))
    else:
        surface = pygame.Surface((out_w, out_h))

    # Fit first frame to get video placement dimensions
    _, x_off, y_off, vid_w, vid_h = _fit_to_output(video_frames[0], out_w, out_h)

    # Grid cell dimensions (in video coordinates, before scaling)
    src_h, src_w = video_frames[0].shape[:2]
    cell_src_w = src_w // grid_cols
    cell_src_h = src_h // grid_rows

    # Grid cell dimensions in output coordinates
    cell_out_w = vid_w // grid_cols
    cell_out_h = vid_h // grid_rows

    # Per-cell state
    cell_count = grid_rows * grid_cols
    motion_history = [[0.0] * smoothing_frames for _ in range(cell_count)]
    last_trigger_time = [-999.0] * cell_count
    active_cells = {}  # (row, col) -> frames_remaining for glow effect

    # Grid timeseries arrays (if collecting)
    if collect_grid_timeseries:
        motion_ts = np.zeros((grid_rows, grid_cols, total_frames), dtype=np.float32)
        hue_ts = np.zeros((grid_rows, grid_cols, total_frames), dtype=np.float32)
        sat_ts = np.zeros((grid_rows, grid_cols, total_frames), dtype=np.float32)
        bright_ts = np.zeros((grid_rows, grid_cols, total_frames), dtype=np.float32)

    # Z-overlay data — supports single tensor or list of (z_flat, row_start, row_end)
    z_pca_axes = None
    z_overlay_multi = None  # list of (z_flat, row_start, row_end) for multi-stem
    z_overlay_single = None  # single z_flat tensor for single-stem
    if z_overlay_data is not None and show_z_overlay:
        try:
            from ..music.acestep_bridge import load_inverse_patch_axes
            axes = load_inverse_patch_axes()
            z_pca_axes = axes.get("contrastive_pca", None)
        except Exception:
            z_pca_axes = None

        if isinstance(z_overlay_data, list):
            z_overlay_multi = z_overlay_data  # [(z_flat, row_start, row_end), ...]
        else:
            z_overlay_single = z_overlay_data

    # Pitch assignment
    pitch_base_range = (48, 84)

    prev_gray = None

    # Font for info bar
    font = None
    small_font = None
    if not collect_events_only:
        try:
            font = pygame.font.SysFont("Arial", 28, bold=True)
            small_font = pygame.font.SysFont("Arial", 20)
        except Exception:
            font = pygame.font.Font(None, 28)
            small_font = pygame.font.Font(None, 20)

    for frame_idx in range(total_frames):
        current_time = frame_idx / config.VIDEO_FPS
        rgb_frame = video_frames[frame_idx]

        # Convert for analysis
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)

        # Fit to output
        fitted_frame, _, _, _, _ = _fit_to_output(rgb_frame, out_w, out_h)

        # Analyze each grid cell
        frame_events = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                cell_idx = row * grid_cols + col
                y1 = row * cell_src_h
                y2 = min((row + 1) * cell_src_h, src_h)
                x1 = col * cell_src_w
                x2 = min((col + 1) * cell_src_w, src_w)

                curr_gray_cell = gray_frame[y1:y2, x1:x2]
                curr_hsv_cell = hsv_frame[y1:y2, x1:x2]

                prev_gray_cell = None
                if prev_gray is not None:
                    prev_gray_cell = prev_gray[y1:y2, x1:x2]

                metrics = _compute_cell_metrics(
                    prev_gray_cell, curr_gray_cell, curr_hsv_cell
                )

                # Update motion smoothing buffer
                motion_history[cell_idx].pop(0)
                motion_history[cell_idx].append(metrics["motion"])
                smoothed_motion = sum(motion_history[cell_idx]) / smoothing_frames

                # Store timeseries
                if collect_grid_timeseries:
                    motion_ts[row, col, frame_idx] = smoothed_motion
                    hue_ts[row, col, frame_idx] = metrics["hue"]
                    sat_ts[row, col, frame_idx] = metrics["saturation"]
                    bright_ts[row, col, frame_idx] = metrics["brightness"]

                # Check if this cell should trigger a note
                time_since_last = current_time - last_trigger_time[cell_idx]
                if (smoothed_motion > motion_threshold and
                        time_since_last >= min_note_interval):

                    last_trigger_time[cell_idx] = current_time

                    # Compute pitch from grid position
                    col_norm = col / max(1, grid_cols - 1)
                    pitch_span = pitch_base_range[1] - pitch_base_range[0]
                    base_pitch = pitch_base_range[0] + col_norm * pitch_span

                    row_norm = 1.0 - (row / max(1, grid_rows - 1))
                    octave_shift = int((row_norm - 0.5) * 12)
                    pitch = int(base_pitch + octave_shift)
                    pitch = max(36, min(96, pitch))

                    force = min(smoothed_motion * 100, 5000.0)

                    bright_norm = metrics["brightness"] / 255.0
                    duration = 0.05 + bright_norm * 0.45 * brightness_scale

                    sat_mod = (metrics["saturation"] / 255.0 - 0.5) * 0.4 + 1.0
                    force *= sat_mod

                    instrument = _hue_to_instrument(metrics["hue"])

                    cx = x_off + col * cell_out_w + cell_out_w // 2
                    cy = y_off + row * cell_out_h + cell_out_h // 2

                    event = {
                        "frame": frame_idx,
                        "time_sec": current_time,
                        "x": float(cx),
                        "y": float(cy),
                        "force": float(force),
                        "instrument": instrument,
                        "duration_sec": duration,
                        "hue": metrics["hue"],
                        "grid_row": row,
                        "grid_col": col,
                    }
                    _collision_events.append(event)
                    frame_events.append(event)
                    active_cells[(row, col)] = 8

        prev_gray = gray_frame

        # Render frame
        if not collect_events_only:
            surf_array = np.transpose(fitted_frame, (1, 0, 2))
            pygame.surfarray.blit_array(surface, surf_array)

            # Z-overlay (subtle tint, alpha ~0.2)
            if z_pca_axes is not None and show_z_overlay:
                if z_overlay_multi is not None:
                    # Multi-stem: draw per-band z-overlays
                    for z_flat_band, z_rs, z_re in z_overlay_multi:
                        _draw_z_overlay(
                            surface, x_off, y_off, cell_out_w, cell_out_h,
                            grid_cols, grid_rows, z_flat_band, z_pca_axes,
                            current_time, alpha=0.15,
                            row_start=z_rs, row_end=z_re,
                        )
                elif z_overlay_single is not None:
                    _draw_z_overlay(
                        surface, x_off, y_off, cell_out_w, cell_out_h,
                        grid_cols, grid_rows, z_overlay_single, z_pca_axes,
                        current_time, alpha=0.2,
                    )

            # Draw stem band borders (colored left-edge indicator per row band)
            if row_to_band and show_stem_bands:
                _draw_stem_bands(surface, x_off, y_off, cell_out_w, cell_out_h,
                                 grid_cols, grid_rows, row_to_band, vid_w)

            # Draw grid overlay
            if show_grid:
                _draw_grid(surface, x_off, y_off, vid_w, vid_h,
                           grid_cols, grid_rows, cell_out_w, cell_out_h,
                           active_cells, frame_events, pal)

            # Draw motion bars
            if show_motion_bars:
                _draw_motion_bars(surface, x_off, y_off, cell_out_w, cell_out_h,
                                  grid_cols, grid_rows, motion_history,
                                  smoothing_frames, motion_threshold)

            # Draw info bar
            if font:
                stem_count = len(stem_bands) if stem_bands else 1
                _draw_info_bar(surface, font, small_font, current_time,
                               target_duration, len(frame_events), pal,
                               stem_count=stem_count)

            # Decay active cell glow
            expired = []
            for key in active_cells:
                active_cells[key] -= 1
                if active_cells[key] <= 0:
                    expired.append(key)
            for key in expired:
                del active_cells[key]

        yield surface

    # Store grid timeseries for retrieval
    if collect_grid_timeseries:
        _grid_timeseries = {
            "motion": motion_ts,
            "hue": hue_ts,
            "saturation": sat_ts,
            "brightness": bright_ts,
        }

    # Hold final frame
    for _ in range(2 * config.VIDEO_FPS):
        yield surface


def _draw_z_overlay(surface, x_off, y_off, cell_w, cell_h,
                    grid_cols, grid_rows, z_data, pca_axes,
                    current_time, alpha=0.2, row_start=0, row_end=None):
    """Draw z-space color tint overlay on grid cells.

    Args:
        z_data: [T_lat, 128] tensor
        pca_axes: list of [128] tensors (at least 3)
        current_time: seconds into video
        row_start, row_end: optional row range for per-band overlay
    """
    import torch

    if row_end is None:
        row_end = grid_rows

    DCAE_LATENT_FPS = 44100 / 4096  # ~10.77

    T_lat = z_data.shape[0]
    z_frame_idx = int(current_time * DCAE_LATENT_FPS)
    z_frame_idx = min(z_frame_idx, T_lat - 1)
    z_frame = z_data[z_frame_idx]

    from ..music.acestep_bridge import z_frame_to_grid_colors
    band_rows = row_end - row_start
    grid_colors = z_frame_to_grid_colors(z_frame, band_rows, grid_cols, pca_axes)

    # Blend onto surface as semi-transparent tint
    alpha_int = int(255 * alpha)
    overlay = pygame.Surface((cell_w, cell_h), pygame.SRCALPHA)

    for ri, r in enumerate(range(row_start, row_end)):
        for c in range(grid_cols):
            cr, cg, cb = int(grid_colors[ri, c, 0]), int(grid_colors[ri, c, 1]), int(grid_colors[ri, c, 2])
            overlay.fill((cr, cg, cb, alpha_int))
            cx = x_off + c * cell_w
            cy = y_off + r * cell_h
            surface.blit(overlay, (cx, cy))


def _draw_stem_bands(surface, x_off, y_off, cell_w, cell_h,
                     grid_cols, grid_rows, row_to_band, vid_w):
    """Draw colored band indicators on the left edge showing which stem owns each row."""
    band_width = 4
    for r in range(grid_rows):
        if r not in row_to_band:
            continue
        band_idx, band_name, color = row_to_band[r]
        y = y_off + r * cell_h
        # Left edge colored bar
        pygame.draw.rect(surface, color,
                         (x_off, y, band_width, cell_h))
        # Right edge colored bar
        pygame.draw.rect(surface, color,
                         (x_off + vid_w - band_width, y, band_width, cell_h))

    # Draw separator lines between bands
    seen_boundaries = set()
    for r in range(1, grid_rows):
        if r in row_to_band and (r - 1) in row_to_band:
            if row_to_band[r][0] != row_to_band[r - 1][0]:
                y = y_off + r * cell_h
                if y not in seen_boundaries:
                    # Use the upper band's color for the separator
                    color = row_to_band[r][2]
                    pygame.draw.line(surface, color,
                                     (x_off, y), (x_off + vid_w, y), 2)
                    seen_boundaries.add(y)


def _draw_grid(surface, x_off, y_off, vid_w, vid_h,
               grid_cols, grid_rows, cell_w, cell_h,
               active_cells, frame_events, pal):
    """Draw grid lines and active cell highlights."""
    # Grid lines (semi-transparent, thinner for high-res grid)
    line_color = (160, 160, 160)
    for col in range(grid_cols + 1):
        x = x_off + col * cell_w
        pygame.draw.line(surface, line_color,
                         (x, y_off), (x, y_off + vid_h), 1)
    for row in range(grid_rows + 1):
        y = y_off + row * cell_h
        pygame.draw.line(surface, line_color,
                         (x_off, y), (x_off + vid_w, y), 1)

    # Active cell glow effects
    for (row, col), remaining in active_cells.items():
        alpha = remaining / 8.0
        cx = x_off + col * cell_w
        cy = y_off + row * cell_h

        hue = 60
        for ev in frame_events:
            if ev["grid_row"] == row and ev["grid_col"] == col:
                hue = ev["hue"]
                break

        glow_color = _hue_to_rgb(hue)
        r = min(255, int(glow_color[0] * alpha + 100 * alpha))
        g = min(255, int(glow_color[1] * alpha + 100 * alpha))
        b = min(255, int(glow_color[2] * alpha + 100 * alpha))

        border_width = max(1, int(2 * alpha))
        rect = pygame.Rect(cx, cy, cell_w, cell_h)
        pygame.draw.rect(surface, (r, g, b), rect, border_width)

        # Small indicator at cell center (only for strong activations)
        if remaining > 5:
            center_x = cx + cell_w // 2
            center_y = cy + cell_h // 2
            radius = max(2, int(4 * alpha))
            pygame.draw.circle(surface, (255, 255, 255), (center_x, center_y), radius)
            pygame.draw.circle(surface, (r, g, b), (center_x, center_y), radius + 2, 1)


def _draw_motion_bars(surface, x_off, y_off, cell_w, cell_h,
                      grid_cols, grid_rows, motion_history,
                      smoothing_frames, motion_threshold):
    """Draw thin motion intensity bars at bottom of each cell."""
    bar_height = 3
    for row in range(grid_rows):
        for col in range(grid_cols):
            cell_idx = row * grid_cols + col
            smoothed = sum(motion_history[cell_idx]) / smoothing_frames
            normalized = min(1.0, smoothed / (motion_threshold * 3))

            bar_x = x_off + col * cell_w + 1
            bar_y = y_off + (row + 1) * cell_h - bar_height - 1
            bar_w = int((cell_w - 2) * normalized)

            if normalized < 0.33:
                color = (0, max(0, min(255, int(200 * normalized * 3))), 50)
            elif normalized < 0.66:
                t = (normalized - 0.33) * 3
                color = (max(0, min(255, int(255 * t))), 200, 50)
            else:
                t = min(1.0, (normalized - 0.66) * 3)
                color = (255, max(0, min(255, int(200 * (1 - t)))), 50)

            if bar_w > 0:
                pygame.draw.rect(surface, color,
                                 (bar_x, bar_y, bar_w, bar_height))


def _draw_info_bar(surface, font, small_font, current_time,
                   total_duration, active_count, pal, stem_count=1):
    """Draw info bar at bottom of screen."""
    bar_height = 60
    bar_y = config.HEIGHT - bar_height
    bar_surf = pygame.Surface((config.WIDTH, bar_height))
    bar_surf.fill((20, 20, 30))
    bar_surf.set_alpha(180)
    surface.blit(bar_surf, (0, bar_y))

    time_text = f"{current_time:.1f}s / {total_duration:.0f}s"
    time_surf = small_font.render(time_text, True, (200, 200, 200))
    surface.blit(time_surf, (20, bar_y + 8))

    if active_count > 0:
        note_text = f"{active_count} notes"
        note_surf = small_font.render(note_text, True, (100, 255, 150))
        surface.blit(note_surf, (20, bar_y + 32))

    title_text = "VIDEO GRID MUSIC"
    if stem_count > 1:
        title_text = f"VIDEO GRID MUSIC ({stem_count} stems)"
    title_surf = font.render(title_text, True, (220, 220, 255))
    title_rect = title_surf.get_rect(center=(config.WIDTH // 2, bar_y + 20))
    surface.blit(title_surf, title_rect)

    # Stem indicator dots on the right
    if stem_count > 1:
        dot_x = config.WIDTH - 80
        for si in range(stem_count):
            color = STEM_BAND_COLORS[si % len(STEM_BAND_COLORS)]
            dot_y = bar_y + 12 + si * 14
            pygame.draw.circle(surface, color, (dot_x, dot_y), 4)
            label = ["bass", "mid", "high", "extra"][min(stem_count - 1 - si, 3)]
            label_surf = small_font.render(label, True, color)
            surface.blit(label_surf, (dot_x + 10, dot_y - 8))

    progress = current_time / max(0.1, total_duration)
    bar_x = config.WIDTH // 2 - 150
    bar_w = 300
    pygame.draw.rect(surface, (60, 60, 80),
                     (bar_x, bar_y + 40, bar_w, 8))
    pygame.draw.rect(surface, (100, 180, 255),
                     (bar_x, bar_y + 40, int(bar_w * progress), 8))


def get_metadata(video_num, params=None):
    params = params or {}
    return {
        "title": f"Video Grid Music #{video_num}",
        "description": "AI-generated music from video motion! Grid analysis drives ACE-Step neural audio. #Shorts #music #AI",
        "tags": ["video", "grid", "music", "AI", "neural", "visualization", "reactive", "shorts"],
        "category": "24",
    }
