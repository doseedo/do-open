"""Plinko / Galton Board scene.

Drops colorful balls through a grid of pegs into bins at the bottom.
Supports parameterization, collision event collection, and music-driven drops.
"""

import random
import math
import pymunk
import pygame
from .. import config
from ..palettes import BALL_COLORS, get_palette

# Default parameters (overridable via params dict)
DEFAULT_PARAMS = {
    "num_rows": 16,
    "pegs_per_row_base": 9,
    "peg_radius": 12,
    "ball_radius": 14,
    "num_balls": 40,
    "drop_interval_frames": 15,
    "ball_elasticity": 0.55,
    "ball_friction": 0.3,
    "peg_elasticity": 0.5,
    "gravity": 900,
    "top_margin": 250,
    "bottom_margin": 300,
    "side_margin": 80,
    "palette": "classic",
}

# Module-level collision event storage
_collision_events = []


def get_collision_events():
    """Return collected collision events and clear the list."""
    global _collision_events
    events = list(_collision_events)
    _collision_events = []
    return events


def _p(params, key):
    """Get param value with fallback to default."""
    return params.get(key, DEFAULT_PARAMS[key])


def create_scene(params=None):
    """Set up the pymunk space with pegs, walls, and bins."""
    params = params or {}
    space = pymunk.Space()
    space.gravity = (0, _p(params, "gravity"))

    w, h = config.WIDTH, config.HEIGHT
    num_rows = _p(params, "num_rows")
    pegs_per_row = _p(params, "pegs_per_row_base")
    peg_radius = _p(params, "peg_radius")
    peg_elasticity = _p(params, "peg_elasticity")
    top_margin = _p(params, "top_margin")
    bottom_margin = _p(params, "bottom_margin")
    side_margin = _p(params, "side_margin")

    peg_area_height = h - top_margin - bottom_margin
    row_spacing = peg_area_height / (num_rows + 1)
    col_spacing = (w - 2 * side_margin) / pegs_per_row

    # Create pegs (collision_type=2)
    pegs = []
    for row in range(num_rows):
        y = top_margin + (row + 1) * row_spacing
        is_offset = row % 2 == 1
        num_pegs = pegs_per_row if not is_offset else pegs_per_row - 1
        start_x = side_margin + (col_spacing / 2 if is_offset else 0)

        for col in range(num_pegs):
            x = start_x + col * col_spacing
            peg_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            peg_body.position = (x, y)
            peg_shape = pymunk.Circle(peg_body, peg_radius)
            peg_shape.elasticity = peg_elasticity
            peg_shape.friction = 0.6
            peg_shape.collision_type = 2
            space.add(peg_body, peg_shape)
            pegs.append((x, y))

    # Walls (left, right, bottom)
    walls = [
        [(side_margin - 20, top_margin - 50), (side_margin - 20, h - 50)],
        [(w - side_margin + 20, top_margin - 50), (w - side_margin + 20, h - 50)],
        [(side_margin - 20, h - 50), (w - side_margin + 20, h - 50)],
    ]
    for a, b in walls:
        seg = pymunk.Segment(space.static_body, a, b, 5)
        seg.elasticity = 0.4
        seg.friction = 0.5
        space.add(seg)

    # Bin dividers
    num_bins = pegs_per_row + 1
    bin_width = (w - 2 * side_margin + 40) / num_bins
    bin_y_start = h - bottom_margin + 40
    bin_dividers = []
    for i in range(num_bins + 1):
        x = side_margin - 20 + i * bin_width
        seg = pymunk.Segment(space.static_body, (x, bin_y_start), (x, h - 50), 3)
        seg.elasticity = 0.3
        seg.friction = 0.5
        space.add(seg)
        bin_dividers.append(x)

    return space, pegs, bin_dividers, bin_y_start


def run(seed=None, params=None, note_schedule=None, collect_events_only=False):
    """Generator that yields pygame surfaces for each frame.

    Args:
        seed: Random seed for reproducibility
        params: Dict of scene parameters (overrides DEFAULT_PARAMS)
        note_schedule: If provided (music-drives-animation mode), drop balls
                      at scheduled times. List of {time_sec, pitch, ...}
        collect_events_only: If True, skip rendering (for event collection pass)
    """
    global _collision_events
    _collision_events = []

    if seed is not None:
        random.seed(seed)

    params = params or {}
    pal = get_palette(_p(params, "palette"))

    if collect_events_only:
        surface = pygame.Surface((1, 1))
    else:
        surface = pygame.Surface((config.WIDTH, config.HEIGHT))

    space, pegs, bin_dividers, bin_y_start = create_scene(params)

    w, h = config.WIDTH, config.HEIGHT
    num_balls = _p(params, "num_balls")
    ball_radius = _p(params, "ball_radius")
    ball_elasticity = _p(params, "ball_elasticity")
    ball_friction = _p(params, "ball_friction")
    peg_radius = _p(params, "peg_radius")
    drop_interval = _p(params, "drop_interval_frames")
    top_margin = _p(params, "top_margin")
    side_margin = _p(params, "side_margin")

    ball_colors = pal["balls"]

    # Set up collision handler for ball-peg collisions
    frame_ref = [0]  # mutable ref for closure

    def _on_post_solve(arbiter, space, data):
        impulse = arbiter.total_impulse
        force = impulse.length
        if force < 50:
            return True
        pts = arbiter.contact_point_set.points
        if pts:
            pos = pts[0].point_a
        else:
            pos = arbiter.shapes[0].body.position
        _collision_events.append({
            "frame": frame_ref[0],
            "time_sec": frame_ref[0] / config.VIDEO_FPS,
            "x": float(pos.x),
            "y": float(pos.y),
            "force": float(force),
        })
        return True

    space.on_collision(1, 2, post_solve=_on_post_solve)

    # Music-driven drop schedule
    schedule_idx = 0
    if note_schedule:
        note_schedule = sorted(note_schedule, key=lambda n: n["time_sec"])
        num_balls = len(note_schedule)

    balls = []
    ball_queue = num_balls
    drop_counter = 0
    frame_idx = 0

    max_frames = 40 * config.VIDEO_FPS
    settle_frames = 0
    settle_threshold = 3 * config.VIDEO_FPS

    while frame_idx < max_frames:
        frame_ref[0] = frame_idx
        current_time = frame_idx / config.VIDEO_FPS

        # Drop balls
        if note_schedule:
            # Music-driven: drop at scheduled times
            while schedule_idx < len(note_schedule):
                note = note_schedule[schedule_idx]
                if note["time_sec"] <= current_time:
                    # Map pitch to X position
                    pitch = note["pitch"]
                    pitch_range = (48, 84)
                    norm = (pitch - pitch_range[0]) / max(1, pitch_range[1] - pitch_range[0])
                    norm = max(0.0, min(1.0, norm))
                    x = side_margin + norm * (w - 2 * side_margin)

                    body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, ball_radius))
                    body.position = (x, top_margin - 60)
                    shape = pymunk.Circle(body, ball_radius)
                    shape.elasticity = ball_elasticity + random.uniform(-0.05, 0.05)
                    shape.friction = ball_friction
                    shape.collision_type = 1
                    space.add(body, shape)
                    color = ball_colors[len(balls) % len(ball_colors)]
                    balls.append((body, shape, color))
                    schedule_idx += 1
                else:
                    break
        else:
            # Default: drop at fixed interval
            if ball_queue > 0:
                drop_counter += 1
                if drop_counter >= drop_interval:
                    drop_counter = 0
                    ball_queue -= 1

                    body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, ball_radius))
                    x = w / 2 + random.uniform(-30, 30)
                    body.position = (x, top_margin - 60)
                    shape = pymunk.Circle(body, ball_radius)
                    shape.elasticity = ball_elasticity + random.uniform(-0.05, 0.05)
                    shape.friction = ball_friction
                    shape.collision_type = 1
                    space.add(body, shape)
                    color = ball_colors[len(balls) % len(ball_colors)]
                    balls.append((body, shape, color))

        # Step physics
        for _ in range(config.SIM_STEPS_PER_FRAME):
            space.step(1.0 / config.SIM_FPS)

        # Check if balls have settled
        all_dropped = (note_schedule and schedule_idx >= len(note_schedule)) or \
                      (not note_schedule and ball_queue == 0)
        if all_dropped and len(balls) > 0:
            max_vel = max(b.velocity.length for b, _, _ in balls)
            if max_vel < 5:
                settle_frames += 1
            else:
                settle_frames = 0
            if settle_frames >= settle_threshold:
                break

        if not collect_events_only:
            # Draw frame
            surface.fill(pal["bg"])

            # Bin dividers
            for x in bin_dividers:
                pygame.draw.line(surface, pal.get("wall", (180, 180, 180)),
                                 (int(x), bin_y_start), (int(x), h - 50), 3)

            # Walls
            pygame.draw.line(surface, pal.get("wall", (100, 100, 100)),
                             (side_margin - 20, h - 50), (w - side_margin + 20, h - 50), 5)
            pygame.draw.line(surface, pal.get("wall", (100, 100, 100)),
                             (side_margin - 20, top_margin - 50), (side_margin - 20, h - 50), 5)
            pygame.draw.line(surface, pal.get("wall", (100, 100, 100)),
                             (w - side_margin + 20, top_margin - 50), (w - side_margin + 20, h - 50), 5)

            # Pegs
            for px, py in pegs:
                pygame.draw.circle(surface, pal.get("peg", (60, 60, 60)),
                                   (int(px), int(py)), peg_radius)

            # Balls
            for body, shape, color in balls:
                bx, by = int(body.position.x), int(body.position.y)
                if 0 <= bx <= w and 0 <= by <= h:
                    pygame.draw.circle(surface, color, (bx, by), ball_radius)
                    highlight_pos = (bx - ball_radius // 3, by - ball_radius // 3)
                    pygame.draw.circle(surface, (255, 255, 255), highlight_pos, ball_radius // 4)

        yield surface
        frame_idx += 1

    # Hold final frame
    for _ in range(2 * config.VIDEO_FPS):
        yield surface


def get_metadata(video_num, params=None):
    params = params or {}
    return {
        "title": f"Plinko Ball Drop #{video_num}",
        "description": "Watch colorful balls bounce through pegs! Which bin gets the most? #Shorts #plinko #satisfying",
        "tags": ["plinko", "physics", "satisfying", "balls", "galton board", "shorts"],
        "category": "24",
    }
