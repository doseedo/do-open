"""Xylophone Bounce scene.

Balls drop from the top and bounce across angled/horizontal colored bars
that represent xylophone keys. Each bar has an assigned pitch. When balls
hit bars, they play notes. Bars flash briefly on impact.
"""

import random
import math
import pymunk
import pygame
from .. import config
from ..palettes import get_palette

# Default parameters (overridable via params dict)
DEFAULT_PARAMS = {
    "num_bars": 12,            # 8-16
    "num_balls": 20,           # 10-35
    "ball_radius": 14,         # 10-18
    "gravity": 600,            # 400-800
    "elasticity": 0.75,        # 0.6-0.9
    "bar_arrangement": "cascade",  # "cascade", "stacked", or "v_shape"
    "palette": "classic",
    "drop_interval_frames": 25,
    "ball_friction": 0.3,
    "bar_thickness": 14,
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


def _lerp_color(c1, c2, t):
    """Linearly interpolate between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _brighten(color, amount=60):
    """Brighten a color by a fixed amount (for flash effects)."""
    return (
        min(255, color[0] + amount),
        min(255, color[1] + amount),
        min(255, color[2] + amount),
    )


# Xylophone gradient: warm (low pitch) to cool (high pitch)
_GRADIENT_COLORS = [
    (220, 50, 50),     # red (lowest)
    (240, 120, 40),    # orange
    (240, 200, 40),    # yellow
    (100, 200, 80),    # green
    (50, 160, 220),    # blue
    (100, 80, 200),    # indigo
    (160, 60, 180),    # violet (highest)
]


def _bar_color(index, total):
    """Get a gradient color for a bar based on its index (0=lowest pitch)."""
    if total <= 1:
        return _GRADIENT_COLORS[0]
    t = index / (total - 1)
    # Map t across the gradient stops
    num_stops = len(_GRADIENT_COLORS)
    segment = t * (num_stops - 1)
    low = int(segment)
    high = min(low + 1, num_stops - 1)
    frac = segment - low
    return _lerp_color(_GRADIENT_COLORS[low], _GRADIENT_COLORS[high], frac)


def _pitch_for_bar(index, total, pitch_range=(48, 84)):
    """Assign a MIDI pitch to a bar. Index 0 = lowest pitch (bottom)."""
    if total <= 1:
        return (pitch_range[0] + pitch_range[1]) // 2
    t = index / (total - 1)
    return int(pitch_range[0] + t * (pitch_range[1] - pitch_range[0]))


def create_scene(params=None):
    """Set up the pymunk space with xylophone bars and walls.

    Returns:
        space: pymunk.Space
        bars: list of dicts, each with keys:
            'shape': pymunk.Segment
            'a', 'b': endpoint tuples
            'color': RGB tuple
            'pitch': MIDI pitch int
            'index': bar index (0=bottom/lowest)
            'flash_until': frame when flash expires (0 initially)
    """
    params = params or {}
    space = pymunk.Space()
    space.gravity = (0, _p(params, "gravity"))

    w, h = config.WIDTH, config.HEIGHT
    num_bars = _p(params, "num_bars")
    elasticity = _p(params, "elasticity")
    bar_thickness = _p(params, "bar_thickness")
    arrangement = _p(params, "bar_arrangement")

    side_margin = 80
    top_y = 200
    bottom_y = 1700
    bar_area_height = bottom_y - top_y

    # Bar widths: bottom bars (low pitch) are wider, top bars (high pitch) shorter
    min_bar_width = 160
    max_bar_width = w - 2 * side_margin - 40  # ~960

    bars = []

    # We use collision_type = 2 for all bars. To identify which bar was hit,
    # we store bar info on the pymunk shape via a custom attribute.
    bar_collision_type = 2

    for i in range(num_bars):
        # i=0 is the BOTTOM bar (lowest pitch, longest)
        # i=num_bars-1 is the TOP bar (highest pitch, shortest)
        frac = i / max(1, num_bars - 1)  # 0 at bottom, 1 at top
        y = bottom_y - frac * bar_area_height

        bar_width = max_bar_width - frac * (max_bar_width - min_bar_width)
        half_w = bar_width / 2
        cx = w / 2  # center X

        if arrangement == "cascade":
            # Alternating left-right tilt, like stairs
            angle_deg = 12 if i % 2 == 0 else -12
            angle_rad = math.radians(angle_deg)
            # Offset center alternately so ball cascades down
            offset_x = 60 * (1 if i % 2 == 0 else -1)
            cx += offset_x
            ax = cx - half_w * math.cos(angle_rad)
            ay = y - half_w * math.sin(angle_rad)
            bx = cx + half_w * math.cos(angle_rad)
            by = y + half_w * math.sin(angle_rad)
        elif arrangement == "v_shape":
            # V-shape: bars angled inward forming a V
            spread = 0.5 * (1.0 - frac)  # more spread at bottom
            angle_deg = 20 * (1 if i % 2 == 0 else -1) * (1.0 - frac * 0.5)
            angle_rad = math.radians(angle_deg)
            offset_x = 120 * (1 if i % 2 == 0 else -1) * (1.0 - frac * 0.3)
            cx += offset_x
            ax = cx - half_w * math.cos(angle_rad)
            ay = y - half_w * math.sin(angle_rad)
            bx = cx + half_w * math.cos(angle_rad)
            by = y + half_w * math.sin(angle_rad)
        else:
            # "stacked": horizontal bars offset left/right alternately
            offset_x = 100 * (1 if i % 2 == 0 else -1)
            cx += offset_x
            ax = cx - half_w
            ay = y
            bx = cx + half_w
            by = y

        # Create pymunk segment
        seg_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        seg = pymunk.Segment(seg_body, (ax, ay), (bx, by), bar_thickness)
        seg.elasticity = elasticity
        seg.friction = 0.5
        seg.collision_type = bar_collision_type
        space.add(seg_body, seg)

        color = _bar_color(i, num_bars)
        pitch = _pitch_for_bar(i, num_bars)

        # Attach bar index to shape for collision identification
        seg.bar_index = i

        bars.append({
            "shape": seg,
            "a": (ax, ay),
            "b": (bx, by),
            "color": color,
            "pitch": pitch,
            "index": i,
            "flash_until": 0,
        })

    # Side walls to keep balls in play
    wall_segs = [
        [(side_margin - 30, top_y - 100), (side_margin - 30, h - 50)],
        [(w - side_margin + 30, top_y - 100), (w - side_margin + 30, h - 50)],
        [(side_margin - 30, h - 50), (w - side_margin + 30, h - 50)],
    ]
    for a, b in wall_segs:
        seg = pymunk.Segment(space.static_body, a, b, 5)
        seg.elasticity = 0.4
        seg.friction = 0.5
        seg.collision_type = 3  # wall
        space.add(seg)

    return space, bars


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

    space, bars = create_scene(params)

    w, h = config.WIDTH, config.HEIGHT
    num_balls = _p(params, "num_balls")
    ball_radius = _p(params, "ball_radius")
    elasticity = _p(params, "elasticity")
    ball_friction = _p(params, "ball_friction")
    bar_thickness = _p(params, "bar_thickness")
    drop_interval = _p(params, "drop_interval_frames")
    side_margin = 80

    ball_colors = pal["balls"]

    # Build a mapping from bar shape id to bar dict for collision lookup
    bar_shape_map = {}
    for bar in bars:
        bar_shape_map[id(bar["shape"])] = bar

    # Set up collision handler for ball-bar collisions
    frame_ref = [0]
    flash_duration = 6  # frames to flash a bar after hit

    def _on_post_solve(arbiter, space, data):
        impulse = arbiter.total_impulse
        force = impulse.length
        if force < 40:
            return True
        pts = arbiter.contact_point_set.points
        if pts:
            pos = pts[0].point_a
        else:
            pos = arbiter.shapes[0].body.position

        # Identify which bar was hit
        hit_bar = None
        for shape in arbiter.shapes:
            sid = id(shape)
            if sid in bar_shape_map:
                hit_bar = bar_shape_map[sid]
                break

        event = {
            "frame": frame_ref[0],
            "time_sec": frame_ref[0] / config.VIDEO_FPS,
            "x": float(pos.x),
            "y": float(pos.y),
            "force": float(force),
        }
        if hit_bar is not None:
            event["extra_pitch"] = hit_bar["pitch"]
            # Trigger flash
            hit_bar["flash_until"] = frame_ref[0] + flash_duration

        _collision_events.append(event)
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

    # ~32 seconds of content + 2 second hold at end
    max_frames = 35 * config.VIDEO_FPS
    settle_frames = 0
    settle_threshold = 3 * config.VIDEO_FPS

    while frame_idx < max_frames:
        frame_ref[0] = frame_idx
        current_time = frame_idx / config.VIDEO_FPS

        # Drop balls
        if note_schedule:
            while schedule_idx < len(note_schedule):
                note = note_schedule[schedule_idx]
                if note["time_sec"] <= current_time:
                    # Map pitch to X position of the closest bar
                    pitch = note["pitch"]
                    # Find the bar whose pitch is closest
                    best_bar = min(bars, key=lambda b: abs(b["pitch"] - pitch))
                    # Drop ball above the midpoint of that bar
                    bar_cx = (best_bar["a"][0] + best_bar["b"][0]) / 2
                    x = bar_cx + random.uniform(-20, 20)

                    body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, ball_radius))
                    body.position = (x, 100)
                    shape = pymunk.Circle(body, ball_radius)
                    shape.elasticity = elasticity + random.uniform(-0.05, 0.05)
                    shape.friction = ball_friction
                    shape.collision_type = 1
                    space.add(body, shape)
                    color = ball_colors[len(balls) % len(ball_colors)]
                    balls.append((body, shape, color))
                    schedule_idx += 1
                else:
                    break
        else:
            # Default: drop at fixed interval from varying positions
            if ball_queue > 0:
                drop_counter += 1
                if drop_counter >= drop_interval:
                    drop_counter = 0
                    ball_queue -= 1

                    body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, ball_radius))
                    # Drop from random X near the top, biased toward center
                    x = w / 2 + random.gauss(0, 120)
                    x = max(side_margin + ball_radius, min(w - side_margin - ball_radius, x))
                    body.position = (x, 100)
                    shape = pymunk.Circle(body, ball_radius)
                    shape.elasticity = elasticity + random.uniform(-0.05, 0.05)
                    shape.friction = ball_friction
                    shape.collision_type = 1
                    space.add(body, shape)
                    color = ball_colors[len(balls) % len(ball_colors)]
                    balls.append((body, shape, color))

        # Step physics
        for _ in range(config.SIM_STEPS_PER_FRAME):
            space.step(1.0 / config.SIM_FPS)

        # Check if all balls have settled
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

            # Draw bars
            for bar in bars:
                ax, ay = bar["a"]
                bx, by = bar["b"]
                color = bar["color"]

                # Flash effect if recently hit
                if frame_idx < bar["flash_until"]:
                    color = _brighten(color, 80)

                # Draw bar as a thick rounded-end line (rounded caps via circles)
                thickness = bar_thickness
                pygame.draw.line(surface, color,
                                 (int(ax), int(ay)), (int(bx), int(by)),
                                 thickness * 2)
                # Rounded end caps
                pygame.draw.circle(surface, color, (int(ax), int(ay)), thickness)
                pygame.draw.circle(surface, color, (int(bx), int(by)), thickness)

                # Subtle darker outline for depth
                outline_color = _lerp_color(color, (0, 0, 0), 0.25)
                pygame.draw.line(surface, outline_color,
                                 (int(ax), int(ay)), (int(bx), int(by)),
                                 thickness * 2 + 4)
                pygame.draw.circle(surface, outline_color, (int(ax), int(ay)), thickness + 2)
                pygame.draw.circle(surface, outline_color, (int(bx), int(by)), thickness + 2)
                # Redraw filled bar on top of outline
                pygame.draw.line(surface, color,
                                 (int(ax), int(ay)), (int(bx), int(by)),
                                 thickness * 2)
                pygame.draw.circle(surface, color, (int(ax), int(ay)), thickness)
                pygame.draw.circle(surface, color, (int(bx), int(by)), thickness)

            # Draw side walls (subtle)
            wall_color = pal.get("wall", (100, 100, 100))
            pygame.draw.line(surface, wall_color,
                             (side_margin - 30, 150), (side_margin - 30, h - 50), 4)
            pygame.draw.line(surface, wall_color,
                             (w - side_margin + 30, 150), (w - side_margin + 30, h - 50), 4)
            # Floor
            pygame.draw.line(surface, wall_color,
                             (side_margin - 30, h - 50), (w - side_margin + 30, h - 50), 5)

            # Draw balls
            for body, shape, color in balls:
                bx, by = int(body.position.x), int(body.position.y)
                if 0 <= bx <= w and 0 <= by <= h:
                    pygame.draw.circle(surface, color, (bx, by), ball_radius)
                    # Highlight for depth (small white circle, upper-left)
                    highlight_x = bx - ball_radius // 3
                    highlight_y = by - ball_radius // 3
                    pygame.draw.circle(surface, (255, 255, 255),
                                       (highlight_x, highlight_y), ball_radius // 4)

        yield surface
        frame_idx += 1

    # Hold final frame for 2 seconds
    for _ in range(2 * config.VIDEO_FPS):
        yield surface


def get_metadata(video_num, params=None):
    params = params or {}
    arrangement = _p(params, "bar_arrangement")
    return {
        "title": f"Xylophone Ball Bounce #{video_num}",
        "description": (
            "Watch balls cascade down colorful xylophone bars! "
            "Every bounce plays a note. #Shorts #xylophone #satisfying #physics"
        ),
        "tags": [
            "xylophone", "physics", "satisfying", "balls", "music",
            "bounce", "asmr", "shorts",
        ],
        "category": "24",
    }
