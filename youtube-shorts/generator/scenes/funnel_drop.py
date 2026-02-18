"""Funnel Drop scene.

Balls cascade through a series of funnel-shaped structures, bottlenecking
and spreading as they flow down. Funnels alternate between centered and
offset positions, creating interesting flow patterns.
"""

import random
import math
import pymunk
import pygame
from .. import config
from ..palettes import get_palette

DEFAULT_PARAMS = {
    "num_balls": 35,        # range 20-60
    "num_funnels": 5,       # range 3-7
    "ball_radius": 12,      # range 8-16
    "gravity": 800,         # range 600-1000
    "funnel_width": 350,    # range 200-500  (width at top of each funnel)
    "funnel_gap": 60,       # range 40-100   (bottleneck gap at bottom)
    "drop_interval": 3,     # range 2-8      (frames between ball drops)
    "palette": "classic",
}

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
    """Set up the pymunk space with funnels, deflectors, walls, and floor."""
    params = params or {}
    space = pymunk.Space()
    space.gravity = (0, _p(params, "gravity"))

    w, h = config.WIDTH, config.HEIGHT
    num_funnels = _p(params, "num_funnels")
    funnel_width_base = _p(params, "funnel_width")
    funnel_gap = _p(params, "funnel_gap")

    side_margin = 40
    top_margin = 180
    bottom_margin = 200

    # Usable vertical space for funnels
    usable_height = h - top_margin - bottom_margin
    funnel_spacing = usable_height / num_funnels
    funnel_height = funnel_spacing * 0.6  # each funnel's V depth

    funnel_segments = []  # store for drawing: list of ((x1,y1),(x2,y2))
    deflector_segments = []  # ramps between funnels

    for i in range(num_funnels):
        # Shrink funnel width for lower funnels
        shrink = 1.0 - (i * 0.08)
        fw = funnel_width_base * shrink
        fg = funnel_gap + i * 4  # gap grows slightly too

        # Funnel center: alternate centered / offset left / offset right
        if i % 3 == 0:
            cx = w / 2
        elif i % 3 == 1:
            cx = w / 2 - 120
        else:
            cx = w / 2 + 120

        # Clamp center so funnel stays on screen
        half_w = fw / 2
        cx = max(side_margin + half_w + 10, min(w - side_margin - half_w - 10, cx))

        # Top of this funnel's V
        fy_top = top_margin + i * funnel_spacing
        fy_bottom = fy_top + funnel_height

        # Left arm of V: from (cx - half_w, fy_top) to (cx - fg/2, fy_bottom)
        left_top = (cx - half_w, fy_top)
        left_bottom = (cx - fg / 2, fy_bottom)

        seg_l = pymunk.Segment(space.static_body, left_top, left_bottom, 4)
        seg_l.elasticity = 0.3
        seg_l.friction = 0.5
        seg_l.collision_type = 2
        space.add(seg_l)
        funnel_segments.append((left_top, left_bottom))

        # Right arm of V: from (cx + half_w, fy_top) to (cx + fg/2, fy_bottom)
        right_top = (cx + half_w, fy_top)
        right_bottom = (cx + fg / 2, fy_bottom)

        seg_r = pymunk.Segment(space.static_body, right_top, right_bottom, 4)
        seg_r.elasticity = 0.3
        seg_r.friction = 0.5
        seg_r.collision_type = 2
        space.add(seg_r)
        funnel_segments.append((right_top, right_bottom))

        # Add deflectors between funnels (not after last funnel)
        if i < num_funnels - 1:
            # Next funnel center
            next_shrink = 1.0 - ((i + 1) * 0.08)
            next_fw = funnel_width_base * next_shrink
            if (i + 1) % 3 == 0:
                next_cx = w / 2
            elif (i + 1) % 3 == 1:
                next_cx = w / 2 - 120
            else:
                next_cx = w / 2 + 120
            next_half_w = next_fw / 2
            next_cx = max(side_margin + next_half_w + 10,
                          min(w - side_margin - next_half_w - 10, next_cx))

            # Deflector sits in the gap between this funnel's bottom and next funnel's top
            defl_y = fy_bottom + (funnel_spacing - funnel_height) * 0.4
            defl_length = 100 + random.randint(0, 60)

            # Direction: guide balls toward next funnel's center
            if next_cx > cx:
                # Next funnel is to the right, angle deflector right
                defl_start = (cx - defl_length * 0.3, defl_y)
                defl_end = (cx + defl_length * 0.7, defl_y + 30)
            elif next_cx < cx:
                # Next funnel is to the left, angle deflector left
                defl_start = (cx + defl_length * 0.3, defl_y)
                defl_end = (cx - defl_length * 0.7, defl_y + 30)
            else:
                # Same center -- add a small centered bump / splitter
                defl_start = (cx - defl_length * 0.5, defl_y + 25)
                defl_end = (cx + defl_length * 0.5, defl_y + 25)

            seg_d = pymunk.Segment(space.static_body, defl_start, defl_end, 3)
            seg_d.elasticity = 0.35
            seg_d.friction = 0.4
            seg_d.collision_type = 2
            space.add(seg_d)
            deflector_segments.append((defl_start, defl_end))

    # Side walls
    wall_left = pymunk.Segment(space.static_body,
                                (side_margin, 0), (side_margin, h - 50), 5)
    wall_left.elasticity = 0.3
    wall_left.friction = 0.5
    space.add(wall_left)

    wall_right = pymunk.Segment(space.static_body,
                                 (w - side_margin, 0), (w - side_margin, h - 50), 5)
    wall_right.elasticity = 0.3
    wall_right.friction = 0.5
    space.add(wall_right)

    # Floor
    floor = pymunk.Segment(space.static_body,
                            (side_margin, h - 50), (w - side_margin, h - 50), 5)
    floor.elasticity = 0.2
    floor.friction = 0.7
    space.add(floor)

    # Collection bins at the bottom
    bin_y_top = h - bottom_margin + 40
    bin_y_bottom = h - 50
    num_bins = 5
    bin_width = (w - 2 * side_margin) / num_bins
    bin_dividers = []
    for i in range(num_bins + 1):
        bx = side_margin + i * bin_width
        seg_b = pymunk.Segment(space.static_body,
                                (bx, bin_y_top), (bx, bin_y_bottom), 3)
        seg_b.elasticity = 0.2
        seg_b.friction = 0.5
        space.add(seg_b)
        bin_dividers.append(bx)

    # Angled guides into the bins
    last_funnel_bottom_y = top_margin + (num_funnels - 1) * funnel_spacing + funnel_height
    guide_y = last_funnel_bottom_y + (bin_y_top - last_funnel_bottom_y) * 0.5

    seg_gl = pymunk.Segment(space.static_body,
                             (side_margin, guide_y - 20),
                             (w / 2 - 40, guide_y + 40), 3)
    seg_gl.elasticity = 0.3
    seg_gl.friction = 0.4
    seg_gl.collision_type = 2
    space.add(seg_gl)
    deflector_segments.append(((side_margin, guide_y - 20), (w / 2 - 40, guide_y + 40)))

    seg_gr = pymunk.Segment(space.static_body,
                             (w - side_margin, guide_y - 20),
                             (w / 2 + 40, guide_y + 40), 3)
    seg_gr.elasticity = 0.3
    seg_gr.friction = 0.4
    seg_gr.collision_type = 2
    space.add(seg_gr)
    deflector_segments.append(((w - side_margin, guide_y - 20), (w / 2 + 40, guide_y + 40)))

    scene_data = {
        "funnel_segments": funnel_segments,
        "deflector_segments": deflector_segments,
        "bin_dividers": bin_dividers,
        "bin_y_top": bin_y_top,
        "side_margin": side_margin,
        "top_margin": top_margin,
    }
    return space, scene_data


def run(seed=None, params=None, note_schedule=None, collect_events_only=False):
    """Generator that yields pygame surfaces for each frame.

    Args:
        seed: Random seed for reproducibility
        params: Dict of scene parameters (overrides DEFAULT_PARAMS)
        note_schedule: If provided, drop balls at scheduled times.
                      List of {time_sec, pitch, ...}
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

    space, scene_data = create_scene(params)

    w, h = config.WIDTH, config.HEIGHT
    num_balls = _p(params, "num_balls")
    ball_radius = _p(params, "ball_radius")
    drop_interval = _p(params, "drop_interval")
    funnel_width_base = _p(params, "funnel_width")
    side_margin = scene_data["side_margin"]
    top_margin = scene_data["top_margin"]
    funnel_segments = scene_data["funnel_segments"]
    deflector_segments = scene_data["deflector_segments"]
    bin_dividers = scene_data["bin_dividers"]
    bin_y_top = scene_data["bin_y_top"]

    ball_colors = pal["balls"]

    # Collision handler: ball (1) vs wall/funnel (2)
    frame_ref = [0]

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

    # First funnel center X for drop spread
    first_funnel_cx = w / 2  # funnel 0 is always centered (i%3==0)
    drop_spread = funnel_width_base * 0.35

    max_frames = 45 * config.VIDEO_FPS
    settle_frames = 0
    settle_threshold = 2 * config.VIDEO_FPS

    while frame_idx < max_frames:
        frame_ref[0] = frame_idx
        current_time = frame_idx / config.VIDEO_FPS

        # Drop balls
        if note_schedule:
            while schedule_idx < len(note_schedule):
                note = note_schedule[schedule_idx]
                if note["time_sec"] <= current_time:
                    pitch = note["pitch"]
                    pitch_range = (48, 84)
                    norm = (pitch - pitch_range[0]) / max(1, pitch_range[1] - pitch_range[0])
                    norm = max(0.0, min(1.0, norm))
                    x = first_funnel_cx - drop_spread + norm * drop_spread * 2

                    body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, ball_radius))
                    body.position = (x, top_margin - 80)
                    shape = pymunk.Circle(body, ball_radius)
                    shape.elasticity = 0.5 + random.uniform(-0.05, 0.05)
                    shape.friction = 0.3
                    shape.collision_type = 1
                    space.add(body, shape)
                    color = ball_colors[len(balls) % len(ball_colors)]
                    balls.append((body, shape, color))
                    schedule_idx += 1
                else:
                    break
        else:
            if ball_queue > 0:
                drop_counter += 1
                if drop_counter >= drop_interval:
                    drop_counter = 0
                    ball_queue -= 1

                    x = first_funnel_cx + random.uniform(-drop_spread, drop_spread)

                    body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, ball_radius))
                    body.position = (x, top_margin - 80)
                    shape = pymunk.Circle(body, ball_radius)
                    shape.elasticity = 0.5 + random.uniform(-0.05, 0.05)
                    shape.friction = 0.3
                    shape.collision_type = 1
                    space.add(body, shape)
                    color = ball_colors[len(balls) % len(ball_colors)]
                    balls.append((body, shape, color))

        # Step physics
        for _ in range(config.SIM_STEPS_PER_FRAME):
            space.step(1.0 / config.SIM_FPS)

        # Check if all balls dropped and settled
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

        # Enforce a minimum duration of ~30s worth of sim frames
        # (balls should have time to cascade through all funnels)

        if not collect_events_only:
            surface.fill(pal["bg"])

            wall_color = pal.get("wall", (100, 100, 100))
            obstacle_color = pal.get("obstacle", (80, 80, 80))

            # Draw funnel walls (thick lines)
            for (x1, y1), (x2, y2) in funnel_segments:
                pygame.draw.line(surface, wall_color,
                                 (int(x1), int(y1)), (int(x2), int(y2)), 5)

            # Draw deflectors
            for (x1, y1), (x2, y2) in deflector_segments:
                pygame.draw.line(surface, obstacle_color,
                                 (int(x1), int(y1)), (int(x2), int(y2)), 4)

            # Draw side walls
            pygame.draw.line(surface, wall_color,
                             (side_margin, 0), (side_margin, h - 50), 5)
            pygame.draw.line(surface, wall_color,
                             (w - side_margin, 0), (w - side_margin, h - 50), 5)

            # Draw floor
            pygame.draw.line(surface, wall_color,
                             (side_margin, h - 50), (w - side_margin, h - 50), 5)

            # Draw bin dividers
            for bx in bin_dividers:
                pygame.draw.line(surface, pal.get("wall", (180, 180, 180)),
                                 (int(bx), bin_y_top), (int(bx), h - 50), 3)

            # Draw balls with highlight
            for body, shape, color in balls:
                bx, by = int(body.position.x), int(body.position.y)
                if 0 <= bx <= w and 0 <= by <= h:
                    pygame.draw.circle(surface, color, (bx, by), ball_radius)
                    # Specular highlight
                    hx = bx - ball_radius // 3
                    hy = by - ball_radius // 3
                    pygame.draw.circle(surface, (255, 255, 255),
                                       (hx, hy), ball_radius // 4)

        yield surface
        frame_idx += 1

    # Hold final frame for 2 seconds
    for _ in range(2 * config.VIDEO_FPS):
        yield surface


def get_metadata(video_num, params=None):
    params = params or {}
    n = _p(params, "num_balls")
    f = _p(params, "num_funnels")
    return {
        "title": f"Funnel Drop #{video_num}",
        "description": (
            f"Watch {n} balls cascade through {f} funnels! "
            "Which bin catches the most? #Shorts #physics #satisfying"
        ),
        "tags": ["funnel drop", "physics", "satisfying", "balls", "cascade", "shorts"],
        "category": "24",
    }
