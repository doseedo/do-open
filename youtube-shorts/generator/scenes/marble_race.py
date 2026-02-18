"""Marble Race scene.

Multiple colored marbles race down a course with ramps, funnels,
spinning obstacles, and bumpers. Supports parameterization and collision events.
"""

import random
import math
import pymunk
import pygame
from .. import config
from ..palettes import BALL_COLORS, get_palette

DEFAULT_PARAMS = {
    "num_marbles": 5,
    "marble_radius": 20,
    "marble_elasticity": 0.6,
    "marble_friction": 0.4,
    "gravity": 800,
    "num_bumper_rows": 3,
    "num_spinners": 2,
    "wall_thickness": 6,
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
    return params.get(key, DEFAULT_PARAMS[key])


def add_ramp(space, x1, y1, x2, y2, segments, wall_thickness=6):
    seg = pymunk.Segment(space.static_body, (x1, y1), (x2, y2), wall_thickness)
    seg.elasticity = 0.4
    seg.friction = 0.5
    seg.collision_type = 3  # ramp
    space.add(seg)
    segments.append(((x1, y1), (x2, y2)))
    return seg


def add_bumper(space, x, y, radius, bumpers_list):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (x, y)
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.8
    shape.friction = 0.3
    shape.collision_type = 4  # bumper
    space.add(body, shape)
    bumpers_list.append((x, y, radius))


def create_scene(params=None):
    params = params or {}
    space = pymunk.Space()
    space.gravity = (0, _p(params, "gravity"))

    w, h = config.WIDTH, config.HEIGHT
    wall_thickness = _p(params, "wall_thickness")
    num_bumper_rows = _p(params, "num_bumper_rows")
    num_spinners = _p(params, "num_spinners")
    segments = []
    bumpers = []
    spinners = []

    margin = 60
    course_left = margin
    course_right = w - margin

    # Starting gate
    gate_y = 200
    gate_width = 400 + random.randint(-50, 100)
    gate_left = (w - gate_width) / 2
    gate_right = gate_left + gate_width

    add_ramp(space, course_left, gate_y - 50, gate_left, gate_y + 40, segments, wall_thickness)
    add_ramp(space, course_right, gate_y - 50, gate_right, gate_y + 40, segments, wall_thickness)

    # Section 1: Alternating ramps (randomized angles)
    y = gate_y + 120
    ramp_gap = 40 + random.randint(0, 30)
    num_ramps = random.randint(2, 4)
    for i in range(num_ramps):
        drop = 70 + random.randint(0, 40)
        if i % 2 == 0:
            add_ramp(space, course_left, y, course_right - ramp_gap, y + drop, segments, wall_thickness)
        else:
            add_ramp(space, course_left + ramp_gap, y, course_right, y + drop, segments, wall_thickness)
        y += drop + 50 + random.randint(0, 20)

    # Section 2: Bumper field
    bumper_y_start = y + 30
    bumper_radius = 20 + random.randint(0, 10)
    for row in range(num_bumper_rows):
        by = bumper_y_start + row * (80 + random.randint(0, 30))
        num_bumps = 4 if row % 2 == 0 else 3
        spacing = (course_right - course_left) / (num_bumps + 1)
        offset = 0 if row % 2 == 0 else spacing / 2
        for col in range(num_bumps):
            bx = course_left + offset + (col + 1) * spacing
            add_bumper(space, bx, by, bumper_radius, bumpers)

    y = bumper_y_start + num_bumper_rows * 100 + 20

    # Section 3: Narrow funnel
    funnel_width = 150 + random.randint(0, 100)
    funnel_left = (w - funnel_width) / 2
    funnel_right = funnel_left + funnel_width
    add_ramp(space, course_left, y, funnel_left, y + 120, segments, wall_thickness)
    add_ramp(space, course_right, y, funnel_right, y + 120, segments, wall_thickness)
    y += 140

    add_ramp(space, funnel_left, y, funnel_left, y + 80, segments, wall_thickness)
    add_ramp(space, funnel_right, y, funnel_right, y + 80, segments, wall_thickness)
    y += 80

    add_ramp(space, funnel_left, y, course_left + 50, y + 80, segments, wall_thickness)
    add_ramp(space, funnel_right, y, course_right - 50, y + 80, segments, wall_thickness)
    y += 100

    # Section 4: Spinning obstacles
    for i in range(num_spinners):
        spinner_y = y + i * 180
        spinner_x = w / 2 + ((-1) ** i) * (80 + random.randint(0, 50))
        spinner_length = 160 + random.randint(0, 80)
        spinner_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        spinner_body.position = (spinner_x, spinner_y)
        spinner_body.angular_velocity = (2.0 + random.random()) * ((-1) ** i)

        half = spinner_length / 2
        seg = pymunk.Segment(spinner_body, (-half, 0), (half, 0), 8)
        seg.elasticity = 0.5
        seg.friction = 0.3
        space.add(spinner_body, seg)
        spinners.append((spinner_body, spinner_length, 8))

    y += num_spinners * 180 + 20

    # Section 5: Final ramps
    add_ramp(space, course_left, y, course_right - ramp_gap, y + 70, segments, wall_thickness)
    y += 110
    add_ramp(space, course_left + ramp_gap, y, course_right, y + 70, segments, wall_thickness)
    y += 110

    finish_y = y + 30

    # Floor and side walls
    add_ramp(space, course_left, h - 60, course_right, h - 60, segments, wall_thickness)
    add_ramp(space, course_left, 0, course_left, h - 60, segments, wall_thickness)
    add_ramp(space, course_right, 0, course_right, h - 60, segments, wall_thickness)

    return space, segments, bumpers, spinners, finish_y


def run(seed=None, params=None, note_schedule=None, collect_events_only=False):
    """Generator that yields pygame surfaces for each frame."""
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

    space, segments, bumpers, spinners, finish_y = create_scene(params)

    w, h = config.WIDTH, config.HEIGHT
    margin = 60
    num_marbles = _p(params, "num_marbles")
    marble_radius = _p(params, "marble_radius")
    marble_elasticity = _p(params, "marble_elasticity")
    marble_friction = _p(params, "marble_friction")
    wall_thickness = _p(params, "wall_thickness")
    ball_colors = pal["balls"]

    # Collision handler
    frame_ref = [0]

    def _on_post_solve(arbiter, space, data):
        impulse = arbiter.total_impulse
        force = impulse.length
        if force < 80:
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

    # Marble-bumper and marble-ramp handlers
    for ctype in [3, 4]:
        space.on_collision(1, ctype, post_solve=_on_post_solve)

    # Create marbles
    marbles = []
    start_x_center = w / 2
    spacing = marble_radius * 3
    start_x = start_x_center - (num_marbles - 1) * spacing / 2

    for i in range(num_marbles):
        body = pymunk.Body(1.5, pymunk.moment_for_circle(1.5, 0, marble_radius))
        body.position = (start_x + i * spacing, 100 + random.uniform(-5, 5))
        shape = pymunk.Circle(body, marble_radius)
        shape.elasticity = marble_elasticity + random.uniform(-0.05, 0.05)
        shape.friction = marble_friction
        shape.collision_type = 1
        space.add(body, shape)
        color = ball_colors[i % len(ball_colors)]
        marbles.append((body, shape, color, False))

    max_frames = 45 * config.VIDEO_FPS
    frame_idx = 0
    finish_order = []
    all_finished = False
    end_hold_start = 0

    font = None
    if not collect_events_only:
        try:
            font = pygame.font.SysFont("Arial", 36, bold=True)
        except Exception:
            font = pygame.font.Font(None, 36)

    while frame_idx < max_frames:
        frame_ref[0] = frame_idx

        for _ in range(config.SIM_STEPS_PER_FRAME):
            space.step(1.0 / config.SIM_FPS)

        # Check finish crossings
        for idx, (body, shape, color, finished) in enumerate(marbles):
            if not finished and body.position.y >= finish_y:
                marbles[idx] = (body, shape, color, True)
                finish_order.append((idx, color))

        if len(finish_order) == num_marbles and not all_finished:
            all_finished = True
            end_hold_start = frame_idx

        if all_finished and frame_idx - end_hold_start >= 3 * config.VIDEO_FPS:
            break

        if not collect_events_only:
            surface.fill(pal["bg"])

            # Ramp segments
            for (x1, y1), (x2, y2) in segments:
                pygame.draw.line(surface, pal.get("wall", (100, 100, 100)),
                                 (int(x1), int(y1)), (int(x2), int(y2)), wall_thickness)

            # Bumpers
            for bx, by, br in bumpers:
                pygame.draw.circle(surface, pal.get("obstacle", (80, 80, 80)),
                                   (int(bx), int(by)), br)
                pygame.draw.circle(surface, pal.get("wall", (100, 100, 100)),
                                   (int(bx), int(by)), br, 3)

            # Spinners
            for spinner_body, length, width in spinners:
                cx, cy = int(spinner_body.position.x), int(spinner_body.position.y)
                angle = spinner_body.angle
                half = length / 2
                dx = math.cos(angle) * half
                dy = math.sin(angle) * half
                p1 = (int(cx - dx), int(cy - dy))
                p2 = (int(cx + dx), int(cy + dy))
                pygame.draw.line(surface, pal.get("obstacle", (80, 80, 80)), p1, p2, width * 2)
                pygame.draw.circle(surface, pal.get("wall", (100, 100, 100)), (cx, cy), 8)

            # Finish line
            finish_green = (46, 204, 113)
            for x in range(margin, w - margin, 40):
                cf = finish_green if (x // 40) % 2 == 0 else (255, 255, 255)
                pygame.draw.rect(surface, cf, (x, finish_y - 5, 20, 10))
            if font:
                ft = font.render("FINISH", True, finish_green)
                tr = ft.get_rect(center=(w // 2, finish_y - 25))
                surface.blit(ft, tr)

            # Marbles
            for body, shape, color, finished in marbles:
                mx, my = int(body.position.x), int(body.position.y)
                if 0 <= mx <= w and 0 <= my <= h:
                    pygame.draw.circle(surface, color, (mx, my), marble_radius)
                    hx = mx - marble_radius // 3
                    hy = my - marble_radius // 3
                    pygame.draw.circle(surface, (255, 255, 255), (hx, hy), marble_radius // 3)

            # Finish overlay
            if finish_order and font:
                for rank, (idx, color) in enumerate(finish_order):
                    label = font.render(f"#{rank + 1}", True, pal.get("text", (40, 40, 40)))
                    oy = 40 + rank * 50
                    pygame.draw.circle(surface, color, (50, oy + 15), 15)
                    surface.blit(label, (75, oy))

        yield surface
        frame_idx += 1


def get_metadata(video_num, params=None):
    params = params or {}
    n = _p(params, "num_marbles")
    return {
        "title": f"Epic Marble Race #{video_num}",
        "description": f"Which marble wins the {n}-way race? Watch till the end! #Shorts #marblerace #satisfying",
        "tags": ["marble race", "physics", "satisfying", "race", "marbles", "shorts"],
        "category": "24",
    }
