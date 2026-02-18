"""Wrecking Ball scene.

A heavy wrecking ball on a string swings into a tower of stacked blocks,
smashing it apart. Satisfying destruction physics with realistic tumbling debris.
Supports parameterization, collision event collection, and music-driven timing.
"""

import random
import math
import pymunk
import pygame
from .. import config
from ..palettes import get_palette

DEFAULT_PARAMS = {
    "tower_cols": 6,        # range 4-8
    "tower_rows": 12,       # range 8-18
    "block_size": 40,       # range 30-50
    "ball_radius": 60,      # range 40-80
    "ball_mass": 25,        # range 15-40
    "string_length": 550,   # range 400-700
    "num_towers": 1,        # range 1-2
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
    """Set up the pymunk space with tower(s), wrecking ball, floor, and walls."""
    params = params or {}
    space = pymunk.Space()
    space.gravity = (0, 900)

    w, h = config.WIDTH, config.HEIGHT
    tower_cols = _p(params, "tower_cols")
    tower_rows = _p(params, "tower_rows")
    block_size = _p(params, "block_size")
    ball_radius = _p(params, "ball_radius")
    ball_mass = _p(params, "ball_mass")
    string_length = _p(params, "string_length")
    num_towers = _p(params, "num_towers")

    block_w = block_size
    block_h = block_size // 2

    floor_y = 1700

    # --- Floor ---
    floor_seg = pymunk.Segment(space.static_body, (0, floor_y), (w, floor_y), 8)
    floor_seg.elasticity = 0.2
    floor_seg.friction = 0.9
    floor_seg.collision_type = 0
    space.add(floor_seg)

    # --- Side walls (keep debris on screen) ---
    left_wall = pymunk.Segment(space.static_body, (0, 0), (0, h), 8)
    left_wall.elasticity = 0.3
    left_wall.friction = 0.5
    space.add(left_wall)

    right_wall = pymunk.Segment(space.static_body, (w, 0), (w, h), 8)
    right_wall.elasticity = 0.3
    right_wall.friction = 0.5
    space.add(right_wall)

    # --- Build tower(s) ---
    blocks = []

    if num_towers == 1:
        # Single tower, slightly right of center (ball swings from left)
        tower_centers = [w * 0.58]
    else:
        # Two towers side by side
        tower_centers = [w * 0.48, w * 0.72]

    for tower_cx in tower_centers:
        tower_base_x = tower_cx - (tower_cols * block_w) / 2.0

        for row in range(tower_rows):
            # Brick-laying pattern: alternate rows offset by half a block width
            offset = (block_w / 2.0) if (row % 2 == 1) else 0.0
            num_in_row = tower_cols if (row % 2 == 0) else tower_cols - 1

            for col in range(num_in_row):
                bx = tower_base_x + offset + col * block_w + block_w / 2.0
                by = floor_y - block_h / 2.0 - row * block_h

                mass = 1.0
                moment = pymunk.moment_for_box(mass, (block_w - 1, block_h - 1))
                body = pymunk.Body(mass, moment)
                body.position = (bx, by)

                shape = pymunk.Poly.create_box(body, (block_w - 1, block_h - 1))
                shape.elasticity = 0.2
                shape.friction = 0.7
                shape.collision_type = 2
                space.add(body, shape)

                # Store color index for rendering
                color_idx = (row + col) % 10
                blocks.append((body, shape, color_idx))

    # --- Wrecking ball ---
    # Pivot point: top-left area of screen
    pivot_x = 100
    pivot_y = 150

    # Ball starts pulled to the far left at a high angle
    # Calculate start position: ball is at the end of the string, pulled left
    start_angle = -math.pi * 0.42  # pulled far left (negative = left of vertical)
    ball_start_x = pivot_x + string_length * math.sin(start_angle)
    ball_start_y = pivot_y + string_length * math.cos(start_angle)

    ball_moment = pymunk.moment_for_circle(ball_mass, 0, ball_radius)
    ball_body = pymunk.Body(ball_mass, ball_moment)
    ball_body.position = (ball_start_x, ball_start_y)
    # Start at rest (will be released by gravity)
    ball_body.velocity = (0, 0)

    ball_shape = pymunk.Circle(ball_body, ball_radius)
    ball_shape.elasticity = 0.15
    ball_shape.friction = 0.5
    ball_shape.collision_type = 1
    space.add(ball_body, ball_shape)

    # Pivot joint (string connection at top)
    pivot_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    pivot_body.position = (pivot_x, pivot_y)
    space.add(pivot_body)

    # Use PinJoint (distance constraint) to simulate the string
    joint = pymunk.PinJoint(pivot_body, ball_body, (0, 0), (0, 0))
    space.add(joint)

    return space, blocks, ball_body, (pivot_x, pivot_y), floor_y


def run(seed=None, params=None, note_schedule=None, collect_events_only=False):
    """Generator that yields pygame surfaces for each frame.

    Args:
        seed: Random seed for reproducibility
        params: Dict of scene parameters (overrides DEFAULT_PARAMS)
        note_schedule: Unused for this scene (kept for interface compatibility)
        collect_events_only: If True, skip rendering (for event collection pass)
    """
    global _collision_events
    _collision_events = []

    if seed is not None:
        random.seed(seed)

    params = params or {}
    pal = get_palette(_p(params, "palette"))
    ball_colors = pal["balls"]

    if collect_events_only:
        surface = pygame.Surface((1, 1))
    else:
        surface = pygame.Surface((config.WIDTH, config.HEIGHT))

    space, blocks, ball_body, pivot_pos, floor_y = create_scene(params)

    w, h = config.WIDTH, config.HEIGHT
    ball_radius = _p(params, "ball_radius")

    # --- Collision handlers ---
    frame_ref = [0]

    def _on_ball_block(arbiter, space, data):
        """Ball (type 1) hits block (type 2)."""
        impulse = arbiter.total_impulse
        force = impulse.length
        if force < 30:
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
            "type": "ball_block",
        })
        return True

    def _on_block_block(arbiter, space, data):
        """Block (type 2) hits block (type 2) -- cascading destruction."""
        impulse = arbiter.total_impulse
        force = impulse.length
        if force < 60:
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
            "type": "block_block",
        })
        return True

    space.on_collision(1, 2, post_solve=_on_ball_block)
    space.on_collision(2, 2, post_solve=_on_block_block)

    # --- Timing ---
    # Anticipation: hold the ball in place for ~2.5 seconds (countdown)
    anticipation_frames = int(2.5 * config.VIDEO_FPS)  # 75 frames at 30fps
    # Total max duration ~30 seconds
    max_frames = 30 * config.VIDEO_FPS
    # Settle detection
    settle_frames = 0
    settle_threshold = int(2.0 * config.VIDEO_FPS)  # 2 seconds of low velocity
    # Track whether ball has been released
    ball_released = False
    # Impact happened flag (for settle detection -- only start checking after impact)
    impact_happened = False

    # Font for countdown
    countdown_font = None
    if not collect_events_only:
        try:
            countdown_font = pygame.font.SysFont("Arial", 120, bold=True)
        except Exception:
            countdown_font = pygame.font.Font(None, 120)

    frame_idx = 0

    while frame_idx < max_frames:
        frame_ref[0] = frame_idx
        current_time = frame_idx / config.VIDEO_FPS

        # --- Anticipation phase: hold ball still ---
        if frame_idx < anticipation_frames:
            # Override velocity to keep ball stationary during countdown
            ball_body.velocity = (0, 0)
            ball_body.angular_velocity = 0
        elif not ball_released:
            ball_released = True
            # Ball is released -- gravity takes over naturally

        # --- Step physics ---
        for _ in range(config.SIM_STEPS_PER_FRAME):
            space.step(1.0 / config.SIM_FPS)

        # --- Check for impact (ball has hit something) ---
        if ball_released and not impact_happened:
            # Check if the ball has started decelerating (hit the tower)
            if len(_collision_events) > 0 and any(
                e.get("type") == "ball_block" for e in _collision_events
            ):
                impact_happened = True

        # --- Settle detection (after impact) ---
        if impact_happened and ball_released:
            max_vel = 0.0
            for body, shape, cidx in blocks:
                v = body.velocity.length
                if v > max_vel:
                    max_vel = v
            # Also check ball
            ball_vel = ball_body.velocity.length
            if ball_vel > max_vel:
                max_vel = ball_vel

            if max_vel < 8.0:
                settle_frames += 1
            else:
                settle_frames = 0

            if settle_frames >= settle_threshold:
                break

        # --- Rendering ---
        if not collect_events_only:
            surface.fill(pal["bg"])

            # Floor
            pygame.draw.line(
                surface,
                pal.get("wall", (100, 100, 100)),
                (0, floor_y),
                (w, floor_y),
                8,
            )

            # Side walls (subtle)
            pygame.draw.line(
                surface,
                pal.get("wall", (100, 100, 100)),
                (0, 0), (0, h), 4,
            )
            pygame.draw.line(
                surface,
                pal.get("wall", (100, 100, 100)),
                (w - 1, 0), (w - 1, h), 4,
            )

            # Blocks (colored bricks, rotate and tumble)
            for body, shape, color_idx in blocks:
                color = ball_colors[color_idx % len(ball_colors)]
                # Get the polygon vertices in world coordinates
                verts = [body.local_to_world(v) for v in shape.get_vertices()]
                points = [(int(v.x), int(v.y)) for v in verts]

                # Only draw if at least partially on screen
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                if max(xs) < -50 or min(xs) > w + 50:
                    continue
                if min(ys) > h + 50:
                    continue

                # Filled block
                if len(points) >= 3:
                    pygame.draw.polygon(surface, color, points)
                    # Slightly darker outline for definition
                    darker = (
                        max(0, color[0] - 40),
                        max(0, color[1] - 40),
                        max(0, color[2] - 40),
                    )
                    pygame.draw.polygon(surface, darker, points, 2)

            # String (thin line from pivot to ball center)
            bx, by = int(ball_body.position.x), int(ball_body.position.y)
            px, py = int(pivot_pos[0]), int(pivot_pos[1])
            pygame.draw.line(
                surface,
                pal.get("wall", (100, 100, 100)),
                (px, py),
                (bx, by),
                3,
            )

            # Pivot point (small circle)
            pygame.draw.circle(
                surface,
                pal.get("obstacle", (80, 80, 80)),
                (px, py),
                10,
            )

            # Wrecking ball -- dark circle with metallic gradient
            ball_color_base = (50, 50, 55)
            ball_color_highlight = (140, 140, 150)
            ball_color_shadow = (30, 30, 35)

            # Main ball
            pygame.draw.circle(surface, ball_color_base, (bx, by), ball_radius)

            # Shadow (lower-right crescent)
            shadow_offset_x = ball_radius // 5
            shadow_offset_y = ball_radius // 5
            pygame.draw.circle(
                surface,
                ball_color_shadow,
                (bx + shadow_offset_x, by + shadow_offset_y),
                ball_radius - 2,
            )
            # Re-draw main to layer correctly
            pygame.draw.circle(surface, ball_color_base, (bx, by), ball_radius - 3)

            # Highlight (upper-left for metallic sheen)
            hl_x = bx - ball_radius // 3
            hl_y = by - ball_radius // 3
            hl_r = ball_radius // 3
            pygame.draw.circle(surface, ball_color_highlight, (hl_x, hl_y), hl_r)

            # Small bright specular
            spec_x = bx - ball_radius // 4
            spec_y = by - ball_radius // 4
            spec_r = max(3, ball_radius // 6)
            pygame.draw.circle(surface, (200, 200, 210), (spec_x, spec_y), spec_r)

            # Ball outline
            pygame.draw.circle(
                surface,
                (35, 35, 40),
                (bx, by),
                ball_radius,
                3,
            )

            # --- Countdown text during anticipation ---
            if frame_idx < anticipation_frames and countdown_font:
                remaining = anticipation_frames - frame_idx
                # 3-2-1 countdown
                countdown_sec = math.ceil(remaining / config.VIDEO_FPS)
                countdown_sec = min(countdown_sec, 3)

                if countdown_sec >= 1:
                    text_str = str(countdown_sec)
                    text_color = pal.get("text", (40, 40, 40))
                    text_surf = countdown_font.render(text_str, True, text_color)
                    text_rect = text_surf.get_rect(center=(w // 2, h // 3))
                    surface.blit(text_surf, text_rect)

        yield surface
        frame_idx += 1

    # --- Hold final frame for 2 seconds ---
    for _ in range(2 * config.VIDEO_FPS):
        yield surface


def get_metadata(video_num, params=None):
    params = params or {}
    num_towers = _p(params, "num_towers")
    tower_desc = "a tower" if num_towers == 1 else "two towers"
    return {
        "title": f"Wrecking Ball Destruction #{video_num}",
        "description": (
            f"A massive wrecking ball smashes into {tower_desc} of blocks! "
            "Watch the satisfying destruction. #Shorts #wreckingball #physics #satisfying"
        ),
        "tags": [
            "wrecking ball",
            "physics",
            "satisfying",
            "destruction",
            "blocks",
            "tower",
            "shorts",
        ],
        "category": "24",
    }
