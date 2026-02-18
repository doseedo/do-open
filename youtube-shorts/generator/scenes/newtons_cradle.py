"""Newton's Cradle scene.

N metallic balls on strings in a line. 1-3 balls are pulled to one side
and released, transferring momentum through the line with satisfying clicks.
Supports parameterization, collision event collection, and music-driven timing.
"""

import random
import math
import pymunk
import pygame
from .. import config
from ..palettes import get_palette

DEFAULT_PARAMS = {
    "num_balls": 7,
    "ball_radius": 40,
    "string_length": 550,
    "pull_balls": 1,
    "damping": 0.999,
    "pull_angle": 0.8,
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


def run(seed=None, params=None, note_schedule=None, collect_events_only=False):
    """Generator that yields pygame surfaces for each frame.

    Args:
        seed: Random seed for reproducibility
        params: Dict of scene parameters (overrides DEFAULT_PARAMS)
        note_schedule: Optional list of {time_sec, pitch, ...} for
                       music-driven impulses (unused in cradle but kept
                       for interface consistency)
        collect_events_only: If True, skip rendering (for event collection pass)
    """
    global _collision_events
    _collision_events = []

    if seed is not None:
        random.seed(seed)

    params = params or {}
    pal = get_palette(_p(params, "palette"))

    w, h = config.WIDTH, config.HEIGHT

    if collect_events_only:
        surface = pygame.Surface((1, 1))
    else:
        surface = pygame.Surface((w, h))

    num_balls = _p(params, "num_balls")
    ball_radius = _p(params, "ball_radius")
    string_length = _p(params, "string_length")
    pull_balls = _p(params, "pull_balls")
    damping = _p(params, "damping")
    pull_angle = _p(params, "pull_angle")

    # Clamp pull_balls to valid range
    pull_balls = max(1, min(pull_balls, num_balls // 2))

    # --- Layout ---
    # Support bar at the top, balls hang below
    support_y = 380
    ball_spacing = ball_radius * 2.02  # just barely touching
    cradle_width = (num_balls - 1) * ball_spacing
    cradle_center_x = w / 2
    cradle_left_x = cradle_center_x - cradle_width / 2

    # Rest position of balls (hanging straight down)
    rest_y = support_y + string_length

    # --- Pymunk setup ---
    space = pymunk.Space()
    space.gravity = (0, 980)
    space.damping = 1.0  # no global damping; we apply our own per-ball

    # Static support body
    support_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    support_body.position = (0, 0)
    space.add(support_body)

    # Collision handler setup
    frame_ref = [0]

    def _on_post_solve(arbiter, space, data):
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
        })
        return True

    space.on_collision(1, 1, post_solve=_on_post_solve)

    # Create balls and string constraints
    balls = []
    ball_mass = 5.0
    for i in range(num_balls):
        anchor_x = cradle_left_x + i * ball_spacing
        anchor_y = support_y

        moment = pymunk.moment_for_circle(ball_mass, 0, ball_radius)
        body = pymunk.Body(ball_mass, moment)

        # Initial position: leftmost pull_balls are displaced
        if i < pull_balls:
            # Displace to the left at pull_angle from vertical
            offset_x = -math.sin(pull_angle) * string_length
            offset_y = math.cos(pull_angle) * string_length
            body.position = (anchor_x + offset_x, anchor_y + offset_y)
        else:
            body.position = (anchor_x, anchor_y + string_length)

        shape = pymunk.Circle(body, ball_radius)
        shape.elasticity = 0.999
        shape.friction = 0.0
        shape.collision_type = 1
        space.add(body, shape)

        # Pin joint acts as string: anchors ball to its support point
        joint = pymunk.PinJoint(
            support_body, body,
            anchor_a=(anchor_x, anchor_y),
            anchor_b=(0, 0),
        )
        joint.distance = string_length
        space.add(joint)

        balls.append({
            "body": body,
            "shape": shape,
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
        })

    # --- Visual helpers ---
    # Support bar dimensions
    bar_left = cradle_left_x - ball_radius - 40
    bar_right = cradle_left_x + cradle_width + ball_radius + 40
    bar_top = support_y - 18
    bar_height = 24

    # Base/stand dimensions
    base_top = rest_y + ball_radius + 60
    base_foot_y = base_top + 140
    base_left = cradle_center_x - cradle_width / 2 - 80
    base_right = cradle_center_x + cradle_width / 2 + 80

    # Metallic ball colors
    ball_base_color = (90, 90, 100)
    ball_highlight_color = (200, 200, 210)
    ball_dark_color = (50, 50, 58)
    string_color = (180, 180, 185)
    bar_color_top = (140, 140, 150)
    bar_color_bottom = (100, 100, 110)
    base_color = (120, 120, 130)

    # --- Simulation loop ---
    max_frames = 35 * config.VIDEO_FPS  # ~35 seconds max
    hold_frames = 2 * config.VIDEO_FPS
    frame_idx = 0
    stopped = False
    stop_threshold_vel = 8.0  # px/s
    low_vel_streak = 0
    low_vel_needed = int(1.5 * config.VIDEO_FPS)  # 1.5s of low velocity to stop

    # Second impulse tracking
    second_impulse_applied = False
    second_impulse_min_frame = 15 * config.VIDEO_FPS
    second_impulse_max_frame = 20 * config.VIDEO_FPS

    while frame_idx < max_frames and not stopped:
        frame_ref[0] = frame_idx

        # Apply velocity damping to each ball
        for ball in balls:
            body = ball["body"]
            vx, vy = body.velocity
            body.velocity = (vx * damping, vy * damping)

        # Step physics (multiple sub-steps for stability)
        dt = 1.0 / config.SIM_FPS
        for _ in range(config.SIM_STEPS_PER_FRAME):
            space.step(dt)

        # Check maximum velocity across all balls
        max_vel = max(b["body"].velocity.length for b in balls)

        # Second impulse: if energy is getting low and we haven't applied it yet,
        # pull rightmost balls and release (apply impulse)
        if (not second_impulse_applied
                and second_impulse_min_frame <= frame_idx <= second_impulse_max_frame
                and max_vel < 60):
            second_impulse_applied = True
            # Apply impulse to rightmost pull_balls balls
            impulse_strength = ball_mass * string_length * math.sin(pull_angle * 0.7) * 3.0
            for i in range(num_balls - pull_balls, num_balls):
                body = balls[i]["body"]
                body.apply_impulse_at_local_point((impulse_strength, 0), (0, 0))

        # Check if cradle has effectively stopped
        if max_vel < stop_threshold_vel:
            low_vel_streak += 1
        else:
            low_vel_streak = 0

        if low_vel_streak >= low_vel_needed:
            stopped = True

        # --- Rendering ---
        if not collect_events_only:
            bg_color = pal["bg"]
            surface.fill(bg_color)

            # Draw support bar with gradient effect
            for row in range(bar_height):
                t = row / max(1, bar_height - 1)
                r = int(bar_color_top[0] * (1 - t) + bar_color_bottom[0] * t)
                g = int(bar_color_top[1] * (1 - t) + bar_color_bottom[1] * t)
                b_c = int(bar_color_top[2] * (1 - t) + bar_color_bottom[2] * t)
                pygame.draw.line(
                    surface, (r, g, b_c),
                    (int(bar_left), int(bar_top + row)),
                    (int(bar_right), int(bar_top + row)),
                )

            # Bar end caps (small vertical rectangles for 3D look)
            cap_width = 12
            pygame.draw.rect(surface, bar_color_bottom,
                             (int(bar_left - cap_width / 2), int(bar_top - 4),
                              cap_width, bar_height + 8))
            pygame.draw.rect(surface, bar_color_bottom,
                             (int(bar_right - cap_width / 2), int(bar_top - 4),
                              cap_width, bar_height + 8))

            # Draw strings
            for ball in balls:
                body = ball["body"]
                bx, by = int(body.position.x), int(body.position.y)
                ax, ay = int(ball["anchor_x"]), int(ball["anchor_y"])
                pygame.draw.line(surface, string_color, (ax, ay), (bx, by), 2)

            # Draw balls with metallic shading
            for ball in balls:
                body = ball["body"]
                bx, by = int(body.position.x), int(body.position.y)

                # Shadow (slightly offset, darker)
                shadow_offset = 4
                pygame.draw.circle(
                    surface, (40, 40, 45),
                    (bx + shadow_offset, by + shadow_offset),
                    ball_radius,
                )

                # Base ball color
                pygame.draw.circle(surface, ball_base_color, (bx, by), ball_radius)

                # Dark edge ring
                pygame.draw.circle(surface, ball_dark_color, (bx, by), ball_radius, 2)

                # Gradient highlight: a lighter elliptical region offset to top-left
                highlight_offset_x = -ball_radius * 0.3
                highlight_offset_y = -ball_radius * 0.3
                highlight_radius = int(ball_radius * 0.55)
                hx = int(bx + highlight_offset_x)
                hy = int(by + highlight_offset_y)

                # Draw concentric circles for smooth highlight gradient
                for ri in range(highlight_radius, 0, -1):
                    t = 1.0 - (ri / highlight_radius)
                    cr = int(ball_base_color[0] + (ball_highlight_color[0] - ball_base_color[0]) * t)
                    cg = int(ball_base_color[1] + (ball_highlight_color[1] - ball_base_color[1]) * t)
                    cb = int(ball_base_color[2] + (ball_highlight_color[2] - ball_base_color[2]) * t)
                    pygame.draw.circle(surface, (cr, cg, cb), (hx, hy), ri)

                # Tiny specular dot
                spec_x = int(bx - ball_radius * 0.2)
                spec_y = int(by - ball_radius * 0.35)
                pygame.draw.circle(surface, (240, 240, 248), (spec_x, spec_y), max(3, ball_radius // 8))

            # Draw base/stand (A-frame supports)
            # Left support leg
            pygame.draw.line(
                surface, base_color,
                (int(bar_left + cap_width / 2), int(bar_top + bar_height)),
                (int(base_left + 20), int(base_foot_y)),
                6,
            )
            # Right support leg
            pygame.draw.line(
                surface, base_color,
                (int(bar_right - cap_width / 2), int(bar_top + bar_height)),
                (int(base_right - 20), int(base_foot_y)),
                6,
            )
            # Cross brace (horizontal bar at base)
            pygame.draw.line(
                surface, base_color,
                (int(base_left + 10), int(base_foot_y)),
                (int(base_right - 10), int(base_foot_y)),
                6,
            )
            # Small feet
            foot_w, foot_h = 30, 8
            pygame.draw.rect(
                surface, base_color,
                (int(base_left + 20 - foot_w / 2), int(base_foot_y),
                 foot_w, foot_h),
            )
            pygame.draw.rect(
                surface, base_color,
                (int(base_right - 20 - foot_w / 2), int(base_foot_y),
                 foot_w, foot_h),
            )

        yield surface
        frame_idx += 1

    # Hold final frame
    for _ in range(hold_frames):
        yield surface


def get_metadata(video_num, params=None):
    params = params or {}
    n = _p(params, "num_balls")
    pull = _p(params, "pull_balls")
    pull_desc = f"{pull} ball{'s' if pull > 1 else ''}"
    return {
        "title": f"Newton's Cradle #{video_num}",
        "description": (
            f"Mesmerizing Newton's Cradle with {n} balls! "
            f"Watch {pull_desc} transfer momentum perfectly. "
            "#Shorts #physics #newtonscradle #satisfying"
        ),
        "tags": [
            "newtons cradle", "physics", "satisfying", "momentum",
            "conservation of energy", "shorts",
        ],
        "category": "24",
    }
