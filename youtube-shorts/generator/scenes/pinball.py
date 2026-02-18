"""Pinball Machine scene.

A ball bounces around bumpers, hits flippers, and scores points in a
retro arcade-style pinball setup. Dark neon palette looks best. Supports
parameterization, collision event collection, and music-driven launches.
"""

import random
import math
import pymunk
import pygame
from .. import config
from ..palettes import get_palette

# Default parameters (overridable via params dict)
DEFAULT_PARAMS = {
    "num_bumpers": 6,          # 4-10
    "num_flippers": 2,
    "ball_radius": 15,         # 12-18
    "gravity": 700,            # 500-900
    "bumper_elasticity": 1.5,  # 1.2-1.8 (>1.0 = bumpers add energy)
    "launch_force": 1100,      # 800-1500
    "palette": "neon",
    "num_lives": 4,            # 3-5 balls before game over
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


def _brighten(color, amount=80):
    """Brighten a color by a fixed amount (for flash effects)."""
    return (
        min(255, color[0] + amount),
        min(255, color[1] + amount),
        min(255, color[2] + amount),
    )


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Playfield geometry constants
# ---------------------------------------------------------------------------
_SIDE_MARGIN = 60
_TOP_MARGIN = 160
_BOTTOM_Y = 1780          # bottom of playfield
_FLIPPER_Y = 1700         # flipper pivot Y
_FLIPPER_LEN = 140
_FLIPPER_GAP = 120        # half-gap between flipper pivots
_CHUTE_WIDTH = 60         # launch chute width
_CHUTE_X = config.WIDTH - _SIDE_MARGIN  # right inner wall X
_BUMPER_RADIUS = 40


def _place_bumpers(rng, num_bumpers, w, h):
    """Generate bumper positions with minimum spacing."""
    min_spacing = _BUMPER_RADIUS * 3.5
    region_left = _SIDE_MARGIN + 80
    region_right = _CHUTE_X - _CHUTE_WIDTH - 80
    region_top = _TOP_MARGIN + 200
    region_bottom = _FLIPPER_Y - 300

    positions = []
    attempts = 0
    while len(positions) < num_bumpers and attempts < 2000:
        x = rng.uniform(region_left, region_right)
        y = rng.uniform(region_top, region_bottom)
        # Check spacing against all existing bumpers
        too_close = False
        for px, py in positions:
            if math.hypot(x - px, y - py) < min_spacing:
                too_close = True
                break
        if not too_close:
            positions.append((x, y))
        attempts += 1
    return positions


def create_scene(params=None, rng=None):
    """Set up the pymunk space with walls, bumpers, flippers, and chute.

    Returns:
        space: pymunk.Space
        bumpers: list of dicts with 'body', 'shape', 'pos', 'color', 'flash_until', 'pitch'
        flippers: list of dicts with 'body', 'shape', 'pivot', 'rest_angle', 'up_angle', 'side'
        walls: list of endpoint pairs for drawing
    """
    params = params or {}
    if rng is None:
        rng = random.Random()

    space = pymunk.Space()
    # Slight tilt: mostly downward gravity with a small X component (like a real table)
    grav = _p(params, "gravity")
    space.gravity = (grav * 0.03, grav)

    w, h = config.WIDTH, config.HEIGHT
    num_bumpers = _p(params, "num_bumpers")
    bumper_elasticity = _p(params, "bumper_elasticity")

    # ---- Walls ----
    wall_thickness = 8
    wall_friction = 0.4
    wall_elasticity = 0.3

    wall_endpoints = []

    def _add_wall(a, b, elast=wall_elasticity):
        seg = pymunk.Segment(space.static_body, a, b, wall_thickness)
        seg.elasticity = elast
        seg.friction = wall_friction
        seg.collision_type = 4  # wall
        space.add(seg)
        wall_endpoints.append((a, b))

    left_x = _SIDE_MARGIN
    right_x = _CHUTE_X
    chute_inner_x = right_x - _CHUTE_WIDTH

    # Top wall
    _add_wall((left_x, _TOP_MARGIN), (right_x, _TOP_MARGIN))
    # Left wall
    _add_wall((left_x, _TOP_MARGIN), (left_x, _BOTTOM_Y))
    # Right wall (stops at chute opening near bottom)
    _add_wall((right_x, _TOP_MARGIN), (right_x, _BOTTOM_Y))

    # Chute inner wall (from top down to just above flipper area)
    chute_opening_y = _FLIPPER_Y - 50
    _add_wall((chute_inner_x, _TOP_MARGIN), (chute_inner_x, chute_opening_y))

    # Chute curved entry at top — small diagonal to direct ball into playfield
    _add_wall((chute_inner_x, _TOP_MARGIN), (chute_inner_x - 60, _TOP_MARGIN - 40), elast=0.6)

    # Bottom — two angled sections forming drain between flippers
    center_x = w / 2 - 20  # slightly left since chute eats right side
    # Left drain wall — angled down toward center
    _add_wall((left_x, _BOTTOM_Y), (center_x - _FLIPPER_GAP - 20, _BOTTOM_Y))
    # Right drain wall
    _add_wall((center_x + _FLIPPER_GAP + 20, _BOTTOM_Y), (chute_inner_x, _BOTTOM_Y))

    # Angled guide walls above flippers
    guide_y = _FLIPPER_Y + 30
    _add_wall((left_x, guide_y + 60), (center_x - _FLIPPER_GAP - 40, guide_y), elast=0.4)
    _add_wall((chute_inner_x, guide_y + 60), (center_x + _FLIPPER_GAP + 40, guide_y), elast=0.4)

    # ---- Bumpers (static circles with high elasticity) ----
    pal = get_palette(_p(params, "palette"))
    bumper_colors = pal["balls"]
    bumper_positions = _place_bumpers(rng, num_bumpers, w, h)
    bumpers = []

    for i, (bx, by) in enumerate(bumper_positions):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (bx, by)
        shape = pymunk.Circle(body, _BUMPER_RADIUS)
        shape.elasticity = bumper_elasticity
        shape.friction = 0.0
        shape.collision_type = 2  # bumper
        space.add(body, shape)

        # Map pitch to Y position (higher on screen = higher pitch)
        region_top = _TOP_MARGIN + 200
        region_bottom = _FLIPPER_Y - 300
        y_frac = 1.0 - (by - region_top) / max(1, region_bottom - region_top)
        pitch = int(48 + y_frac * 36)  # MIDI 48-84

        bumpers.append({
            "body": body,
            "shape": shape,
            "pos": (bx, by),
            "color": bumper_colors[i % len(bumper_colors)],
            "flash_until": 0,
            "pitch": pitch,
            "index": i,
        })

    # ---- Flippers (kinematic segments) ----
    center_x_flip = center_x
    flippers = []

    for side in ("left", "right"):
        if side == "left":
            pivot_x = center_x_flip - _FLIPPER_GAP
            rest_angle = math.radians(30)   # angled downward
            up_angle = math.radians(-30)     # swung upward
        else:
            pivot_x = center_x_flip + _FLIPPER_GAP
            rest_angle = math.radians(180 - 30)
            up_angle = math.radians(180 + 30)

        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = (pivot_x, _FLIPPER_Y)
        body.angle = rest_angle

        # Flipper shape: segment from origin outward
        shape = pymunk.Segment(body, (0, 0), (_FLIPPER_LEN, 0), 10)
        shape.elasticity = 0.5
        shape.friction = 0.8
        shape.collision_type = 3  # flipper
        space.add(body, shape)

        flippers.append({
            "body": body,
            "shape": shape,
            "pivot": (pivot_x, _FLIPPER_Y),
            "rest_angle": rest_angle,
            "up_angle": up_angle,
            "side": side,
            "active_until": 0,
        })

    return space, bumpers, flippers, wall_endpoints


def _launch_ball(space, params, rng):
    """Create and launch a ball from the chute."""
    ball_radius = _p(params, "ball_radius")
    mass = 1
    moment = pymunk.moment_for_circle(mass, 0, ball_radius)
    body = pymunk.Body(mass, moment)
    # Start in the chute, near the bottom
    chute_cx = _CHUTE_X - _CHUTE_WIDTH / 2
    body.position = (chute_cx, _FLIPPER_Y - 20)
    shape = pymunk.Circle(body, ball_radius)
    shape.elasticity = 0.6
    shape.friction = 0.3
    shape.collision_type = 1  # ball
    space.add(body, shape)

    # Launch upward with some force
    launch_force = _p(params, "launch_force")
    body.apply_impulse_at_local_point((0, -launch_force))

    return body, shape


def run(seed=None, params=None, note_schedule=None, collect_events_only=False):
    """Generator that yields pygame surfaces for each frame.

    Args:
        seed: Random seed for reproducibility
        params: Dict of scene parameters (overrides DEFAULT_PARAMS)
        note_schedule: If provided, launch balls at scheduled times.
                      List of {time_sec, pitch, ...}
        collect_events_only: If True, skip rendering (for event collection pass)
    """
    global _collision_events
    _collision_events = []

    rng = random.Random(seed)
    if seed is not None:
        random.seed(seed)

    params = params or {}
    pal = get_palette(_p(params, "palette"))

    if collect_events_only:
        surface = pygame.Surface((1, 1))
    else:
        surface = pygame.Surface((config.WIDTH, config.HEIGHT))

    space, bumpers, flippers, wall_endpoints = create_scene(params, rng)

    w, h = config.WIDTH, config.HEIGHT
    ball_radius = _p(params, "ball_radius")
    num_lives = _p(params, "num_lives")

    ball_colors = pal["balls"]
    bg_color = pal["bg"]
    wall_color = pal.get("wall", (40, 40, 60))
    text_color = pal.get("text", (220, 220, 255))

    # Build bumper shape lookup for collision handler
    bumper_shape_map = {}
    for bumper in bumpers:
        bumper_shape_map[id(bumper["shape"])] = bumper

    # Collision handler: ball (1) vs bumper (2)
    frame_ref = [0]
    score_ref = [0]
    flash_duration = 8  # frames

    def _on_ball_bumper(arbiter, space, data):
        impulse = arbiter.total_impulse
        force = impulse.length
        if force < 30:
            return True
        pts = arbiter.contact_point_set.points
        if pts:
            pos = pts[0].point_a
        else:
            pos = arbiter.shapes[0].body.position

        # Identify which bumper was hit
        hit_bumper = None
        for shape in arbiter.shapes:
            sid = id(shape)
            if sid in bumper_shape_map:
                hit_bumper = bumper_shape_map[sid]
                break

        event = {
            "frame": frame_ref[0],
            "time_sec": frame_ref[0] / config.VIDEO_FPS,
            "x": float(pos.x),
            "y": float(pos.y),
            "force": float(force),
        }
        if hit_bumper is not None:
            event["extra_pitch"] = hit_bumper["pitch"]
            hit_bumper["flash_until"] = frame_ref[0] + flash_duration
            # Score: 100 per bumper hit
            score_ref[0] += 100

        _collision_events.append(event)
        return True

    space.on_collision(1, 2, post_solve=_on_ball_bumper)

    # Also record ball-flipper collisions (1 vs 3)
    def _on_ball_flipper(arbiter, space, data):
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
            "extra_pitch": 36,  # low thump for flipper
        })
        score_ref[0] += 10
        return True

    space.on_collision(1, 3, post_solve=_on_ball_flipper)

    # ---- Game state ----
    lives_used = 0
    ball_body = None
    ball_shape = None
    ball_color = ball_colors[0]
    ball_active = False
    relaunch_cooldown = 0  # frames to wait before relaunch
    ball_trail = []  # list of (x, y) positions for trail effect

    # Music-driven schedule
    schedule_idx = 0
    if note_schedule:
        note_schedule = sorted(note_schedule, key=lambda n: n["time_sec"])
        num_lives = len(note_schedule)

    # Flipper auto-activation state
    flipper_cooldowns = [0, 0]  # frames until next possible activation
    flipper_active_duration = 10  # frames flipper stays up

    frame_idx = 0
    game_over = False
    game_over_frame = -1

    # Duration: ~32s active play + 2s hold = ~34s
    max_frames = 35 * config.VIDEO_FPS

    # Font for score (initialized lazily)
    pygame.font.init()
    try:
        score_font = pygame.font.SysFont("monospace", 64, bold=True)
        lives_font = pygame.font.SysFont("monospace", 40, bold=True)
        game_over_font = pygame.font.SysFont("monospace", 80, bold=True)
    except Exception:
        score_font = pygame.font.Font(None, 64)
        lives_font = pygame.font.Font(None, 40)
        game_over_font = pygame.font.Font(None, 80)

    def _launch_new_ball():
        nonlocal ball_body, ball_shape, ball_color, ball_active, lives_used, ball_trail
        ball_body, ball_shape = _launch_ball(space, params, rng)
        ball_color = ball_colors[lives_used % len(ball_colors)]
        ball_active = True
        lives_used += 1
        ball_trail = []

    def _remove_ball():
        nonlocal ball_body, ball_shape, ball_active
        if ball_body is not None and ball_shape is not None:
            try:
                space.remove(ball_shape, ball_body)
            except Exception:
                pass
        ball_body = None
        ball_shape = None
        ball_active = False

    def _update_flippers(frame):
        """Activate flippers based on ball proximity or periodic timer."""
        nonlocal flipper_cooldowns
        for i, flipper in enumerate(flippers):
            # Check if ball is near flipper area
            ball_nearby = False
            if ball_active and ball_body is not None:
                bx, by = ball_body.position
                if by > _FLIPPER_Y - 200 and abs(bx - flipper["pivot"][0]) < 300:
                    ball_nearby = True

            # Periodic activation or proximity-based
            if flipper_cooldowns[i] <= 0:
                if ball_nearby:
                    flipper["active_until"] = frame + flipper_active_duration
                    flipper_cooldowns[i] = rng.randint(20, 40)
                elif frame % rng.randint(60, 90) < 3:
                    flipper["active_until"] = frame + flipper_active_duration
                    flipper_cooldowns[i] = rng.randint(40, 70)
            else:
                flipper_cooldowns[i] -= 1

            # Animate flipper angle
            if frame < flipper["active_until"]:
                # Swing up
                target = flipper["up_angle"]
            else:
                # Rest position
                target = flipper["rest_angle"]

            # Smooth rotation toward target
            current = flipper["body"].angle
            diff = target - current
            # Normalize angle difference
            while diff > math.pi:
                diff -= 2 * math.pi
            while diff < -math.pi:
                diff += 2 * math.pi

            speed = 0.35  # radians per frame
            if abs(diff) < speed:
                flipper["body"].angle = target
                flipper["body"].angular_velocity = 0
            else:
                flipper["body"].angular_velocity = math.copysign(speed * config.SIM_FPS, diff)

    # ---- Main loop ----
    while frame_idx < max_frames:
        frame_ref[0] = frame_idx
        current_time = frame_idx / config.VIDEO_FPS

        # ---- Ball launch logic ----
        if not ball_active and not game_over:
            if relaunch_cooldown > 0:
                relaunch_cooldown -= 1
            else:
                if note_schedule:
                    # Music-driven launches
                    if schedule_idx < len(note_schedule):
                        note = note_schedule[schedule_idx]
                        if note["time_sec"] <= current_time:
                            _launch_new_ball()
                            schedule_idx += 1
                elif lives_used < num_lives:
                    _launch_new_ball()
                else:
                    if not game_over:
                        game_over = True
                        game_over_frame = frame_idx

        # ---- Check if ball drained ----
        if ball_active and ball_body is not None:
            bx, by = ball_body.position
            if by > _BOTTOM_Y + 100 or by > h + 50:
                _remove_ball()
                if (note_schedule and schedule_idx >= len(note_schedule)) or \
                   (not note_schedule and lives_used >= num_lives):
                    game_over = True
                    game_over_frame = frame_idx
                else:
                    relaunch_cooldown = int(1.0 * config.VIDEO_FPS)  # 1 second pause

            # Also check if ball escaped sideways (shouldn't happen but safety)
            elif bx < -50 or bx > w + 50:
                _remove_ball()
                relaunch_cooldown = int(0.5 * config.VIDEO_FPS)

        # ---- Update flippers ----
        _update_flippers(frame_idx)

        # ---- Step physics ----
        for _ in range(config.SIM_STEPS_PER_FRAME):
            space.step(1.0 / config.SIM_FPS)

        # ---- Update ball trail ----
        if ball_active and ball_body is not None:
            bx, by = ball_body.position
            ball_trail.append((int(bx), int(by)))
            # Keep last 12 positions
            if len(ball_trail) > 12:
                ball_trail = ball_trail[-12:]

        # ---- End condition: game over and hold ----
        if game_over and frame_idx - game_over_frame > 2 * config.VIDEO_FPS:
            break

        # ---- Rendering ----
        if not collect_events_only:
            surface.fill(bg_color)

            # Draw playfield border glow (subtle)
            border_glow = pygame.Surface((w, h), pygame.SRCALPHA)

            # Draw walls
            for (ax, ay), (bx, by) in wall_endpoints:
                pygame.draw.line(surface, wall_color,
                                 (int(ax), int(ay)), (int(bx), int(by)), 6)
                # Brighter inner line for neon effect
                bright_wall = _brighten(wall_color, 30)
                pygame.draw.line(surface, bright_wall,
                                 (int(ax), int(ay)), (int(bx), int(by)), 2)

            # Draw launch chute (thin rectangle on right)
            chute_color = _brighten(wall_color, 15)
            chute_rect = pygame.Rect(
                int(_CHUTE_X - _CHUTE_WIDTH), int(_TOP_MARGIN),
                int(_CHUTE_WIDTH), int(_FLIPPER_Y - _TOP_MARGIN - 50)
            )
            pygame.draw.rect(surface, chute_color, chute_rect, 2)

            # Draw bumpers
            for bumper in bumpers:
                bx, by = bumper["pos"]
                color = bumper["color"]
                ibx, iby = int(bx), int(by)

                # Flash effect
                if frame_idx < bumper["flash_until"]:
                    color = _brighten(color, 100)

                # Outer glow ring (larger, semi-transparent)
                glow_surf = pygame.Surface(
                    (_BUMPER_RADIUS * 4 + 10, _BUMPER_RADIUS * 4 + 10),
                    pygame.SRCALPHA
                )
                glow_center = (_BUMPER_RADIUS * 2 + 5, _BUMPER_RADIUS * 2 + 5)
                glow_color = (color[0], color[1], color[2], 60)
                pygame.draw.circle(glow_surf, glow_color,
                                   glow_center, _BUMPER_RADIUS + 15)
                surface.blit(glow_surf,
                             (ibx - _BUMPER_RADIUS * 2 - 5,
                              iby - _BUMPER_RADIUS * 2 - 5))

                # Main bumper circle
                pygame.draw.circle(surface, color, (ibx, iby), _BUMPER_RADIUS)

                # Inner highlight ring
                highlight = _brighten(color, 50)
                pygame.draw.circle(surface, highlight, (ibx, iby),
                                   _BUMPER_RADIUS - 8, 3)

                # Center dot
                pygame.draw.circle(surface, (255, 255, 255), (ibx, iby), 6)

            # Draw flippers
            flipper_color = pal["balls"][2] if len(pal["balls"]) > 2 else (255, 255, 0)
            for flipper in flippers:
                body = flipper["body"]
                px, py = body.position
                angle = body.angle
                # Endpoint of flipper
                ex = px + _FLIPPER_LEN * math.cos(angle)
                ey = py + _FLIPPER_LEN * math.sin(angle)

                # Draw thick flipper line
                pygame.draw.line(surface, flipper_color,
                                 (int(px), int(py)), (int(ex), int(ey)), 16)
                # Bright center line
                bright_flip = _brighten(flipper_color, 40)
                pygame.draw.line(surface, bright_flip,
                                 (int(px), int(py)), (int(ex), int(ey)), 6)
                # Pivot circle
                pygame.draw.circle(surface, bright_flip, (int(px), int(py)), 12)
                # Tip circle
                pygame.draw.circle(surface, flipper_color, (int(ex), int(ey)), 10)

            # Draw ball trail
            if ball_active and len(ball_trail) > 1:
                trail_len = len(ball_trail)
                for ti in range(trail_len - 1):
                    alpha = int(40 + 80 * (ti / trail_len))
                    radius = max(3, int(ball_radius * (0.3 + 0.5 * ti / trail_len)))
                    tx, ty = ball_trail[ti]
                    if 0 <= tx <= w and 0 <= ty <= h:
                        trail_surf = pygame.Surface(
                            (radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA
                        )
                        trail_color = (ball_color[0], ball_color[1], ball_color[2], alpha)
                        pygame.draw.circle(trail_surf, trail_color,
                                           (radius + 1, radius + 1), radius)
                        surface.blit(trail_surf, (tx - radius - 1, ty - radius - 1))

            # Draw ball
            if ball_active and ball_body is not None:
                bx, by = int(ball_body.position.x), int(ball_body.position.y)
                if 0 <= bx <= w and 0 <= by <= h:
                    # Outer glow
                    glow_r = ball_radius + 8
                    ball_glow = pygame.Surface(
                        (glow_r * 2 + 2, glow_r * 2 + 2), pygame.SRCALPHA
                    )
                    pygame.draw.circle(ball_glow,
                                       (255, 255, 255, 40),
                                       (glow_r + 1, glow_r + 1), glow_r)
                    surface.blit(ball_glow, (bx - glow_r - 1, by - glow_r - 1))

                    # Main ball
                    pygame.draw.circle(surface, ball_color, (bx, by), ball_radius)
                    # Specular highlight
                    hx = bx - ball_radius // 3
                    hy = by - ball_radius // 3
                    pygame.draw.circle(surface, (255, 255, 255),
                                       (hx, hy), ball_radius // 3)

            # Draw score at top
            score_text = score_font.render(f"{score_ref[0]:,}", True, text_color)
            score_rect = score_text.get_rect(centerx=w // 2 - 30, top=40)
            surface.blit(score_text, score_rect)

            # "SCORE" label above the number
            label_text = lives_font.render("SCORE", True, text_color)
            label_rect = label_text.get_rect(centerx=w // 2 - 30, top=10)
            surface.blit(label_text, label_rect)

            # Draw lives remaining (small circles at top-right)
            lives_remaining = num_lives - lives_used
            for li in range(lives_remaining):
                lx = w - 80 - li * 35
                ly = 50
                pygame.draw.circle(surface, ball_colors[0], (lx, ly), 12)
                pygame.draw.circle(surface, (255, 255, 255), (lx - 3, ly - 3), 4)

            # Ball number indicator
            ball_label = lives_font.render(
                f"BALL {min(lives_used, num_lives)}/{num_lives}", True, text_color
            )
            ball_label_rect = ball_label.get_rect(left=40, top=50)
            surface.blit(ball_label, ball_label_rect)

            # Game over text
            if game_over:
                go_text = game_over_font.render("GAME OVER", True, (255, 50, 50))
                go_rect = go_text.get_rect(center=(w // 2 - 20, h // 2 - 60))
                surface.blit(go_text, go_rect)

                final_score_text = score_font.render(
                    f"FINAL SCORE: {score_ref[0]:,}", True, text_color
                )
                fs_rect = final_score_text.get_rect(center=(w // 2 - 20, h // 2 + 30))
                surface.blit(final_score_text, fs_rect)

        yield surface
        frame_idx += 1

    # Hold final frame for 2 seconds
    for _ in range(2 * config.VIDEO_FPS):
        yield surface


def get_metadata(video_num, params=None):
    params = params or {}
    return {
        "title": f"Pinball Machine #{video_num}",
        "description": (
            "Watch the ball bounce off bumpers and rack up points! "
            "Retro arcade pinball physics. "
            "#Shorts #pinball #arcade #satisfying #physics"
        ),
        "tags": [
            "pinball", "arcade", "physics", "satisfying", "bumpers",
            "flippers", "retro", "neon", "shorts",
        ],
        "category": "24",
    }
