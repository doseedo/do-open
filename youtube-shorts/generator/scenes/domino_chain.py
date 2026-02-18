"""Domino Chain Reaction scene.

Dominoes standing upright in various patterns (straight, curve, zigzag, spiral).
The first domino is tipped over, causing a chain reaction as each domino
knocks over the next. Supports parameterization, collision event collection,
and music-driven timing.
"""

import random
import math
import pymunk
import pygame
from .. import config
from ..palettes import get_palette

DEFAULT_PARAMS = {
    "num_dominoes": 50,       # range 30-80
    "domino_height": 55,      # range 40-70
    "domino_width": 10,       # range 8-15
    "spacing_factor": 2.0,    # range 1.5-3.0, multiplied by domino_width for spacing
    "gravity": 980,           # range 800-1200
    "pattern": "zigzag",      # choices: "straight", "curve", "zigzag", "spiral"
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


def _generate_path(pattern, num_dominoes, w, h):
    """Generate a list of (x, y, angle) positions for dominoes.

    Each domino stands upright, so 'angle' is the rotation of its upright axis
    relative to the path tangent (dominoes face perpendicular to the path).
    Returns list of (x, y, facing_angle) where facing_angle is the angle
    the domino's thin face points along (perpendicular to its tall axis).
    """
    margin_x = 120
    margin_top = 200
    margin_bottom = 250
    usable_w = w - 2 * margin_x
    usable_h = h - margin_top - margin_bottom

    positions = []

    if pattern == "straight":
        # Dominoes in a straight vertical line going top to bottom
        spacing = usable_h / (num_dominoes - 1) if num_dominoes > 1 else 0
        x_center = w / 2
        for i in range(num_dominoes):
            y = margin_top + i * spacing
            positions.append((x_center, y, 0.0))  # angle 0 = facing right

    elif pattern == "curve":
        # Dominoes along a sine wave path going downward
        spacing = usable_h / (num_dominoes - 1) if num_dominoes > 1 else 0
        amplitude = usable_w * 0.35
        frequency = 2.5  # number of half-waves
        for i in range(num_dominoes):
            t = i / max(1, num_dominoes - 1)
            y = margin_top + i * spacing
            x = w / 2 + amplitude * math.sin(t * frequency * math.pi)
            # Tangent direction for facing
            if i < num_dominoes - 1:
                t2 = (i + 1) / max(1, num_dominoes - 1)
                y2 = margin_top + (i + 1) * spacing
                x2 = w / 2 + amplitude * math.sin(t2 * frequency * math.pi)
                angle = math.atan2(y2 - y, x2 - x)
            else:
                # Use previous tangent
                t2 = (i - 1) / max(1, num_dominoes - 1)
                y2 = margin_top + (i - 1) * spacing
                x2 = w / 2 + amplitude * math.sin(t2 * frequency * math.pi)
                angle = math.atan2(y - y2, x - x2)
            positions.append((x, y, angle))

    elif pattern == "zigzag":
        # Dominoes go left-right in zigzag rows going down the screen
        # Determine how many rows and dominoes per row
        num_rows = max(3, int(math.sqrt(num_dominoes / 2)))
        per_row = num_dominoes // num_rows
        remainder = num_dominoes - per_row * num_rows

        row_height = usable_h / num_rows
        idx = 0
        for row in range(num_rows):
            row_count = per_row + (1 if row < remainder else 0)
            if row_count == 0:
                continue
            y_center = margin_top + row * row_height + row_height / 2

            left_x = margin_x + 40
            right_x = w - margin_x - 40

            going_right = (row % 2 == 0)

            for j in range(row_count):
                t = j / max(1, row_count - 1)
                if going_right:
                    x = left_x + t * (right_x - left_x)
                else:
                    x = right_x - t * (right_x - left_x)
                # Slight vertical offset along the row for realism
                y = y_center + (t - 0.5) * row_height * 0.3
                angle = 0.0 if going_right else math.pi
                positions.append((x, y, angle))
                idx += 1

            # Add a connecting domino between rows (turning corner)
            if row < num_rows - 1 and idx < num_dominoes:
                corner_x = right_x if going_right else left_x
                corner_y = y_center + row_height * 0.4
                # Facing downward
                positions.append((corner_x, corner_y, math.pi / 2))
                idx += 1

        # Trim to exact count
        positions = positions[:num_dominoes]

    elif pattern == "spiral":
        # Dominoes placed in a spiral from top-center going outward and down
        center_x = w / 2
        center_y = h / 2 - 50
        max_radius = min(usable_w, usable_h) * 0.42
        total_turns = 3.0

        for i in range(num_dominoes):
            t = i / max(1, num_dominoes - 1)
            angle = t * total_turns * 2 * math.pi
            radius = 60 + t * (max_radius - 60)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            # Tangent angle (perpendicular to radius for spiral)
            tangent = angle + math.pi / 2
            positions.append((x, y, tangent))

    return positions


def create_scene(params=None):
    """Set up the pymunk space with dominoes, floor, and walls."""
    params = params or {}
    space = pymunk.Space()
    space.gravity = (0, _p(params, "gravity"))
    # Improve collision detection for thin shapes
    space.collision_slop = 0.5

    w, h = config.WIDTH, config.HEIGHT
    num_dominoes = _p(params, "num_dominoes")
    domino_height = _p(params, "domino_height")
    domino_width = _p(params, "domino_width")
    spacing_factor = _p(params, "spacing_factor")
    pattern = _p(params, "pattern")

    # Generate path positions
    raw_positions = _generate_path(pattern, num_dominoes, w, h)

    # Build floor and walls (static)
    floor_y = h - 100
    floor_seg = pymunk.Segment(space.static_body, (0, floor_y), (w, floor_y), 5)
    floor_seg.elasticity = 0.1
    floor_seg.friction = 0.8
    floor_seg.collision_type = 2
    space.add(floor_seg)

    # Side walls
    left_wall = pymunk.Segment(space.static_body, (20, 0), (20, floor_y), 5)
    left_wall.elasticity = 0.2
    left_wall.friction = 0.5
    left_wall.collision_type = 2
    space.add(left_wall)

    right_wall = pymunk.Segment(space.static_body, (w - 20, 0), (w - 20, floor_y), 5)
    right_wall.elasticity = 0.2
    right_wall.friction = 0.5
    right_wall.collision_type = 2
    space.add(right_wall)

    # Create dominoes
    # For patterns other than "straight", dominoes stand on little local platforms
    # or we place them at (x,y) standing upright with the path defining their position
    dominoes = []
    half_w = domino_width / 2
    half_h = domino_height / 2
    mass = 1.0
    moment = pymunk.moment_for_box(mass, (domino_width, domino_height))

    for i, (px, py, path_angle) in enumerate(raw_positions):
        body = pymunk.Body(mass, moment)

        if pattern == "straight":
            # Standing upright on floor, spaced vertically means we
            # reinterpret: place them horizontally spaced instead
            # Actually for vertical shorts, "straight" = a line going down-screen
            # Dominoes stand perpendicular to path direction
            # Place domino with bottom on a local floor segment
            domino_x = px
            domino_y = py
            # The domino stands upright: its tall axis is vertical (angle=0 in pymunk)
            body.position = (domino_x, domino_y - half_h)
            body.angle = 0
        elif pattern == "spiral":
            # Domino stands upright at position, rotated to face along tangent
            body.position = (px, py - half_h)
            # Rotate so the thin face points along path_angle
            body.angle = 0
        else:
            body.position = (px, py - half_h)
            body.angle = 0

        # Box shape: vertices for a rectangle centered at body
        shape = pymunk.Poly.create_box(body, (domino_width, domino_height))
        shape.elasticity = 0.15
        shape.friction = 0.6
        shape.collision_type = 1
        space.add(body, shape)
        dominoes.append((body, shape, i))

    # For non-straight patterns, we need little floor segments under each domino
    # so they can stand upright at their positions
    local_floors = []
    if pattern in ("curve", "zigzag", "spiral"):
        for i, (px, py, path_angle) in enumerate(raw_positions):
            # Small static platform under each domino
            platform_half = domino_width * spacing_factor * 0.8
            seg = pymunk.Segment(
                space.static_body,
                (px - platform_half, py),
                (px + platform_half, py),
                3,
            )
            seg.elasticity = 0.1
            seg.friction = 0.8
            seg.collision_type = 2
            space.add(seg)
            local_floors.append(((px - platform_half, py), (px + platform_half, py)))

    return space, dominoes, raw_positions, floor_y, local_floors


def run(seed=None, params=None, note_schedule=None, collect_events_only=False):
    """Generator that yields pygame surfaces for each frame.

    Args:
        seed: Random seed for reproducibility
        params: Dict of scene parameters (overrides DEFAULT_PARAMS)
        note_schedule: If provided, not used for domino (chain is automatic)
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

    space, dominoes, raw_positions, floor_y, local_floors = create_scene(params)

    w, h = config.WIDTH, config.HEIGHT
    num_dominoes = _p(params, "num_dominoes")
    domino_height = _p(params, "domino_height")
    domino_width = _p(params, "domino_width")
    pattern = _p(params, "pattern")
    ball_colors = pal["balls"]

    # Assign gradient colors across the chain
    domino_colors = []
    for i in range(num_dominoes):
        color_idx = int(i / max(1, num_dominoes - 1) * (len(ball_colors) - 1))
        color_idx = min(color_idx, len(ball_colors) - 1)
        # Interpolate between adjacent colors for smooth gradient
        t_full = i / max(1, num_dominoes - 1) * (len(ball_colors) - 1)
        idx_a = int(t_full)
        idx_b = min(idx_a + 1, len(ball_colors) - 1)
        frac = t_full - idx_a
        ca = ball_colors[idx_a]
        cb = ball_colors[idx_b]
        color = (
            int(ca[0] + (cb[0] - ca[0]) * frac),
            int(ca[1] + (cb[1] - ca[1]) * frac),
            int(ca[2] + (cb[2] - ca[2]) * frac),
        )
        domino_colors.append(color)

    # Collision event handler: domino-domino (1 vs 1)
    frame_ref = [0]

    def _on_domino_domino(arbiter, space, data):
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
            "type": "domino_domino",
        })
        return True

    def _on_domino_floor(arbiter, space, data):
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
            "type": "domino_floor",
        })
        return True

    space.on_collision(1, 1, post_solve=_on_domino_domino)
    space.on_collision(1, 2, post_solve=_on_domino_floor)

    # Let the dominoes settle for a moment before tipping the first one
    settle_sim_steps = int(0.5 * config.SIM_FPS)  # 0.5 seconds of settling
    for _ in range(settle_sim_steps):
        space.step(1.0 / config.SIM_FPS)

    # Tip the first domino with an angular impulse
    if dominoes:
        first_body = dominoes[0][0]
        # Determine tip direction based on pattern
        if pattern == "straight":
            # Tip forward (downward on screen = positive y direction)
            first_body.apply_impulse_at_local_point((0, -80), (0, -domino_height / 2))
            first_body.angular_velocity = 2.5
        elif pattern == "curve":
            _, _, path_angle = raw_positions[0]
            # Tip in the direction of the path
            tip_force = 80
            fx = math.cos(path_angle) * tip_force
            fy = math.sin(path_angle) * tip_force
            first_body.apply_impulse_at_local_point((fx, -abs(fy)), (0, -domino_height / 2))
            first_body.angular_velocity = 2.5
        elif pattern == "zigzag":
            # Tip to the right (first row goes left to right)
            first_body.apply_impulse_at_local_point((80, 0), (0, -domino_height / 2))
            first_body.angular_velocity = 2.5
        elif pattern == "spiral":
            # Tip along the spiral tangent
            _, _, path_angle = raw_positions[0]
            first_body.apply_impulse_at_local_point((60, -40), (0, -domino_height / 2))
            first_body.angular_velocity = 2.5

    frame_idx = 0
    max_frames = 35 * config.VIDEO_FPS  # up to 35 seconds
    settle_frames = 0
    settle_threshold = 3 * config.VIDEO_FPS  # hold 3 seconds after settling
    chain_started = True  # We already tipped the first domino
    velocity_threshold = 2.0
    # Don't check settle until enough time for chain to propagate
    # (~0.15s per domino is typical chain speed)
    min_chain_time = max(8, int(num_dominoes * 0.15)) * config.VIDEO_FPS

    while frame_idx < max_frames:
        frame_ref[0] = frame_idx

        # Step physics
        for _ in range(config.SIM_STEPS_PER_FRAME):
            space.step(1.0 / config.SIM_FPS)

        # Check if all dominoes have settled
        if chain_started and len(dominoes) > 0:
            max_vel = max(b.velocity.length for b, _, _ in dominoes)
            max_ang = max(abs(b.angular_velocity) for b, _, _ in dominoes)
            if max_vel < velocity_threshold and max_ang < 0.5:
                settle_frames += 1
            else:
                settle_frames = 0
            # Only start checking after some time (give chain time to propagate)
            if frame_idx > min_chain_time and settle_frames >= settle_threshold:
                break

        if not collect_events_only:
            surface.fill(pal["bg"])

            # Draw floor
            pygame.draw.line(
                surface,
                pal.get("wall", (100, 100, 100)),
                (0, floor_y),
                (w, floor_y),
                5,
            )

            # Draw side walls
            pygame.draw.line(
                surface,
                pal.get("wall", (100, 100, 100)),
                (20, 0),
                (20, floor_y),
                4,
            )
            pygame.draw.line(
                surface,
                pal.get("wall", (100, 100, 100)),
                (w - 20, 0),
                (w - 20, floor_y),
                4,
            )

            # Draw local floor platforms (for non-straight patterns)
            if pattern in ("curve", "zigzag", "spiral"):
                for (ax, ay), (bx, by) in local_floors:
                    pygame.draw.line(
                        surface,
                        pal.get("wall", (100, 100, 100)),
                        (int(ax), int(ay)),
                        (int(bx), int(by)),
                        3,
                    )

            # Draw dominoes
            for body, shape, idx in dominoes:
                if idx >= len(domino_colors):
                    continue
                color = domino_colors[idx]

                # Get the four corners of the domino box in world coordinates
                verts = shape.get_vertices()
                world_verts = [body.local_to_world(v) for v in verts]
                points = [(int(v.x), int(v.y)) for v in world_verts]

                # Check if any vertex is on screen
                any_visible = any(
                    -50 <= px <= w + 50 and -50 <= py <= h + 50
                    for px, py in points
                )
                if not any_visible:
                    continue

                # Main domino body
                if len(points) >= 3:
                    pygame.draw.polygon(surface, color, points)

                    # 3D bevel effect: lighter top edge
                    lighter = (
                        min(255, color[0] + 50),
                        min(255, color[1] + 50),
                        min(255, color[2] + 50),
                    )
                    darker = (
                        max(0, color[0] - 40),
                        max(0, color[1] - 40),
                        max(0, color[2] - 40),
                    )

                    # Draw top edge highlight (the edge that is highest on screen)
                    # Sort vertices by y to find top edge
                    sorted_verts = sorted(points, key=lambda p: p[1])
                    top_two = sorted_verts[:2]
                    bottom_two = sorted_verts[2:]

                    pygame.draw.line(surface, lighter, top_two[0], top_two[1], 2)
                    # Draw bottom edge shadow
                    pygame.draw.line(surface, darker, bottom_two[0], bottom_two[1], 2)

                    # Outline
                    pygame.draw.polygon(surface, darker, points, 1)

        yield surface
        frame_idx += 1

    # Hold final frame for 2 seconds
    for _ in range(2 * config.VIDEO_FPS):
        yield surface


def get_metadata(video_num, params=None):
    params = params or {}
    pattern = _p(params, "pattern")
    n = _p(params, "num_dominoes")
    pattern_label = {
        "straight": "Straight Line",
        "curve": "Curving Wave",
        "zigzag": "Zigzag",
        "spiral": "Spiral",
    }.get(pattern, pattern.title())
    return {
        "title": f"{pattern_label} Domino Chain #{video_num}",
        "description": (
            f"Watch {n} dominoes fall in a satisfying {pattern_label.lower()} "
            f"chain reaction! #Shorts #dominoes #satisfying #chainreaction"
        ),
        "tags": [
            "dominoes", "chain reaction", "physics", "satisfying",
            "domino effect", pattern, "shorts",
        ],
        "category": "24",
    }
