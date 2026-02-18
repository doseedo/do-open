"""Sand Cascade scene.

Many small particles cascade through a maze of angled deflectors, creating
satisfying flowing streams. Particles are divided into colored groups that
pour from different positions at the top, splitting and merging as they
bounce off alternating deflector platforms.
"""

import random
import math
import pymunk
import pygame
from .. import config
from ..palettes import get_palette

DEFAULT_PARAMS = {
    "num_particles": 400,       # range 200-600
    "particle_radius": 4,       # range 3-6
    "num_levels": 6,            # range 4-8
    "gravity": 600,             # range 400-800
    "pour_rate": 8,             # range 5-15, particles added per frame
    "num_colors": 3,            # range 2-5
    "palette": "ocean",
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


def _build_deflectors(space, num_levels, w, h):
    """Create angled deflector platforms across multiple levels.

    Each level has 2-4 angled segments that split and redirect particle streams.
    Alternating levels deflect left and right.

    Returns list of deflector endpoint pairs for drawing.
    """
    deflectors = []

    top_margin = 180
    bottom_margin = 350
    usable_height = h - top_margin - bottom_margin
    level_spacing = usable_height / (num_levels + 1)
    side_margin = 60

    for level in range(num_levels):
        y_center = top_margin + (level + 1) * level_spacing
        going_right = (level % 2 == 0)

        # Determine number of segments at this level (2-4)
        # More segments in middle levels, fewer at top and bottom
        if level == 0 or level == num_levels - 1:
            num_segments = 2
        else:
            num_segments = 3 if num_levels <= 5 else 4

        # Distribute segments across the width
        segment_zone_width = (w - 2 * side_margin) / num_segments
        angle_deg = 22 + random.uniform(-3, 3)  # 20-30 degrees
        angle_rad = math.radians(angle_deg)

        for seg_idx in range(num_segments):
            zone_left = side_margin + seg_idx * segment_zone_width
            zone_center_x = zone_left + segment_zone_width / 2

            # Segment length: roughly 60-80% of zone width
            seg_length = segment_zone_width * random.uniform(0.55, 0.75)
            half_len = seg_length / 2

            # Alternate tilt direction based on level
            if going_right:
                dx = math.cos(angle_rad) * half_len
                dy = math.sin(angle_rad) * half_len
            else:
                dx = math.cos(math.pi - angle_rad) * half_len
                dy = math.sin(math.pi - angle_rad) * half_len

            # Offset alternate segments slightly for visual variety
            y_offset = random.uniform(-10, 10)

            ax = zone_center_x - dx
            ay = y_center + y_offset - dy
            bx = zone_center_x + dx
            by = y_center + y_offset + dy

            seg = pymunk.Segment(space.static_body, (ax, ay), (bx, by), 4)
            seg.elasticity = random.uniform(0.1, 0.3)
            seg.friction = random.uniform(0.3, 0.5)
            seg.collision_type = 2
            space.add(seg)
            deflectors.append(((ax, ay), (bx, by)))

    return deflectors


def create_scene(params=None):
    """Set up the pymunk space with deflectors, walls, and collection floor."""
    params = params or {}
    space = pymunk.Space()
    space.gravity = (0, _p(params, "gravity"))

    particle_radius = _p(params, "particle_radius")
    num_particles = _p(params, "num_particles")

    # Spatial hash for broad-phase optimization
    space.use_spatial_hash(particle_radius * 2, num_particles * 2)
    space.iterations = 5

    w, h = config.WIDTH, config.HEIGHT
    num_levels = _p(params, "num_levels")

    # Build deflector platforms
    deflectors = _build_deflectors(space, num_levels, w, h)

    # Side walls
    wall_thickness = 5
    left_wall = pymunk.Segment(space.static_body, (30, 0), (30, h - 80), wall_thickness)
    left_wall.elasticity = 0.2
    left_wall.friction = 0.4
    left_wall.collision_type = 2
    space.add(left_wall)

    right_wall = pymunk.Segment(space.static_body, (w - 30, 0), (w - 30, h - 80), wall_thickness)
    right_wall.elasticity = 0.2
    right_wall.friction = 0.4
    right_wall.collision_type = 2
    space.add(right_wall)

    # Collection floor at the bottom
    floor_y = h - 80
    floor_seg = pymunk.Segment(space.static_body, (30, floor_y), (w - 30, floor_y), wall_thickness)
    floor_seg.elasticity = 0.05
    floor_seg.friction = 0.6
    floor_seg.collision_type = 2
    space.add(floor_seg)

    # Small angled walls at the very bottom corners to funnel particles inward
    funnel_left = pymunk.Segment(space.static_body, (30, floor_y), (80, floor_y - 40), 3)
    funnel_left.elasticity = 0.1
    funnel_left.friction = 0.4
    funnel_left.collision_type = 2
    space.add(funnel_left)

    funnel_right = pymunk.Segment(space.static_body, (w - 30, floor_y), (w - 80, floor_y - 40), 3)
    funnel_right.elasticity = 0.1
    funnel_right.friction = 0.4
    funnel_right.collision_type = 2
    space.add(funnel_right)

    return space, deflectors, floor_y


def run(seed=None, params=None, note_schedule=None, collect_events_only=False):
    """Generator that yields pygame surfaces for each frame.

    Args:
        seed: Random seed for reproducibility
        params: Dict of scene parameters (overrides DEFAULT_PARAMS)
        note_schedule: Not used for this scene (ambient particle flow)
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

    space, deflectors, floor_y = create_scene(params)

    w, h = config.WIDTH, config.HEIGHT
    num_particles = _p(params, "num_particles")
    particle_radius = _p(params, "particle_radius")
    pour_rate = _p(params, "pour_rate")
    num_colors = _p(params, "num_colors")

    ball_colors = pal["balls"]
    # Select the color groups from the palette
    color_groups = []
    for i in range(num_colors):
        color_groups.append(ball_colors[i % len(ball_colors)])

    # Compute pour positions for each color group, spread across the top
    pour_zone_width = (w - 200) / num_colors
    pour_positions = []
    for i in range(num_colors):
        cx = 100 + pour_zone_width * (i + 0.5)
        pour_positions.append(cx)

    # Density-band collision event tracking
    num_bands = 8
    band_height = h / num_bands
    density_check_interval = 15  # frames
    density_threshold = 5  # particles per band to trigger event

    frame_ref = [0]

    # Simple collision handler for deflector contacts (for event collection)
    def _on_post_solve(arbiter, space, data):
        # We use density-band approach instead of per-collision events
        return True

    space.on_collision(1, 2, post_solve=_on_post_solve)

    particles = []  # list of (body, shape, color)
    particles_spawned = 0
    frame_idx = 0

    # Pour duration: ~20 seconds, settle for ~10s, hold 2s
    pour_duration_frames = 20 * config.VIDEO_FPS  # 600 frames at 30fps
    max_frames = 33 * config.VIDEO_FPS  # up to 33 seconds before hold

    settle_frames = 0
    settle_threshold = 3 * config.VIDEO_FPS  # 3 seconds of stillness to stop early

    # Precompute deflector drawing values
    wall_color = pal.get("wall", (100, 100, 100))
    obstacle_color = pal.get("obstacle", (80, 80, 80))
    bg_color = pal["bg"]

    while frame_idx < max_frames:
        frame_ref[0] = frame_idx

        # Pour particles from top
        if frame_idx < pour_duration_frames and particles_spawned < num_particles:
            to_spawn = min(pour_rate, num_particles - particles_spawned)
            for _ in range(to_spawn):
                # Pick a color group (round-robin for even distribution)
                group_idx = particles_spawned % num_colors
                color = color_groups[group_idx]
                cx = pour_positions[group_idx]

                # Add slight random offset to pour position
                x = cx + random.uniform(-25, 25)
                y = random.uniform(40, 80)

                mass = 0.1
                moment = pymunk.moment_for_circle(mass, 0, particle_radius)
                body = pymunk.Body(mass, moment)
                body.position = (x, y)
                # Small initial downward velocity with slight horizontal spread
                body.velocity = (random.uniform(-20, 20), random.uniform(30, 80))

                shape = pymunk.Circle(body, particle_radius)
                shape.elasticity = 0.15
                shape.friction = 0.2
                shape.collision_type = 1
                space.add(body, shape)

                particles.append((body, shape, color))
                particles_spawned += 1

        # Step physics
        for _ in range(config.SIM_STEPS_PER_FRAME):
            space.step(1.0 / config.SIM_FPS)

        # Density-band collision events (every density_check_interval frames)
        if frame_idx % density_check_interval == 0 and len(particles) > 0:
            band_counts = [0] * num_bands
            for body, _, _ in particles:
                py = body.position.y
                band_idx = int(py / band_height)
                if 0 <= band_idx < num_bands:
                    band_counts[band_idx] += 1

            for band_idx, count in enumerate(band_counts):
                if count >= density_threshold:
                    band_y_mid = (band_idx + 0.5) * band_height
                    # X position: average x of particles in this band
                    band_xs = []
                    for body, _, _ in particles:
                        py = body.position.y
                        bi = int(py / band_height)
                        if bi == band_idx:
                            band_xs.append(body.position.x)
                    avg_x = sum(band_xs) / len(band_xs) if band_xs else w / 2

                    _collision_events.append({
                        "frame": frame_idx,
                        "time_sec": frame_idx / config.VIDEO_FPS,
                        "x": float(avg_x),
                        "y": float(band_y_mid),
                        "force": float(count),
                        "type": "density_band",
                        "band": band_idx,
                    })

        # Check if particles have settled (only after pouring is done)
        all_poured = particles_spawned >= num_particles or frame_idx >= pour_duration_frames
        if all_poured and len(particles) > 0:
            max_vel = max(b.velocity.length for b, _, _ in particles)
            if max_vel < 3.0:
                settle_frames += 1
            else:
                settle_frames = 0
            if settle_frames >= settle_threshold:
                break
        else:
            settle_frames = 0

        if not collect_events_only:
            surface.fill(bg_color)

            # Draw deflector platforms as thick lines
            for (ax, ay), (bx, by) in deflectors:
                pygame.draw.line(surface, obstacle_color,
                                 (int(ax), int(ay)), (int(bx), int(by)), 6)
                # Lighter top edge for subtle 3D effect
                lighter = (
                    min(255, obstacle_color[0] + 30),
                    min(255, obstacle_color[1] + 30),
                    min(255, obstacle_color[2] + 30),
                )
                pygame.draw.line(surface, lighter,
                                 (int(ax), int(ay) - 2), (int(bx), int(by) - 2), 2)

            # Draw side walls
            pygame.draw.line(surface, wall_color, (30, 0), (30, floor_y), 4)
            pygame.draw.line(surface, wall_color, (w - 30, 0), (w - 30, floor_y), 4)

            # Draw floor
            pygame.draw.line(surface, wall_color, (30, floor_y), (w - 30, floor_y), 4)

            # Draw funnel walls
            pygame.draw.line(surface, wall_color, (30, floor_y), (80, floor_y - 40), 3)
            pygame.draw.line(surface, wall_color, (w - 30, floor_y), (w - 80, floor_y - 40), 3)

            # Draw particles (only those within screen bounds)
            for body, shape, color in particles:
                bx_pos = body.position.x
                by_pos = body.position.y
                ix = int(bx_pos)
                iy = int(by_pos)
                if 0 <= ix <= w and 0 <= iy <= h:
                    pygame.draw.circle(surface, color, (ix, iy), particle_radius)

        yield surface
        frame_idx += 1

    # Hold final frame for 2 seconds
    for _ in range(2 * config.VIDEO_FPS):
        yield surface


def get_metadata(video_num, params=None):
    params = params or {}
    n = _p(params, "num_particles")
    levels = _p(params, "num_levels")
    return {
        "title": f"Sand Cascade #{video_num}",
        "description": (
            f"Watch {n} particles cascade through {levels} levels of deflectors "
            f"in beautiful flowing streams! #Shorts #satisfying #physics #sand #cascade"
        ),
        "tags": [
            "sand", "cascade", "physics", "satisfying", "particles",
            "flowing", "simulation", "shorts",
        ],
        "category": "24",
    }
