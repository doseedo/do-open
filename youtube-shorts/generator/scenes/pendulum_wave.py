"""Pendulum Wave scene.

N pendulums of different lengths swing from a bar at the top. Each has a slightly
different period, creating mesmerizing wave patterns as they go in and out of sync.

Uses ANALYTICAL motion (not pymunk) for clean wave patterns:
    x(t) = amplitude * sin(2*pi*t / period_i)
"""

import random
import math
import pygame
from .. import config
from ..palettes import get_palette

# Default parameters (overridable via params dict)
DEFAULT_PARAMS = {
    "num_pendulums": 15,       # range 10-25
    "min_period": 1.7,         # range 1.5-2.0 — period in seconds for the shortest pendulum
    "max_period": 3.3,         # range 2.5-4.0 — period in seconds for the longest pendulum
    "ball_radius": 16,         # range 12-22
    "amplitude": 180,          # range 100-250 — horizontal swing amplitude in pixels
    "string_length": 600,      # range 400-800 — length of the longest string
    "palette": "neon",
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

    w, h = config.WIDTH, config.HEIGHT

    if collect_events_only:
        surface = pygame.Surface((1, 1))
    else:
        surface = pygame.Surface((w, h))

    num_pendulums = _p(params, "num_pendulums")
    min_period = _p(params, "min_period")
    max_period = _p(params, "max_period")
    ball_radius = _p(params, "ball_radius")
    amplitude = _p(params, "amplitude")
    max_string_length = _p(params, "string_length")
    ball_colors = pal["balls"]

    # Support bar geometry
    bar_y = 150
    bar_left = 60
    bar_right = w - 60
    bar_height = 14

    # Pendulum pivot points — evenly spaced across the bar
    side_margin = 100
    usable_width = w - 2 * side_margin
    if num_pendulums > 1:
        spacing = usable_width / (num_pendulums - 1)
    else:
        spacing = 0
    pivot_xs = [side_margin + i * spacing for i in range(num_pendulums)]
    center_x = w / 2

    # Compute periods for each pendulum: i=0 is shortest period (leftmost)
    # Period increases left to right
    periods = []
    for i in range(num_pendulums):
        if num_pendulums > 1:
            t = i / (num_pendulums - 1)
        else:
            t = 0.0
        period_i = min_period + t * (max_period - min_period)
        periods.append(period_i)

    # String lengths proportional to period^2 (T = 2*pi*sqrt(L/g))
    # Normalize so the longest string = max_string_length
    max_period_val = max(periods)
    string_lengths = []
    for period_i in periods:
        # L proportional to T^2
        ratio = (period_i / max_period_val) ** 2
        string_lengths.append(ratio * max_string_length)

    # Trail history: store last N positions per pendulum for trail effect
    trail_length = 8
    trails = [[] for _ in range(num_pendulums)]

    # Previous sin values for zero-crossing detection (collision events)
    prev_sins = [None] * num_pendulums

    # Animation timing
    anim_duration = 35.0  # seconds of animation
    hold_duration = 2.0   # seconds to hold final frame
    total_anim_frames = int(anim_duration * config.VIDEO_FPS)
    hold_frames = int(hold_duration * config.VIDEO_FPS)

    # All pendulums start released from the right (phase = pi/2 so sin starts at +1)
    initial_phase = math.pi / 2

    # Pre-build glow surface for balls (only if rendering)
    glow_surf = None
    if not collect_events_only:
        glow_radius = ball_radius + 8
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)

    frame_idx = 0

    while frame_idx < total_anim_frames:
        current_time = frame_idx / config.VIDEO_FPS

        # Compute positions for each pendulum
        positions = []
        for i in range(num_pendulums):
            period_i = periods[i]
            # Analytical: angle as a function of time
            # sin(2*pi*t / T + initial_phase), starting from max deflection
            sin_val = math.sin(2 * math.pi * current_time / period_i + initial_phase)
            # Horizontal displacement
            dx = amplitude * sin_val
            ball_x = pivot_xs[i] + dx

            # Compute ball y from string length and displacement
            # For a pendulum: if string length = L and horizontal displacement = dx,
            # then ball_y = pivot_y + sqrt(L^2 - dx^2) (approximately)
            # Clamp dx to string length to avoid math domain error
            L = string_lengths[i]
            if abs(dx) >= L:
                ball_y = bar_y + bar_height
            else:
                ball_y = bar_y + bar_height + math.sqrt(L * L - dx * dx)

            positions.append((ball_x, ball_y))

            # Zero-crossing detection for collision events
            if prev_sins[i] is not None:
                # Detect sign change (crossing center)
                if prev_sins[i] * sin_val < 0:
                    # Pendulum crossed center
                    # Map index to pitch: leftmost = low, rightmost = high
                    pitch = 48 + int(i * 36 / max(1, num_pendulums - 1))
                    # Estimate force from velocity (derivative of sin at crossing is max)
                    velocity = abs(2 * math.pi * amplitude / period_i)
                    _collision_events.append({
                        "frame": frame_idx,
                        "time_sec": current_time,
                        "x": float(pivot_xs[i]),
                        "y": float(ball_y),
                        "force": float(velocity),
                        "pitch": pitch,
                        "pendulum_index": i,
                    })
            prev_sins[i] = sin_val

        if not collect_events_only:
            # Update trails
            for i in range(num_pendulums):
                trails[i].append(positions[i])
                if len(trails[i]) > trail_length:
                    trails[i].pop(0)

            # --- Drawing ---
            bg_color = pal["bg"]
            surface.fill(bg_color)

            # Support bar — metallic gradient rectangle
            bar_rect = pygame.Rect(bar_left, bar_y, bar_right - bar_left, bar_height)
            # Draw gradient: lighter at top, darker at bottom
            wall_color = pal.get("wall", (100, 100, 100))
            for row in range(bar_height):
                t = row / max(1, bar_height - 1)
                # Lighten top, darken bottom
                r = min(255, int(wall_color[0] + (1 - t) * 60))
                g = min(255, int(wall_color[1] + (1 - t) * 60))
                b = min(255, int(wall_color[2] + (1 - t) * 60))
                pygame.draw.line(surface, (r, g, b),
                                 (bar_left, bar_y + row),
                                 (bar_right, bar_y + row))
            # Bar edge highlights
            pygame.draw.rect(surface, (min(255, wall_color[0] + 80),
                                       min(255, wall_color[1] + 80),
                                       min(255, wall_color[2] + 80)),
                             bar_rect, 2)

            # Bar mounting brackets (small rectangles at ends)
            bracket_w, bracket_h = 20, 24
            for bx in [bar_left - 5, bar_right - bracket_w + 5]:
                bracket_rect = pygame.Rect(bx, bar_y - 5, bracket_w, bracket_h)
                pygame.draw.rect(surface, wall_color, bracket_rect)
                pygame.draw.rect(surface, (min(255, wall_color[0] + 40),
                                           min(255, wall_color[1] + 40),
                                           min(255, wall_color[2] + 40)),
                                 bracket_rect, 2)

            # Draw trails (faded previous positions)
            for i in range(num_pendulums):
                color = ball_colors[i % len(ball_colors)]
                trail = trails[i]
                for j, (tx, ty) in enumerate(trail[:-1]):  # skip current position
                    # Fade: older = more transparent
                    alpha = int(30 + 60 * (j / max(1, len(trail) - 1)))
                    trail_r = max(3, ball_radius - 4)
                    trail_surf = pygame.Surface((trail_r * 2, trail_r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(trail_surf, (color[0], color[1], color[2], alpha),
                                       (trail_r, trail_r), trail_r)
                    surface.blit(trail_surf, (int(tx) - trail_r, int(ty) - trail_r))

            # Draw strings and balls
            string_color = (160, 160, 180) if pal["bg"][0] < 100 else (120, 120, 120)
            for i in range(num_pendulums):
                px = pivot_xs[i]
                py = bar_y + bar_height
                bx, by = positions[i]

                # String: thin line from pivot to ball
                pygame.draw.line(surface, string_color,
                                 (int(px), int(py)),
                                 (int(bx), int(by)), 2)

                # Ball glow effect (larger, semi-transparent circle behind ball)
                color = ball_colors[i % len(ball_colors)]
                glow_radius = ball_radius + 8
                glow_surf.fill((0, 0, 0, 0))
                # Draw multiple concentric circles for smooth glow
                for g in range(3):
                    gr = glow_radius - g * 2
                    ga = 25 + g * 10  # increasing alpha toward center of glow
                    pygame.draw.circle(glow_surf, (color[0], color[1], color[2], ga),
                                       (glow_radius, glow_radius), gr)
                surface.blit(glow_surf, (int(bx) - glow_radius, int(by) - glow_radius))

                # Ball: filled circle
                pygame.draw.circle(surface, color,
                                   (int(bx), int(by)), ball_radius)

                # Highlight on ball (specular reflection)
                hx = int(bx) - ball_radius // 3
                hy = int(by) - ball_radius // 3
                pygame.draw.circle(surface, (255, 255, 255), (hx, hy), ball_radius // 4)

            # Small pivot dots on bar
            for px in pivot_xs:
                pygame.draw.circle(surface, string_color,
                                   (int(px), bar_y + bar_height), 3)

        yield surface
        frame_idx += 1

    # Hold final frame
    for _ in range(hold_frames):
        yield surface


def get_metadata(video_num, params=None):
    params = params or {}
    n = _p(params, "num_pendulums")
    return {
        "title": f"Pendulum Wave #{video_num}",
        "description": f"Watch {n} pendulums create mesmerizing wave patterns! #Shorts #pendulumwave #satisfying #physics",
        "tags": ["pendulum wave", "physics", "satisfying", "mesmerizing", "wave", "shorts"],
        "category": "24",
    }
