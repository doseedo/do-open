"""Scene registry — maps scene names to their modules."""

SCENE_REGISTRY = {}


def register_scene(name, module):
    """Register a scene module by name."""
    SCENE_REGISTRY[name] = module


def get_scene(name):
    """Get a registered scene module by name."""
    return SCENE_REGISTRY.get(name)


def list_scenes():
    """Return list of registered scene names."""
    return list(SCENE_REGISTRY.keys())


# Import and register all built-in scenes
from . import plinko, marble_race
register_scene("plinko", plinko)
register_scene("marble_race", marble_race)

# New scenes — imported as they're created
try:
    from . import domino_chain
    register_scene("domino_chain", domino_chain)
except ImportError:
    pass

try:
    from . import xylophone_bounce
    register_scene("xylophone_bounce", xylophone_bounce)
except ImportError:
    pass

try:
    from . import pendulum_wave
    register_scene("pendulum_wave", pendulum_wave)
except ImportError:
    pass

try:
    from . import newtons_cradle
    register_scene("newtons_cradle", newtons_cradle)
except ImportError:
    pass

try:
    from . import funnel_drop
    register_scene("funnel_drop", funnel_drop)
except ImportError:
    pass

try:
    from . import wrecking_ball
    register_scene("wrecking_ball", wrecking_ball)
except ImportError:
    pass

try:
    from . import pinball
    register_scene("pinball", pinball)
except ImportError:
    pass

try:
    from . import sand_cascade
    register_scene("sand_cascade", sand_cascade)
except ImportError:
    pass

try:
    from . import video_grid_music
    register_scene("video_grid_music", video_grid_music)
except ImportError:
    pass
