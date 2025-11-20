"""
Big Band Style Profiles

This module provides style configuration profiles for different big band arrangers.
"""

from .base_profile import StyleProfile
from .basie_profile import BASIE_STYLE
from .ellington_profile import ELLINGTON_STYLE
from .thad_jones_profile import THAD_JONES_STYLE

# Style registry
STYLE_REGISTRY = {
    "basie": BASIE_STYLE,
    "ellington": ELLINGTON_STYLE,
    "thad_jones": THAD_JONES_STYLE,
    "modern": THAD_JONES_STYLE,  # Alias
}

def get_style(name: str) -> StyleProfile:
    """Get style profile by name."""
    name_lower = name.lower()
    if name_lower not in STYLE_REGISTRY:
        raise ValueError(
            f"Unknown style: {name}. Available: {list(STYLE_REGISTRY.keys())}"
        )
    return STYLE_REGISTRY[name_lower]

def list_styles() -> list:
    """List all available style names."""
    return list(STYLE_REGISTRY.keys())

__all__ = [
    'StyleProfile',
    'BASIE_STYLE',
    'ELLINGTON_STYLE',
    'THAD_JONES_STYLE',
    'STYLE_REGISTRY',
    'get_style',
    'list_styles',
]
