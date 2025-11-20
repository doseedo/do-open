"""
Data Module - Musical Style Database
====================================

Provides comprehensive style definitions with complete parameter sets
for 105+ musical styles across all genres and eras.

Author: Agent 7 - Style Database Curator
"""

from .style_database import StyleDatabase, StyleMetadata, STYLE_METADATA
from .style_generator import generate_all_styles, BASE_PARAMETERS_TEMPLATE

__all__ = [
    'StyleDatabase',
    'StyleMetadata',
    'STYLE_METADATA',
    'generate_all_styles',
    'BASE_PARAMETERS_TEMPLATE',
]
