"""
Storage Module for Musical Program Synthesis

Provides persistent storage for analyzed examples, training data,
and system state for adaptive learning.
"""

from .example_database import ExampleDatabase

__all__ = ['ExampleDatabase']
