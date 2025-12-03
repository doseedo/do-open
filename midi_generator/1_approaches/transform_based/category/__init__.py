"""
Categorical Cross-Track Layer.

Provides category-theoretic structures for modeling cross-track relationships:
- TrackFunctor: Represents a track as functor F: Time → PitchSpace
- MultiTrackSpace: Categorical product with selective unification
- NaturalTransform: Cross-track dependencies as natural transformations
"""

from .track_functor import TrackFunctor, NaturalTransform, MultiTrackSpace

__all__ = ['TrackFunctor', 'NaturalTransform', 'MultiTrackSpace']
