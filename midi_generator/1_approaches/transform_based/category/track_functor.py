"""
TrackFunctor - Category-Theoretic Representation of Musical Tracks
===================================================================

Each track is modeled as a functor F: TimeCategory -> PitchSpace

TimeCategory:
- Objects: Time points t₀, t₁, t₂, ...
- Morphisms: Time intervals [t₁, t₂) representing durations

PitchSpace:
- Objects: Pitch configurations (sets of active pitches)
- Morphisms: Pitch transformations (D24 group elements, voice leading)

Track as Functor:
- F(t) = pitch configuration at time t
- F(f: t₁ → t₂) = how pitches transform over the interval

Natural Transformation η: F ⇒ G:
- For each time point t, a pitch transformation η_t: F(t) → G(t)
- Naturality: For any time morphism f: t₁ → t₂,
              G(f) ∘ η_{t₁} = η_{t₂} ∘ F(f)

This captures: "If track F changes by transformation X over interval I,
and track G follows F via η, then G must change by the same X."

Author: Dosedo Architecture v2
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# CATEGORICAL PRIMITIVES
# =============================================================================

@dataclass(frozen=True)
class TimePoint:
    """Object in the time category."""
    t: int  # Time step

    def __repr__(self):
        return f"T{self.t}"


@dataclass(frozen=True)
class TimeInterval:
    """Morphism in the time category: f: t₁ → t₂"""
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start

    def compose(self, other: 'TimeInterval') -> 'TimeInterval':
        """Compose intervals: (t₁→t₂) ∘ (t₂→t₃) = (t₁→t₃)"""
        assert self.end == other.start, "Intervals must be composable"
        return TimeInterval(self.start, other.end)

    def __repr__(self):
        return f"[{self.start}→{self.end}]"


@dataclass
class PitchConfiguration:
    """Object in the pitch space category."""
    pitches: np.ndarray  # Active pitches at this time
    velocities: Optional[np.ndarray] = None

    def __hash__(self):
        return hash(self.pitches.tobytes())

    def __eq__(self, other):
        if not isinstance(other, PitchConfiguration):
            return False
        return np.array_equal(self.pitches, other.pitches)

    def __repr__(self):
        return f"Pitch({list(self.pitches)})"


@dataclass
class PitchMorphism:
    """Morphism in pitch space: a transformation between pitch configurations."""
    transform_type: str  # 'd24', 'voice_leading', 'identity'
    transform_id: int    # Element index in transform group
    param: Optional[float] = None

    def apply(self, config: PitchConfiguration) -> PitchConfiguration:
        """Apply this morphism to a pitch configuration."""
        if self.transform_type == 'identity':
            return config
        elif self.transform_type == 'd24':
            # Apply D24 transform
            from core.groups.d24_group import D24Group
            d24 = D24Group()
            new_pitches = d24.apply_to_pitch_array(self.transform_id, config.pitches)
            return PitchConfiguration(new_pitches, config.velocities)
        else:
            return config

    def compose(self, other: 'PitchMorphism') -> 'PitchMorphism':
        """Compose morphisms."""
        if self.transform_type == 'identity':
            return other
        if other.transform_type == 'identity':
            return self
        if self.transform_type == other.transform_type == 'd24':
            from core.groups.d24_group import D24Group
            d24 = D24Group()
            composed = d24.compose(self.transform_id, other.transform_id)
            return PitchMorphism('d24', composed)
        # Default: return self (approximate)
        return self

    def __repr__(self):
        if self.transform_type == 'identity':
            return "id"
        return f"{self.transform_type}[{self.transform_id}]"


# =============================================================================
# TRACK FUNCTOR
# =============================================================================

class TrackFunctor:
    """
    Represents a track as functor F: TimeCategory → PitchSpace.

    F(t) = pitch configuration at time t
    F(f: t₁→t₂) = how pitches transform over the interval

    Usage:
        functor = TrackFunctor.from_objects(track_id, objects)
        config_at_t = functor(TimePoint(t))
        morphism = functor.morphism(TimeInterval(t1, t2))
    """

    def __init__(self, track_id: str):
        """
        Initialize empty functor.

        Args:
            track_id: Unique identifier for this track
        """
        self.track_id = track_id

        # F on objects: time -> pitch configuration
        self._object_map: Dict[int, PitchConfiguration] = {}

        # F on morphisms: interval -> pitch morphism
        self._morphism_map: Dict[Tuple[int, int], PitchMorphism] = {}

        # Cached morphisms for common intervals
        self._morphism_cache: Dict[Tuple[int, int], PitchMorphism] = {}

    def __call__(self, t: TimePoint) -> Optional[PitchConfiguration]:
        """Apply functor to time point: F(t)."""
        return self._object_map.get(t.t)

    def morphism(self, interval: TimeInterval) -> PitchMorphism:
        """Apply functor to time interval: F(f)."""
        key = (interval.start, interval.end)

        if key in self._morphism_map:
            return self._morphism_map[key]

        # Compute morphism from configurations
        config_start = self._object_map.get(interval.start)
        config_end = self._object_map.get(interval.end)

        if config_start is None or config_end is None:
            return PitchMorphism('identity', 0)

        # Find transformation between configurations
        morphism = self._compute_morphism(config_start, config_end)
        self._morphism_map[key] = morphism

        return morphism

    def _compute_morphism(
        self,
        source: PitchConfiguration,
        target: PitchConfiguration
    ) -> PitchMorphism:
        """Compute the morphism between two pitch configurations."""
        if len(source.pitches) == 0 or len(target.pitches) == 0:
            return PitchMorphism('identity', 0)

        if np.array_equal(source.pitches, target.pitches):
            return PitchMorphism('identity', 0)

        # Try to find D24 transform
        from core.groups.d24_group import D24Group
        d24 = D24Group()

        # Check pitch class transformation
        source_pc = source.pitches % 12
        target_pc = target.pitches % 12

        if len(source_pc) == len(target_pc):
            for t_id in range(24):
                transformed = d24.apply_to_pitch_class_array(t_id, source_pc)
                if np.array_equal(transformed, target_pc):
                    return PitchMorphism('d24', t_id)

        # Default to voice leading
        return PitchMorphism('voice_leading', 0)

    def set_configuration(self, t: int, config: PitchConfiguration):
        """Set pitch configuration at time t."""
        self._object_map[t] = config

    def set_morphism(self, interval: TimeInterval, morphism: PitchMorphism):
        """Explicitly set morphism for an interval."""
        self._morphism_map[(interval.start, interval.end)] = morphism

    @classmethod
    def from_objects(cls, track_id: str, objects: List) -> 'TrackFunctor':
        """
        Build functor from list of FactoredObjectV2.

        Args:
            track_id: Track identifier
            objects: List of FactoredObjectV2 from this track

        Returns:
            TrackFunctor representing the track
        """
        functor = cls(track_id)

        # Sort objects by start time
        sorted_objects = sorted(objects, key=lambda o: o.start_time)

        for obj in sorted_objects:
            # Create pitch configuration
            if hasattr(obj, 'pitch_class') and hasattr(obj, 'octave'):
                pitches = obj.pitch_class.astype(np.int32) + obj.octave.astype(np.int32) * 12
            else:
                pitches = obj.pitches

            velocities = obj.velocity if hasattr(obj, 'velocity') else None
            config = PitchConfiguration(pitches, velocities)

            # Set for all time points in this object's span
            functor.set_configuration(obj.start_time, config)

        return functor

    @property
    def time_points(self) -> List[int]:
        """Get all time points where functor is defined."""
        return sorted(self._object_map.keys())

    def __repr__(self):
        return f"TrackFunctor({self.track_id}, {len(self._object_map)} points)"


# =============================================================================
# NATURAL TRANSFORMATION
# =============================================================================

class NaturalTransform:
    """
    Natural transformation η: F ⇒ G between track functors.

    For each time point t:
        η_t: F(t) → G(t) is a pitch morphism

    Naturality condition:
        For any time morphism f: t₁ → t₂,
        G(f) ∘ η_{t₁} = η_{t₂} ∘ F(f)

    This means: "The transformation commutes with time evolution."

    Usage:
        eta = NaturalTransform.discover(functor_f, functor_g)
        if eta.is_natural:
            print(f"Consistent transformation: {eta.component_at(0)}")
    """

    def __init__(self, source: TrackFunctor, target: TrackFunctor):
        """
        Initialize natural transformation.

        Args:
            source: Source functor F
            target: Target functor G
        """
        self.source = source
        self.target = target

        # Components: η_t for each time point
        self._components: Dict[int, PitchMorphism] = {}

        # Whether naturality holds
        self._is_natural: Optional[bool] = None
        self._naturality_violations: List[Tuple[int, int]] = []

    def component_at(self, t: int) -> Optional[PitchMorphism]:
        """Get component η_t."""
        return self._components.get(t)

    def set_component(self, t: int, morphism: PitchMorphism):
        """Set component η_t."""
        self._components[t] = morphism
        self._is_natural = None  # Invalidate cache

    @property
    def is_natural(self) -> bool:
        """Check if naturality condition holds."""
        if self._is_natural is None:
            self._check_naturality()
        return self._is_natural

    def _check_naturality(self):
        """
        Verify naturality condition for all consecutive time pairs.

        For each pair (t₁, t₂) with interval f: t₁ → t₂,
        check that G(f) ∘ η_{t₁} = η_{t₂} ∘ F(f)
        """
        self._naturality_violations = []

        # Get common time points
        source_times = set(self.source.time_points)
        target_times = set(self.target.time_points)
        common_times = sorted(source_times & target_times)

        if len(common_times) < 2:
            self._is_natural = True
            return

        # Check consecutive pairs
        for i in range(len(common_times) - 1):
            t1, t2 = common_times[i], common_times[i + 1]

            eta_t1 = self._components.get(t1)
            eta_t2 = self._components.get(t2)

            if eta_t1 is None or eta_t2 is None:
                continue

            # F(f) and G(f)
            interval = TimeInterval(t1, t2)
            f_morphism = self.source.morphism(interval)
            g_morphism = self.target.morphism(interval)

            # Check: G(f) ∘ η_{t₁} = η_{t₂} ∘ F(f)
            # Compose and compare (approximate)
            left = g_morphism.compose(eta_t1)
            right = eta_t2.compose(f_morphism)

            if left.transform_type != right.transform_type or left.transform_id != right.transform_id:
                self._naturality_violations.append((t1, t2))

        self._is_natural = len(self._naturality_violations) == 0

    @classmethod
    def discover(
        cls,
        source: TrackFunctor,
        target: TrackFunctor
    ) -> 'NaturalTransform':
        """
        Discover natural transformation between two tracks.

        Args:
            source: Source track functor F
            target: Target track functor G

        Returns:
            NaturalTransform (possibly non-natural if no consistent transform exists)
        """
        eta = cls(source, target)

        # Get common time points
        source_times = set(source.time_points)
        target_times = set(target.time_points)
        common_times = sorted(source_times & target_times)

        if not common_times:
            return eta

        # For each common time point, find the morphism
        for t in common_times:
            source_config = source(TimePoint(t))
            target_config = target(TimePoint(t))

            if source_config is None or target_config is None:
                continue

            # Find morphism between configurations
            morphism = source._compute_morphism(source_config, target_config)
            eta.set_component(t, morphism)

        # Check naturality
        eta._check_naturality()

        return eta

    def consistency_score(self) -> float:
        """
        Compute how "natural" this transformation is (0-1).

        1.0 = fully natural (all squares commute)
        0.0 = highly inconsistent
        """
        if len(self._components) <= 1:
            return 1.0

        if self._is_natural is None:
            self._check_naturality()

        n_pairs = max(1, len(self._components) - 1)
        n_violations = len(self._naturality_violations)

        return 1.0 - (n_violations / n_pairs)

    def __repr__(self):
        natural_str = "natural" if self.is_natural else "non-natural"
        return f"NaturalTransform({self.source.track_id}⇒{self.target.track_id}, {natural_str})"


# =============================================================================
# MULTI-TRACK SPACE
# =============================================================================

class UnificationMode(Enum):
    """Mode for track component linking."""
    UNIFIED = auto()    # Changes propagate between linked tracks
    SEPARATED = auto()  # Tracks evolve independently


@dataclass
class ComponentLink:
    """A link between track components."""
    track_a: str
    track_b: str
    component: str  # 'pitch', 'rhythm', 'velocity'
    transform: Optional[PitchMorphism] = None  # Transform from A to B


class MultiTrackSpace:
    """
    Categorical product of track functors with selective unification.

    Provides two views:
    1. Unified: Limit in product category (changes propagate)
    2. Separated: Work in individual factor categories (independent)

    Usage:
        space = MultiTrackSpace()
        space.add_track(track_functor)
        space.unify('piano', 'strings', 'pitch')  # Link pitch components

        # Changes to piano pitch now propagate to strings
        space.propagate_change('piano', 'pitch', transpose_5)
    """

    def __init__(self):
        """Initialize empty multi-track space."""
        self.tracks: Dict[str, TrackFunctor] = {}
        self.links: List[ComponentLink] = []
        self.natural_transforms: Dict[Tuple[str, str], NaturalTransform] = {}

    def add_track(self, functor: TrackFunctor):
        """Add a track functor to the space."""
        self.tracks[functor.track_id] = functor

    def get_track(self, track_id: str) -> Optional[TrackFunctor]:
        """Get track by ID."""
        return self.tracks.get(track_id)

    def unify(self, track_a: str, track_b: str, component: str):
        """
        Unify a specific component between two tracks.

        After unification, changes to this component in track_a
        will propagate to track_b (and vice versa).

        Args:
            track_a: First track ID
            track_b: Second track ID
            component: 'pitch', 'rhythm', or 'velocity'
        """
        if track_a not in self.tracks or track_b not in self.tracks:
            raise ValueError(f"Track not found: {track_a} or {track_b}")

        # Discover the natural transformation between them
        eta = NaturalTransform.discover(
            self.tracks[track_a],
            self.tracks[track_b]
        )

        # Store the transformation
        self.natural_transforms[(track_a, track_b)] = eta

        # Create the link
        # Use the most common transform component as the link transform
        transforms = list(eta._components.values())
        if transforms:
            common_transform = max(set(transforms), key=transforms.count)
        else:
            common_transform = PitchMorphism('identity', 0)

        link = ComponentLink(
            track_a=track_a,
            track_b=track_b,
            component=component,
            transform=common_transform
        )
        self.links.append(link)

    def separate(self, track_a: str, track_b: str, component: str):
        """
        Separate a specific component between tracks.

        Removes the unification, so changes no longer propagate.
        """
        self.links = [
            link for link in self.links
            if not (link.track_a == track_a and link.track_b == track_b and
                    link.component == component)
        ]

        key = (track_a, track_b)
        if key in self.natural_transforms:
            del self.natural_transforms[key]

    def get_linked_tracks(self, track_id: str, component: str) -> List[Tuple[str, PitchMorphism]]:
        """
        Get all tracks linked to the given track for a component.

        Returns list of (track_id, transform) pairs.
        """
        linked = []

        for link in self.links:
            if link.component != component:
                continue

            if link.track_a == track_id:
                linked.append((link.track_b, link.transform))
            elif link.track_b == track_id:
                # Compute inverse transform
                inverse = PitchMorphism(
                    link.transform.transform_type,
                    link.transform.transform_id  # Approximate inverse
                )
                linked.append((link.track_a, inverse))

        return linked

    def propagate_change(
        self,
        source_track: str,
        component: str,
        change: PitchMorphism,
        apply_fn: Optional[Callable[[str, PitchMorphism], None]] = None
    ) -> List[str]:
        """
        Propagate a change through unified components.

        Args:
            source_track: Track where change originates
            component: Which component changed
            change: The transformation applied
            apply_fn: Optional function to apply the change (track_id, morphism)

        Returns:
            List of affected track IDs
        """
        affected = [source_track]
        visited = {source_track}

        # BFS through linked tracks
        queue = [(source_track, change)]

        while queue:
            current_track, current_change = queue.pop(0)

            for linked_id, link_transform in self.get_linked_tracks(current_track, component):
                if linked_id in visited:
                    continue

                visited.add(linked_id)
                affected.append(linked_id)

                # Compose the change with the link transform
                propagated_change = link_transform.compose(current_change)

                # Apply if function provided
                if apply_fn:
                    apply_fn(linked_id, propagated_change)

                # Continue propagation
                queue.append((linked_id, propagated_change))

        return affected

    def get_unified_view(self) -> Dict[str, Set[str]]:
        """
        Get the unified view: groups of tracks linked by component.

        Returns:
            Dict mapping component -> set of track IDs that are unified
        """
        from collections import defaultdict

        components = defaultdict(set)

        for link in self.links:
            components[link.component].add(link.track_a)
            components[link.component].add(link.track_b)

        return dict(components)

    def get_separation_graph(self) -> Dict[str, List[str]]:
        """
        Get tracks that are NOT linked (separated).

        Returns:
            Dict mapping track_id -> list of non-linked track_ids
        """
        all_tracks = set(self.tracks.keys())
        linked_pairs = {(l.track_a, l.track_b) for l in self.links}
        linked_pairs.update({(l.track_b, l.track_a) for l in self.links})

        separated = {}
        for track in all_tracks:
            not_linked = [
                other for other in all_tracks
                if other != track and (track, other) not in linked_pairs
            ]
            separated[track] = not_linked

        return separated

    def discover_all_natural_transforms(self) -> List[NaturalTransform]:
        """
        Discover natural transformations between all track pairs.

        Returns:
            List of NaturalTransform objects
        """
        transforms = []
        track_ids = list(self.tracks.keys())

        for i, track_a in enumerate(track_ids):
            for track_b in track_ids[i+1:]:
                eta = NaturalTransform.discover(
                    self.tracks[track_a],
                    self.tracks[track_b]
                )
                transforms.append(eta)
                self.natural_transforms[(track_a, track_b)] = eta

        return transforms

    def __repr__(self):
        return f"MultiTrackSpace({len(self.tracks)} tracks, {len(self.links)} links)"
