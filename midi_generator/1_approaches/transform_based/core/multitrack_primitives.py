"""
Multitrack primitive transforms for cross-track relationship discovery.

These primitives are different from single-object transforms:
- They compare ENTIRE tracks or track segments
- They discover arranging relationships like "trombone = trumpet - fifth"
- They require access to multiple objects simultaneously

Author: Agent - Multitrack Primitives
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CrossTrackRelationship:
    """
    A discovered relationship between two tracks.

    Example:
        source_track: "trumpet"
        target_track: "trombone"
        transform: "transpose_semitone(-7)"
        error: 0.002
        meaning: "Trombone plays trumpet part, fifth below"
    """
    piece_id: str
    source_track_id: str
    target_track_id: str
    transform_name: str
    transform_amount: float
    error: float
    section: Optional[str] = None  # For section-aware derivations


def find_cross_track_relationships(
    piece_objects: List,
    transforms: List[Dict],
    max_error: float = 0.03,
    same_section_only: bool = False
) -> List[CrossTrackRelationship]:
    """
    Find cross-track relationships within a piece.

    This discovers patterns like:
    - "Trombone = Trumpet - fifth"
    - "Sax doubles trumpet octave up"
    - "Bass plays piano left hand root notes"

    Args:
        piece_objects: All objects from a single piece
        transforms: Available transforms to test
        max_error: Maximum error for valid relationship
        same_section_only: Only compare within same time range

    Returns:
        List of discovered cross-track relationships
    """
    from core.numpy_transforms import NumpyTransformLibrary
    lib = NumpyTransformLibrary()

    relationships = []

    # Group objects by track
    objects_by_track = {}
    for obj in piece_objects:
        if obj.track_id not in objects_by_track:
            objects_by_track[obj.track_id] = []
        objects_by_track[obj.track_id].append(obj)

    # Get piece ID (same for all objects)
    piece_id = piece_objects[0].piece_id if piece_objects else None
    if not piece_id:
        return relationships

    # Compare each pair of tracks
    track_ids = list(objects_by_track.keys())
    for i, source_track_id in enumerate(track_ids):
        for target_track_id in track_ids[i+1:]:  # Avoid duplicates
            # Get objects from each track
            source_objs = objects_by_track[source_track_id]
            target_objs = objects_by_track[target_track_id]

            # Find best transform relating these tracks
            best_transform_name = None
            best_transform_amount = None
            best_error = float('inf')
            match_count = 0

            # Try each transform
            for transform in transforms:
                total_error = 0.0
                valid_comparisons = 0

                # Compare overlapping segments
                for source_obj in source_objs:
                    for target_obj in target_objs:
                        # Check if they overlap in time (same section)
                        if same_section_only:
                            # Require temporal overlap or adjacency
                            overlap = (
                                (source_obj.start_time <= target_obj.start_time < source_obj.end_time) or
                                (target_obj.start_time <= source_obj.start_time < target_obj.end_time) or
                                (abs(source_obj.start_time - target_obj.start_time) < 16)  # Within 1 bar
                            )
                            if not overlap:
                                continue

                        # Check if same duration (for valid comparison)
                        source_dur = source_obj.end_time - source_obj.start_time
                        target_dur = target_obj.end_time - target_obj.start_time
                        if source_dur != target_dur:
                            continue

                        try:
                            # Apply transform to source
                            source_expanded = np.expand_dims(source_obj.tensor, 0)
                            transformed = lib.apply_transform(
                                source_expanded,
                                transform['name'],
                                transform['amount']
                            )[0]

                            # Compare with target
                            error = np.mean((target_obj.tensor - transformed) ** 2)
                            total_error += error
                            valid_comparisons += 1

                        except Exception:
                            continue

                # Average error across all comparisons
                if valid_comparisons > 0:
                    avg_error = total_error / valid_comparisons

                    if avg_error < best_error:
                        best_error = avg_error
                        best_transform_name = transform['name']
                        best_transform_amount = transform['amount']
                        match_count = valid_comparisons

            # Record if good match found
            if best_error < max_error and match_count >= 2:  # At least 2 matching segments
                relationships.append(CrossTrackRelationship(
                    piece_id=piece_id,
                    source_track_id=source_track_id,
                    target_track_id=target_track_id,
                    transform_name=best_transform_name,
                    transform_amount=best_transform_amount,
                    error=best_error
                ))

                # Also add reverse relationship
                # (If trombone = trumpet - 5th, then trumpet = trombone + 5th)
                # But we need to compute the inverse transform
                # For simplicity, we'll discover both directions independently

    return relationships


def get_common_instruments():
    """
    Get list of common big band instruments for filtering.

    Returns:
        List of instrument name patterns
    """
    return [
        'trumpet', 'trombone', 'sax', 'saxophone',
        'bass', 'drums', 'percussion', 'piano',
        'guitar', 'clarinet', 'flute',
        'brass', 'reed', 'rhythm'
    ]


def identify_track_instrument(track_id: str) -> str:
    """
    Identify instrument type from track ID.

    Args:
        track_id: Track identifier string

    Returns:
        Instrument category or 'unknown'
    """
    track_lower = track_id.lower()

    instruments = get_common_instruments()
    for inst in instruments:
        if inst in track_lower:
            return inst

    # Check for generic patterns
    if 'inst_' in track_lower:
        return 'melodic'
    if 'track_' in track_lower:
        return 'unknown'

    return 'unknown'


def create_track_derive_primitive(source_track: str, target_track: str) -> Dict:
    """
    Create a TrackDerive primitive specification.

    This is a meta-primitive that encodes cross-track relationships.

    Args:
        source_track: Source track name/pattern
        target_track: Target track name/pattern

    Returns:
        Primitive specification dict
    """
    return {
        'name': 'track_derive',
        'source_track': source_track,
        'target_track': target_track,
        'is_multitrack': True
    }


def format_cross_track_relationship(rel: CrossTrackRelationship) -> str:
    """
    Format a cross-track relationship as a readable string.

    Args:
        rel: CrossTrackRelationship to format

    Returns:
        Human-readable string

    Example:
        "trombone = transpose_semitone(-7)(trumpet)  # Fifth below"
    """
    # Get instrument names
    source_inst = identify_track_instrument(rel.source_track_id)
    target_inst = identify_track_instrument(rel.target_track_id)

    # Format transform
    transform_str = f"{rel.transform_name}({rel.transform_amount})"

    # Create readable description
    description = f"{target_inst} = {transform_str}({source_inst})"

    # Add musical interpretation
    if rel.transform_name == 'transpose_semitone':
        interval = int(rel.transform_amount)
        if interval == 7:
            description += "  # Fifth up"
        elif interval == -7:
            description += "  # Fifth down"
        elif interval == 12:
            description += "  # Octave up"
        elif interval == -12:
            description += "  # Octave down"
        elif interval == 5:
            description += "  # Fourth up"
        elif interval == -5:
            description += "  # Fourth down"
        elif interval == 3:
            description += "  # Minor third up"
        elif interval == -3:
            description += "  # Minor third down"

    return description
