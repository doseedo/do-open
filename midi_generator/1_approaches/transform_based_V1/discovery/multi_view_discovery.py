"""
Multi-View Pattern Discovery with MDL Vocabulary Selection

Architecture:
  Phase 0: Multi-view pair generation (orchestration, temporal, full, cross-piece)
  Phase 1: Compound mining per view (GPU batched)
  Phase 2: MDL vocabulary selection (unified across views)
  Phase 3: Full derivation with expanded vocabulary

The key insight: different "views" ask different musical questions.
- Same-time cross-track: "How do instruments relate at this moment?" (orchestration)
- Same-track cross-time: "How does this part evolve?" (temporal structure)
- Cross-track cross-time: "Call and response patterns?" (interaction)
- Cross-piece same-instrument: "Stylistic idioms?" (composer voice)

MDL unifies discoveries across all views.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from pathlib import Path
import json
import time
from enum import Enum, auto


class ViewType(Enum):
    """The four musical views for pair generation."""
    SAME_TIME_CROSS_TRACK = auto()    # Orchestration: instruments at same moment
    SAME_TRACK_CROSS_TIME = auto()    # Temporal: evolution within a part
    CROSS_TRACK_CROSS_TIME = auto()   # Full: any object to any object
    CROSS_PIECE_SAME_INSTRUMENT = auto()  # Style: same instrument across pieces


@dataclass
class MusicalObject:
    """A segment from a specific track at a specific time."""
    piece_id: str
    track_id: str
    bar_start: int
    bar_end: int
    pitch_classes: torch.Tensor  # [12] pitch class vector
    intervals: torch.Tensor      # [12] interval class vector
    contour: torch.Tensor        # [8] melodic contour
    rhythm: torch.Tensor         # [16] rhythm pattern

    # Metadata for view-based pairing
    instrument_family: str = ""  # brass, reeds, rhythm, strings, etc.

    @property
    def feature_vector(self) -> torch.Tensor:
        """Combined feature vector for compound testing."""
        return torch.cat([self.pitch_classes, self.intervals, self.contour, self.rhythm])

    @property
    def uid(self) -> str:
        """Unique identifier."""
        return f"{self.piece_id}:{self.track_id}:{self.bar_start}-{self.bar_end}"


@dataclass
class Transform:
    """A Lewinian transformation."""
    name: str
    param: int

    def apply_to_pitch_classes(self, pc: torch.Tensor) -> torch.Tensor:
        """Apply transform to pitch class vector."""
        if self.name == "T":  # Transposition
            return torch.roll(pc, shifts=self.param)
        elif self.name == "I":  # Inversion (around axis)
            # Invert then transpose to axis
            inverted = torch.flip(pc, dims=[0])
            return torch.roll(inverted, shifts=self.param)
        elif self.name == "M":  # Multiplication (M5, M7)
            # Multiplicative transform mod 12
            indices = torch.arange(12) * self.param % 12
            return pc[indices]
        return pc

    def apply(self, obj: torch.Tensor) -> torch.Tensor:
        """Apply transform to feature vector."""
        # Only transforms pitch-related features (first 24 dims)
        result = obj.clone()
        result[:12] = self.apply_to_pitch_classes(obj[:12])
        # Intervals transform similarly for transposition
        if self.name == "T":
            pass  # Intervals invariant under transposition
        elif self.name == "I":
            result[12:24] = torch.flip(obj[12:24], dims=[0])
        return result

    def __repr__(self):
        return f"{self.name}({self.param})"


@dataclass
class Compound:
    """A compound transformation (sequence of primitives)."""
    transforms: Tuple[Transform, ...]

    def apply(self, obj: torch.Tensor) -> torch.Tensor:
        """Apply compound to feature vector."""
        result = obj
        for t in self.transforms:
            result = t.apply(result)
        return result

    @property
    def name(self) -> str:
        if not self.transforms:
            return "id"
        return " . ".join(str(t) for t in self.transforms)

    @property
    def cost(self) -> float:
        """MDL cost of this compound."""
        if not self.transforms:
            return 0.0
        # Base cost per transform + parameter costs
        cost = 0.0
        for t in self.transforms:
            cost += 1.0  # 1 bit for transform type
            if t.name == "T":
                cost += 3.58  # log2(12) for transposition
            elif t.name == "I":
                cost += 3.58  # log2(12) for inversion axis
            elif t.name == "M":
                cost += 1.0   # log2(2) for M5 or M7
        return cost

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


@dataclass
class ViewPairs:
    """Pairs generated from a specific view."""
    view_type: ViewType
    pairs: List[Tuple[int, int]]  # Indices into object list
    description: str


@dataclass
class CompoundStats:
    """Statistics for a compound across views."""
    compound: Compound
    total_matches: int = 0
    matches_by_view: Dict[ViewType, int] = field(default_factory=dict)
    example_pairs: List[Tuple[str, str]] = field(default_factory=list)  # UIDs
    exemplar_cost: float = 48.0  # Default: 48 dims * 1 bit each (will be updated)

    @property
    def compression_ratio(self) -> float:
        """
        Compression ratio scoring: coverage / (transform_cost + exemplar_cost)

        Higher is better - more coverage per bit of description.
        This is the key MDL insight: we want transforms that explain
        many objects with minimal description cost.
        """
        if self.total_matches < 2:
            return 0.0

        # Transform cost: bits to describe the transform
        transform_bits = self.compound.cost
        if transform_bits == 0:  # Identity
            transform_bits = 0.1  # Small cost for identity

        # Total cost = transform + one exemplar
        total_cost = transform_bits + self.exemplar_cost

        # Coverage = number of objects explained
        coverage = self.total_matches

        return coverage / total_cost

    @property
    def stitch_utility(self) -> float:
        """
        Stitch-style utility: abstraction_size × num_uses

        Balances generality (many uses) vs specificity (captures structure).
        Larger transforms that are used often get higher scores.
        """
        # Size = number of operations (how much structure it captures)
        transform_size = len(self.compound.transforms) if self.compound.transforms else 0.5

        # Uses = how often it appears
        num_uses = self.total_matches

        return transform_size * num_uses

    @property
    def mdl_benefit(self) -> float:
        """Combined MDL benefit using compression ratio."""
        return self.compression_ratio


class MultiViewDiscovery:
    """
    Multi-view pattern discovery system.

    Generates pairs from multiple musical perspectives, mines compounds
    on each view, then unifies via MDL.
    """

    def __init__(
        self,
        device: str = "cuda",
        match_threshold: float = 0.15,
        samples_per_view: int = 50000,
        batch_size: int = 4096,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.match_threshold = match_threshold
        self.samples_per_view = samples_per_view
        self.batch_size = batch_size

        # Will be populated
        self.objects: List[MusicalObject] = []
        self.object_features: torch.Tensor = None  # [N, D] on GPU

        # Index structures for view-based pairing
        self.by_piece: Dict[str, List[int]] = defaultdict(list)
        self.by_track: Dict[str, List[int]] = defaultdict(list)
        self.by_piece_bar: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        self.by_instrument: Dict[str, List[int]] = defaultdict(list)

        # Results
        self.compound_stats: Dict[str, CompoundStats] = {}
        self.vocabulary: List[Compound] = []

    def build_primitive_compounds(self) -> List[Compound]:
        """Build the set of compounds to test."""
        compounds = [Compound(tuple())]  # Identity

        # Single transforms
        for i in range(12):
            compounds.append(Compound((Transform("T", i),)))
        for i in range(12):
            compounds.append(Compound((Transform("I", i),)))
        compounds.append(Compound((Transform("M", 5),)))
        compounds.append(Compound((Transform("M", 7),)))

        # Common compound patterns
        # T then I (retrograde-inversion patterns)
        for t in [0, 3, 4, 5, 7]:  # Common transpositions
            for inv in [0, 6]:  # Common inversion axes
                compounds.append(Compound((Transform("T", t), Transform("I", inv))))

        # I then T
        for inv in [0, 6]:
            for t in [0, 3, 4, 5, 7]:
                compounds.append(Compound((Transform("I", inv), Transform("T", t))))

        return compounds

    def add_objects(self, objects: List[MusicalObject]):
        """Add musical objects and build indices."""
        start_idx = len(self.objects)
        self.objects.extend(objects)

        for i, obj in enumerate(objects):
            idx = start_idx + i
            self.by_piece[obj.piece_id].append(idx)
            self.by_track[f"{obj.piece_id}:{obj.track_id}"].append(idx)
            self.by_piece_bar[(obj.piece_id, obj.bar_start)].append(idx)
            self.by_instrument[obj.instrument_family].append(idx)

        # Rebuild feature tensor
        if self.objects:
            features = torch.stack([obj.feature_vector for obj in self.objects])
            self.object_features = features.to(self.device)

    # =========================================================================
    # PHASE 0: Multi-View Pair Generation
    # =========================================================================

    def generate_view_pairs(self, view_type: ViewType) -> ViewPairs:
        """Generate pairs for a specific view."""
        if view_type == ViewType.SAME_TIME_CROSS_TRACK:
            return self._generate_orchestration_pairs()
        elif view_type == ViewType.SAME_TRACK_CROSS_TIME:
            return self._generate_temporal_pairs()
        elif view_type == ViewType.CROSS_TRACK_CROSS_TIME:
            return self._generate_full_pairs()
        elif view_type == ViewType.CROSS_PIECE_SAME_INSTRUMENT:
            return self._generate_style_pairs()
        else:
            raise ValueError(f"Unknown view type: {view_type}")

    def _generate_orchestration_pairs(self) -> ViewPairs:
        """View 1: Same-timestep, cross-track pairs."""
        pairs = []

        for (piece_id, bar), indices in self.by_piece_bar.items():
            if len(indices) < 2:
                continue
            # All pairs of different tracks at same time
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_i, idx_j = indices[i], indices[j]
                    # Only if different tracks
                    if self.objects[idx_i].track_id != self.objects[idx_j].track_id:
                        pairs.append((idx_i, idx_j))

        # Sample if too many
        if len(pairs) > self.samples_per_view:
            indices = np.random.choice(len(pairs), self.samples_per_view, replace=False)
            pairs = [pairs[i] for i in indices]

        return ViewPairs(
            view_type=ViewType.SAME_TIME_CROSS_TRACK,
            pairs=pairs,
            description=f"Orchestration pairs (same time, different tracks): {len(pairs)}"
        )

    def _generate_temporal_pairs(self) -> ViewPairs:
        """View 2: Same-track, cross-time pairs."""
        pairs = []

        for track_key, indices in self.by_track.items():
            if len(indices) < 2:
                continue
            # Sort by bar position
            sorted_indices = sorted(indices, key=lambda i: self.objects[i].bar_start)

            # Pairs within reasonable distance (not too far apart)
            max_distance = 32  # bars
            for i in range(len(sorted_indices)):
                for j in range(i + 1, len(sorted_indices)):
                    idx_i, idx_j = sorted_indices[i], sorted_indices[j]
                    bar_dist = abs(self.objects[idx_j].bar_start - self.objects[idx_i].bar_start)
                    if bar_dist <= max_distance:
                        pairs.append((idx_i, idx_j))

        # Sample if too many
        if len(pairs) > self.samples_per_view:
            indices = np.random.choice(len(pairs), self.samples_per_view, replace=False)
            pairs = [pairs[i] for i in indices]

        return ViewPairs(
            view_type=ViewType.SAME_TRACK_CROSS_TIME,
            pairs=pairs,
            description=f"Temporal pairs (same track, different times): {len(pairs)}"
        )

    def _generate_full_pairs(self) -> ViewPairs:
        """View 3: Cross-track, cross-time pairs within piece."""
        pairs = []

        for piece_id, indices in self.by_piece.items():
            if len(indices) < 2:
                continue

            # Sample pairs from this piece
            n = len(indices)
            max_pairs_per_piece = self.samples_per_view // max(1, len(self.by_piece))

            if n * (n - 1) // 2 <= max_pairs_per_piece:
                # Take all pairs
                for i in range(n):
                    for j in range(i + 1, n):
                        pairs.append((indices[i], indices[j]))
            else:
                # Random sampling
                for _ in range(max_pairs_per_piece):
                    i, j = np.random.choice(n, 2, replace=False)
                    pairs.append((indices[i], indices[j]))

        # Final sample if needed
        if len(pairs) > self.samples_per_view:
            sample_indices = np.random.choice(len(pairs), self.samples_per_view, replace=False)
            pairs = [pairs[i] for i in sample_indices]

        return ViewPairs(
            view_type=ViewType.CROSS_TRACK_CROSS_TIME,
            pairs=pairs,
            description=f"Full pairs (cross-track, cross-time): {len(pairs)}"
        )

    def _generate_style_pairs(self) -> ViewPairs:
        """View 4: Cross-piece, same-instrument pairs."""
        pairs = []

        for instrument, indices in self.by_instrument.items():
            if len(indices) < 2:
                continue

            # Group by piece
            by_piece_for_inst = defaultdict(list)
            for idx in indices:
                by_piece_for_inst[self.objects[idx].piece_id].append(idx)

            piece_ids = list(by_piece_for_inst.keys())
            if len(piece_ids) < 2:
                continue

            # Sample cross-piece pairs
            max_pairs_per_inst = self.samples_per_view // max(1, len(self.by_instrument))

            for _ in range(min(max_pairs_per_inst, len(indices) * 10)):
                # Pick two different pieces
                p1, p2 = np.random.choice(len(piece_ids), 2, replace=False)
                piece1, piece2 = piece_ids[p1], piece_ids[p2]

                # Pick one object from each
                idx1 = np.random.choice(by_piece_for_inst[piece1])
                idx2 = np.random.choice(by_piece_for_inst[piece2])
                pairs.append((idx1, idx2))

        # Final sample if needed
        if len(pairs) > self.samples_per_view:
            sample_indices = np.random.choice(len(pairs), self.samples_per_view, replace=False)
            pairs = [pairs[i] for i in sample_indices]

        return ViewPairs(
            view_type=ViewType.CROSS_PIECE_SAME_INSTRUMENT,
            pairs=pairs,
            description=f"Style pairs (same instrument, cross-piece): {len(pairs)}"
        )

    # =========================================================================
    # PHASE 1: Compound Mining (GPU Batched)
    # =========================================================================

    def mine_compounds_for_view(
        self,
        view_pairs: ViewPairs,
        compounds: List[Compound]
    ) -> Dict[str, int]:
        """Mine compound frequencies for a specific view."""
        print(f"\n  Mining {view_pairs.description}")

        if not view_pairs.pairs:
            return {}

        # Convert pairs to tensors
        pairs_array = np.array(view_pairs.pairs)
        source_indices = torch.tensor(pairs_array[:, 0], device=self.device)
        target_indices = torch.tensor(pairs_array[:, 1], device=self.device)

        # Get features
        source_features = self.object_features[source_indices]  # [P, D]
        target_features = self.object_features[target_indices]  # [P, D]

        compound_matches = defaultdict(int)

        # Process in batches
        n_pairs = len(view_pairs.pairs)

        for compound in compounds:
            # Apply compound to all sources
            if not compound.transforms:
                transformed = source_features
            else:
                # Apply on CPU for complex transforms, could optimize later
                transformed = torch.stack([
                    compound.apply(source_features[i])
                    for i in range(n_pairs)
                ])

            # Compute distances
            # Normalize for cosine-like comparison
            transformed_norm = F.normalize(transformed, dim=1)
            target_norm = F.normalize(target_features, dim=1)

            # Cosine similarity
            similarity = (transformed_norm * target_norm).sum(dim=1)

            # Count matches
            matches = (similarity > (1 - self.match_threshold)).sum().item()

            if matches > 0:
                compound_matches[compound.name] = matches

        return dict(compound_matches)

    # =========================================================================
    # PHASE 2: MDL Vocabulary Selection (with Stitch utility)
    # =========================================================================

    def select_vocabulary(
        self,
        min_compression_ratio: float = 0.01,
        min_stitch_utility: float = 2.0,
        max_vocab_size: int = 50
    ) -> List[Compound]:
        """
        Select vocabulary using combined MDL criteria:
        1. Compression ratio: coverage / (transform_cost + exemplar_cost)
        2. Stitch utility: abstraction_size × num_uses

        A compound is selected if it passes BOTH thresholds.
        """
        print("\n" + "="*60)
        print("PHASE 2: MDL Vocabulary Selection")
        print("="*60)
        print(f"  Min compression ratio: {min_compression_ratio}")
        print(f"  Min Stitch utility: {min_stitch_utility}")

        # Filter by both criteria
        candidates = []
        for stats in self.compound_stats.values():
            cr = stats.compression_ratio
            su = stats.stitch_utility

            if cr >= min_compression_ratio and su >= min_stitch_utility:
                # Combined score: geometric mean of both metrics
                combined_score = (cr * su) ** 0.5
                candidates.append((stats, combined_score, cr, su))

        # Sort by combined score
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Take top candidates up to max vocab size
        vocabulary = []
        total_compression = 0.0

        print(f"\n  {'Compound':<30} {'Matches':>8} {'CR':>8} {'Utility':>8} {'Score':>8}")
        print("  " + "-" * 70)

        for stats, score, cr, su in candidates[:max_vocab_size]:
            vocabulary.append(stats.compound)
            total_compression += cr

            # Show which views contributed
            view_breakdown = ", ".join(
                f"{v.name.split('_')[0][:4]}: {c}"
                for v, c in stats.matches_by_view.items()
            )
            print(f"  {stats.compound.name:<30} {stats.total_matches:>8} "
                  f"{cr:>8.3f} {su:>8.1f} {score:>8.3f}")

        print(f"\nSelected {len(vocabulary)} compounds")
        print(f"Total compression ratio: {total_compression:.2f}")

        # Also show rejected compounds with high matches but low scores
        rejected_high_match = [
            s for s in self.compound_stats.values()
            if s.total_matches > 50 and s not in [c[0] for c in candidates[:max_vocab_size]]
        ]
        if rejected_high_match:
            print(f"\nRejected (high matches but low scores):")
            for stats in sorted(rejected_high_match, key=lambda s: s.total_matches, reverse=True)[:5]:
                print(f"  {stats.compound.name}: {stats.total_matches} matches, "
                      f"CR={stats.compression_ratio:.3f}, utility={stats.stitch_utility:.1f}")

        self.vocabulary = vocabulary
        return vocabulary

    # =========================================================================
    # PHASE 3: Greedy Covering Assignment (COSIATEC-style)
    # =========================================================================

    def greedy_cover_assignment(self) -> Dict[int, Tuple[Compound, int, float]]:
        """
        COSIATEC-style greedy covering assignment.

        Instead of assigning each object independently, we iteratively:
        1. Find the transform that best covers REMAINING objects
        2. Assign those objects and remove from consideration
        3. Repeat until no beneficial transforms remain

        Returns: {target_idx: (compound, source_idx, similarity)}
        """
        print("\n" + "="*60)
        print("PHASE 3: Greedy Covering Assignment")
        print("="*60)

        if not self.vocabulary:
            print("  No vocabulary selected!")
            return {}

        n = len(self.objects)
        remaining = set(range(n))  # Objects not yet assigned
        assignments = {}  # target_idx -> (compound, source_idx, similarity)

        iteration = 0
        max_iterations = 100  # Safety limit
        assigned = set()  # Track assigned objects for source optimization

        while remaining and iteration < max_iterations:
            iteration += 1

            # Find best transform for remaining objects
            best_compound = None
            best_score = 0
            best_matches = []  # [(source_idx, target_idx, similarity), ...]

            for compound in self.vocabulary:
                # Find matches among remaining objects
                # Use assigned objects as sources once we have some
                matches = self._find_matches_for_remaining(
                    compound, remaining,
                    assigned if len(assigned) > 10 else None
                )

                if not matches:
                    continue

                # Score = coverage / (transform_cost + exemplar_cost)
                transform_cost = compound.cost if compound.cost > 0 else 0.1
                exemplar_cost = 48.0  # Feature vector size
                score = len(matches) / (transform_cost + exemplar_cost)

                if score > best_score:
                    best_score = score
                    best_compound = compound
                    best_matches = matches

            if not best_matches or best_score < 0.01:
                break  # No more useful transforms

            # Assign and remove from consideration
            for source_idx, target_idx, similarity in best_matches:
                if target_idx in remaining:
                    assignments[target_idx] = (best_compound, source_idx, similarity)
                    remaining.discard(target_idx)
                    assigned.add(target_idx)  # Now this can serve as a source

            if iteration % 10 == 0 or len(best_matches) > 100:
                print(f"  Iteration {iteration}: {best_compound.name} "
                      f"assigned {len(best_matches)} objects, "
                      f"{len(remaining)} remaining")

        # Remaining objects: no good transform found
        literals = len(remaining)
        print(f"\n  Assigned {len(assignments)} objects via transforms")
        print(f"  {literals} objects remain as literals (no beneficial transform)")

        return assignments

    def _find_matches_for_remaining(
        self,
        compound: Compound,
        remaining: Set[int],
        assigned: Optional[Set[int]] = None
    ) -> List[Tuple[int, int, float]]:
        """
        Find all (source, target, similarity) matches for remaining objects.

        Sources can be ANY object (including already-assigned ones, since they
        can serve as exemplars). Targets are only REMAINING objects.

        Optimization: if assigned set is large, we only use assigned objects
        as potential sources (they're the exemplars).
        """
        matches = []

        remaining_list = list(remaining)
        if not remaining_list:
            return matches

        # Get features for remaining objects (targets)
        remaining_indices = torch.tensor(remaining_list, device=self.device)
        remaining_features = self.object_features[remaining_indices]  # [R, D]

        # Determine source set
        # If we have assigned objects, prefer them as sources (they're exemplars)
        # Otherwise use all objects
        if assigned and len(assigned) > 0:
            source_list = list(assigned)
        else:
            source_list = list(range(len(self.objects)))

        source_indices_tensor = torch.tensor(source_list, device=self.device)
        source_features = self.object_features[source_indices_tensor]  # [S, D]

        # Apply compound to sources
        if not compound.transforms:
            transformed = source_features
        else:
            transformed = torch.stack([
                compound.apply(source_features[i])
                for i in range(len(source_list))
            ])

        # Normalize
        transformed_norm = F.normalize(transformed, dim=1)  # [S, D]
        remaining_norm = F.normalize(remaining_features, dim=1)  # [R, D]

        # Similarity: [S, R] - each source to each remaining target
        similarities = torch.mm(transformed_norm, remaining_norm.t())

        # Find matches above threshold
        match_mask = similarities > (1 - self.match_threshold)

        # Extract matches
        source_local, target_local = match_mask.nonzero(as_tuple=True)

        for i in range(len(source_local)):
            source_idx = source_list[source_local[i].item()]
            target_idx = remaining_list[target_local[i].item()]
            sim = similarities[source_local[i], target_local[i]].item()

            # Don't match object to itself
            if source_idx != target_idx:
                matches.append((source_idx, target_idx, sim))

        return matches

    def derive_all(self) -> Dict[Tuple[int, int], Compound]:
        """
        Legacy interface - returns derivations as (source, target) -> compound.
        Now uses greedy covering internally.
        """
        assignments = self.greedy_cover_assignment()

        # Convert to legacy format
        derivations = {}
        for target_idx, (compound, source_idx, _) in assignments.items():
            derivations[(source_idx, target_idx)] = compound

        return derivations

    # =========================================================================
    # Main Discovery Pipeline
    # =========================================================================

    def discover(self) -> Dict:
        """Run full multi-view discovery pipeline."""
        print("\n" + "="*60)
        print("MULTI-VIEW PATTERN DISCOVERY")
        print("="*60)
        print(f"Objects: {len(self.objects)}")
        print(f"Device: {self.device}")

        start_time = time.time()

        # Build compounds
        compounds = self.build_primitive_compounds()
        print(f"Testing {len(compounds)} compounds")

        # Phase 0 & 1: Generate pairs and mine for each view
        print("\n" + "="*60)
        print("PHASE 0-1: Multi-View Pair Generation & Compound Mining")
        print("="*60)

        all_view_stats = {}

        for view_type in ViewType:
            # Generate pairs
            view_pairs = self.generate_view_pairs(view_type)
            print(f"\n{view_pairs.description}")

            if not view_pairs.pairs:
                continue

            # Mine compounds
            matches = self.mine_compounds_for_view(view_pairs, compounds)
            all_view_stats[view_type] = matches

            # Show top compounds for this view
            sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)[:10]
            for name, count in sorted_matches:
                print(f"    {name}: {count}")

        # Merge statistics across views
        print("\n" + "="*60)
        print("Merging statistics across views...")

        # Find compound objects
        compound_by_name = {c.name: c for c in compounds}

        for compound_name in set().union(*(set(v.keys()) for v in all_view_stats.values())):
            compound = compound_by_name[compound_name]
            stats = CompoundStats(compound=compound)

            for view_type, matches in all_view_stats.items():
                if compound_name in matches:
                    count = matches[compound_name]
                    stats.matches_by_view[view_type] = count
                    stats.total_matches += count

            self.compound_stats[compound_name] = stats

        # Phase 2: Vocabulary selection
        self.select_vocabulary()

        # Phase 3: Full derivation
        derivations = self.derive_all()

        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed:.1f}s")

        return {
            "vocabulary": [c.name for c in self.vocabulary],
            "compound_stats": {
                name: {
                    "total_matches": s.total_matches,
                    "mdl_benefit": s.mdl_benefit,
                    "views": {v.name: c for v, c in s.matches_by_view.items()}
                }
                for name, s in self.compound_stats.items()
            },
            "derivations_count": len(derivations),
            "elapsed_time": elapsed
        }


# =============================================================================
# MIDI Loading Utilities
# =============================================================================

def extract_objects_from_midi(midi_path: Path, segment_bars: int = 4) -> List[MusicalObject]:
    """
    Extract MusicalObjects from a MIDI file.

    Segments each track into bar-aligned chunks and extracts features.
    """
    try:
        import pretty_midi
    except ImportError:
        print("pretty_midi required: pip install pretty_midi")
        return []

    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return []

    objects = []
    piece_id = midi_path.stem

    # Estimate tempo and bar length
    tempo = midi.estimate_tempo() if midi.estimate_tempo() > 0 else 120
    bar_duration = 4 * 60 / tempo  # Assuming 4/4

    # Get total duration
    total_duration = midi.get_end_time()
    total_bars = int(total_duration / bar_duration) + 1

    for track_idx, instrument in enumerate(midi.instruments):
        if instrument.is_drum:
            continue

        track_id = f"track_{track_idx}"

        # Guess instrument family
        program = instrument.program
        if program < 8:
            family = "piano"
        elif program < 16:
            family = "chromatic_percussion"
        elif program < 24:
            family = "organ"
        elif program < 32:
            family = "guitar"
        elif program < 40:
            family = "bass"
        elif program < 48:
            family = "strings"
        elif program < 56:
            family = "ensemble"
        elif program < 64:
            family = "brass"
        elif program < 72:
            family = "reed"
        elif program < 80:
            family = "pipe"
        elif program < 88:
            family = "synth_lead"
        elif program < 96:
            family = "synth_pad"
        else:
            family = "other"

        # Segment into bars
        for bar_start in range(0, total_bars - segment_bars + 1, segment_bars):
            bar_end = bar_start + segment_bars

            start_time = bar_start * bar_duration
            end_time = bar_end * bar_duration

            # Get notes in this segment
            notes = [n for n in instrument.notes
                    if start_time <= n.start < end_time]

            if not notes:
                continue

            # Extract features
            pitch_classes = torch.zeros(12)
            for n in notes:
                pitch_classes[n.pitch % 12] += n.end - n.start
            if pitch_classes.sum() > 0:
                pitch_classes = pitch_classes / pitch_classes.sum()

            # Interval classes
            intervals = torch.zeros(12)
            sorted_notes = sorted(notes, key=lambda n: n.start)
            for i in range(len(sorted_notes) - 1):
                interval = (sorted_notes[i+1].pitch - sorted_notes[i].pitch) % 12
                intervals[interval] += 1
            if intervals.sum() > 0:
                intervals = intervals / intervals.sum()

            # Melodic contour (simplified)
            contour = torch.zeros(8)
            if len(sorted_notes) > 1:
                pitches = [n.pitch for n in sorted_notes]
                for i in range(min(len(pitches) - 1, 8)):
                    diff = pitches[i+1] - pitches[i]
                    contour[i] = np.tanh(diff / 12)  # Normalize

            # Rhythm pattern (onset positions within segment)
            rhythm = torch.zeros(16)
            for n in notes:
                pos = int((n.start - start_time) / bar_duration * 16) % 16
                rhythm[pos] = 1

            obj = MusicalObject(
                piece_id=piece_id,
                track_id=track_id,
                bar_start=bar_start,
                bar_end=bar_end,
                pitch_classes=pitch_classes,
                intervals=intervals,
                contour=contour,
                rhythm=rhythm,
                instrument_family=family
            )
            objects.append(obj)

    return objects


def load_corpus(corpus_dir: Path, segment_bars: int = 4) -> List[MusicalObject]:
    """Load all MIDI files from a directory."""
    all_objects = []

    midi_files = list(corpus_dir.glob("**/*.mid")) + list(corpus_dir.glob("**/*.midi"))
    print(f"Found {len(midi_files)} MIDI files")

    for midi_path in midi_files:
        objects = extract_objects_from_midi(midi_path, segment_bars)
        all_objects.extend(objects)
        print(f"  {midi_path.name}: {len(objects)} objects")

    return all_objects


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-View Pattern Discovery")
    parser.add_argument("corpus_dir", type=Path, help="Directory with MIDI files")
    parser.add_argument("--segment-bars", type=int, default=4, help="Bars per segment")
    parser.add_argument("--samples-per-view", type=int, default=50000, help="Max pairs per view")
    parser.add_argument("--threshold", type=float, default=0.15, help="Match threshold")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file")

    args = parser.parse_args()

    # Load corpus
    print("Loading corpus...")
    objects = load_corpus(args.corpus_dir, args.segment_bars)
    print(f"Total objects: {len(objects)}")

    if not objects:
        print("No objects extracted!")
        return

    # Run discovery
    discovery = MultiViewDiscovery(
        samples_per_view=args.samples_per_view,
        match_threshold=args.threshold
    )
    discovery.add_objects(objects)

    results = discovery.discover()

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "="*60)
    print("FINAL VOCABULARY")
    print("="*60)
    for compound_name in results["vocabulary"]:
        stats = results["compound_stats"][compound_name]
        print(f"  {compound_name}: {stats['total_matches']} matches, "
              f"benefit={stats['mdl_benefit']:.1f}")


if __name__ == "__main__":
    main()
