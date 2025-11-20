"""
Bidirectional Workflow Coordinator - Agent 09
==============================================

Coordinates bidirectional workflows between MIDI and parameters:
1. MIDI → Parameters (Analysis/Extraction)
2. Parameters → MIDI (Generation)

Also provides advanced workflows:
- Style Transfer: Extract params from MIDI A, apply to MIDI B
- Parameter Modification: Tweak parameters and regenerate
- Comparison: Compare parameters between multiple MIDI files
- Blending: Interpolate parameters between two styles

Author: Agent 09 - HarmonyModule Integration Lead
Date: 2025-11-20
License: MIT
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import mido
import time
import json

# Internal imports
try:
    from midi_generator.integration.parameter_prediction_api import (
        ParameterPredictionAPI,
        ParameterAnalysisResult
    )
    PREDICTION_API_AVAILABLE = True
except ImportError:
    print("Warning: ParameterPredictionAPI not available")
    PREDICTION_API_AVAILABLE = False

try:
    from midi_generator.integration.hierarchical_model_wrapper import HierarchicalPrediction
    MODEL_WRAPPER_AVAILABLE = True
except ImportError:
    print("Warning: HierarchicalPrediction not available")
    MODEL_WRAPPER_AVAILABLE = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class StyleTransferResult:
    """Result from style transfer operation"""
    source_midi: Path  # Where we got parameters from
    target_midi: Path  # What we're applying to
    output_midi: Optional[Path]  # Generated result
    transferred_parameters: Dict[str, Any]
    success: bool
    processing_time_ms: float
    notes: List[str]


@dataclass
class ParameterBlendResult:
    """Result from parameter blending"""
    source_a: str  # First parameter source
    source_b: str  # Second parameter source
    blend_weight: float  # 0.0 = all A, 1.0 = all B
    blended_parameters: Dict[str, Any]
    output_midi: Optional[Path]


@dataclass
class ParameterComparisonResult:
    """Result from comparing parameters between files"""
    files: List[Path]
    parameters: List[Dict[str, Any]]
    differences: Dict[str, Any]
    similarity_score: float  # 0.0 = completely different, 1.0 = identical


# ============================================================================
# Main Bidirectional Workflow
# ============================================================================

class BidirectionalWorkflow:
    """
    Coordinates bidirectional workflows between MIDI and parameters.

    Provides high-level operations:
    1. Extract parameters from MIDI
    2. Generate MIDI from parameters
    3. Style transfer (extract → apply)
    4. Parameter modification workflows
    5. Parameter blending and interpolation
    6. Comparison and analysis

    Usage:
        >>> workflow = BidirectionalWorkflow()
        >>>
        >>> # Analysis: MIDI → Parameters
        >>> params = workflow.extract_parameters("song.mid")
        >>>
        >>> # Generation: Parameters → MIDI
        >>> workflow.generate_from_parameters(params, "output.mid")
        >>>
        >>> # Style Transfer
        >>> workflow.transfer_style("jazz.mid", "target.mid", "output.mid")
    """

    def __init__(
        self,
        prediction_api: Optional[ParameterPredictionAPI] = None,
        models_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize bidirectional workflow coordinator

        Args:
            prediction_api: Parameter prediction API instance
            models_dir: Directory with trained models
            output_dir: Default output directory for generated files
        """
        # Initialize prediction API
        if prediction_api:
            self.prediction_api = prediction_api
        elif PREDICTION_API_AVAILABLE:
            self.prediction_api = ParameterPredictionAPI(models_dir=models_dir)
        else:
            self.prediction_api = None
            print("⚠ ParameterPredictionAPI not available")

        # Output directory
        self.output_dir = Path(output_dir) if output_dir else Path('./output')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Workflow history
        self.history: List[Dict[str, Any]] = []

        print(f"BidirectionalWorkflow initialized")
        print(f"  Output directory: {self.output_dir}")

    # ========================================================================
    # Workflow 1: MIDI → Parameters (Analysis)
    # ========================================================================

    def extract_parameters(
        self,
        midi_file: Union[str, Path],
        format: str = 'hierarchical'
    ) -> Union[Dict[str, Any], HierarchicalPrediction]:
        """
        Extract parameters from MIDI file

        Args:
            midi_file: Path to MIDI file
            format: 'hierarchical', 'flat', or 'both'

        Returns:
            Parameters in requested format

        Example:
            >>> params = workflow.extract_parameters("song.mid")
            >>> print(params['genre.primary'])  # 'jazz'
            >>> print(params['tempo.bpm'])      # 140.0
        """
        if self.prediction_api is None:
            raise RuntimeError("Prediction API not available")

        result = self.prediction_api.analyze_midi(midi_file)

        if format == 'hierarchical':
            return result.hierarchical
        elif format == 'flat':
            return result.parameters
        elif format == 'both':
            return {
                'hierarchical': result.hierarchical,
                'flat': result.parameters
            }
        else:
            raise ValueError(f"Unknown format: {format}")

    def extract_batch(
        self,
        midi_files: List[Union[str, Path]],
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract parameters from multiple MIDI files

        Args:
            midi_files: List of MIDI files
            save_results: Save results to JSON files

        Returns:
            List of parameter dictionaries
        """
        if self.prediction_api is None:
            raise RuntimeError("Prediction API not available")

        results = self.prediction_api.analyze_batch(midi_files)

        parameters_list = [r.parameters for r in results]

        if save_results:
            output_file = self.output_dir / 'batch_analysis.json'
            with open(output_file, 'w') as f:
                json.dump(parameters_list, f, indent=2)
            print(f"✓ Saved batch results to {output_file}")

        return parameters_list

    # ========================================================================
    # Workflow 2: Parameters → MIDI (Generation)
    # ========================================================================

    def generate_from_parameters(
        self,
        parameters: Dict[str, Any],
        output_file: Optional[Union[str, Path]] = None,
        length_bars: int = 16,
        use_harmony_api: bool = True
    ) -> Path:
        """
        Generate MIDI from parameters

        Args:
            parameters: Parameter dictionary
            output_file: Output MIDI file path
            length_bars: Length in bars/measures
            use_harmony_api: Use HarmonyModuleAPI for generation

        Returns:
            Path to generated MIDI file

        Example:
            >>> params = {'genre.primary': 'jazz', 'tempo.bpm': 140, ...}
            >>> output = workflow.generate_from_parameters(params, "output.mid")
        """
        start_time = time.time()

        if output_file is None:
            output_file = self.output_dir / f"generated_{int(time.time())}.mid"
        else:
            output_file = Path(output_file)

        print(f"Generating MIDI from parameters...")

        # Extract key parameters
        genre = parameters.get('genre.primary', 'jazz')
        tempo = parameters.get('tempo.bpm', 120)
        key = parameters.get('key.tonic', 'C')
        mode = parameters.get('key.mode', 'major')
        complexity = parameters.get('complexity.overall', 0.5)

        print(f"  Genre: {genre}")
        print(f"  Tempo: {tempo} BPM")
        print(f"  Key: {key} {mode}")
        print(f"  Complexity: {complexity:.2f}")

        if use_harmony_api:
            # Use HarmonyModuleAPI for generation
            try:
                from midi_generator.api.unified_api import HarmonyModuleAPI

                api = HarmonyModuleAPI(output_dir=str(self.output_dir))

                # Generate using quick_fusion
                composition = api.quick_fusion(
                    harmony=genre,
                    rhythm=genre,
                    tempo=int(tempo),
                    key=f"{key}m" if mode == 'minor' else key,
                    measures=length_bars
                )

                # Export
                api.composition = composition
                output_path = api.export(output_file.name, overwrite=True)
                output_file = Path(output_path)

                print(f"  ✓ Generated using HarmonyModuleAPI")

            except Exception as e:
                print(f"  ✗ HarmonyModuleAPI generation failed: {e}")
                print(f"  ⚠ Falling back to basic generation")
                output_file = self._generate_basic_midi(parameters, output_file, length_bars)
        else:
            # Use basic generation
            output_file = self._generate_basic_midi(parameters, output_file, length_bars)

        elapsed = (time.time() - start_time) * 1000
        print(f"  ✓ Generated in {elapsed:.1f}ms")
        print(f"  ✓ Saved to: {output_file}\n")

        # Add to history
        self._add_to_history('generate', {
            'parameters': parameters,
            'output': str(output_file),
            'time_ms': elapsed
        })

        return output_file

    def _generate_basic_midi(
        self,
        parameters: Dict[str, Any],
        output_file: Path,
        length_bars: int
    ) -> Path:
        """
        Generate basic MIDI file from parameters (fallback method)

        This creates a simple MIDI file based on parameters when
        HarmonyModuleAPI is not available.
        """
        tempo = int(parameters.get('tempo.bpm', 120))
        key_tonic = parameters.get('key.tonic', 'C')

        # Create MIDI file
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Set tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))

        # Generate simple chord progression (placeholder)
        ticks_per_beat = mid.ticks_per_beat
        beats_per_bar = 4
        ticks_per_bar = ticks_per_beat * beats_per_bar

        # Simple C major scale chord progression
        chords = [
            [60, 64, 67],  # C major
            [65, 69, 72],  # F major
            [67, 71, 74],  # G major
            [60, 64, 67],  # C major
        ]

        time_offset = 0
        for bar in range(length_bars):
            chord = chords[bar % len(chords)]

            # Note on
            for pitch in chord:
                track.append(mido.Message('note_on', note=pitch, velocity=80, time=time_offset))
                time_offset = 0

            # Note off
            time_offset = ticks_per_bar
            for pitch in chord:
                track.append(mido.Message('note_off', note=pitch, velocity=0, time=time_offset))
                time_offset = 0

        # Save
        mid.save(str(output_file))
        print(f"  ✓ Generated basic MIDI")

        return output_file

    # ========================================================================
    # Workflow 3: Style Transfer
    # ========================================================================

    def transfer_style(
        self,
        source_midi: Union[str, Path],
        target_midi: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        parameters_to_transfer: Optional[List[str]] = None,
        preserve_melody: bool = True
    ) -> StyleTransferResult:
        """
        Transfer style from source MIDI to target MIDI

        Extracts parameters from source and applies them to target,
        optionally preserving melodic content.

        Args:
            source_midi: File to extract style from
            target_midi: File to apply style to
            output_file: Output file path
            parameters_to_transfer: Specific parameters to transfer (None = all)
            preserve_melody: Keep melody from target

        Returns:
            Style transfer result

        Example:
            >>> # Transfer jazz style to a classical piece
            >>> result = workflow.transfer_style(
            ...     "jazz_sample.mid",
            ...     "classical_piece.mid",
            ...     "jazz_classical.mid"
            ... )
        """
        start_time = time.time()

        source_midi = Path(source_midi)
        target_midi = Path(target_midi)

        if output_file is None:
            output_file = self.output_dir / f"style_transfer_{source_midi.stem}_to_{target_midi.stem}.mid"
        else:
            output_file = Path(output_file)

        print(f"\nStyle Transfer:")
        print(f"  Source: {source_midi.name}")
        print(f"  Target: {target_midi.name}")
        print(f"  Output: {output_file.name}")

        notes_list = []

        try:
            # Step 1: Extract parameters from source
            print(f"\n  [1/3] Extracting style from source...")
            source_params = self.extract_parameters(source_midi, format='flat')
            notes_list.append(f"Extracted {len(source_params)} parameters from source")

            # Step 2: Optionally filter parameters
            if parameters_to_transfer:
                source_params = {k: v for k, v in source_params.items()
                               if k in parameters_to_transfer}
                notes_list.append(f"Transferring {len(source_params)} selected parameters")

            # Step 3: Extract target parameters (for melody preservation)
            if preserve_melody:
                print(f"  [2/3] Extracting melody from target...")
                target_params = self.extract_parameters(target_midi, format='flat')

                # Keep melodic parameters from target
                melodic_params = [k for k in target_params.keys() if 'melody' in k]
                for param in melodic_params:
                    if param in target_params:
                        source_params[param] = target_params[param]

                notes_list.append(f"Preserved {len(melodic_params)} melodic parameters from target")

            # Step 4: Generate with transferred parameters
            print(f"  [3/3] Generating output...")
            output_path = self.generate_from_parameters(
                source_params,
                output_file,
                length_bars=16  # TODO: Extract from target
            )

            elapsed = (time.time() - start_time) * 1000

            result = StyleTransferResult(
                source_midi=source_midi,
                target_midi=target_midi,
                output_midi=output_path,
                transferred_parameters=source_params,
                success=True,
                processing_time_ms=elapsed,
                notes=notes_list
            )

            print(f"\n  ✓ Style transfer complete ({elapsed:.1f}ms)")

        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            print(f"\n  ✗ Style transfer failed: {e}")

            result = StyleTransferResult(
                source_midi=source_midi,
                target_midi=target_midi,
                output_midi=None,
                transferred_parameters={},
                success=False,
                processing_time_ms=elapsed,
                notes=[f"Error: {str(e)}"]
            )

        # Add to history
        self._add_to_history('style_transfer', {
            'source': str(source_midi),
            'target': str(target_midi),
            'output': str(output_file) if result.success else None,
            'success': result.success,
            'time_ms': elapsed
        })

        return result

    # ========================================================================
    # Workflow 4: Parameter Blending
    # ========================================================================

    def blend_parameters(
        self,
        params_a: Union[Dict[str, Any], str, Path],
        params_b: Union[Dict[str, Any], str, Path],
        weight: float = 0.5,
        output_file: Optional[Union[str, Path]] = None
    ) -> ParameterBlendResult:
        """
        Blend/interpolate between two parameter sets

        Args:
            params_a: First parameter set (or MIDI file to extract from)
            params_b: Second parameter set (or MIDI file to extract from)
            weight: Blend weight (0.0 = all A, 1.0 = all B, 0.5 = 50/50)
            output_file: Output MIDI file

        Returns:
            Blending result

        Example:
            >>> # 70% jazz, 30% classical
            >>> result = workflow.blend_parameters(
            ...     "jazz.mid",
            ...     "classical.mid",
            ...     weight=0.3
            ... )
        """
        print(f"\nBlending parameters (weight={weight:.2f})...")

        # Extract parameters if MIDI files provided
        if isinstance(params_a, (str, Path)):
            print(f"  Extracting from: {Path(params_a).name}")
            params_a = self.extract_parameters(params_a, format='flat')
            source_a = str(params_a)
        else:
            source_a = "dict_a"

        if isinstance(params_b, (str, Path)):
            print(f"  Extracting from: {Path(params_b).name}")
            params_b = self.extract_parameters(params_b, format='flat')
            source_b = str(params_b)
        else:
            source_b = "dict_b"

        # Blend parameters
        blended = {}

        # Get all parameter keys
        all_keys = set(params_a.keys()) | set(params_b.keys())

        for key in all_keys:
            val_a = params_a.get(key)
            val_b = params_b.get(key)

            # Skip if either is missing
            if val_a is None or val_b is None:
                blended[key] = val_a if val_a is not None else val_b
                continue

            # Blend based on type
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                # Numeric: interpolate
                blended[key] = val_a * (1 - weight) + val_b * weight
            elif isinstance(val_a, str) and isinstance(val_b, str):
                # Categorical: choose based on weight
                blended[key] = val_a if weight < 0.5 else val_b
            else:
                # Other: default to A
                blended[key] = val_a

        print(f"  ✓ Blended {len(blended)} parameters")

        # Generate output if requested
        output_path = None
        if output_file or True:  # Always generate
            if output_file is None:
                output_file = self.output_dir / f"blend_{weight:.2f}_{int(time.time())}.mid"

            output_path = self.generate_from_parameters(blended, output_file)

        return ParameterBlendResult(
            source_a=source_a,
            source_b=source_b,
            blend_weight=weight,
            blended_parameters=blended,
            output_midi=output_path
        )

    # ========================================================================
    # Workflow 5: Parameter Comparison
    # ========================================================================

    def compare_parameters(
        self,
        midi_files: List[Union[str, Path]],
        parameters_to_compare: Optional[List[str]] = None
    ) -> ParameterComparisonResult:
        """
        Compare parameters across multiple MIDI files

        Args:
            midi_files: List of MIDI files to compare
            parameters_to_compare: Specific parameters to compare (None = all)

        Returns:
            Comparison result with differences and similarity

        Example:
            >>> files = ["jazz1.mid", "jazz2.mid", "classical.mid"]
            >>> result = workflow.compare_parameters(files)
            >>> print(f"Similarity: {result.similarity_score:.2f}")
        """
        print(f"\nComparing {len(midi_files)} files...")

        # Extract parameters from all files
        all_params = []
        for midi_file in midi_files:
            params = self.extract_parameters(midi_file, format='flat')
            all_params.append(params)
            print(f"  ✓ Extracted from {Path(midi_file).name}")

        # Filter to requested parameters
        if parameters_to_compare:
            all_params = [{k: p.get(k) for k in parameters_to_compare}
                         for p in all_params]

        # Calculate differences
        differences = {}
        all_keys = set(all_params[0].keys())

        for key in all_keys:
            values = [p.get(key) for p in all_params if p.get(key) is not None]

            if not values:
                continue

            if isinstance(values[0], (int, float)):
                differences[key] = {
                    'type': 'numeric',
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
            elif isinstance(values[0], str):
                differences[key] = {
                    'type': 'categorical',
                    'unique_values': list(set(values)),
                    'counts': {v: values.count(v) for v in set(values)},
                    'values': values
                }

        # Calculate similarity score (0-1)
        similarity = self._calculate_similarity(all_params)

        print(f"  ✓ Similarity score: {similarity:.2f}")

        return ParameterComparisonResult(
            files=[Path(f) for f in midi_files],
            parameters=all_params,
            differences=differences,
            similarity_score=similarity
        )

    def _calculate_similarity(self, params_list: List[Dict[str, Any]]) -> float:
        """Calculate similarity score between parameter sets"""
        if len(params_list) < 2:
            return 1.0

        # Get common keys
        common_keys = set(params_list[0].keys())
        for params in params_list[1:]:
            common_keys &= set(params.keys())

        if not common_keys:
            return 0.0

        # Calculate pairwise similarities
        similarities = []

        for i in range(len(params_list)):
            for j in range(i + 1, len(params_list)):
                params_a = params_list[i]
                params_b = params_list[j]

                matches = 0
                total = 0

                for key in common_keys:
                    val_a = params_a.get(key)
                    val_b = params_b.get(key)

                    if val_a is None or val_b is None:
                        continue

                    total += 1

                    if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                        # Numeric: use normalized distance
                        diff = abs(val_a - val_b)
                        max_diff = max(abs(val_a), abs(val_b), 1.0)
                        similarity = 1.0 - min(diff / max_diff, 1.0)
                        matches += similarity
                    elif isinstance(val_a, str) and isinstance(val_b, str):
                        # Categorical: exact match
                        if val_a == val_b:
                            matches += 1

                if total > 0:
                    similarities.append(matches / total)

        return np.mean(similarities) if similarities else 0.0

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _add_to_history(self, operation: str, data: Dict[str, Any]):
        """Add operation to workflow history"""
        self.history.append({
            'timestamp': time.time(),
            'operation': operation,
            'data': data
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """Get workflow operation history"""
        return self.history.copy()

    def clear_history(self):
        """Clear workflow history"""
        self.history.clear()
        print("✓ History cleared")


# ============================================================================
# Convenience Functions
# ============================================================================

def transfer_style(source: Union[str, Path], target: Union[str, Path], output: Union[str, Path]) -> Path:
    """
    Quick style transfer function

    Example:
        >>> transfer_style("jazz.mid", "target.mid", "output.mid")
    """
    workflow = BidirectionalWorkflow()
    result = workflow.transfer_style(source, target, output)
    return result.output_midi


def blend_styles(file_a: Union[str, Path], file_b: Union[str, Path], weight: float = 0.5) -> Path:
    """
    Quick style blending function

    Example:
        >>> blend_styles("jazz.mid", "classical.mid", weight=0.3)
    """
    workflow = BidirectionalWorkflow()
    result = workflow.blend_parameters(file_a, file_b, weight)
    return result.output_midi


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Bidirectional Workflow Coordinator - Agent 09")
    print("=" * 60)

    # Create workflow
    workflow = BidirectionalWorkflow()

    print("\n✓ Bidirectional workflow ready!")
    print("\nAvailable operations:")
    print("  1. Extract parameters from MIDI")
    print("  2. Generate MIDI from parameters")
    print("  3. Transfer style between files")
    print("  4. Blend parameters")
    print("  5. Compare parameters")
