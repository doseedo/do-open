"""
HarmonyModule Integration - Agent 09
=====================================

Integrates parameter prediction and generation with the existing HarmonyModuleAPI.

This module extends HarmonyModuleAPI with:
- Automatic parameter extraction from MIDI
- Parameter-driven generation
- Style transfer capabilities
- Real-time parameter prediction
- Bidirectional workflows

Author: Agent 09 - HarmonyModule Integration Lead
Date: 2025-11-20
License: MIT
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import time

# Internal imports
try:
    from midi_generator.api.unified_api import HarmonyModuleAPI
    HARMONY_API_AVAILABLE = True
except ImportError:
    print("Warning: HarmonyModuleAPI not available")
    HARMONY_API_AVAILABLE = False

try:
    from midi_generator.integration.parameter_prediction_api import (
        ParameterPredictionAPI,
        ParameterAnalysisResult
    )
    from midi_generator.integration.bidirectional_workflow import BidirectionalWorkflow
    from midi_generator.integration.hierarchical_model_wrapper import (
        HierarchicalMTLWrapper,
        HierarchicalPrediction
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    print("Warning: Integration modules not available")
    INTEGRATION_AVAILABLE = False


# ============================================================================
# Extended HarmonyModule API with ML Integration
# ============================================================================

class EnhancedHarmonyModuleAPI:
    """
    Enhanced HarmonyModuleAPI with ML-powered parameter prediction.

    This class extends the base HarmonyModuleAPI with:
    - Automatic style extraction from MIDI
    - ML-based parameter prediction
    - Intelligent style transfer
    - Parameter-driven generation
    - Real-time analysis and generation

    Usage:
        >>> api = EnhancedHarmonyModuleAPI()
        >>>
        >>> # Extract and analyze
        >>> params = api.analyze_and_extract("song.mid")
        >>>
        >>> # Generate with extracted style
        >>> api.generate_with_style("song.mid", "output.mid")
        >>>
        >>> # Style transfer
        >>> api.transfer_style_intelligent("source.mid", "target.mid", "output.mid")
    """

    def __init__(
        self,
        output_dir: str = "./output",
        models_dir: Optional[Path] = None,
        enable_ml: bool = True
    ):
        """
        Initialize enhanced API

        Args:
            output_dir: Directory for output files
            models_dir: Directory containing trained ML models
            enable_ml: Enable ML-powered features
        """
        # Initialize base HarmonyModule API
        if HARMONY_API_AVAILABLE:
            self.harmony_api = HarmonyModuleAPI(output_dir=output_dir)
            print("✓ HarmonyModuleAPI initialized")
        else:
            self.harmony_api = None
            print("⚠ HarmonyModuleAPI not available")

        # Initialize ML components
        self.enable_ml = enable_ml and INTEGRATION_AVAILABLE

        if self.enable_ml:
            self.prediction_api = ParameterPredictionAPI(models_dir=models_dir)
            self.workflow = BidirectionalWorkflow(
                prediction_api=self.prediction_api,
                output_dir=Path(output_dir)
            )
            print("✓ ML components initialized")
        else:
            self.prediction_api = None
            self.workflow = None
            print("⚠ ML features disabled")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"EnhancedHarmonyModuleAPI ready!")
        print(f"  ML-powered: {'Yes' if self.enable_ml else 'No'}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}\n")

    # ========================================================================
    # ML-Enhanced Analysis Methods
    # ========================================================================

    def analyze_and_extract(
        self,
        midi_file: Union[str, Path],
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze MIDI file and extract all parameters using ML

        Args:
            midi_file: Path to MIDI file
            detailed: Return detailed analysis with confidence scores

        Returns:
            Dictionary of extracted parameters

        Example:
            >>> params = api.analyze_and_extract("song.mid")
            >>> print(f"Genre: {params['genre.primary']}")
            >>> print(f"Tempo: {params['tempo.bpm']}")
        """
        if not self.enable_ml or self.prediction_api is None:
            print("⚠ ML not available, using basic analysis")
            return self._analyze_basic(midi_file)

        print(f"\nAnalyzing: {Path(midi_file).name}")

        result = self.prediction_api.analyze_midi(midi_file)

        if detailed:
            return result.to_dict()
        else:
            return result.parameters

    def analyze_batch(
        self,
        midi_files: List[Union[str, Path]],
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple MIDI files

        Args:
            midi_files: List of MIDI files
            save_results: Save results to JSON

        Returns:
            List of parameter dictionaries
        """
        if not self.enable_ml or self.workflow is None:
            print("⚠ ML not available")
            return []

        return self.workflow.extract_batch(midi_files, save_results=save_results)

    def _analyze_basic(self, midi_file: Union[str, Path]) -> Dict[str, Any]:
        """Basic analysis without ML (fallback)"""
        if self.harmony_api is None:
            return {}

        info = self.harmony_api.load_midi(str(midi_file))

        return {
            'tempo.bpm': info.get('tempo', 120),
            'time_signature': info.get('time_signature', (4, 4)),
            'num_tracks': info.get('num_tracks', 1)
        }

    # ========================================================================
    # ML-Enhanced Generation Methods
    # ========================================================================

    def generate_with_style(
        self,
        style_source: Union[str, Path, Dict[str, Any]],
        output_file: Optional[Union[str, Path]] = None,
        length_bars: int = 16,
        modifications: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate new MIDI using style extracted from source

        Args:
            style_source: MIDI file to extract style from, or parameter dict
            output_file: Output file path
            length_bars: Length in bars
            modifications: Optional parameter modifications

        Returns:
            Path to generated file

        Example:
            >>> # Generate 32 bars using jazz style, but at 140 BPM
            >>> api.generate_with_style(
            ...     "jazz_sample.mid",
            ...     "output.mid",
            ...     length_bars=32,
            ...     modifications={'tempo.bpm': 140}
            ... )
        """
        if not self.enable_ml or self.workflow is None:
            print("⚠ ML not available, using default generation")
            return self._generate_default(output_file, length_bars)

        print(f"\nGenerating with extracted style...")

        # Extract parameters if MIDI file provided
        if isinstance(style_source, (str, Path)):
            print(f"  Extracting style from: {Path(style_source).name}")
            parameters = self.workflow.extract_parameters(style_source, format='flat')
        else:
            parameters = style_source

        # Apply modifications
        if modifications:
            print(f"  Applying {len(modifications)} modifications")
            parameters.update(modifications)

        # Generate
        output_path = self.workflow.generate_from_parameters(
            parameters,
            output_file,
            length_bars=length_bars
        )

        return output_path

    def generate_from_description(
        self,
        genre: str,
        tempo: int,
        complexity: float = 0.5,
        energy: float = 0.7,
        key: str = "C",
        mode: str = "major",
        length_bars: int = 16,
        output_file: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Generate MIDI from high-level description

        Args:
            genre: Musical genre
            tempo: Tempo in BPM
            complexity: Complexity level (0-1)
            energy: Energy level (0-1)
            key: Key tonic (C, D, E, etc.)
            mode: Key mode (major, minor)
            length_bars: Length in bars
            output_file: Output file path

        Returns:
            Path to generated file

        Example:
            >>> api.generate_from_description(
            ...     genre='jazz',
            ...     tempo=140,
            ...     complexity=0.7,
            ...     energy=0.8,
            ...     key='Bb',
            ...     mode='major'
            ... )
        """
        # Build parameter dictionary
        parameters = {
            'genre.primary': genre,
            'tempo.bpm': tempo,
            'complexity.overall': complexity,
            'energy.level': energy,
            'key.tonic': key,
            'key.mode': mode
        }

        # Use base API if ML not available
        if not self.enable_ml or self.harmony_api is None:
            if output_file is None:
                output_file = self.output_dir / f"{genre}_{tempo}bpm.mid"

            # Use HarmonyModuleAPI
            if self.harmony_api:
                composition = self.harmony_api.quick_fusion(
                    harmony=genre,
                    rhythm=genre,
                    tempo=tempo,
                    key=f"{key}m" if mode == 'minor' else key,
                    measures=length_bars
                )
                self.harmony_api.composition = composition
                return Path(self.harmony_api.export(Path(output_file).name, overwrite=True))

        # Generate with ML
        return self.generate_with_style(
            parameters,
            output_file,
            length_bars=length_bars
        )

    def _generate_default(self, output_file: Optional[Path], length_bars: int) -> Path:
        """Default generation without ML (fallback)"""
        if output_file is None:
            output_file = self.output_dir / f"generated_{int(time.time())}.mid"

        if self.harmony_api:
            composition = self.harmony_api.quick_fusion(
                harmony='jazz',
                rhythm='jazz',
                tempo=120,
                key='C',
                measures=length_bars
            )
            self.harmony_api.composition = composition
            return Path(self.harmony_api.export(Path(output_file).name, overwrite=True))

        return output_file

    # ========================================================================
    # Style Transfer Methods
    # ========================================================================

    def transfer_style_intelligent(
        self,
        source_midi: Union[str, Path],
        target_midi: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        preserve_melody: bool = True,
        preserve_structure: bool = True,
        parameters_to_transfer: Optional[List[str]] = None
    ) -> Path:
        """
        Intelligent style transfer with ML-based analysis

        Args:
            source_midi: File to extract style from
            target_midi: File to apply style to
            output_file: Output file path
            preserve_melody: Keep melody from target
            preserve_structure: Keep structure from target
            parameters_to_transfer: Specific parameters to transfer

        Returns:
            Path to generated file

        Example:
            >>> # Transfer jazz harmony to classical piece, keep melody
            >>> api.transfer_style_intelligent(
            ...     "jazz.mid",
            ...     "classical.mid",
            ...     "jazz_classical.mid",
            ...     preserve_melody=True
            ... )
        """
        if not self.enable_ml or self.workflow is None:
            print("⚠ ML not available")
            return Path(output_file) if output_file else self.output_dir / "output.mid"

        result = self.workflow.transfer_style(
            source_midi,
            target_midi,
            output_file,
            parameters_to_transfer=parameters_to_transfer,
            preserve_melody=preserve_melody
        )

        if result.success:
            print(f"✓ Style transfer successful!")
            return result.output_midi
        else:
            print(f"✗ Style transfer failed")
            for note in result.notes:
                print(f"  - {note}")
            raise RuntimeError("Style transfer failed")

    def blend_styles(
        self,
        midi_a: Union[str, Path],
        midi_b: Union[str, Path],
        weight: float = 0.5,
        output_file: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Blend styles from two MIDI files

        Args:
            midi_a: First MIDI file
            midi_b: Second MIDI file
            weight: Blend weight (0.0 = all A, 1.0 = all B)
            output_file: Output file path

        Returns:
            Path to generated file

        Example:
            >>> # 70% jazz, 30% funk
            >>> api.blend_styles("jazz.mid", "funk.mid", weight=0.3)
        """
        if not self.enable_ml or self.workflow is None:
            print("⚠ ML not available")
            return Path(output_file) if output_file else self.output_dir / "output.mid"

        result = self.workflow.blend_parameters(
            midi_a,
            midi_b,
            weight=weight,
            output_file=output_file
        )

        return result.output_midi

    # ========================================================================
    # Comparison and Analysis Methods
    # ========================================================================

    def compare_styles(
        self,
        midi_files: List[Union[str, Path]],
        parameters_to_compare: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare styles across multiple MIDI files

        Args:
            midi_files: List of MIDI files
            parameters_to_compare: Specific parameters to compare

        Returns:
            Comparison results

        Example:
            >>> files = ["jazz1.mid", "jazz2.mid", "classical.mid"]
            >>> result = api.compare_styles(files)
            >>> print(f"Similarity: {result['similarity_score']:.2f}")
        """
        if not self.enable_ml or self.workflow is None:
            print("⚠ ML not available")
            return {}

        result = self.workflow.compare_parameters(
            midi_files,
            parameters_to_compare=parameters_to_compare
        )

        return {
            'files': [str(f) for f in result.files],
            'similarity_score': result.similarity_score,
            'differences': result.differences,
            'parameters': result.parameters
        }

    def suggest_modifications(
        self,
        midi_file: Union[str, Path],
        target_genre: Optional[str] = None,
        target_complexity: Optional[float] = None,
        target_energy: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Suggest parameter modifications to achieve target style

        Args:
            midi_file: MIDI file to analyze
            target_genre: Target genre
            target_complexity: Target complexity (0-1)
            target_energy: Target energy (0-1)

        Returns:
            Suggested modifications

        Example:
            >>> # Suggest how to make piece more complex
            >>> suggestions = api.suggest_modifications(
            ...     "simple.mid",
            ...     target_complexity=0.8
            ... )
        """
        if not self.enable_ml or self.prediction_api is None:
            print("⚠ ML not available")
            return {}

        # Extract current parameters
        current = self.analyze_and_extract(midi_file)

        # Build suggestions
        suggestions = {}

        if target_genre and current.get('genre.primary') != target_genre:
            suggestions['genre.primary'] = {
                'current': current.get('genre.primary'),
                'suggested': target_genre,
                'reason': f"Change genre to {target_genre}"
            }

        if target_complexity is not None:
            current_complexity = current.get('complexity.overall', 0.5)
            if abs(current_complexity - target_complexity) > 0.1:
                suggestions['complexity.overall'] = {
                    'current': current_complexity,
                    'suggested': target_complexity,
                    'delta': target_complexity - current_complexity,
                    'reason': f"Adjust complexity from {current_complexity:.2f} to {target_complexity:.2f}"
                }

        if target_energy is not None:
            current_energy = current.get('energy.level', 0.5)
            if abs(current_energy - target_energy) > 0.1:
                suggestions['energy.level'] = {
                    'current': current_energy,
                    'suggested': target_energy,
                    'delta': target_energy - current_energy,
                    'reason': f"Adjust energy from {current_energy:.2f} to {target_energy:.2f}"
                }

        return {
            'current_parameters': current,
            'suggestions': suggestions,
            'num_suggestions': len(suggestions)
        }

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}

        if self.prediction_api:
            stats['prediction'] = self.prediction_api.get_performance_stats()

        if self.workflow:
            stats['workflow_history'] = len(self.workflow.get_history())

        return stats

    def clear_caches(self):
        """Clear all caches"""
        if self.prediction_api:
            self.prediction_api.clear_cache()

        print("✓ Caches cleared")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_enhanced_api(output_dir: str = "./output", enable_ml: bool = True) -> EnhancedHarmonyModuleAPI:
    """
    Create enhanced HarmonyModule API instance

    Args:
        output_dir: Output directory
        enable_ml: Enable ML features

    Returns:
        Enhanced API instance
    """
    return EnhancedHarmonyModuleAPI(output_dir=output_dir, enable_ml=enable_ml)


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("HarmonyModule Integration - Agent 09")
    print("=" * 60)

    # Create enhanced API
    api = create_enhanced_api()

    print("\n✓ Enhanced HarmonyModule API ready!")
    print("\nNew ML-powered features:")
    print("  ✓ Automatic parameter extraction")
    print("  ✓ Style-driven generation")
    print("  ✓ Intelligent style transfer")
    print("  ✓ Style blending")
    print("  ✓ Style comparison")
    print("  ✓ Modification suggestions")
