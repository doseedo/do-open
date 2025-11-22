"""
Musical DNA v2.0 - Hierarchical 300D Architecture
=================================================

Agent 3: DNA Expansion & Hierarchical Architecture

This module implements the expanded 300D Musical DNA with hierarchical structure:
- GLOBAL level (60D): Musical context and style
- SECTIONAL level (140D): Musical content
- LOCAL level (100D): Implementation details

Key Features:
- Hierarchical organization (global → sectional → local)
- Backward compatibility with v1.0 (120D)
- Automatic version detection and migration
- Enhanced interpretability through hierarchy

Author: Agent 3 - DNA Expansion & Hierarchical Architecture
Date: 2025-11-22
Version: 2.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time
import numpy as np
import warnings


# =============================================================================
# Musical DNA v2.0 - Hierarchical 300D
# =============================================================================

@dataclass
class MusicalDNA:
    """
    Hierarchical Musical DNA (300D) - Version 2.0

    Represents the complete "Musical DNA" of a MIDI file with hierarchical structure:

    GLOBAL LEVEL (60D) - Musical Context & Style
    ├── key_context: 12D        Tonal center, modulations, key stability
    ├── tempo_feel: 8D          Tempo, variations, metric feel
    ├── genre_style: 20D        Jazz/classical/latin style markers
    └── form_structure: 20D     Overall form, sections, proportions

    SECTIONAL LEVEL (140D) - Musical Content
    ├── harmony: 60D            Chords, progressions, voice leading
    ├── melody: 40D             Contour, motifs, phrasing
    └── rhythm: 40D             Syncopation, groove, subdivision

    LOCAL LEVEL (100D) - Implementation Details
    ├── voicing: 30D            Spacing, doubling, register
    ├── texture: 30D            Density, independence, layering
    └── orchestration: 40D      Instrumentation, balance, articulation

    Total: 300 interpretable parameters organized hierarchically
    """

    # Version control
    version: str = "2.0"

    # GLOBAL LEVEL (60D) - Musical context
    key_context_params: np.ndarray      # 12D: tonal center, modulations
    tempo_feel_params: np.ndarray       # 8D: tempo, variations, feel
    genre_style_params: np.ndarray      # 20D: style classification
    form_structure_params: np.ndarray   # 20D: form, sections, proportions

    # SECTIONAL LEVEL (140D) - Musical content
    harmony_params: np.ndarray          # 60D: chords, progressions (expanded from 30D)
    melody_params: np.ndarray           # 40D: contour, motifs, phrasing (NEW)
    rhythm_params: np.ndarray           # 40D: syncopation, groove (expanded from 20D)

    # LOCAL LEVEL (100D) - Implementation details
    voicing_params: np.ndarray          # 30D: spacing, doubling, register (NEW)
    texture_params: np.ndarray          # 30D: density, independence (expanded from 20D)
    orchestration_params: np.ndarray    # 40D: instrumentation, balance (expanded from 25D)

    # Metadata
    source_file: Optional[str] = None
    extraction_timestamp: Optional[str] = None

    def __post_init__(self):
        """Validate dimensions"""
        expected_dims = {
            'key_context_params': 12,
            'tempo_feel_params': 8,
            'genre_style_params': 20,
            'form_structure_params': 20,
            'harmony_params': 60,
            'melody_params': 40,
            'rhythm_params': 40,
            'voicing_params': 30,
            'texture_params': 30,
            'orchestration_params': 40,
        }

        for param_name, expected_dim in expected_dims.items():
            param = getattr(self, param_name)
            if len(param) != expected_dim:
                raise ValueError(
                    f"{param_name} has {len(param)} dimensions, expected {expected_dim}"
                )

    def to_vector(self) -> np.ndarray:
        """
        Flatten all parameters into single 300D vector.

        Returns:
            1D numpy array of shape [300]

        Layout:
            [0:12]    key_context
            [12:20]   tempo_feel
            [20:40]   genre_style
            [40:60]   form_structure
            [60:120]  harmony
            [120:160] melody
            [160:200] rhythm
            [200:230] voicing
            [230:260] texture
            [260:300] orchestration
        """
        return np.concatenate([
            # Global (60D)
            self.key_context_params,
            self.tempo_feel_params,
            self.genre_style_params,
            self.form_structure_params,
            # Sectional (140D)
            self.harmony_params,
            self.melody_params,
            self.rhythm_params,
            # Local (100D)
            self.voicing_params,
            self.texture_params,
            self.orchestration_params,
        ])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary (for JSON serialization).

        Returns:
            Dictionary with all parameters and metadata
        """
        return {
            'version': self.version,
            # Global level
            'global': {
                'key_context': self.key_context_params.tolist(),
                'tempo_feel': self.tempo_feel_params.tolist(),
                'genre_style': self.genre_style_params.tolist(),
                'form_structure': self.form_structure_params.tolist(),
            },
            # Sectional level
            'sectional': {
                'harmony': self.harmony_params.tolist(),
                'melody': self.melody_params.tolist(),
                'rhythm': self.rhythm_params.tolist(),
            },
            # Local level
            'local': {
                'voicing': self.voicing_params.tolist(),
                'texture': self.texture_params.tolist(),
                'orchestration': self.orchestration_params.tolist(),
            },
            # Metadata
            'metadata': {
                'source_file': self.source_file,
                'extraction_timestamp': self.extraction_timestamp,
            }
        }

    def to_hierarchical_dict(self) -> Dict[str, np.ndarray]:
        """
        Get parameters organized by hierarchical level.

        Returns:
            Dictionary with keys 'global', 'sectional', 'local'
        """
        return {
            'global': np.concatenate([
                self.key_context_params,
                self.tempo_feel_params,
                self.genre_style_params,
                self.form_structure_params,
            ]),  # 60D
            'sectional': np.concatenate([
                self.harmony_params,
                self.melody_params,
                self.rhythm_params,
            ]),  # 140D
            'local': np.concatenate([
                self.voicing_params,
                self.texture_params,
                self.orchestration_params,
            ]),  # 100D
        }

    @classmethod
    def from_vector(cls, vector: np.ndarray, source_file: Optional[str] = None) -> 'MusicalDNA':
        """
        Create MusicalDNA from 300D vector.

        Args:
            vector: 1D array of shape [300]
            source_file: Optional source MIDI file path

        Returns:
            MusicalDNA instance

        Raises:
            ValueError: If vector is not 300D
        """
        if len(vector) != 300:
            raise ValueError(f"Expected 300D vector, got {len(vector)}D")

        return cls(
            # Global (60D)
            key_context_params=vector[0:12],
            tempo_feel_params=vector[12:20],
            genre_style_params=vector[20:40],
            form_structure_params=vector[40:60],
            # Sectional (140D)
            harmony_params=vector[60:120],
            melody_params=vector[120:160],
            rhythm_params=vector[160:200],
            # Local (100D)
            voicing_params=vector[200:230],
            texture_params=vector[230:260],
            orchestration_params=vector[260:300],
            # Metadata
            source_file=source_file,
            extraction_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    @classmethod
    def from_dict(cls, dna_dict: Dict[str, Any]) -> 'MusicalDNA':
        """
        Create from dictionary.

        Args:
            dna_dict: Dictionary with hierarchical structure

        Returns:
            MusicalDNA instance

        Raises:
            ValueError: If dictionary structure is invalid
        """
        version = dna_dict.get('version', '2.0')

        if version != '2.0':
            warnings.warn(
                f"Loading DNA with version {version}, expected 2.0. "
                f"Use migrate_120d_to_300d() if loading v1.0"
            )

        global_params = dna_dict['global']
        sectional_params = dna_dict['sectional']
        local_params = dna_dict['local']
        metadata = dna_dict.get('metadata', {})

        return cls(
            # Global
            key_context_params=np.array(global_params['key_context']),
            tempo_feel_params=np.array(global_params['tempo_feel']),
            genre_style_params=np.array(global_params['genre_style']),
            form_structure_params=np.array(global_params['form_structure']),
            # Sectional
            harmony_params=np.array(sectional_params['harmony']),
            melody_params=np.array(sectional_params['melody']),
            rhythm_params=np.array(sectional_params['rhythm']),
            # Local
            voicing_params=np.array(local_params['voicing']),
            texture_params=np.array(local_params['texture']),
            orchestration_params=np.array(local_params['orchestration']),
            # Metadata
            source_file=metadata.get('source_file'),
            extraction_timestamp=metadata.get('extraction_timestamp'),
        )

    @classmethod
    def from_zeros(cls, source_file: Optional[str] = None) -> 'MusicalDNA':
        """
        Create zero-initialized DNA (useful for templates).

        Args:
            source_file: Optional source file path

        Returns:
            MusicalDNA with all parameters set to zero
        """
        return cls(
            # Global (60D)
            key_context_params=np.zeros(12),
            tempo_feel_params=np.zeros(8),
            genre_style_params=np.zeros(20),
            form_structure_params=np.zeros(20),
            # Sectional (140D)
            harmony_params=np.zeros(60),
            melody_params=np.zeros(40),
            rhythm_params=np.zeros(40),
            # Local (100D)
            voicing_params=np.zeros(30),
            texture_params=np.zeros(30),
            orchestration_params=np.zeros(40),
            # Metadata
            source_file=source_file,
            extraction_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def save(self, path: Path):
        """
        Save DNA to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'MusicalDNA':
        """
        Load DNA from JSON file with automatic version detection.

        Args:
            path: Input file path

        Returns:
            MusicalDNA instance (v2.0)

        Note:
            If loading v1.0 (120D), it will be automatically migrated to v2.0 (300D)
        """
        with open(path, 'r') as f:
            dna_dict = json.load(f)

        version = dna_dict.get('version', '1.0')

        if version == '1.0':
            # Auto-migrate from v1.0 to v2.0
            from midi_generator.learning.dna_migration import migrate_120d_to_300d
            warnings.warn(
                "Loading v1.0 (120D) DNA. Auto-migrating to v2.0 (300D). "
                "Original data will be preserved and expanded with zeros for new parameters."
            )
            return migrate_120d_to_300d(dna_dict)
        else:
            return cls.from_dict(dna_dict)

    def get_global_params(self) -> np.ndarray:
        """Get all global-level parameters (60D)."""
        return np.concatenate([
            self.key_context_params,
            self.tempo_feel_params,
            self.genre_style_params,
            self.form_structure_params,
        ])

    def get_sectional_params(self) -> np.ndarray:
        """Get all sectional-level parameters (140D)."""
        return np.concatenate([
            self.harmony_params,
            self.melody_params,
            self.rhythm_params,
        ])

    def get_local_params(self) -> np.ndarray:
        """Get all local-level parameters (100D)."""
        return np.concatenate([
            self.voicing_params,
            self.texture_params,
            self.orchestration_params,
        ])

    def summary(self) -> str:
        """
        Generate human-readable summary of DNA.

        Returns:
            Formatted summary string
        """
        lines = [
            "="*70,
            "Musical DNA v2.0 - Hierarchical 300D Summary",
            "="*70,
            "",
            "GLOBAL LEVEL (60D) - Musical Context:",
            f"  Key Context:     {self.key_context_params.shape} → {np.mean(self.key_context_params):.3f} (mean)",
            f"  Tempo Feel:      {self.tempo_feel_params.shape} → {np.mean(self.tempo_feel_params):.3f} (mean)",
            f"  Genre Style:     {self.genre_style_params.shape} → {np.mean(self.genre_style_params):.3f} (mean)",
            f"  Form Structure:  {self.form_structure_params.shape} → {np.mean(self.form_structure_params):.3f} (mean)",
            "",
            "SECTIONAL LEVEL (140D) - Musical Content:",
            f"  Harmony:         {self.harmony_params.shape} → {np.mean(self.harmony_params):.3f} (mean)",
            f"  Melody:          {self.melody_params.shape} → {np.mean(self.melody_params):.3f} (mean)",
            f"  Rhythm:          {self.rhythm_params.shape} → {np.mean(self.rhythm_params):.3f} (mean)",
            "",
            "LOCAL LEVEL (100D) - Implementation Details:",
            f"  Voicing:         {self.voicing_params.shape} → {np.mean(self.voicing_params):.3f} (mean)",
            f"  Texture:         {self.texture_params.shape} → {np.mean(self.texture_params):.3f} (mean)",
            f"  Orchestration:   {self.orchestration_params.shape} → {np.mean(self.orchestration_params):.3f} (mean)",
            "",
            "Metadata:",
            f"  Source:          {self.source_file or 'N/A'}",
            f"  Timestamp:       {self.extraction_timestamp or 'N/A'}",
            f"  Version:         {self.version}",
            "="*70,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MusicalDNA(version={self.version}, dims=300D, source={self.source_file})"


# =============================================================================
# Utility Functions
# =============================================================================

def create_random_dna(seed: Optional[int] = None) -> MusicalDNA:
    """
    Create random DNA for testing.

    Args:
        seed: Optional random seed for reproducibility

    Returns:
        MusicalDNA with random parameters
    """
    if seed is not None:
        np.random.seed(seed)

    return MusicalDNA(
        # Global (60D)
        key_context_params=np.random.randn(12),
        tempo_feel_params=np.random.randn(8),
        genre_style_params=np.random.randn(20),
        form_structure_params=np.random.randn(20),
        # Sectional (140D)
        harmony_params=np.random.randn(60),
        melody_params=np.random.randn(40),
        rhythm_params=np.random.randn(40),
        # Local (100D)
        voicing_params=np.random.randn(30),
        texture_params=np.random.randn(30),
        orchestration_params=np.random.randn(40),
        # Metadata
        source_file="random_test.mid",
        extraction_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )


def validate_dna(dna: MusicalDNA) -> Dict[str, Any]:
    """
    Validate DNA structure and contents.

    Args:
        dna: MusicalDNA instance to validate

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    # Check version
    if dna.version != '2.0':
        results['warnings'].append(f"Unexpected version: {dna.version}")

    # Check dimensions
    try:
        vector = dna.to_vector()
        if len(vector) != 300:
            results['valid'] = False
            results['errors'].append(f"Vector dimension mismatch: {len(vector)} != 300")
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Failed to convert to vector: {e}")

    # Check for NaN or Inf
    vector = dna.to_vector()
    if np.any(np.isnan(vector)):
        results['warnings'].append("Contains NaN values")
    if np.any(np.isinf(vector)):
        results['warnings'].append("Contains Inf values")

    # Compute statistics
    results['stats'] = {
        'mean': float(np.mean(vector)),
        'std': float(np.std(vector)),
        'min': float(np.min(vector)),
        'max': float(np.max(vector)),
        'global_mean': float(np.mean(dna.get_global_params())),
        'sectional_mean': float(np.mean(dna.get_sectional_params())),
        'local_mean': float(np.mean(dna.get_local_params())),
    }

    return results


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Musical DNA v2.0 - Hierarchical 300D Architecture")
    print("="*70)

    # Test 1: Create from zeros
    print("\nTest 1: Create zero DNA")
    dna = MusicalDNA.from_zeros(source_file="test.mid")
    print(dna.summary())

    # Test 2: Create random DNA
    print("\nTest 2: Create random DNA")
    random_dna = create_random_dna(seed=42)
    print(random_dna.summary())

    # Test 3: Convert to vector and back
    print("\nTest 3: Vector conversion")
    vector = random_dna.to_vector()
    print(f"Vector shape: {vector.shape}")
    print(f"Expected: (300,)")

    reconstructed = MusicalDNA.from_vector(vector)
    diff = np.max(np.abs(vector - reconstructed.to_vector()))
    print(f"Reconstruction error: {diff:.10f}")
    assert diff < 1e-10, "Reconstruction failed!"
    print("✅ Vector conversion successful")

    # Test 4: Save and load
    print("\nTest 4: Save and load")
    temp_path = Path("/tmp/test_dna_v2.json")
    random_dna.save(temp_path)
    print(f"Saved to: {temp_path}")

    loaded_dna = MusicalDNA.load(temp_path)
    diff = np.max(np.abs(random_dna.to_vector() - loaded_dna.to_vector()))
    print(f"Save/load error: {diff:.10f}")
    assert diff < 1e-10, "Save/load failed!"
    print("✅ Save/load successful")

    # Test 5: Validation
    print("\nTest 5: Validation")
    validation = validate_dna(random_dna)
    print(f"Valid: {validation['valid']}")
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
    print(f"Stats: {validation['stats']}")

    # Test 6: Hierarchical access
    print("\nTest 6: Hierarchical parameter access")
    print(f"Global params:    {dna.get_global_params().shape} = (60,)")
    print(f"Sectional params: {dna.get_sectional_params().shape} = (140,)")
    print(f"Local params:     {dna.get_local_params().shape} = (100,)")
    total = (dna.get_global_params().shape[0] +
             dna.get_sectional_params().shape[0] +
             dna.get_local_params().shape[0])
    print(f"Total: {total} = 300")
    assert total == 300, "Hierarchical dimension mismatch!"
    print("✅ Hierarchical access successful")

    print("\n" + "="*70)
    print("All tests passed! MusicalDNA v2.0 ready for use.")
    print("="*70)
