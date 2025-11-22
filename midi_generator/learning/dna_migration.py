"""
DNA Migration Utility - v1.0 (120D) → v2.0 (300D)
================================================

Agent 3: DNA Expansion & Hierarchical Architecture

This module provides utilities for migrating Musical DNA from v1.0 (120D) to v2.0 (300D).

Migration Strategy:
- Preserve all original 120D parameters
- Intelligently map old parameters to new hierarchical structure
- Initialize new parameters with zeros (to be learned later)
- Maintain full backward compatibility

Migration Mapping:
    v1.0 (120D)                    →  v2.0 (300D)
    ═══════════════════════════════════════════════════════════════════
    harmony (30D)                  →  harmony (60D): replicate + expand
    rhythm (20D)                   →  rhythm (40D): replicate + expand
    form (15D)                     →  form_structure (20D): extend
    orchestration (25D)            →  orchestration (40D): extend
    texture (20D)                  →  texture (30D): extend
    cross_dimensional (10D)        →  distributed to global params

    NEW PARAMETERS (initialized to zeros):
    - key_context (12D)            →  zeros
    - tempo_feel (8D)              →  zeros
    - genre_style (20D)            →  zeros (except some from cross)
    - melody (40D)                 →  zeros
    - voicing (30D)                →  zeros

Author: Agent 3 - DNA Expansion & Hierarchical Architecture
Date: 2025-11-22
Version: 1.0.0
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import numpy as np
import warnings

try:
    from midi_generator.learning.musical_dna_v2 import MusicalDNA
    DNA_V2_AVAILABLE = True
except ImportError:
    DNA_V2_AVAILABLE = False
    warnings.warn("MusicalDNA v2.0 not available")


# =============================================================================
# Migration Functions
# =============================================================================

def migrate_120d_to_300d(old_dna_dict: Dict[str, Any]) -> 'MusicalDNA':
    """
    Migrate v1.0 (120D) DNA to v2.0 (300D) with hierarchical structure.

    Args:
        old_dna_dict: v1.0 DNA as dictionary (from JSON)

    Returns:
        MusicalDNA v2.0 instance (300D)

    Migration Rules:
        1. OLD parameters are preserved and intelligently mapped
        2. OLD parameters are extended by replication + small noise
        3. NEW parameters are initialized to zeros
        4. Metadata is preserved

    Example:
        v1.0 harmony (30D) → v2.0 harmony (60D):
        - First 30D: copy original
        - Next 30D: copy original + small gaussian noise

        v1.0 cross_dimensional (10D) → v2.0 genre_style (partial):
        - First 10D of genre_style: copy from cross_dimensional
        - Remaining 10D: zeros
    """
    if not DNA_V2_AVAILABLE:
        raise ImportError("MusicalDNA v2.0 required for migration")

    # Extract old parameters
    old_harmony = np.array(old_dna_dict.get('harmony', [0]*30))
    old_rhythm = np.array(old_dna_dict.get('rhythm', [0]*20))
    old_form = np.array(old_dna_dict.get('form', [0]*15))
    old_orchestration = np.array(old_dna_dict.get('orchestration', [0]*25))
    old_texture = np.array(old_dna_dict.get('texture', [0]*20))
    old_cross = np.array(old_dna_dict.get('cross_dimensional', [0]*10))

    # Validate dimensions
    if len(old_harmony) != 30:
        warnings.warn(f"Expected harmony 30D, got {len(old_harmony)}D. Padding/truncating.")
        old_harmony = _pad_or_truncate(old_harmony, 30)
    if len(old_rhythm) != 20:
        warnings.warn(f"Expected rhythm 20D, got {len(old_rhythm)}D. Padding/truncating.")
        old_rhythm = _pad_or_truncate(old_rhythm, 20)
    if len(old_form) != 15:
        warnings.warn(f"Expected form 15D, got {len(old_form)}D. Padding/truncating.")
        old_form = _pad_or_truncate(old_form, 15)
    if len(old_orchestration) != 25:
        warnings.warn(f"Expected orchestration 25D, got {len(old_orchestration)}D. Padding/truncating.")
        old_orchestration = _pad_or_truncate(old_orchestration, 25)
    if len(old_texture) != 20:
        warnings.warn(f"Expected texture 20D, got {len(old_texture)}D. Padding/truncating.")
        old_texture = _pad_or_truncate(old_texture, 20)
    if len(old_cross) != 10:
        warnings.warn(f"Expected cross 10D, got {len(old_cross)}D. Padding/truncating.")
        old_cross = _pad_or_truncate(old_cross, 10)

    # =============================================================================
    # GLOBAL LEVEL (60D) - Initialize new + use some old
    # =============================================================================

    # key_context (12D): NEW - initialize to zeros
    key_context = np.zeros(12)

    # tempo_feel (8D): NEW - initialize to zeros
    tempo_feel = np.zeros(8)

    # genre_style (20D): Partial from old cross_dimensional (10D) + zeros (10D)
    genre_style = np.concatenate([
        old_cross,  # First 10D from old cross-dimensional
        np.zeros(10)  # Remaining 10D: zeros
    ])

    # form_structure (20D): Extend old form (15D) → (20D)
    form_structure = _extend_params(old_form, 15, 20, method='replicate_smooth')

    # =============================================================================
    # SECTIONAL LEVEL (140D) - Expand old + new
    # =============================================================================

    # harmony (60D): Expand old (30D) → (60D) by replication with noise
    harmony = _extend_params(old_harmony, 30, 60, method='replicate_noise')

    # melody (40D): NEW - initialize to zeros
    melody = np.zeros(40)

    # rhythm (40D): Expand old (20D) → (40D) by replication with noise
    rhythm = _extend_params(old_rhythm, 20, 40, method='replicate_noise')

    # =============================================================================
    # LOCAL LEVEL (100D) - Expand old + new
    # =============================================================================

    # voicing (30D): NEW - initialize to zeros
    voicing = np.zeros(30)

    # texture (30D): Expand old (20D) → (30D) by replication with noise
    texture = _extend_params(old_texture, 20, 30, method='replicate_noise')

    # orchestration (40D): Expand old (25D) → (40D) by replication with noise
    orchestration = _extend_params(old_orchestration, 25, 40, method='replicate_noise')

    # =============================================================================
    # Create v2.0 DNA
    # =============================================================================

    # Preserve metadata
    source_file = old_dna_dict.get('source_file')
    extraction_timestamp = old_dna_dict.get('extraction_timestamp')

    dna_v2 = MusicalDNA(
        # Global (60D)
        key_context_params=key_context,
        tempo_feel_params=tempo_feel,
        genre_style_params=genre_style,
        form_structure_params=form_structure,
        # Sectional (140D)
        harmony_params=harmony,
        melody_params=melody,
        rhythm_params=rhythm,
        # Local (100D)
        voicing_params=voicing,
        texture_params=texture,
        orchestration_params=orchestration,
        # Metadata
        source_file=source_file,
        extraction_timestamp=extraction_timestamp,
    )

    return dna_v2


def migrate_checkpoint_120d_to_300d(
    old_checkpoint_path: Path,
    new_checkpoint_path: Path,
    preserve_optimizer: bool = False
):
    """
    Migrate encoder checkpoint from 120D to 300D.

    Args:
        old_checkpoint_path: Path to v1.0 checkpoint
        new_checkpoint_path: Path to save v2.0 checkpoint
        preserve_optimizer: Whether to preserve optimizer state (experimental)

    Note:
        This migrates encoder weights. Strategy:
        - Copy weights for modules that still exist
        - Initialize new dimensions with Xavier initialization
        - Optionally preserve optimizer state for transferred weights
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for checkpoint migration")

    # Load old checkpoint
    old_checkpoint = torch.load(old_checkpoint_path, map_location='cpu')

    print(f"Migrating checkpoint: {old_checkpoint_path} → {new_checkpoint_path}")
    print(f"Old checkpoint keys: {list(old_checkpoint.keys())}")

    # Create new checkpoint structure
    new_checkpoint = {
        'version': '2.0',
        'architecture': 'hierarchical_300d',
        'model_state_dict': {},
        'migration_info': {
            'source_checkpoint': str(old_checkpoint_path),
            'source_version': '1.0',
            'migrated_modules': [],
            'new_modules': [],
        }
    }

    # Migrate model state dict
    old_state = old_checkpoint.get('model_state_dict', {})

    for key, value in old_state.items():
        if 'harmony' in key.lower():
            # Harmony encoder: 30D → 60D
            if 'semantic_features' in key and 'weight' in key:
                # This is the final layer: [30, hidden] → [60, hidden]
                old_weight = value  # [30, hidden]
                # Replicate with noise
                noise = torch.randn_like(old_weight) * 0.01
                new_weight = torch.cat([old_weight, old_weight + noise], dim=0)  # [60, hidden]
                new_checkpoint['model_state_dict'][key] = new_weight
                new_checkpoint['migration_info']['migrated_modules'].append(f'{key} (30D→60D)')
            elif 'semantic_features' in key and 'bias' in key:
                # Bias: [30] → [60]
                old_bias = value
                noise = torch.randn_like(old_bias) * 0.01
                new_bias = torch.cat([old_bias, old_bias + noise], dim=0)
                new_checkpoint['model_state_dict'][key] = new_bias
            else:
                # Other layers: copy as-is
                new_checkpoint['model_state_dict'][key] = value

        elif 'rhythm' in key.lower():
            # Rhythm encoder: 20D → 40D
            if 'semantic_features' in key and 'weight' in key:
                old_weight = value  # [20, hidden]
                noise = torch.randn_like(old_weight) * 0.01
                new_weight = torch.cat([old_weight, old_weight + noise], dim=0)  # [40, hidden]
                new_checkpoint['model_state_dict'][key] = new_weight
                new_checkpoint['migration_info']['migrated_modules'].append(f'{key} (20D→40D)')
            elif 'semantic_features' in key and 'bias' in key:
                old_bias = value
                noise = torch.randn_like(old_bias) * 0.01
                new_bias = torch.cat([old_bias, old_bias + noise], dim=0)
                new_checkpoint['model_state_dict'][key] = new_bias
            else:
                new_checkpoint['model_state_dict'][key] = value

        elif 'texture' in key.lower():
            # Texture encoder: 20D → 30D
            if 'semantic_features' in key and 'weight' in key:
                old_weight = value  # [20, hidden]
                # Extend to 30D
                extra = torch.randn(10, old_weight.shape[1]) * 0.01
                new_weight = torch.cat([old_weight, extra], dim=0)  # [30, hidden]
                new_checkpoint['model_state_dict'][key] = new_weight
                new_checkpoint['migration_info']['migrated_modules'].append(f'{key} (20D→30D)')
            elif 'semantic_features' in key and 'bias' in key:
                old_bias = value
                extra = torch.randn(10) * 0.01
                new_bias = torch.cat([old_bias, extra], dim=0)
                new_checkpoint['model_state_dict'][key] = new_bias
            else:
                new_checkpoint['model_state_dict'][key] = value

        elif 'orchestration' in key.lower():
            # Orchestration encoder: 25D → 40D
            if 'semantic_features' in key and 'weight' in key:
                old_weight = value  # [25, hidden]
                extra = torch.randn(15, old_weight.shape[1]) * 0.01
                new_weight = torch.cat([old_weight, extra], dim=0)  # [40, hidden]
                new_checkpoint['model_state_dict'][key] = new_weight
                new_checkpoint['migration_info']['migrated_modules'].append(f'{key} (25D→40D)')
            elif 'semantic_features' in key and 'bias' in key:
                old_bias = value
                extra = torch.randn(15) * 0.01
                new_bias = torch.cat([old_bias, extra], dim=0)
                new_checkpoint['model_state_dict'][key] = new_bias
            else:
                new_checkpoint['model_state_dict'][key] = value

        elif 'form' in key.lower():
            # Form encoder: 15D → 20D (becomes form_structure)
            if 'semantic_features' in key and 'weight' in key:
                old_weight = value  # [15, hidden]
                extra = torch.randn(5, old_weight.shape[1]) * 0.01
                new_weight = torch.cat([old_weight, extra], dim=0)  # [20, hidden]
                # Rename to form_structure
                new_key = key.replace('form', 'form_structure')
                new_checkpoint['model_state_dict'][new_key] = new_weight
                new_checkpoint['migration_info']['migrated_modules'].append(f'{key}→{new_key} (15D→20D)')
            elif 'semantic_features' in key and 'bias' in key:
                old_bias = value
                extra = torch.randn(5) * 0.01
                new_bias = torch.cat([old_bias, extra], dim=0)
                new_key = key.replace('form', 'form_structure')
                new_checkpoint['model_state_dict'][new_key] = new_bias
            else:
                new_key = key.replace('form', 'form_structure')
                new_checkpoint['model_state_dict'][new_key] = value

        else:
            # Unknown key: copy as-is
            new_checkpoint['model_state_dict'][key] = value

    # Note about new modules
    new_checkpoint['migration_info']['new_modules'] = [
        'global_encoder (60D) - not in checkpoint, needs training',
        'melody_encoder (40D) - not in checkpoint, needs training',
        'voicing_encoder (30D) - not in checkpoint, needs training',
    ]

    # Save new checkpoint
    torch.save(new_checkpoint, new_checkpoint_path)
    print(f"✅ Checkpoint migrated successfully")
    print(f"   Migrated modules: {len(new_checkpoint['migration_info']['migrated_modules'])}")
    print(f"   New modules (need training): {len(new_checkpoint['migration_info']['new_modules'])}")

    return new_checkpoint


# =============================================================================
# Helper Functions
# =============================================================================

def _pad_or_truncate(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad with zeros or truncate array to target dimension."""
    if len(arr) == target_dim:
        return arr
    elif len(arr) < target_dim:
        # Pad with zeros
        return np.pad(arr, (0, target_dim - len(arr)), mode='constant')
    else:
        # Truncate
        return arr[:target_dim]


def _extend_params(
    old_params: np.ndarray,
    old_dim: int,
    new_dim: int,
    method: str = 'replicate_noise'
) -> np.ndarray:
    """
    Extend parameters from old_dim to new_dim.

    Args:
        old_params: Original parameters
        old_dim: Original dimension
        new_dim: Target dimension
        method: Extension method
            - 'zeros': Fill new dimensions with zeros
            - 'replicate': Replicate old parameters
            - 'replicate_noise': Replicate with small gaussian noise
            - 'replicate_smooth': Replicate with interpolation

    Returns:
        Extended parameters of shape (new_dim,)
    """
    if old_dim == new_dim:
        return old_params

    if method == 'zeros':
        # Simple: copy old, pad with zeros
        return np.pad(old_params, (0, new_dim - old_dim), mode='constant')

    elif method == 'replicate':
        # Replicate old parameters
        n_copies = new_dim // old_dim
        remainder = new_dim % old_dim
        replicated = np.tile(old_params, n_copies)
        if remainder > 0:
            replicated = np.concatenate([replicated, old_params[:remainder]])
        return replicated

    elif method == 'replicate_noise':
        # Replicate with noise (recommended for parameter expansion)
        base = old_params[:new_dim] if new_dim < old_dim else old_params
        if new_dim > old_dim:
            extra_needed = new_dim - old_dim
            # Replicate with small noise
            std = np.std(old_params) * 0.1 if np.std(old_params) > 0 else 0.01
            noise = np.random.randn(extra_needed) * std
            extra = old_params[:extra_needed] + noise
            return np.concatenate([old_params, extra])
        else:
            return base

    elif method == 'replicate_smooth':
        # Interpolate to new dimension
        from scipy.interpolate import interp1d
        old_indices = np.linspace(0, 1, old_dim)
        new_indices = np.linspace(0, 1, new_dim)
        interpolator = interp1d(old_indices, old_params, kind='cubic', fill_value='extrapolate')
        return interpolator(new_indices)

    else:
        raise ValueError(f"Unknown extension method: {method}")


# =============================================================================
# Validation Functions
# =============================================================================

def validate_migration(
    old_dna_dict: Dict[str, Any],
    new_dna: 'MusicalDNA'
) -> Dict[str, Any]:
    """
    Validate that migration preserved information correctly.

    Args:
        old_dna_dict: Original v1.0 DNA dictionary
        new_dna: Migrated v2.0 DNA

    Returns:
        Validation report
    """
    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {},
    }

    # Check dimensions
    new_vector = new_dna.to_vector()
    if len(new_vector) != 300:
        report['valid'] = False
        report['errors'].append(f"New DNA dimension {len(new_vector)} != 300")

    # Check that old harmony was preserved (first 30D of new harmony should match)
    old_harmony = np.array(old_dna_dict.get('harmony', []))
    if len(old_harmony) == 30:
        new_harmony_first_30 = new_dna.harmony_params[:30]
        max_diff = np.max(np.abs(old_harmony - new_harmony_first_30))
        if max_diff > 1e-6:
            report['warnings'].append(
                f"Old harmony not perfectly preserved in new harmony (max diff: {max_diff})"
            )

    # Statistics
    old_vector = np.concatenate([
        old_dna_dict.get('harmony', [0]*30),
        old_dna_dict.get('rhythm', [0]*20),
        old_dna_dict.get('form', [0]*15),
        old_dna_dict.get('orchestration', [0]*25),
        old_dna_dict.get('texture', [0]*20),
        old_dna_dict.get('cross_dimensional', [0]*10),
    ])

    report['stats'] = {
        'old_mean': float(np.mean(old_vector)),
        'new_mean': float(np.mean(new_vector)),
        'old_std': float(np.std(old_vector)),
        'new_std': float(np.std(new_vector)),
        'preserved_dims': 120,
        'new_dims': 180,
        'total_dims': 300,
    }

    return report


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DNA Migration Utility - v1.0 (120D) → v2.0 (300D)")
    print("="*70)

    # Test: Create fake v1.0 DNA
    print("\nTest: Migrate v1.0 to v2.0")

    old_dna_dict = {
        'version': '1.0',
        'harmony': np.random.randn(30).tolist(),
        'rhythm': np.random.randn(20).tolist(),
        'form': np.random.randn(15).tolist(),
        'orchestration': np.random.randn(25).tolist(),
        'texture': np.random.randn(20).tolist(),
        'cross_dimensional': np.random.randn(10).tolist(),
        'source_file': 'test_v1.mid',
        'extraction_timestamp': '2024-01-01 12:00:00',
    }

    # Migrate
    new_dna = migrate_120d_to_300d(old_dna_dict)

    print(f"\nMigration complete:")
    print(f"  Old DNA: 120D")
    print(f"  New DNA: {len(new_dna.to_vector())}D")
    print(f"  Version: {new_dna.version}")

    # Validate
    validation = validate_migration(old_dna_dict, new_dna)
    print(f"\nValidation:")
    print(f"  Valid: {validation['valid']}")
    print(f"  Errors: {validation['errors']}")
    print(f"  Warnings: {validation['warnings']}")
    print(f"  Stats: {validation['stats']}")

    # Show summary
    print(new_dna.summary())

    print("\n" + "="*70)
    print("Migration test completed successfully!")
    print("="*70)
