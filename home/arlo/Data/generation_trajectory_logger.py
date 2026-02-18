#!/usr/bin/env python3
"""
Generation Trajectory Logger for ACE-Step Model Debugging

Saves latent trajectory data, conditioning, parameters, and audio outputs
for detailed analysis and debugging of the generation process.

Usage:
    from generation_trajectory_logger import GenerationLogger

    logger = GenerationLogger(debug_mode=True, save_every_n_steps=10)

    # During generation, collect trajectory data
    trajectory = {
        'timesteps': timesteps_list,
        'latents': latents_list,
        'additional_data': {...}
    }

    # Log the generation
    metadata = logger.log_generation(
        conditioning={...},
        params={...},
        trajectory=trajectory,
        audio_output_path='path/to/output.wav',
        user_rating=5
    )
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import warnings


class GenerationLogger:
    """
    Logger for saving generation trajectories and metadata for debugging and analysis.

    Attributes:
        debug_mode: Whether to save generation data (default: False)
        save_every_n_steps: Sparsity for trajectory saving (default: 10)
        base_dir: Root directory for saving data
        trajectories_dir: Directory for trajectory .npz files
        metadata_dir: Directory for metadata JSON files
        audio_dir: Directory for audio outputs (optional, can use external paths)
    """

    def __init__(self,
                 debug_mode: bool = False,
                 save_every_n_steps: int = 10,
                 base_dir: str = "/mnt/msdd/generation_debug"):
        """
        Initialize the generation logger.

        Args:
            debug_mode: If True, saves generation data. If False, logger is no-op.
            save_every_n_steps: Save every Nth step of trajectory (for memory efficiency)
            base_dir: Base directory for saving all debug data
        """
        self.debug_mode = debug_mode
        self.save_every_n_steps = save_every_n_steps
        self.base_dir = Path(base_dir)

        if self.debug_mode:
            # Create directory structure
            self.trajectories_dir = self.base_dir / "trajectories"
            self.metadata_dir = self.base_dir / "metadata"
            self.audio_dir = self.base_dir / "audio"

            self.trajectories_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            self.audio_dir.mkdir(parents=True, exist_ok=True)

            print(f"[GenerationLogger] Debug mode enabled. Saving to: {self.base_dir}")
        else:
            print("[GenerationLogger] Debug mode disabled. No data will be saved.")

    def compress_latent(self, latent: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Compress latent tensor for efficient storage.

        Stores statistics and optionally quantized representation.

        Args:
            latent: Latent tensor/array to compress

        Returns:
            Dictionary with compressed representation
        """
        # Convert to numpy if torch tensor
        if isinstance(latent, torch.Tensor):
            latent = latent.detach().cpu().numpy()

        return {
            'shape': latent.shape,
            'dtype': str(latent.dtype),
            'mean': float(np.mean(latent)),
            'std': float(np.std(latent)),
            'min': float(np.min(latent)),
            'max': float(np.max(latent)),
            'norm': float(np.linalg.norm(latent)),
            # Optionally store quantized version for very sparse sampling
            'quantiles': {
                '0.01': float(np.quantile(latent, 0.01)),
                '0.25': float(np.quantile(latent, 0.25)),
                '0.50': float(np.quantile(latent, 0.50)),
                '0.75': float(np.quantile(latent, 0.75)),
                '0.99': float(np.quantile(latent, 0.99)),
            }
        }

    def log_generation(self,
                      conditioning: Dict[str, Any],
                      params: Dict[str, Any],
                      trajectory: Optional[Dict[str, Any]] = None,
                      audio_output_path: Optional[str] = None,
                      user_rating: Optional[float] = None,
                      additional_metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Log a complete generation with trajectory, conditioning, and parameters.

        Args:
            conditioning: Dictionary of conditioning inputs (MIDI, lyrics, etc.)
            params: Generation parameters (CFG scale, steps, seed, etc.)
            trajectory: Optional dictionary containing:
                - 'timesteps': List/array of timestep values
                - 'latents': List of latent tensors at each timestep
                - Additional trajectory data (noise levels, intermediate outputs, etc.)
            audio_output_path: Path to generated audio file (will be copied if not in audio_dir)
            user_rating: Optional user rating/feedback (1-5 scale or None)
            additional_metadata: Any additional metadata to save

        Returns:
            Metadata dictionary with paths and generation_id, or None if debug_mode is False
        """
        if not self.debug_mode:
            return None

        generation_id = str(uuid4())[:8]  # Short UUID for readability
        timestamp = datetime.now().isoformat()

        # Prepare metadata
        metadata = {
            'generation_id': generation_id,
            'timestamp': timestamp,
            'conditioning': self._serialize_conditioning(conditioning),
            'params': self._serialize_params(params),
            'user_rating': user_rating,
        }

        # Save trajectory if provided
        if trajectory is not None:
            trajectory_path = self._save_trajectory(generation_id, trajectory)
            metadata['trajectory_path'] = str(trajectory_path)
            metadata['trajectory_stats'] = self._compute_trajectory_stats(trajectory)

        # Handle audio output
        if audio_output_path is not None:
            audio_path = self._handle_audio(generation_id, audio_output_path)
            metadata['audio_path'] = str(audio_path)

        # Add additional metadata
        if additional_metadata is not None:
            metadata['additional'] = additional_metadata

        # Save metadata JSON
        metadata_path = self.metadata_dir / f"{generation_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        metadata['metadata_path'] = str(metadata_path)

        print(f"[GenerationLogger] Saved generation {generation_id}")
        print(f"  - Metadata: {metadata_path}")
        if 'trajectory_path' in metadata:
            print(f"  - Trajectory: {metadata['trajectory_path']}")
        if 'audio_path' in metadata:
            print(f"  - Audio: {metadata['audio_path']}")

        return metadata

    def _save_trajectory(self, generation_id: str, trajectory: Dict[str, Any]) -> Path:
        """
        Save trajectory data to compressed .npz file.

        Args:
            generation_id: Unique identifier for this generation
            trajectory: Dictionary with timesteps, latents, and other data

        Returns:
            Path to saved trajectory file
        """
        trajectory_path = self.trajectories_dir / f"{generation_id}.npz"

        # Extract and downsample trajectory data
        save_dict = {}

        # Handle timesteps
        if 'timesteps' in trajectory:
            timesteps = trajectory['timesteps']
            if isinstance(timesteps, torch.Tensor):
                timesteps = timesteps.detach().cpu().numpy()
            save_dict['timesteps'] = np.array(timesteps)[::self.save_every_n_steps]

        # Handle latents - save both full and compressed versions
        if 'latents' in trajectory:
            latents = trajectory['latents']

            # Downsample
            sparse_latents = latents[::self.save_every_n_steps]

            # Convert to numpy and stack
            latents_numpy = []
            for lat in sparse_latents:
                if isinstance(lat, torch.Tensor):
                    lat = lat.detach().cpu().numpy()
                latents_numpy.append(lat)

            # Stack into single array for efficient storage
            try:
                save_dict['latents'] = np.stack(latents_numpy, axis=0)
            except (ValueError, TypeError) as e:
                # If stacking fails, save as object array
                warnings.warn(f"Could not stack latents, saving as object array: {e}")
                save_dict['latents'] = np.array(latents_numpy, dtype=object)

            # Save compressed statistics for each latent
            compressed_stats = [self.compress_latent(lat) for lat in sparse_latents]
            save_dict['latent_stats'] = np.array(compressed_stats, dtype=object)

        # Handle any additional trajectory data
        for key, value in trajectory.items():
            if key not in ['timesteps', 'latents']:
                # Try to save additional data
                try:
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    elif isinstance(value, list):
                        # Downsample lists
                        value = value[::self.save_every_n_steps]
                        if len(value) > 0 and isinstance(value[0], torch.Tensor):
                            value = [v.detach().cpu().numpy() for v in value]

                    save_dict[key] = value
                except Exception as e:
                    warnings.warn(f"Could not save trajectory key '{key}': {e}")

        # Save compressed
        np.savez_compressed(trajectory_path, **save_dict)

        return trajectory_path

    def _serialize_conditioning(self, conditioning: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize conditioning data for JSON storage.

        Converts tensors to lists, extracts shapes, etc.
        """
        serialized = {}

        for key, value in conditioning.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = {
                    'type': 'tensor',
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'device': str(value.device),
                    'mean': float(value.float().mean()),
                    'std': float(value.float().std()),
                }
            elif isinstance(value, np.ndarray):
                serialized[key] = {
                    'type': 'ndarray',
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'mean': float(np.mean(value)),
                    'std': float(np.std(value)),
                }
            elif isinstance(value, (str, int, float, bool, type(None))):
                serialized[key] = value
            elif isinstance(value, (list, tuple)):
                serialized[key] = list(value)
            elif isinstance(value, dict):
                serialized[key] = self._serialize_conditioning(value)
            else:
                serialized[key] = str(value)

        return serialized

    def _serialize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize generation parameters for JSON storage.
        """
        serialized = {}

        for key, value in params.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serialized[key] = value
            elif isinstance(value, (list, tuple)):
                serialized[key] = list(value)
            elif isinstance(value, dict):
                serialized[key] = self._serialize_params(value)
            elif isinstance(value, torch.Tensor):
                serialized[key] = value.detach().cpu().tolist()
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = str(value)

        return serialized

    def _compute_trajectory_stats(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute summary statistics about the trajectory.
        """
        stats = {}

        if 'timesteps' in trajectory:
            timesteps = trajectory['timesteps']
            if isinstance(timesteps, torch.Tensor):
                timesteps = timesteps.detach().cpu().numpy()
            stats['num_steps'] = len(timesteps)
            stats['timestep_range'] = [float(np.min(timesteps)), float(np.max(timesteps))]

        if 'latents' in trajectory:
            latents = trajectory['latents']
            stats['num_latents_saved'] = len(latents[::self.save_every_n_steps])

            # Compute norm trajectory
            norms = []
            for lat in latents[::self.save_every_n_steps]:
                if isinstance(lat, torch.Tensor):
                    norms.append(float(torch.norm(lat)))
                else:
                    norms.append(float(np.linalg.norm(lat)))

            stats['latent_norms'] = {
                'min': float(np.min(norms)),
                'max': float(np.max(norms)),
                'mean': float(np.mean(norms)),
                'std': float(np.std(norms)),
            }

        return stats

    def _handle_audio(self, generation_id: str, audio_output_path: str) -> Path:
        """
        Handle audio output - copy to audio_dir or store reference.

        Args:
            generation_id: Unique identifier
            audio_output_path: Path to audio file

        Returns:
            Path where audio is stored/referenced
        """
        audio_output_path = Path(audio_output_path)

        # If audio is already in our audio_dir, just return the path
        if audio_output_path.parent == self.audio_dir:
            return audio_output_path

        # Otherwise, copy it to our audio_dir
        dest_path = self.audio_dir / f"{generation_id}{audio_output_path.suffix}"

        try:
            import shutil
            shutil.copy2(audio_output_path, dest_path)
            return dest_path
        except Exception as e:
            warnings.warn(f"Could not copy audio file: {e}")
            # Just store reference to original path
            return audio_output_path

    def load_generation(self, generation_id: str) -> Dict[str, Any]:
        """
        Load a previously saved generation by ID.

        Args:
            generation_id: The generation ID to load

        Returns:
            Dictionary with metadata, trajectory (if exists), etc.
        """
        if not self.debug_mode:
            raise RuntimeError("Cannot load generations when debug_mode is False")

        # Load metadata
        metadata_path = self.metadata_dir / f"{generation_id}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata found for generation {generation_id}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load trajectory if exists
        if 'trajectory_path' in metadata:
            trajectory_path = Path(metadata['trajectory_path'])
            if trajectory_path.exists():
                trajectory_data = np.load(trajectory_path, allow_pickle=True)
                metadata['trajectory_data'] = {key: trajectory_data[key] for key in trajectory_data.files}

        return metadata

    def list_generations(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all saved generations.

        Args:
            limit: Maximum number of generations to return (newest first)

        Returns:
            List of metadata dictionaries
        """
        if not self.debug_mode:
            return []

        metadata_files = sorted(
            self.metadata_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if limit is not None:
            metadata_files = metadata_files[:limit]

        generations = []
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                generations.append(metadata)

        return generations


# Singleton instance for easy import
_global_logger: Optional[GenerationLogger] = None


def get_logger(debug_mode: bool = False,
               save_every_n_steps: int = 10,
               base_dir: str = "/mnt/msdd/generation_debug") -> GenerationLogger:
    """
    Get or create global GenerationLogger instance.

    Args:
        debug_mode: Enable debug logging
        save_every_n_steps: Sparsity for trajectory saving
        base_dir: Base directory for debug data

    Returns:
        Global GenerationLogger instance
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = GenerationLogger(
            debug_mode=debug_mode,
            save_every_n_steps=save_every_n_steps,
            base_dir=base_dir
        )

    return _global_logger


def set_debug_mode(enabled: bool):
    """
    Enable or disable debug mode for the global logger.

    Args:
        enabled: Whether to enable debug mode
    """
    global _global_logger

    if _global_logger is not None:
        _global_logger.debug_mode = enabled
        if enabled:
            print(f"[GenerationLogger] Debug mode enabled. Saving to: {_global_logger.base_dir}")
        else:
            print("[GenerationLogger] Debug mode disabled.")


if __name__ == "__main__":
    # Example usage
    print("=== GenerationLogger Test ===\n")

    # Create logger with debug mode enabled
    logger = GenerationLogger(debug_mode=True, save_every_n_steps=5)

    # Simulate a generation trajectory
    num_steps = 50
    timesteps = np.linspace(1000, 0, num_steps)
    latents = [np.random.randn(4, 128, 64) for _ in range(num_steps)]

    trajectory = {
        'timesteps': timesteps,
        'latents': latents,
        'noise_levels': np.linspace(1.0, 0.0, num_steps),
    }

    conditioning = {
        'midi_pr': np.random.rand(128, 100),
        'text': "test lyrics",
        'style': "pop",
    }

    params = {
        'cfg_scale': 7.5,
        'num_steps': num_steps,
        'seed': 42,
        'sampler': 'ddpm',
    }

    # Log the generation
    metadata = logger.log_generation(
        conditioning=conditioning,
        params=params,
        trajectory=trajectory,
        audio_output_path=None,
        user_rating=4.5
    )

    print(f"\n=== Logged Generation ===")
    print(f"Generation ID: {metadata['generation_id']}")
    print(f"Saved {len(latents[::5])} trajectory points (every 5th step)")

    # Test loading
    print(f"\n=== Testing Load ===")
    loaded = logger.load_generation(metadata['generation_id'])
    print(f"Loaded generation {loaded['generation_id']}")
    print(f"Trajectory data keys: {list(loaded.get('trajectory_data', {}).keys())}")

    # List all generations
    print(f"\n=== All Generations ===")
    all_gens = logger.list_generations(limit=5)
    print(f"Found {len(all_gens)} generation(s)")
