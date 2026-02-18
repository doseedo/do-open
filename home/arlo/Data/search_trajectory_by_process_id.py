#!/usr/bin/env python3
"""
Helper script to search debug trajectories by frontend process_id.

Usage:
    python search_trajectory_by_process_id.py <process_id>
    python search_trajectory_by_process_id.py 612d2c00-9785-4064-ba3c-ef153c941d21
"""

import sys
import json
from pathlib import Path
from generation_trajectory_logger import GenerationLogger


def search_by_process_id(process_id: str):
    """
    Search for trajectory data associated with a frontend process_id.

    Args:
        process_id: The UUID from the frontend file path
    """
    logger = GenerationLogger(debug_mode=True)

    print(f"Searching for trajectories with process_id: {process_id}\n")

    found = []
    all_generations = logger.list_generations()

    for gen in all_generations:
        gen_process_id = gen.get('additional', {}).get('process_id')
        if gen_process_id == process_id:
            found.append(gen)

    if not found:
        print(f"❌ No trajectories found for process_id: {process_id}")
        print(f"\nSearched {len(all_generations)} total generations.")
        return

    print(f"✅ Found {len(found)} trajectory/trajectories:\n")

    for i, gen in enumerate(found, 1):
        print(f"{'='*80}")
        print(f"Result {i}/{len(found)}")
        print(f"{'='*80}")
        print(f"Generation ID: {gen['generation_id']}")
        print(f"Timestamp: {gen['timestamp']}")
        print(f"Process ID: {gen.get('additional', {}).get('process_id')}")
        print(f"\nPaths:")
        print(f"  Trajectory: {gen.get('trajectory_path', 'N/A')}")
        print(f"  Audio: {gen.get('audio_path', 'N/A')}")
        print(f"  Metadata: {gen.get('metadata_path', 'N/A')}")
        print(f"  Output Dir: {gen.get('additional', {}).get('output_dir', 'N/A')}")

        print(f"\nParameters:")
        params = gen.get('params', {})
        print(f"  Steps: {params.get('steps')}")
        print(f"  Seed: {params.get('seed')}")
        print(f"  CFG Weight: {params.get('cfg_weight')}")
        print(f"  Noise Level: {params.get('noise_level')}")
        print(f"  Fast Mode: {params.get('fast_mode_variant', 'None')}")

        print(f"\nConditioning:")
        cond = gen.get('conditioning', {})
        print(f"  Group: {cond.get('group')}")
        print(f"  Subgroup: {cond.get('subgroup')}")
        print(f"  Piano Roll: {cond.get('piano_roll_shape')}")

        print(f"\nTrajectory Stats:")
        stats = gen.get('trajectory_stats', {})
        print(f"  Steps: {stats.get('num_steps')}")
        print(f"  Saved Points: {stats.get('num_latents_saved')}")
        if 'latent_norms' in stats:
            norms = stats['latent_norms']
            print(f"  Latent Norms: min={norms.get('min', 0):.2f}, "
                  f"max={norms.get('max', 0):.2f}, "
                  f"mean={norms.get('mean', 0):.2f}")

        print(f"\nAdditional:")
        additional = gen.get('additional', {})
        print(f"  Voice Index: {additional.get('voice_idx')}")
        print(f"  MIDI Duration: {additional.get('midi_end_time')}s")
        print(f"  Audio Duration: {additional.get('actual_duration')}s")

        print()


def list_recent_generations(limit=10):
    """List recent generations with their process_ids."""
    logger = GenerationLogger(debug_mode=True)

    print(f"Recent {limit} generations:\n")
    print(f"{'Generation ID':<12} {'Process ID':<38} {'Timestamp':<20} {'Steps':<6}")
    print("-" * 80)

    for gen in logger.list_generations(limit=limit):
        gen_id = gen['generation_id']
        process_id = gen.get('additional', {}).get('process_id', 'N/A')[:36]
        timestamp = gen['timestamp'].split('T')[1][:8]  # Just time
        steps = gen.get('params', {}).get('steps', '?')

        print(f"{gen_id:<12} {process_id:<38} {timestamp:<20} {steps:<6}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_trajectory_by_process_id.py <process_id>")
        print("   or: python search_trajectory_by_process_id.py --list")
        print("\nExample:")
        print("  python search_trajectory_by_process_id.py 612d2c00-9785-4064-ba3c-ef153c941d21")
        print("  python search_trajectory_by_process_id.py --list  # List recent generations")
        sys.exit(1)

    if sys.argv[1] == "--list":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        list_recent_generations(limit)
    else:
        process_id = sys.argv[1]
        search_by_process_id(process_id)
