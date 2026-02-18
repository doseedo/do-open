#!/usr/bin/env python3
"""
Analyze user feedback logs with trajectory data.

This script combines:
1. Frontend feedback logs (likes/dislikes from localStorage)
2. Backend trajectory debug data (latent evolution, parameters)

Usage:
    python analyze_feedback_with_trajectories.py feedback_log.json
    python analyze_feedback_with_trajectories.py --stats
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from generation_trajectory_logger import GenerationLogger


def load_frontend_feedback_log(filepath: str) -> List[Dict]:
    """
    Load feedback log exported from frontend.

    Args:
        filepath: Path to JSON file downloaded from frontend

    Returns:
        List of feedback entries
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def find_trajectory_for_feedback(process_id: str, logger: GenerationLogger) -> Dict[str, Any]:
    """
    Find the trajectory data for a given process_id.

    Args:
        process_id: Frontend process ID
        logger: GenerationLogger instance

    Returns:
        Generation metadata dict or None
    """
    for gen in logger.list_generations():
        if gen.get('additional', {}).get('process_id') == process_id:
            return gen
    return None


def analyze_feedback_patterns(feedback_log: List[Dict], logger: GenerationLogger):
    """
    Analyze patterns between user feedback and generation parameters.

    Args:
        feedback_log: List of feedback entries from frontend
        logger: GenerationLogger instance
    """
    print("=" * 80)
    print("FEEDBACK + TRAJECTORY ANALYSIS")
    print("=" * 80)
    print()

    # Collect data for analysis
    liked_trajectories = []
    disliked_trajectories = []
    missing_trajectories = []

    for entry in feedback_log:
        process_id = entry.get('processId')
        feedback = entry.get('feedback')
        timestamp = entry.get('timestamp')
        track_name = entry.get('trackName', 'Unknown')

        if not process_id:
            print(f"⚠️  No process_id for: {track_name}")
            missing_trajectories.append(entry)
            continue

        # Find corresponding trajectory
        traj = find_trajectory_for_feedback(process_id, logger)

        if traj:
            entry_with_traj = {
                'feedback_entry': entry,
                'trajectory': traj
            }

            if feedback == 'like':
                liked_trajectories.append(entry_with_traj)
            elif feedback == 'dislike':
                disliked_trajectories.append(entry_with_traj)

            print(f"{'✅' if feedback == 'like' else '❌'} {feedback.upper()}: {track_name[:40]}")
            print(f"   Process ID: {process_id}")
            print(f"   Trajectory: {traj['generation_id']}")
            print(f"   Timestamp: {timestamp}")
            print()
        else:
            print(f"❓ {feedback.upper()}: {track_name[:40]} - NO TRAJECTORY FOUND")
            print(f"   Process ID: {process_id}")
            print()
            missing_trajectories.append(entry)

    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    total = len(feedback_log)
    likes = len([e for e in feedback_log if e.get('feedback') == 'like'])
    dislikes = len([e for e in feedback_log if e.get('feedback') == 'dislike'])

    print(f"Total feedback entries: {total}")
    print(f"  Likes: {likes} ({likes/total*100:.1f}%)")
    print(f"  Dislikes: {dislikes} ({dislikes/total*100:.1f}%)")
    print(f"  With trajectories: {len(liked_trajectories) + len(disliked_trajectories)}")
    print(f"  Missing trajectories: {len(missing_trajectories)}")
    print()

    if liked_trajectories or disliked_trajectories:
        print("=" * 80)
        print("PARAMETER COMPARISON: LIKED vs DISLIKED")
        print("=" * 80)
        print()

        compare_parameters(liked_trajectories, disliked_trajectories)

    if liked_trajectories or disliked_trajectories:
        print("=" * 80)
        print("TRAJECTORY CHARACTERISTICS")
        print("=" * 80)
        print()

        compare_trajectories(liked_trajectories, disliked_trajectories)


def compare_parameters(liked: List[Dict], disliked: List[Dict]):
    """Compare generation parameters between liked and disliked tracks."""

    def extract_params(entries):
        params = {
            'steps': [],
            'cfg_weight': [],
            'noise_level': [],
            'seed': [],
            'pitch_fidelity_boost': [],
            'onset_guidance_boost': [],
        }

        for entry in entries:
            traj_params = entry['trajectory'].get('params', {})
            for key in params.keys():
                val = traj_params.get(key)
                if val is not None:
                    params[key].append(val)

        return params

    liked_params = extract_params(liked)
    disliked_params = extract_params(disliked)

    print(f"{'Parameter':<25} {'Liked (avg)':<15} {'Disliked (avg)':<15} {'Difference'}")
    print("-" * 80)

    for param in liked_params.keys():
        liked_vals = liked_params[param]
        disliked_vals = disliked_params[param]

        if liked_vals and disliked_vals:
            liked_avg = np.mean(liked_vals)
            disliked_avg = np.mean(disliked_vals)
            diff = liked_avg - disliked_avg
            diff_pct = (diff / disliked_avg * 100) if disliked_avg != 0 else 0

            print(f"{param:<25} {liked_avg:<15.2f} {disliked_avg:<15.2f} "
                  f"{diff:+.2f} ({diff_pct:+.1f}%)")
        elif liked_vals:
            print(f"{param:<25} {np.mean(liked_vals):<15.2f} {'N/A':<15} N/A")
        elif disliked_vals:
            print(f"{param:<25} {'N/A':<15} {np.mean(disliked_vals):<15.2f} N/A")

    print()


def compare_trajectories(liked: List[Dict], disliked: List[Dict]):
    """Compare trajectory characteristics between liked and disliked tracks."""

    def extract_trajectory_stats(entries):
        stats = {
            'latent_norm_mean': [],
            'latent_norm_std': [],
            'latent_norm_min': [],
            'latent_norm_max': [],
            'num_steps': [],
        }

        for entry in entries:
            traj_stats = entry['trajectory'].get('trajectory_stats', {})
            if 'latent_norms' in traj_stats:
                norms = traj_stats['latent_norms']
                stats['latent_norm_mean'].append(norms.get('mean', 0))
                stats['latent_norm_std'].append(norms.get('std', 0))
                stats['latent_norm_min'].append(norms.get('min', 0))
                stats['latent_norm_max'].append(norms.get('max', 0))

            if 'num_steps' in traj_stats:
                stats['num_steps'].append(traj_stats['num_steps'])

        return stats

    liked_stats = extract_trajectory_stats(liked)
    disliked_stats = extract_trajectory_stats(disliked)

    print(f"{'Trajectory Metric':<25} {'Liked (avg)':<15} {'Disliked (avg)':<15} {'Difference'}")
    print("-" * 80)

    for metric in liked_stats.keys():
        liked_vals = liked_stats[metric]
        disliked_vals = disliked_stats[metric]

        if liked_vals and disliked_vals:
            liked_avg = np.mean(liked_vals)
            disliked_avg = np.mean(disliked_vals)
            diff = liked_avg - disliked_avg
            diff_pct = (diff / disliked_avg * 100) if disliked_avg != 0 else 0

            print(f"{metric:<25} {liked_avg:<15.2f} {disliked_avg:<15.2f} "
                  f"{diff:+.2f} ({diff_pct:+.1f}%)")

    print()


def export_analysis_csv(feedback_log: List[Dict], logger: GenerationLogger, output_path: str):
    """
    Export combined feedback + trajectory data as CSV for external analysis.

    Args:
        feedback_log: Frontend feedback log
        logger: GenerationLogger instance
        output_path: Output CSV file path
    """
    import csv

    rows = []

    for entry in feedback_log:
        process_id = entry.get('processId')
        traj = find_trajectory_for_feedback(process_id, logger) if process_id else None

        row = {
            'timestamp': entry.get('timestamp'),
            'process_id': process_id,
            'track_name': entry.get('trackName'),
            'feedback': entry.get('feedback'),
            'audio_url': entry.get('audioUrl'),
        }

        if traj:
            params = traj.get('params', {})
            row.update({
                'generation_id': traj.get('generation_id'),
                'steps': params.get('steps'),
                'cfg_weight': params.get('cfg_weight'),
                'noise_level': params.get('noise_level'),
                'seed': params.get('seed'),
                'pitch_fidelity_boost': params.get('pitch_fidelity_boost'),
                'onset_guidance_boost': params.get('onset_guidance_boost'),
                'pitch_snap_strength': params.get('pitch_snap_strength'),
                'fast_mode_variant': params.get('fast_mode_variant'),
            })

            traj_stats = traj.get('trajectory_stats', {})
            if 'latent_norms' in traj_stats:
                norms = traj_stats['latent_norms']
                row.update({
                    'latent_norm_mean': norms.get('mean'),
                    'latent_norm_std': norms.get('std'),
                    'latent_norm_min': norms.get('min'),
                    'latent_norm_max': norms.get('max'),
                })

            cond = traj.get('conditioning', {})
            row.update({
                'group': cond.get('group'),
                'subgroup': cond.get('subgroup'),
            })

        rows.append(row)

    # Write CSV
    if rows:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"✅ Exported {len(rows)} entries to: {output_path}")


def show_usage():
    """Show usage instructions."""
    print("""
Usage:
    python analyze_feedback_with_trajectories.py <feedback_log.json>
    python analyze_feedback_with_trajectories.py <feedback_log.json> --export output.csv
    python analyze_feedback_with_trajectories.py --help

Examples:
    # Analyze feedback log
    python analyze_feedback_with_trajectories.py ~/Downloads/doseedo_feedback_log_2025-11-08.json

    # Export to CSV for spreadsheet analysis
    python analyze_feedback_with_trajectories.py feedback.json --export analysis.csv

How to get feedback log:
    1. In the frontend, use the browser console:
       > import { downloadLog } from './utils/feedbackLogger'
       > downloadLog()

    2. Or download from localStorage:
       const log = JSON.parse(localStorage.getItem('doseedo_generation_feedback_log'))
       console.log(JSON.stringify(log, null, 2))
       Copy and save to a file
""")


if __name__ == "__main__":
    if len(sys.argv) < 2 or '--help' in sys.argv:
        show_usage()
        sys.exit(0)

    feedback_file = sys.argv[1]

    if not Path(feedback_file).exists():
        print(f"❌ File not found: {feedback_file}")
        sys.exit(1)

    # Load feedback log
    print(f"📂 Loading feedback log: {feedback_file}")
    feedback_log = load_frontend_feedback_log(feedback_file)
    print(f"✅ Loaded {len(feedback_log)} feedback entries\n")

    # Initialize trajectory logger
    logger = GenerationLogger(debug_mode=True)

    # Analyze
    analyze_feedback_patterns(feedback_log, logger)

    # Export if requested
    if '--export' in sys.argv:
        export_idx = sys.argv.index('--export')
        if len(sys.argv) > export_idx + 1:
            output_csv = sys.argv[export_idx + 1]
            print()
            export_analysis_csv(feedback_log, logger, output_csv)
        else:
            print("❌ --export requires an output filename")
