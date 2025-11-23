#!/usr/bin/env python
"""
Analyze improvement distribution from discovery iteration to predict:
1. How many primitives needed to reach target quality (80%, 90%, 95%, 99%)
2. What threshold to set for each quality target
3. Power law parameters (α, β) of the improvement distribution

Run this after completing iteration 1 to calibrate your discovery pipeline.

Usage:
    python scripts/analyze_improvements.py --log discovery_iteration_1.log
"""

import re
import numpy as np
import argparse
from scipy.optimize import curve_fit
from scipy.special import zeta
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def parse_improvements_from_log(log_file: str):
    """Extract improvement values from discovery log."""
    improvements = []
    baseline_error = None

    with open(log_file, 'r') as f:
        for line in f:
            # Parse baseline error
            if 'baseline' in line.lower() and 'error' in line.lower():
                match = re.search(r'(\d+\.\d+)', line)
                if match:
                    baseline_error = float(match.group(1))

            # Parse improvement from diagnostic lines
            # Format: [PID 12345] name: transform=1.2s, error=0.3s, total=1.5s, imp=0.000234
            match = re.search(r'imp=([-\d.e]+)', line)
            if match:
                imp = float(match.group(1))
                improvements.append(imp)

    improvements = np.array(improvements)

    # Filter out negative/zero improvements for power law fitting
    positive_improvements = improvements[improvements > 0]

    return improvements, positive_improvements, baseline_error


def fit_power_law(improvements):
    """
    Fit power law: improvement(k) = α / k^β

    Returns:
        α: scaling constant
        β: power law exponent
    """
    # Sort descending
    sorted_imp = np.sort(improvements)[::-1]
    ranks = np.arange(1, len(sorted_imp) + 1)

    # Fit power law
    def power_law(k, alpha, beta):
        return alpha / (k ** beta)

    try:
        params, _ = curve_fit(
            power_law,
            ranks,
            sorted_imp,
            p0=[sorted_imp[0], 1.0],  # Initial guess
            bounds=([0, 0.1], [np.inf, 2.0])  # α > 0, 0.1 < β < 2.0
        )
        alpha, beta = params

        # Compute R² goodness of fit
        predictions = power_law(ranks, alpha, beta)
        ss_res = np.sum((sorted_imp - predictions) ** 2)
        ss_tot = np.sum((sorted_imp - np.mean(sorted_imp)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return alpha, beta, r_squared

    except Exception as e:
        print(f"Warning: Power law fitting failed: {e}")
        return None, None, 0.0


def estimate_primitives_needed(alpha, beta, baseline_error, target_quality):
    """
    Estimate how many primitives needed to reach target quality.

    Quality = 1 - (final_error / baseline_error)

    We need: Σ(k=1 to n) α/k^β ≥ (1 - target_quality) * baseline_error

    For β ≈ 1: Σ 1/k ≈ ln(n) + γ (where γ ≈ 0.577)
    For β ≠ 1: Use numerical solution
    """
    required_improvement = target_quality * baseline_error

    if beta is None or alpha is None:
        return None

    # Numerical search for n
    n = 1
    cumulative = 0

    while cumulative < required_improvement and n < 100000:
        cumulative += alpha / (n ** beta)
        n += 1

    if n >= 100000:
        return None  # Couldn't reach target

    return n


def plot_improvement_distribution(improvements, alpha, beta, baseline_error, output_path='improvement_analysis.png'):
    """Create visualization of improvement distribution and power law fit."""
    sorted_imp = np.sort(improvements)[::-1]
    ranks = np.arange(1, len(sorted_imp) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Improvement vs Rank (log-log)
    ax = axes[0, 0]
    ax.loglog(ranks, sorted_imp, 'b.', alpha=0.5, label='Actual improvements')

    if alpha is not None and beta is not None:
        predicted = alpha / (ranks ** beta)
        ax.loglog(ranks, predicted, 'r-', linewidth=2, label=f'Power law fit: α/{{"k^{beta:.2f}"}}')

    ax.set_xlabel('Rank (k)')
    ax.set_ylabel('Improvement')
    ax.set_title('Improvement Distribution (Log-Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative improvement
    ax = axes[0, 1]
    cumulative = np.cumsum(sorted_imp)
    ax.plot(ranks, cumulative, 'b-', linewidth=2)

    if baseline_error:
        quality = cumulative / baseline_error
        ax2 = ax.twinx()
        ax2.plot(ranks, quality * 100, 'g--', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Quality (%)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        # Mark quality targets
        for target in [0.8, 0.9, 0.95, 0.99]:
            if alpha and beta:
                n = estimate_primitives_needed(alpha, beta, baseline_error, target)
                if n:
                    ax.axvline(n, color='red', linestyle=':', alpha=0.5)
                    ax.text(n, cumulative[min(n-1, len(cumulative)-1)],
                           f'{target*100:.0f}%', rotation=90, va='bottom')

    ax.set_xlabel('Number of primitives (k)')
    ax.set_ylabel('Cumulative improvement')
    ax.set_title('Cumulative Improvement vs Library Size')
    ax.grid(True, alpha=0.3)

    # Plot 3: Histogram of improvements
    ax = axes[1, 0]
    ax.hist(improvements, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Improvement value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Improvements')
    ax.axvline(0, color='red', linestyle='--', label='Zero improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Threshold analysis
    ax = axes[1, 1]
    percentiles = np.arange(1, 100)
    threshold_values = np.percentile(improvements, 100 - percentiles)

    ax.semilogy(percentiles, threshold_values, 'b-', linewidth=2)
    ax.set_xlabel('Top X% of candidates accepted')
    ax.set_ylabel('Threshold (log scale)')
    ax.set_title('Threshold vs Acceptance Rate')
    ax.grid(True, alpha=0.3)

    # Mark current threshold
    current_threshold = 0.0001
    ax.axhline(current_threshold, color='red', linestyle='--',
              label=f'Current threshold: {current_threshold}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze improvement distribution')
    parser.add_argument('--log', type=str,
                       default='discovery_cpu_test.log',
                       help='Path to discovery log file')
    parser.add_argument('--output', type=str,
                       default='improvement_analysis.png',
                       help='Output path for visualization')

    args = parser.parse_args()

    print("="*70)
    print("IMPROVEMENT DISTRIBUTION ANALYSIS")
    print("="*70)

    # Parse log file
    print(f"\nParsing log file: {args.log}")
    all_improvements, positive_improvements, baseline_error = parse_improvements_from_log(args.log)

    if len(all_improvements) == 0:
        print("ERROR: No improvements found in log file!")
        print("Make sure the log contains lines like: [PID xxx] name: ... imp=0.123")
        return

    print(f"\n✓ Found {len(all_improvements)} compositions tested")
    print(f"  Positive improvements: {len(positive_improvements)} ({len(positive_improvements)/len(all_improvements)*100:.1f}%)")
    print(f"  Negative improvements: {np.sum(all_improvements < 0)} ({np.sum(all_improvements < 0)/len(all_improvements)*100:.1f}%)")
    print(f"  Zero improvements: {np.sum(all_improvements == 0)} ({np.sum(all_improvements == 0)/len(all_improvements)*100:.1f}%)")

    if baseline_error:
        print(f"\n✓ Baseline error: {baseline_error:.6f}")
    else:
        print("\nWarning: Could not find baseline error in log")
        baseline_error = 1.0  # Default

    # Statistics
    print(f"\nImprovement statistics:")
    print(f"  Min: {np.min(all_improvements):.8f}")
    print(f"  Max: {np.max(all_improvements):.8f}")
    print(f"  Mean: {np.mean(all_improvements):.8f}")
    print(f"  Median: {np.median(all_improvements):.8f}")
    print(f"  Std: {np.std(all_improvements):.8f}")

    # Current threshold analysis
    current_threshold = 0.0001
    accepted = np.sum(all_improvements > current_threshold)
    print(f"\nCurrent threshold: {current_threshold}")
    print(f"  Candidates accepted: {accepted}/{len(all_improvements)} ({accepted/len(all_improvements)*100:.1f}%)")

    # Fit power law
    if len(positive_improvements) > 10:
        print(f"\nFitting power law to {len(positive_improvements)} positive improvements...")
        alpha, beta, r_squared = fit_power_law(positive_improvements)

        if alpha and beta:
            print(f"\n✓ Power law fit: improvement(k) = {alpha:.6f} / k^{beta:.3f}")
            print(f"  R² goodness of fit: {r_squared:.4f}")

            # Predictions
            print(f"\n{'='*70}")
            print("PREDICTIONS FOR DIFFERENT QUALITY TARGETS")
            print(f"{'='*70}")
            print(f"{'Target':<10} {'Primitives':<12} {'Weakest Imp':<15} {'Threshold':<15} {'Feasible?':<10}")
            print(f"{'-'*70}")

            for target in [0.80, 0.90, 0.95, 0.99]:
                n = estimate_primitives_needed(alpha, beta, baseline_error, target)

                if n:
                    weakest_imp = alpha / (n ** beta)
                    threshold = weakest_imp * 0.5  # Safety margin
                    feasible = "✅" if n < 500 else "⚠️" if n < 1000 else "❌"

                    print(f"{target*100:>6.0f}%    {n:<12} {weakest_imp:<15.8f} {threshold:<15.8f} {feasible}")
                else:
                    print(f"{target*100:>6.0f}%    {'Unreachable':<12}")

            # Recommendations
            print(f"\n{'='*70}")
            print("RECOMMENDATIONS")
            print(f"{'='*70}")

            n_99 = estimate_primitives_needed(alpha, beta, baseline_error, 0.99)

            if n_99 and n_99 < 300:
                print("\n✅ 99% quality is FEASIBLE!")
                threshold_99 = (alpha / (n_99 ** beta)) * 0.5
                print(f"   Set threshold to: {threshold_99:.8f}")
                print(f"   Expected primitives: {n_99}")
                print(f"   Estimated iterations: {n_99 // 30 + 1}")

            elif n_99 and n_99 < 600:
                print("\n⚠️  99% quality is DIFFICULT but possible")
                threshold_99 = (alpha / (n_99 ** beta)) * 0.5
                print(f"   Set threshold to: {threshold_99:.8f}")
                print(f"   Expected primitives: {n_99}")
                print(f"   Estimated time: {(n_99 // 30 + 1) * 60 / 60:.1f} hours")
                print("\n   Alternative: Target 95% quality instead:")

                n_95 = estimate_primitives_needed(alpha, beta, baseline_error, 0.95)
                if n_95:
                    threshold_95 = (alpha / (n_95 ** beta)) * 0.5
                    print(f"   Threshold: {threshold_95:.8f}")
                    print(f"   Primitives: {n_95}")
                    print(f"   Time: {(n_95 // 30 + 1) * 60 / 60:.1f} hours")
            else:
                print("\n❌ 99% quality is IMPRACTICAL")
                print("   Improvement distribution is too steep (β too high)")
                print("\n   Recommended target: 90-95% quality")

                n_90 = estimate_primitives_needed(alpha, beta, baseline_error, 0.90)
                if n_90:
                    threshold_90 = (alpha / (n_90 ** beta)) * 0.5
                    print(f"\n   For 90% quality:")
                    print(f"   Threshold: {threshold_90:.8f}")
                    print(f"   Primitives: {n_90}")
                    print(f"   Time: {(n_90 // 30 + 1) * 60 / 60:.1f} hours")

            # Plot
            plot_improvement_distribution(all_improvements, alpha, beta, baseline_error, args.output)

        else:
            print("\n⚠️  Power law fitting failed (not enough data or poor fit)")
    else:
        print(f"\n⚠️  Not enough positive improvements ({len(positive_improvements)}) to fit power law")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("\n1. Review the visualization: improvement_analysis.png")
    print("2. Update threshold in cpu_discovery_pipeline.py line 106")
    print("3. Run full discovery with updated threshold")
    print(f"\n   python scripts/start_discovery_cpu.py --iterations 50 --cores 56")
    print()


if __name__ == '__main__':
    main()
