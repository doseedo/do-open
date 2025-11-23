#!/usr/bin/env python3
"""
Parameter Coverage Validator
============================

Validates that the parameter system meets the 500-800 target and has proper
coverage across all musical domains.

This will be used by Agent 10 after Agents 3-9 complete their work.

Author: Agent 10 - Validation & Documentation
"""

from typing import Dict, List, Tuple
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from parameters.universal_registry import UniversalParameterRegistry, MusicalImpact
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    print("Warning: UniversalParameterRegistry not available")


class ParameterCoverageValidator:
    """
    Validates parameter coverage against requirements.

    Requirements:
    - Total: 500-800 parameters
    - Harmony: 150 parameters
    - Melody: 100 parameters
    - Rhythm: 100 parameters
    - Structure: 50 parameters
    - Instrumentation: 50 parameters
    - Dynamics: 50 parameters
    """

    def __init__(self):
        """Initialize validator."""
        if not REGISTRY_AVAILABLE:
            raise RuntimeError("UniversalParameterRegistry not available")

        self.registry = UniversalParameterRegistry()
        self.requirements = {
            'total': (500, 800),
            'domains': {
                'harmony': (100, 200),  # Flexible range
                'melody': (80, 120),
                'rhythm': (80, 120),
                'structure': (40, 60),
                'instrumentation': (40, 60),
                'dynamics': (40, 60),
                'global': (10, 30),
            }
        }

    def validate(self) -> Dict:
        """
        Run complete validation.

        Returns:
            Dict with validation results
        """
        results = {
            'passed': True,
            'total_params': 0,
            'domain_coverage': {},
            'impact_distribution': {},
            'issues': [],
            'warnings': [],
        }

        # Count total parameters
        total = len(self.registry.parameters)
        results['total_params'] = total

        # Check total count
        min_total, max_total = self.requirements['total']
        if total < min_total:
            results['passed'] = False
            results['issues'].append(
                f"Too few parameters: {total} < {min_total} (need {min_total - total} more)"
            )
        elif total > max_total:
            results['warnings'].append(
                f"Too many parameters: {total} > {max_total} (simplify by {total - max_total})"
            )

        # Check domain coverage
        for domain, (min_count, max_count) in self.requirements['domains'].items():
            domain_params = self.registry.get_by_domain(domain)
            count = len(domain_params)

            results['domain_coverage'][domain] = {
                'count': count,
                'min_required': min_count,
                'max_allowed': max_count,
                'status': 'ok'
            }

            if count < min_count:
                results['passed'] = False
                results['domain_coverage'][domain]['status'] = 'insufficient'
                results['issues'].append(
                    f"{domain}: {count} < {min_count} (need {min_count - count} more)"
                )
            elif count > max_count:
                results['domain_coverage'][domain]['status'] = 'excessive'
                results['warnings'].append(
                    f"{domain}: {count} > {max_count} (reduce by {count - max_count})"
                )

        # Check impact distribution
        impact_counts = {}
        for spec in self.registry.parameters.values():
            impact = spec.impact.value
            impact_counts[impact] = impact_counts.get(impact, 0) + 1

        results['impact_distribution'] = impact_counts

        # Warn if too many low-impact parameters
        low_impact = impact_counts.get('low', 0) + impact_counts.get('minimal', 0)
        if low_impact > total * 0.3:
            results['warnings'].append(
                f"Too many low-impact parameters: {low_impact}/{total} ({low_impact/total*100:.0f}%)"
            )

        # Check for duplicate names
        duplicates = self._check_duplicates()
        if duplicates:
            results['passed'] = False
            results['issues'].extend([f"Duplicate parameter: {d}" for d in duplicates])

        return results

    def _check_duplicates(self) -> List[str]:
        """Check for duplicate parameter names."""
        seen = set()
        duplicates = []

        for name in self.registry.parameters:
            if name in seen:
                duplicates.append(name)
            seen.add(name)

        return duplicates

    def print_report(self, results: Dict):
        """Print validation report."""
        print("\n" + "=" * 70)
        print("PARAMETER COVERAGE VALIDATION REPORT")
        print("=" * 70)

        # Overall status
        status = "✅ PASSED" if results['passed'] else "❌ FAILED"
        print(f"\n**Status:** {status}")
        print(f"**Total Parameters:** {results['total_params']}")

        min_total, max_total = self.requirements['total']
        print(f"**Target Range:** {min_total}-{max_total}")

        # Domain coverage
        print(f"\n📊 **Domain Coverage:**")
        print(f"{'Domain':20s} {'Count':>6s} {'Required':>10s} {'Status':>12s}")
        print("-" * 50)

        for domain, info in results['domain_coverage'].items():
            status_symbol = {
                'ok': '✓',
                'insufficient': '✗',
                'excessive': '⚠'
            }[info['status']]

            print(f"{domain:20s} {info['count']:6d} "
                  f"{info['min_required']:4d}-{info['max_allowed']:<4d} "
                  f"{status_symbol} {info['status']:>11s}")

        # Impact distribution
        print(f"\n📈 **Impact Distribution:**")
        for impact, count in sorted(results['impact_distribution'].items()):
            pct = count / results['total_params'] * 100
            print(f"   {impact:10s}: {count:4d} ({pct:5.1f}%)")

        # Issues
        if results['issues']:
            print(f"\n❌ **Issues ({len(results['issues'])}):**")
            for issue in results['issues']:
                print(f"   - {issue}")

        # Warnings
        if results['warnings']:
            print(f"\n⚠️  **Warnings ({len(results['warnings'])}):**")
            for warning in results['warnings']:
                print(f"   - {warning}")

        # Recommendations
        print(f"\n💡 **Recommendations:**")
        if results['passed']:
            print("   - Parameter coverage looks good!")
            print("   - Consider reviewing low-impact parameters")
        else:
            print("   - Address all issues above before finalizing")
            print("   - Focus on domains that are insufficient")
            print("   - Extract parameters from core module refactoring")

        print("\n" + "=" * 70)

    def save_report(self, results: Dict, filepath: str):
        """Save validation report to JSON."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Saved validation report to: {filepath}")


def main():
    """Run validation."""
    print("Parameter Coverage Validator")
    print("=" * 70)

    if not REGISTRY_AVAILABLE:
        print("ERROR: UniversalParameterRegistry not available")
        print("Make sure you're running from the correct directory")
        return 1

    validator = ParameterCoverageValidator()
    results = validator.validate()
    validator.print_report(results)
    validator.save_report(results, 'parameter_validation_report.json')

    return 0 if results['passed'] else 1


if __name__ == "__main__":
    sys.exit(main())
