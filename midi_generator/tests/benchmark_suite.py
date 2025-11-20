#!/usr/bin/env python3
"""
AGENT 20: Comprehensive Benchmark Suite
========================================

Benchmarks big band generator against professional recordings
and establishes quality baselines.

Tests include:
- Basie Swing Test (reference: "One O'Clock Jump")
- Ellington Exotic Test (reference: "Caravan")
- Modern Jazz Test (reference: "A Child is Born" - Thad Jones)
- Bebop Test (reference: Charlie Parker style)

Author: Agent 20 - Master Testing & Benchmarking Lead
Date: 2025-11-20

Usage:
    python benchmark_suite.py
    python benchmark_suite.py --style basie
    python benchmark_suite.py --full
"""

import sys
import os
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import time

try:
    from midi_generator.tools.big_band.generate_professional import (
        ProfessionalBigBandGenerator,
        ProfessionalBigBandConfig
    )
    from midi_generator.tests.validation_tests import ArrangementValidator, ValidationResult
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the correct directory and dependencies are installed")
    sys.exit(1)


@dataclass
class BenchmarkTest:
    """Definition of a benchmark test."""
    name: str
    description: str
    reference: str
    config: ProfessionalBigBandConfig
    metrics: List[str]
    target_score: float

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'reference': self.reference,
            'config': asdict(self.config),
            'metrics': self.metrics,
            'target_score': self.target_score
        }


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    test_name: str
    passed: bool
    overall_score: float
    metric_scores: Dict[str, float]
    generation_time: float
    validation_results: Dict[str, ValidationResult]
    notes_generated: int

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'overall_score': self.overall_score,
            'metric_scores': self.metric_scores,
            'generation_time': self.generation_time,
            'notes_generated': self.notes_generated,
            'validation_details': {
                name: {
                    'passed': result.passed,
                    'score': result.score,
                    'message': result.message
                }
                for name, result in self.validation_results.items()
            }
        }


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for big band generator.

    Compares generated arrangements against professional standards
    across multiple styles and eras.
    """

    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite."""
        self.validator = ArrangementValidator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.benchmark_tests = self._define_benchmarks()

    def _define_benchmarks(self) -> List[BenchmarkTest]:
        """Define all benchmark tests."""
        tests = []

        # Test 1: Count Basie Swing
        tests.append(BenchmarkTest(
            name="Basie Swing Test",
            description="Tests authentic Count Basie swing style with riff-based arrangement",
            reference="One O'Clock Jump (Count Basie)",
            config=ProfessionalBigBandConfig(
                tempo=180,
                key=0,  # C
                form_type="aaba",
                progression_type="jazz_blues",
                output_name="basie_swing_test",
                swing_ratio=0.62,
                ticks_per_beat=480
            ),
            metrics=[
                "swing_accuracy",
                "riff_usage",
                "section_balance",
                "dynamic_range"
            ],
            target_score=0.85
        ))

        # Test 2: Duke Ellington Exotic
        tests.append(BenchmarkTest(
            name="Ellington Exotic Test",
            description="Tests sophisticated Ellington-style harmony and orchestration",
            reference="Caravan (Duke Ellington)",
            config=ProfessionalBigBandConfig(
                tempo=120,
                key=0,  # C
                form_type="aaba",
                progression_type="rhythm_changes",
                output_name="ellington_exotic_test",
                swing_ratio=0.63,
                ticks_per_beat=480
            ),
            metrics=[
                "harmony_complexity",
                "voice_leading",
                "articulation_variety",
                "texture_density"
            ],
            target_score=0.85
        ))

        # Test 3: Thad Jones Modern
        tests.append(BenchmarkTest(
            name="Modern Jazz Test",
            description="Tests contemporary big band writing with wide voicings",
            reference="A Child is Born (Thad Jones)",
            config=ProfessionalBigBandConfig(
                tempo=80,
                key=5,  # F
                form_type="aaba",
                progression_type="ii_V_I",
                output_name="thad_modern_test",
                swing_ratio=0.60,
                ticks_per_beat=480
            ),
            metrics=[
                "voice_spacing",
                "modern_harmony",
                "dynamic_shaping",
                "voice_leading"
            ],
            target_score=0.80
        ))

        # Test 4: Bebop Fast
        tests.append(BenchmarkTest(
            name="Bebop Fast Test",
            description="Tests bebop melody generation at fast tempo",
            reference="Ko-Ko (Charlie Parker)",
            config=ProfessionalBigBandConfig(
                tempo=240,
                key=0,  # C
                form_type="blues",
                progression_type="jazz_blues",
                output_name="bebop_fast_test",
                swing_ratio=0.58,  # Lighter swing at fast tempo
                ticks_per_beat=480
            ),
            metrics=[
                "melodic_vocabulary",
                "swing_accuracy",
                "harmonic_rhythm",
                "bebop_language"
            ],
            target_score=0.80
        ))

        # Test 5: Ballad
        tests.append(BenchmarkTest(
            name="Ballad Test",
            description="Tests slow, lyrical arrangement with rich harmony",
            reference="Lush Life (Billy Strayhorn/Duke Ellington)",
            config=ProfessionalBigBandConfig(
                tempo=60,
                key=2,  # D
                form_type="aaba",
                progression_type="ii_V_I",
                output_name="ballad_test",
                swing_ratio=0.67,  # Heavier swing at slow tempo
                ticks_per_beat=480
            ),
            metrics=[
                "voice_leading",
                "harmonic_richness",
                "dynamic_shaping",
                "section_balance"
            ],
            target_score=0.85
        ))

        return tests

    def run_benchmark(self, test: BenchmarkTest, generate_midi: bool = True) -> BenchmarkResult:
        """
        Run a single benchmark test.

        Args:
            test: BenchmarkTest to run
            generate_midi: Whether to export MIDI file

        Returns:
            BenchmarkResult
        """
        print(f"\n{'='*80}")
        print(f"RUNNING BENCHMARK: {test.name}")
        print(f"{'='*80}")
        print(f"Reference: {test.reference}")
        print(f"Description: {test.description}")
        print(f"Target Score: {test.target_score:.2f}")
        print()

        # Generate arrangement
        start_time = time.time()
        generator = ProfessionalBigBandGenerator(test.config)

        try:
            result = generator.generate()
            generation_time = time.time() - start_time
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return BenchmarkResult(
                test_name=test.name,
                passed=False,
                overall_score=0.0,
                metric_scores={},
                generation_time=0.0,
                validation_results={},
                notes_generated=0
            )

        arrangement = result['arrangement']

        # Count total notes
        total_notes = sum(
            len(notes) for notes in arrangement.values()
            if isinstance(notes, list)
        )

        print(f"✓ Generation complete: {generation_time:.2f}s")
        print(f"✓ Total notes: {total_notes}")
        print()

        # Export MIDI if requested
        if generate_midi:
            midi_file = str(self.output_dir / f"{test.config.output_name}.mid")
            try:
                generator.export_midi(result, midi_file)
                print(f"✓ MIDI exported: {midi_file}")
            except Exception as e:
                print(f"⚠ MIDI export failed: {e}")

        # Run validations
        print("\nRunning validation tests...")
        validation_results = {}

        # 1. Voice leading
        if 'saxes' in arrangement and arrangement['saxes']:
            vl_result = self.validator.validate_voice_leading(arrangement, 'saxes')
            validation_results['voice_leading'] = vl_result
            print(f"  Voice Leading: {vl_result.message}")

        # 2. Harmony
        if 'progression' in result:
            # Convert to ChordEvents if needed
            progression = result['progression']
            from midi_generator.analysis.midi_analyzer import ChordEvent

            chord_events = []
            for i, chord in enumerate(progression):
                ce = ChordEvent(
                    start_time=i * 4.0,
                    duration=4.0,
                    root=chord.root,
                    quality=chord.quality,
                    pitches=[chord.root],
                    bass_note=chord.root,
                    confidence=1.0
                )
                chord_events.append(ce)

            harmony_result = self.validator.validate_harmony(chord_events, 'bebop')
            validation_results['harmony'] = harmony_result
            print(f"  Harmony: {harmony_result.message}")

        # 3. Form
        if 'form' in result:
            form_result = self.validator.validate_form(result, result['form'])
            validation_results['form'] = form_result
            print(f"  Form: {form_result.message}")

        # 4. Swing accuracy
        if 'lead' in arrangement and arrangement['lead']:
            swing_result = self.validator.measure_swing_accuracy(
                arrangement['lead'],
                test.config.swing_ratio
            )
            validation_results['swing'] = swing_result
            print(f"  Swing: {swing_result.message}")

        # 5. Overall authenticity
        authenticity_result = self.validator.measure_authenticity(arrangement)
        validation_results['authenticity'] = authenticity_result
        print(f"  Authenticity: {authenticity_result.message}")

        # Calculate overall score
        metric_scores = {
            name: result.score
            for name, result in validation_results.items()
        }

        if metric_scores:
            overall_score = sum(metric_scores.values()) / len(metric_scores)
        else:
            overall_score = 0.0

        passed = overall_score >= test.target_score

        # Create result
        benchmark_result = BenchmarkResult(
            test_name=test.name,
            passed=passed,
            overall_score=overall_score,
            metric_scores=metric_scores,
            generation_time=generation_time,
            validation_results=validation_results,
            notes_generated=total_notes
        )

        # Print summary
        print(f"\n{'='*80}")
        print(f"BENCHMARK RESULT: {test.name}")
        print(f"{'='*80}")
        print(f"Overall Score: {overall_score:.3f} / {test.target_score:.2f}")
        print(f"Status: {'✅ PASSED' if passed else '❌ FAILED'}")
        print(f"Generation Time: {generation_time:.2f}s")
        print(f"Notes Generated: {total_notes}")
        print("\nMetric Breakdown:")
        for metric, score in metric_scores.items():
            status = "✓" if score >= 0.7 else "✗"
            print(f"  {status} {metric}: {score:.3f}")
        print(f"{'='*80}")

        return benchmark_result

    def run_all_benchmarks(self, generate_midi: bool = True) -> List[BenchmarkResult]:
        """
        Run all benchmark tests.

        Args:
            generate_midi: Whether to export MIDI files

        Returns:
            List of BenchmarkResults
        """
        results = []

        print("\n" + "=" * 80)
        print("AGENT 20: COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 80)
        print(f"Total tests: {len(self.benchmark_tests)}")
        print(f"Output directory: {self.output_dir}")
        print()

        for i, test in enumerate(self.benchmark_tests, 1):
            print(f"\n[Test {i}/{len(self.benchmark_tests)}]")
            result = self.run_benchmark(test, generate_midi)
            results.append(result)

        # Generate summary report
        self._print_summary(results)

        # Save results to JSON
        self._save_results(results)

        return results

    def _print_summary(self, results: List[BenchmarkResult]):
        """Print summary of all benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUITE SUMMARY")
        print("=" * 80)

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        avg_score = sum(r.overall_score for r in results) / total_tests if total_tests > 0 else 0
        total_time = sum(r.generation_time for r in results)
        total_notes = sum(r.notes_generated for r in results)

        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({100*passed_tests/total_tests:.1f}%)")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Average Score: {avg_score:.3f}")
        print(f"Total Generation Time: {total_time:.2f}s")
        print(f"Total Notes Generated: {total_notes}")

        print("\nIndividual Results:")
        print("-" * 80)
        for result in results:
            status = "✅" if result.passed else "❌"
            print(f"{status} {result.test_name:30s} Score: {result.overall_score:.3f}")

        print("\nMetric Averages:")
        print("-" * 80)

        # Collect all unique metrics
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metric_scores.keys())

        # Calculate average for each metric
        for metric in sorted(all_metrics):
            scores = [r.metric_scores.get(metric, 0) for r in results if metric in r.metric_scores]
            if scores:
                avg = sum(scores) / len(scores)
                print(f"  {metric:30s} {avg:.3f}")

        print("\n" + "=" * 80)

        if passed_tests == total_tests:
            print("🎉 ALL BENCHMARKS PASSED!")
        elif passed_tests >= total_tests * 0.8:
            print("✅ MOST BENCHMARKS PASSED")
        else:
            print("⚠️  IMPROVEMENT NEEDED")

        print("=" * 80)

    def _save_results(self, results: List[BenchmarkResult]):
        """Save benchmark results to JSON file."""
        output_file = self.output_dir / "benchmark_results.json"

        data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(results),
            'passed': sum(1 for r in results if r.passed),
            'average_score': sum(r.overall_score for r in results) / len(results) if results else 0,
            'results': [r.to_dict() for r in results]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run benchmark suite for big band generator"
    )
    parser.add_argument(
        '--style',
        choices=['basie', 'ellington', 'modern', 'bebop', 'ballad', 'all'],
        default='all',
        help='Which benchmark test to run'
    )
    parser.add_argument(
        '--no-midi',
        action='store_true',
        help='Skip MIDI file generation'
    )
    parser.add_argument(
        '--output-dir',
        default='benchmark_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Create benchmark suite
    suite = BenchmarkSuite(output_dir=args.output_dir)

    # Run requested benchmarks
    if args.style == 'all':
        suite.run_all_benchmarks(generate_midi=not args.no_midi)
    else:
        # Find specific test
        style_map = {
            'basie': 'Basie Swing Test',
            'ellington': 'Ellington Exotic Test',
            'modern': 'Modern Jazz Test',
            'bebop': 'Bebop Fast Test',
            'ballad': 'Ballad Test'
        }

        test_name = style_map.get(args.style)
        test = next((t for t in suite.benchmark_tests if t.name == test_name), None)

        if test:
            result = suite.run_benchmark(test, generate_midi=not args.no_midi)
            suite._save_results([result])
        else:
            print(f"Error: Test '{args.style}' not found")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
