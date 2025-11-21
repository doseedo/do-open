#!/usr/bin/env python3
"""
Musical Constraint Validator - Interactive Demo
Agent 8: Constraint Validator

Demonstrates all major features of the constraint validation system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constraints.musical_validator import (
    MusicalConstraintValidator,
    CounterpointValidator,
    pitch_to_name
)

from constraints.advanced_constraints import (
    JazzVoiceLeadingValidator,
    PerformancePracticeValidator,
    create_validator_for_style
)

from constraints.integration import (
    XGBoostConstraintIntegration,
    ConstraintPostProcessor,
    ConstraintValidationMetrics
)


def demo_voice_leading():
    """Demonstrate voice leading validation and correction"""
    print("\n" + "="*80)
    print("DEMO 1: VOICE LEADING VALIDATION & AUTOMATIC CORRECTION")
    print("="*80)

    validator = MusicalConstraintValidator(style='common_practice')

    # Example with parallel fifths
    print("\n📝 Testing 4-part harmony with parallel fifths:")
    voices = [
        [48, 50, 52],  # Bass: C3 -> D3 -> E3
        [55, 57, 59],  # Tenor: G3 -> A3 -> B3 (parallel 5th with bass!)
        [64, 65, 67],  # Alto
        [72, 74, 76],  # Soprano
    ]

    print("  Bass:    ", [pitch_to_name(n) for n in voices[0]])
    print("  Tenor:   ", [pitch_to_name(n) for n in voices[1]])
    print("  Alto:    ", [pitch_to_name(n) for n in voices[2]])
    print("  Soprano: ", [pitch_to_name(n) for n in voices[3]])

    result = validator.validate_voice_leading(voices)

    print(f"\n{result.get_summary()}")
    print("\n🔍 Violations found:")
    for v in result.violations[:3]:  # Show first 3
        print(f"  • {v}")

    # Automatic correction
    print("\n🔧 Attempting automatic correction...")
    fixed = validator.fix_voice_leading(voices, result)
    result_after = validator.validate_voice_leading(fixed)

    print(f"After correction: {result_after.get_summary()}")
    print("\n✨ Corrected voices:")
    print("  Tenor:   ", [pitch_to_name(n) for n in fixed[1]], " (adjusted)")


def demo_instrument_ranges():
    """Demonstrate instrument range validation"""
    print("\n" + "="*80)
    print("DEMO 2: INSTRUMENT RANGE VALIDATION")
    print("="*80)

    validator = MusicalConstraintValidator()

    print("\n🎻 Violin part with some notes too high:")
    violin_part = [55, 60, 67, 72, 84, 96, 110]  # 110 is G7 - very high!

    print("  Original: ", [pitch_to_name(n) for n in violin_part])

    result = validator.validate_range(violin_part, 'violin')
    print(f"\n{result.get_summary()}")

    if not result.is_valid:
        print("\n🔍 Issues:")
        for v in result.violations:
            print(f"  • {v.description}")

        print("\n🔧 Auto-correcting out-of-range notes...")
        fixed = validator.fix_out_of_range(violin_part, 'violin')
        print("  Fixed:    ", [pitch_to_name(n) for n in fixed])


def demo_jazz_voicings():
    """Demonstrate jazz voicing validation"""
    print("\n" + "="*80)
    print("DEMO 3: JAZZ VOICING VALIDATION")
    print("="*80)

    jazz_validator = JazzVoiceLeadingValidator(style='bebop')

    print("\n🎹 Dm7 Rootless Voicing (Piano left hand):")

    # Valid rootless voicing: F-A-C-E (3rd, 5th, 7th, 9th)
    dm7_good = [53, 57, 60, 64]
    print("  Voicing: ", [pitch_to_name(n) for n in dm7_good], " (F-A-C-E)")

    result = jazz_validator.validate_jazz_voicing(dm7_good, 'Dm7', 'rootless')
    print(f"  {result.get_summary()}")

    # Invalid: has root
    print("\n❌ Invalid rootless voicing (contains root D):")
    dm7_bad = [50, 53, 57, 60]  # D-F-A-C (has root!)
    print("  Voicing: ", [pitch_to_name(n) for n in dm7_bad], " (D-F-A-C)")

    result = jazz_validator.validate_jazz_voicing(dm7_bad, 'Dm7', 'rootless')
    print(f"  {result.get_summary()}")
    for v in result.violations:
        print(f"    • {v.description}")


def demo_performance_practice():
    """Demonstrate performance practice validation"""
    print("\n" + "="*80)
    print("DEMO 4: PERFORMANCE PRACTICE")
    print("="*80)

    perf_validator = PerformancePracticeValidator()

    # Piano hand span
    print("\n🎹 Piano hand span validation:")
    comfortable = [60, 64, 67, 72]  # C-E-G-C (octave span)
    print(f"  Comfortable chord: {[pitch_to_name(n) for n in comfortable]}")
    result = perf_validator.validate_piano_hand_span(comfortable, 'right')
    print(f"  {result.get_summary()}")

    too_wide = [48, 52, 55, 59, 64]  # 16 semitone span
    print(f"\n  Wide chord: {[pitch_to_name(n) for n in too_wide]}")
    result = perf_validator.validate_piano_hand_span(too_wide, 'right')
    print(f"  {result.get_summary()}")
    for v in result.violations:
        print(f"    • {v.description}")

    # Breathing
    print("\n🎺 Wind instrument breathing validation:")
    long_phrase = [(72, 1.0)] * 10  # 10 beats without rest
    print("  Phrase length: 10 beats without rest")
    result = perf_validator.validate_breathing(long_phrase, 'trumpet')
    print(f"  {result.get_summary()}")
    for v in result.violations:
        print(f"    • {v.description}")


def demo_xgboost_integration():
    """Demonstrate XGBoost integration"""
    print("\n" + "="*80)
    print("DEMO 5: XGBOOST INTEGRATION (Phase 2)")
    print("="*80)

    print("\n🤖 Simulating XGBoost parameter predictions...")

    # Simulated predictions from XGBoost model
    predictions = {
        'voices': [
            [48, 50, 52],
            [60, 62, 64],
            [64, 66, 68],
            [72, 74, 76],
        ],
        'instrument_parts': {
            'trumpet': [60, 64, 67, 72, 76],
            'trombone': [48, 50, 52, 55, 57],
        },
        'key': 60,
    }

    xgb_integration = XGBoostConstraintIntegration(style='jazz')

    # Validate predictions
    result, corrected = xgb_integration.validate_predicted_parameters(predictions)
    print(f"\nValidation: {result.get_summary()}")

    # Extract constraint features for model training
    features = xgb_integration.get_constraint_features(predictions)
    print("\n📊 Constraint features (for XGBoost training):")
    print(f"  constraint_score: {features['constraint_score']:.2%}")
    print(f"  num_violations: {features['num_violations']}")
    print(f"  num_warnings: {features['num_warnings']}")

    # Calculate loss
    loss = xgb_integration.constraint_violation_loss(predictions)
    print(f"\n💰 Constraint violation loss: {loss:.3f}")


def demo_post_processing():
    """Demonstrate post-processing pipeline"""
    print("\n" + "="*80)
    print("DEMO 6: POST-PROCESSING PIPELINE")
    print("="*80)

    print("\n🔄 Running parameter validation through multi-stage pipeline...")

    params = {
        'voices': [[48, 50], [55, 57]],  # Has parallel motion
        'instrument_parts': {
            'violin': [40, 60, 110],  # 40 too low, 110 too high
        }
    }

    processor = ConstraintPostProcessor(style='common_practice')
    processor.create_default_pipeline()

    processed, history = processor.process(params)

    print(f"\nProcessed through {len(history)} pipeline stages:")
    for i, result in enumerate(history, 1):
        print(f"  Stage {i}: {result.get_summary()}")


def demo_validation_metrics():
    """Demonstrate validation metrics"""
    print("\n" + "="*80)
    print("DEMO 7: VALIDATION METRICS (Model Evaluation)")
    print("="*80)

    # Simulate batch of predictions
    predictions_batch = [
        {'voices': [[48, 50], [60, 62], [64, 66], [72, 74]]},
        {'voices': [[48, 55], [55, 62], [64, 71], [72, 79]]},  # Some issues
        {'voices': [[48, 50], [60, 62], [64, 66], [72, 74]]},
    ]

    print("\n📈 Evaluating batch of 3 predictions...")

    satisfaction_rate = ConstraintValidationMetrics.constraint_satisfaction_rate(
        predictions_batch
    )
    avg_score = ConstraintValidationMetrics.average_constraint_score(
        predictions_batch
    )
    violations = ConstraintValidationMetrics.violation_distribution(
        predictions_batch
    )

    print(f"\n✅ Constraint satisfaction rate: {satisfaction_rate:.1%}")
    print(f"📊 Average constraint score: {avg_score:.1%}")
    print(f"\n🔍 Violation distribution:")
    for vtype, count in violations.items():
        print(f"  {vtype}: {count}")


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("🎵 MUSICAL CONSTRAINT VALIDATOR - INTERACTIVE DEMO")
    print("Agent 8: Constraint Validator")
    print("Part of Musical Program Synthesis System (Phase 2)")
    print("="*80)

    demo_voice_leading()
    demo_instrument_ranges()
    demo_jazz_voicings()
    demo_performance_practice()
    demo_xgboost_integration()
    demo_post_processing()
    demo_validation_metrics()

    print("\n" + "="*80)
    print("✨ DEMO COMPLETE!")
    print("\n📚 For more information, see:")
    print("  - README.md for full documentation")
    print("  - test_constraints.py for comprehensive tests")
    print("  - musical_validator.py for core implementation")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
