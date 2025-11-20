"""
Example usage of the Safety Monitor system.

This script demonstrates how to use the SafetyMonitor during parameter expansion
in the self-expanding music generation framework.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from safety.safety_monitor import SafetyMonitor, SafetyConfig


def example_safe_expansion():
    """Example: Safe parameter expansion with monitoring"""

    print("\n" + "=" * 70)
    print("EXAMPLE 1: Safe Parameter Expansion")
    print("=" * 70 + "\n")

    # Create safety monitor with custom config
    config = SafetyConfig()
    config.num_stability_tests = 5  # Reduce for example
    config.num_quality_test_files = 5

    monitor = SafetyMonitor(config)

    # Step 1: Create checkpoint before expansion
    print("STEP 1: Creating checkpoint before expansion...")
    checkpoint_id = monitor.checkpoint_system(
        description="Pre-expansion: Adding harmony.voice_leading parameter"
    )
    print(f"Checkpoint created: {checkpoint_id}\n")

    # Step 2: Simulate adding new parameter
    # (In real usage, this would be done by the parameter expansion agent)
    print("STEP 2: Adding new parameter...")
    print("(Simulated: Adding harmony.voice_leading to registry)")
    print("(Simulated: Training XGBoost model)")
    print("(Simulated: Deploying to production)\n")

    # Step 3: Monitor the expansion
    print("STEP 3: Monitoring expansion...")
    result = monitor.monitor_expansion(
        param_name='harmony.voice_leading',
        checkpoint_id=checkpoint_id
    )

    # Step 4: Decision based on results
    if result.safe:
        print("\n✅ EXPANSION APPROVED")
        print("The new parameter is safe to deploy.")

        if result.warnings:
            print("\nWarnings to address:")
            for warning in result.warnings:
                print(f"  - {warning}")
    else:
        print("\n❌ EXPANSION REJECTED")
        print("Critical issues detected. Rolling back...\n")

        # Rollback to checkpoint
        monitor.rollback_to_checkpoint(checkpoint_id)

    # Step 5: Generate report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    report = monitor.generate_report(checkpoint_id)
    print(report)


def example_unsafe_expansion_with_rollback():
    """Example: Unsafe expansion that gets rolled back"""

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Unsafe Expansion with Rollback")
    print("=" * 70 + "\n")

    monitor = SafetyMonitor()

    # Create checkpoint
    checkpoint_id = monitor.checkpoint_system(
        description="Testing rollback functionality"
    )

    # Simulate problematic expansion
    print("Simulating expansion that breaks existing functionality...\n")

    # Monitor would detect issues
    print("Monitoring detects critical issues:")
    print("  ❌ 5 existing parameters broken")
    print("  ❌ Generator crashes on 30% of test cases")
    print("  ❌ Quality degraded by 0.15 points\n")

    # Decision: Rollback
    print("DECISION: Rolling back to safe checkpoint...\n")
    monitor.rollback_to_checkpoint(checkpoint_id)

    print("✅ System restored to working state")


def example_checkpoint_management():
    """Example: Managing multiple checkpoints"""

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Checkpoint Management")
    print("=" * 70 + "\n")

    monitor = SafetyMonitor()

    # Create multiple checkpoints
    print("Creating checkpoints for different expansion phases...\n")

    checkpoints = []

    # Phase 1: Harmony parameters
    cp1 = monitor.checkpoint_system("Phase 1: Harmony parameters")
    checkpoints.append(cp1)
    print(f"Checkpoint 1: {cp1}")

    # Phase 2: Rhythm parameters
    cp2 = monitor.checkpoint_system("Phase 2: Rhythm parameters")
    checkpoints.append(cp2)
    print(f"Checkpoint 2: {cp2}")

    # Phase 3: Texture parameters
    cp3 = monitor.checkpoint_system("Phase 3: Texture parameters")
    checkpoints.append(cp3)
    print(f"Checkpoint 3: {cp3}")

    # View checkpoint history
    print("\n" + "-" * 70)
    print("CHECKPOINT HISTORY")
    print("-" * 70)

    history = monitor.get_checkpoint_history()
    for i, checkpoint in enumerate(history[-5:], 1):
        print(f"{i}. [{checkpoint.id}] {checkpoint.description}")
        print(f"   Status: {checkpoint.status.value}")
        print(f"   Parameters: {checkpoint.registry_snapshot['count']}")
        print(f"   Timestamp: {checkpoint.timestamp}")
        print()

    # Rollback to specific phase
    print("\nRolling back to Phase 2 checkpoint...")
    monitor.rollback_to_checkpoint(cp2)
    print("✅ Rolled back to Phase 2 state")


def example_quality_monitoring():
    """Example: Quality baseline and monitoring"""

    print("\n" + "=" * 70)
    print("EXAMPLE 4: Quality Monitoring")
    print("=" * 70 + "\n")

    monitor = SafetyMonitor()

    # Check baseline quality
    print("BASELINE QUALITY")
    print("-" * 70)
    baseline = monitor.quality_monitor.baseline_quality
    if baseline:
        print(f"Current baseline: {baseline:.3f}")
    else:
        print("Establishing baseline...")
        baseline = monitor.quality_monitor.establish_baseline()
        print(f"Baseline established: {baseline:.3f}")

    # Test current quality
    print("\nCURRENT QUALITY")
    print("-" * 70)
    quality_test = monitor.quality_monitor.test_current_quality()

    print(f"Average quality: {quality_test.get('avg_quality', 0.0):.3f}")
    print(f"Degradation: {quality_test.get('degradation', 0.0):.3f}")
    print(f"Status: {quality_test.get('status', 'unknown')}")

    if quality_test.get('qualities'):
        print(f"Test samples: {len(quality_test['qualities'])}")
        print(f"Min quality: {min(quality_test['qualities']):.3f}")
        print(f"Max quality: {max(quality_test['qualities']):.3f}")


def example_conflict_detection():
    """Example: Parameter conflict detection"""

    print("\n" + "=" * 70)
    print("EXAMPLE 5: Conflict Detection")
    print("=" * 70 + "\n")

    monitor = SafetyMonitor()

    # Test for conflicts with a hypothetical new parameter
    print("Testing for conflicts with new parameter: harmony.chord_complexity\n")

    conflicts = monitor.parameter_tester.test_parameter_conflicts('harmony.chord_complexity')

    if conflicts['has_conflicts']:
        print(f"❌ {conflicts['num_conflicts']} conflicts detected:")
        for conflict in conflicts['conflicts']:
            print(f"\n  Type: {conflict['type']}")
            print(f"  Conflicts with: {conflict['param']}")
            if 'similarity' in conflict:
                print(f"  Similarity: {conflict['similarity']:.2f}")
            print(f"  Severity: {conflict.get('severity', 'unknown')}")
    else:
        print("✅ No conflicts detected")


def main():
    """Run all examples"""

    print("\n" + "=" * 70)
    print("SAFETY MONITOR - USAGE EXAMPLES")
    print("=" * 70)

    examples = [
        ("Safe Expansion", example_safe_expansion),
        ("Unsafe Expansion with Rollback", example_unsafe_expansion_with_rollback),
        ("Checkpoint Management", example_checkpoint_management),
        ("Quality Monitoring", example_quality_monitoring),
        ("Conflict Detection", example_conflict_detection)
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")

    print("\nRun specific example with: python example_usage.py <number>")
    print("Run all examples with: python example_usage.py all\n")

    # Check command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        if arg == 'all':
            # Run all examples
            for name, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"\n❌ Error in example '{name}': {e}\n")
        else:
            # Run specific example
            try:
                index = int(arg) - 1
                if 0 <= index < len(examples):
                    name, func = examples[index]
                    func()
                else:
                    print(f"Invalid example number. Choose 1-{len(examples)}")
            except ValueError:
                print("Invalid argument. Use a number (1-5) or 'all'")
    else:
        # Run first example by default
        print("Running default example (Safe Expansion)...\n")
        example_safe_expansion()


if __name__ == '__main__':
    main()
