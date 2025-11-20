#!/usr/bin/env python3
"""
AGENT 34: Integration Testing Coordinator - Demo
=================================================

Demonstrates the comprehensive integration testing framework.

This demo shows:
1. Running the complete integration test suite
2. Running specific test suites
3. Filtering tests by priority
4. Analyzing test results
5. Performance monitoring
6. Custom test execution

Author: Agent 34 - Integration Testing Coordinator
License: MIT
"""

import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from midi_generator.testing.integration_test_coordinator import (
        IntegrationTestCoordinator,
        TestPriority,
        TestStatus,
        IntegrationTestReport
    )
    COORDINATOR_AVAILABLE = True
except ImportError:
    COORDINATOR_AVAILABLE = False
    print("ERROR: Integration Test Coordinator not available")
    sys.exit(1)


def demo_basic_usage():
    """Demo 1: Basic usage of the integration test coordinator"""
    print("=" * 80)
    print("DEMO 1: Basic Integration Test Execution")
    print("=" * 80)

    # Initialize coordinator
    coordinator = IntegrationTestCoordinator(
        test_data_dir=Path("tests/test_data"),
        output_dir=Path("tests/integration/results"),
        verbose=True
    )

    # Run all tests (skip slow tests for demo)
    print("\n▶ Running all integration tests (fast mode)...\n")
    report = coordinator.run_all_tests(skip_slow=True)

    # Display summary
    print("\n" + "=" * 80)
    print("TEST EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed: {report.total_passed} ({report.overall_pass_rate*100:.1f}%)")
    print(f"Failed: {report.total_failed}")
    print(f"Duration: {report.total_duration:.2f}s")
    print(f"Status: {'✓ SUCCESS' if report.is_success() else '✗ FAILURE'}")

    return report


def demo_specific_suite():
    """Demo 2: Running a specific test suite"""
    print("\n" + "=" * 80)
    print("DEMO 2: Running Specific Test Suite")
    print("=" * 80)

    coordinator = IntegrationTestCoordinator(verbose=True)

    # Run only feature extraction tests
    print("\n▶ Running Feature Extraction Pipeline tests...\n")
    suite_result = coordinator.test_feature_extraction_pipeline()

    print("\n" + "=" * 80)
    print("SUITE RESULTS")
    print("=" * 80)
    print(suite_result.summary())

    # Show individual test results
    print("\nIndividual Tests:")
    for test in suite_result.tests:
        status_symbol = "✓" if test.is_success() else "✗"
        print(f"  {status_symbol} {test.test_name} ({test.duration:.2f}s)")
        if test.metrics:
            for key, value in test.metrics.items():
                print(f"      {key}: {value}")

    return suite_result


def demo_analyze_results(report: IntegrationTestReport):
    """Demo 3: Analyzing test results in detail"""
    print("\n" + "=" * 80)
    print("DEMO 3: Detailed Result Analysis")
    print("=" * 80)

    # Analyze by priority
    print("\n▶ Tests by Priority:")

    priority_counts = {
        TestPriority.CRITICAL: 0,
        TestPriority.HIGH: 0,
        TestPriority.MEDIUM: 0,
        TestPriority.LOW: 0
    }

    for suite in report.suites:
        for test in suite.tests:
            if test.priority in priority_counts:
                priority_counts[test.priority] += 1

    for priority, count in priority_counts.items():
        print(f"  {priority.value.upper()}: {count} tests")

    # Analyze by status
    print("\n▶ Tests by Status:")

    status_counts = {
        TestStatus.PASSED: 0,
        TestStatus.FAILED: 0,
        TestStatus.SKIPPED: 0,
        TestStatus.ERROR: 0
    }

    for suite in report.suites:
        for test in suite.tests:
            if test.status in status_counts:
                status_counts[test.status] += 1

    for status, count in status_counts.items():
        print(f"  {status.value.upper()}: {count} tests")

    # Show failed tests
    print("\n▶ Failed/Error Tests:")

    failed_tests = [
        (suite.suite_name, test)
        for suite in report.suites
        for test in suite.tests
        if test.status in [TestStatus.FAILED, TestStatus.ERROR]
    ]

    if failed_tests:
        for suite_name, test in failed_tests:
            print(f"\n  ✗ {suite_name}: {test.test_name}")
            print(f"    Error: {test.error_message}")
            print(f"    Priority: {test.priority.value}")
            print(f"    Agent: {test.agent_tested}")
    else:
        print("  None - All tests passed! ✓")

    # Show slowest tests
    print("\n▶ Slowest Tests (Top 5):")

    all_tests = [
        (suite.suite_name, test)
        for suite in report.suites
        for test in suite.tests
    ]

    slowest = sorted(all_tests, key=lambda x: x[1].duration, reverse=True)[:5]

    for suite_name, test in slowest:
        print(f"  {test.duration:.2f}s - {suite_name}: {test.test_name}")

    # Show agents tested
    print("\n▶ Agents Tested:")

    agents_tested = set()
    for suite in report.suites:
        for test in suite.tests:
            if test.agent_tested:
                for agent in test.agent_tested.split(','):
                    agents_tested.add(agent.strip())

    for agent in sorted(agents_tested):
        print(f"  • {agent}")


def demo_performance_metrics(report: IntegrationTestReport):
    """Demo 4: Performance metrics analysis"""
    print("\n" + "=" * 80)
    print("DEMO 4: Performance Metrics")
    print("=" * 80)

    print("\n▶ Test Suite Performance:")

    for suite in report.suites:
        tests_per_second = suite.total_count / suite.total_duration if suite.total_duration > 0 else 0
        avg_test_time = suite.total_duration / suite.total_count if suite.total_count > 0 else 0

        print(f"\n  {suite.suite_name}:")
        print(f"    Duration: {suite.total_duration:.2f}s")
        print(f"    Tests: {suite.total_count}")
        print(f"    Avg per test: {avg_test_time:.2f}s")
        print(f"    Tests/second: {tests_per_second:.2f}")

    print("\n▶ Overall Performance:")
    print(f"  Total duration: {report.total_duration:.2f}s")
    print(f"  Total tests: {report.total_tests}")
    print(f"  Average per test: {report.total_duration / report.total_tests:.2f}s")

    # Check performance targets
    print("\n▶ Performance Targets:")

    fast_mode_target = 120  # 2 minutes
    meets_target = report.total_duration < fast_mode_target

    print(f"  Target (fast mode): < {fast_mode_target}s")
    print(f"  Actual: {report.total_duration:.2f}s")
    print(f"  Status: {'✓ PASS' if meets_target else '✗ FAIL'}")


def demo_custom_test_workflow():
    """Demo 5: Custom test workflow"""
    print("\n" + "=" * 80)
    print("DEMO 5: Custom Test Workflow")
    print("=" * 80)

    coordinator = IntegrationTestCoordinator(verbose=False)

    print("\n▶ Running custom test workflow...")

    # Step 1: Run critical tests only
    print("\n  Step 1: Running critical tests...")
    critical_tests = []

    suite = coordinator.test_feature_extraction_pipeline()
    critical_tests.extend([t for t in suite.tests if t.priority == TestPriority.CRITICAL])

    suite = coordinator.test_training_data_generation()
    critical_tests.extend([t for t in suite.tests if t.priority == TestPriority.CRITICAL])

    critical_passed = sum(1 for t in critical_tests if t.is_success())
    print(f"    Critical tests: {critical_passed}/{len(critical_tests)} passed")

    # Step 2: If critical tests pass, run high priority
    if critical_passed == len(critical_tests):
        print("\n  Step 2: Critical tests passed, running high priority tests...")

        high_tests = []
        suite = coordinator.test_feature_extraction_pipeline()
        high_tests.extend([t for t in suite.tests if t.priority == TestPriority.HIGH])

        high_passed = sum(1 for t in high_tests if t.is_success())
        print(f"    High priority tests: {high_passed}/{len(high_tests)} passed")
    else:
        print("\n  Step 2: Critical tests failed, skipping remaining tests")

    # Step 3: Summary
    print("\n  Step 3: Workflow Summary")
    print(f"    Status: {'✓ SUCCESS' if critical_passed == len(critical_tests) else '✗ FAILURE'}")


def demo_ci_cd_workflow():
    """Demo 6: CI/CD workflow simulation"""
    print("\n" + "=" * 80)
    print("DEMO 6: CI/CD Workflow Simulation")
    print("=" * 80)

    print("\n▶ Simulating CI/CD pipeline...\n")

    coordinator = IntegrationTestCoordinator(
        verbose=False,  # CI mode - less verbose
        output_dir=Path("tests/integration/results/ci")
    )

    # Run fast tests (skip slow for CI)
    print("  [CI] Running fast integration tests...")
    report = coordinator.run_all_tests(skip_slow=True)

    # Check results
    print(f"  [CI] Tests completed in {report.total_duration:.2f}s")
    print(f"  [CI] Results: {report.total_passed}/{report.total_tests} passed")

    # Determine CI status
    ci_success = report.is_success()

    print(f"\n  [CI] Status: {'✓ PASS - Ready to merge' if ci_success else '✗ FAIL - Blocking merge'}")

    if not ci_success:
        print("\n  [CI] Failed tests:")
        for suite in report.suites:
            for test in suite.tests:
                if test.status == TestStatus.FAILED:
                    print(f"    • {test.test_name}: {test.error_message}")

    # Exit with appropriate code (for real CI)
    exit_code = 0 if ci_success else 1
    print(f"\n  [CI] Exit code: {exit_code}")

    return exit_code


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("AGENT 34: Integration Testing Coordinator - Demo Suite")
    print("=" * 80)
    print("\nThis demo showcases the comprehensive integration testing framework")
    print("for the Musical Program Synthesis system.\n")

    if not COORDINATOR_AVAILABLE:
        print("ERROR: Integration Test Coordinator not available")
        return 1

    try:
        # Run demos in sequence
        print("\n" + "▶" * 40)
        print("Starting Demo Suite...")
        print("▶" * 40)

        # Demo 1: Basic usage
        report = demo_basic_usage()

        # Demo 2: Specific suite
        demo_specific_suite()

        # Demo 3: Analyze results
        demo_analyze_results(report)

        # Demo 4: Performance metrics
        demo_performance_metrics(report)

        # Demo 5: Custom workflow
        demo_custom_test_workflow()

        # Demo 6: CI/CD workflow
        exit_code = demo_ci_cd_workflow()

        # Final summary
        print("\n" + "=" * 80)
        print("DEMO SUITE COMPLETE")
        print("=" * 80)
        print("\nAll demos executed successfully!")
        print("\nNext steps:")
        print("  1. Review test results in tests/integration/results/")
        print("  2. Run specific test suites with pytest")
        print("  3. Integrate into your CI/CD pipeline")
        print("  4. Add custom tests for your features")
        print("\nFor more information:")
        print("  • Documentation: midi_generator/AGENT_34_INTEGRATION_TESTING.md")
        print("  • Test README: tests/README.md")
        print("  • Source code: midi_generator/testing/integration_test_coordinator.py")

        return exit_code

    except Exception as e:
        print(f"\n✗ Error running demos: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
