#!/usr/bin/env python3
"""
Agent 16: Expansion Orchestrator - Comprehensive Demo
======================================================

This demo showcases the complete self-expansion workflow orchestrated by Agent 16.

Workflow Demonstrated:
1. Inverse MIDI analysis
2. Gap detection
3. Parameter proposal (LLM)
4. Validation
5. Code generation (LLM)
6. Training data generation
7. Model training
8. Quality verification
9. Deployment/rollback

Usage:
    # With real API (requires Anthropic API key):
    export ANTHROPIC_API_KEY='your-key'
    python agent16_orchestration_demo.py --real

    # Mock mode (no API required):
    python agent16_orchestration_demo.py --mock

Author: Agent 16 - Expansion Orchestrator Demo
License: MIT
"""

import sys
import os
from pathlib import Path
import argparse
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_generator.orchestration import (
    ExpansionOrchestrator,
    ExpansionStatus,
    ExpansionStage
)


def print_banner(text: str, char: str = '='):
    """Print a formatted banner"""
    width = 80
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")


def print_stage_header(stage_num: int, stage_name: str):
    """Print stage header"""
    print(f"\n{'─' * 80}")
    print(f"  STAGE {stage_num}: {stage_name}")
    print(f"{'─' * 80}")


def demo_single_expansion(orchestrator: ExpansionOrchestrator, mock_mode: bool):
    """
    Demo 1: Single MIDI file expansion
    """
    print_banner("DEMO 1: Single MIDI File Expansion")

    # Create a mock MIDI file path
    midi_file = Path('examples/test_inputs/jazz_piano.mid')

    print(f"📝 Configuration:")
    print(f"   Input MIDI: {midi_file}")
    print(f"   Mock mode: {mock_mode}")
    print(f"   Auto-approve: True")
    print(f"   Max expansions: 2")
    print(f"   Min improvement: 5%")

    # Run expansion
    print(f"\n🚀 Starting expansion workflow...\n")
    start_time = time.time()

    result = orchestrator.expand_from_midi(
        input_midi=midi_file,
        auto_approve=True,  # Auto-approve for demo
        max_expansions=2,
        min_improvement=0.05
    )

    elapsed = time.time() - start_time

    # Print results
    print_banner("EXPANSION RESULTS", '=')

    if result.success:
        print(f"✅ SUCCESS!")
        print(f"\n📊 Metrics:")
        print(f"   Parameters deployed: {len(result.expansions_deployed)}")
        print(f"   Quality improvement: {result.quality_improvement:+.3f} ({result.quality_improvement*100:+.1f}%)")
        print(f"   Initial quality: {result.initial_quality:.3f}")
        print(f"   Final quality: {result.final_quality:.3f}")
        print(f"   Total time: {elapsed:.2f}s")

        print(f"\n📝 Deployed Parameters:")
        for i, param in enumerate(result.expansions_deployed, 1):
            print(f"   {i}. {param}")

        # Print expansion details
        if result.expansion_details:
            print(f"\n📋 Expansion Details:")
            for i, exp in enumerate(result.expansion_details, 1):
                print(f"\n   Parameter {i}: {exp.parameter_name}")
                print(f"   Status: {exp.status.value}")
                print(f"   Stage: {exp.stage.value}")
                if exp.error:
                    print(f"   Error: {exp.error}")

    else:
        print(f"❌ FAILED")
        print(f"\n📊 Metrics:")
        print(f"   Quality improvement: {result.quality_improvement:+.3f}")
        print(f"   Failure reason: {result.failure_reason}")
        print(f"   Total time: {elapsed:.2f}s")

    print(f"\n💾 Checkpoint ID: {result.checkpoint_id}")

    return result


def demo_batch_expansion(orchestrator: ExpansionOrchestrator):
    """
    Demo 2: Batch expansion from multiple MIDI files
    """
    print_banner("DEMO 2: Batch Expansion from Multiple MIDI Files")

    # Create mock MIDI file paths
    midi_files = [
        Path('examples/test_inputs/jazz_piano.mid'),
        Path('examples/test_inputs/classical_quartet.mid'),
        Path('examples/test_inputs/fusion_drums.mid'),
        Path('examples/test_inputs/modal_improv.mid'),
        Path('examples/test_inputs/bebop_head.mid'),
    ]

    print(f"📝 Configuration:")
    print(f"   MIDI files: {len(midi_files)}")
    for i, midi in enumerate(midi_files, 1):
        print(f"      {i}. {midi.name}")
    print(f"   Max expansions per file: 2")

    # Run batch expansion
    print(f"\n🚀 Starting batch expansion...\n")
    start_time = time.time()

    results = orchestrator.batch_expand_from_dataset(
        midi_files=midi_files,
        max_expansions_per_file=2
    )

    elapsed = time.time() - start_time

    # Print summary
    print_banner("BATCH EXPANSION SUMMARY", '=')

    successful = sum(1 for r in results.values() if r.success)
    failed = len(results) - successful
    total_params = sum(len(r.expansions_deployed) for r in results.values())

    print(f"📊 Overall Metrics:")
    print(f"   Total files: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total parameters added: {total_params}")
    print(f"   Total time: {elapsed:.2f}s")

    # Per-file results
    print(f"\n📋 Per-File Results:")
    for filename, result in results.items():
        status = "✅" if result.success else "❌"
        print(f"\n   {status} {Path(filename).name}")
        print(f"      Parameters: {len(result.expansions_deployed)}")
        print(f"      Quality: {result.initial_quality:.3f} → {result.final_quality:.3f}")
        if result.expansions_deployed:
            for param in result.expansions_deployed:
                print(f"         - {param}")

    return results


def demo_progressive_expansion(orchestrator: ExpansionOrchestrator):
    """
    Demo 3: Progressive expansion with quality monitoring
    """
    print_banner("DEMO 3: Progressive Expansion with Quality Monitoring")

    # Simulate progressive expansion epochs
    max_epochs = 3
    midi_files = [
        Path('examples/test_inputs/epoch_test_1.mid'),
        Path('examples/test_inputs/epoch_test_2.mid'),
        Path('examples/test_inputs/epoch_test_3.mid'),
    ]

    print(f"📝 Configuration:")
    print(f"   Max epochs: {max_epochs}")
    print(f"   Files per epoch: {len(midi_files)}")
    print(f"   Convergence threshold: <1% avg improvement")

    all_results = []

    for epoch in range(1, max_epochs + 1):
        print_banner(f"EPOCH {epoch}/{max_epochs}", '─')

        # Run batch expansion
        results = orchestrator.batch_expand_from_dataset(
            midi_files=midi_files,
            max_expansions_per_file=1
        )

        all_results.append(results)

        # Get statistics
        stats = orchestrator.get_expansion_statistics()

        print(f"\n📊 Epoch {epoch} Statistics:")
        print(f"   Total parameters: {stats['total_parameters_added']}")
        print(f"   Avg improvement: {stats['avg_quality_improvement']:.3f}")

        # Check convergence
        if stats['avg_quality_improvement'] < 0.01:
            print(f"\n✅ Converged! Average improvement below threshold.")
            break

        # Brief pause between epochs
        if epoch < max_epochs:
            print(f"\n⏸  Preparing next epoch...")
            time.sleep(1)

    # Final summary
    print_banner("PROGRESSIVE EXPANSION COMPLETE", '=')

    final_stats = orchestrator.get_expansion_statistics()
    print(f"📊 Final Statistics:")
    print(f"   Total epochs: {len(all_results)}")
    print(f"   Total expansions: {final_stats['total_expansions']}")
    print(f"   Successful: {final_stats['successful_expansions']}")
    print(f"   Total parameters: {final_stats['total_parameters_added']}")
    print(f"   Avg improvement: {final_stats['avg_quality_improvement']:.3f}")
    print(f"   Total time: {final_stats['total_time']:.1f}s")

    return all_results


def demo_safety_mechanisms(orchestrator: ExpansionOrchestrator):
    """
    Demo 4: Safety mechanisms (checkpoints and rollback)
    """
    print_banner("DEMO 4: Safety Mechanisms - Checkpoints & Rollback")

    from midi_generator.orchestration import SafetyMonitor

    # Create safety monitor
    safety_monitor = SafetyMonitor()

    print(f"📝 Testing checkpoint system...\n")

    # Create checkpoint 1
    print(f"1️⃣ Creating checkpoint 1...")
    checkpoint1 = safety_monitor.create_checkpoint("Checkpoint 1 - Initial state")
    print(f"   ✅ Checkpoint ID: {checkpoint1.checkpoint_id}")
    print(f"   Parameters: {checkpoint1.parameter_count}")
    print(f"   Models: {checkpoint1.model_count}")

    time.sleep(0.5)

    # Create checkpoint 2
    print(f"\n2️⃣ Creating checkpoint 2...")
    checkpoint2 = safety_monitor.create_checkpoint("Checkpoint 2 - After changes")
    print(f"   ✅ Checkpoint ID: {checkpoint2.checkpoint_id}")
    print(f"   Parameters: {checkpoint2.parameter_count}")
    print(f"   Models: {checkpoint2.model_count}")

    # Simulate rollback
    print(f"\n3️⃣ Simulating rollback to checkpoint 1...")
    safety_monitor.rollback_to_checkpoint(checkpoint1)
    print(f"   ✅ Rollback complete!")

    print(f"\n📋 Checkpoint History:")
    for i, cp in enumerate(safety_monitor.checkpoints, 1):
        print(f"   {i}. {cp.checkpoint_id}")
        print(f"      Description: {cp.description}")
        print(f"      Timestamp: {cp.timestamp}")
        print(f"      Files backed up: {len(cp.backup_paths)}")

    print(f"\n✅ Safety mechanisms working correctly!")


def demo_expansion_statistics(orchestrator: ExpansionOrchestrator):
    """
    Demo 5: Expansion statistics and monitoring
    """
    print_banner("DEMO 5: Expansion Statistics & Monitoring")

    # Get current statistics
    stats = orchestrator.get_expansion_statistics()

    print(f"📊 Current System Statistics:\n")
    print(f"   Total expansions run: {stats['total_expansions']}")
    print(f"   Successful: {stats['successful_expansions']}")
    print(f"   Failed: {stats['failed_expansions']}")
    print(f"   Total parameters added: {stats['total_parameters_added']}")
    print(f"   Average quality improvement: {stats['avg_quality_improvement']:.3f}")
    print(f"   Total time spent: {stats['total_time']:.1f}s")

    # Calculate derived metrics
    if stats['total_expansions'] > 0:
        success_rate = stats['successful_expansions'] / stats['total_expansions']
        avg_time_per_expansion = stats['total_time'] / stats['total_expansions']

        print(f"\n📈 Derived Metrics:")
        print(f"   Success rate: {success_rate*100:.1f}%")
        print(f"   Avg time per expansion: {avg_time_per_expansion:.1f}s")

        if stats['successful_expansions'] > 0:
            avg_params_per_success = stats['total_parameters_added'] / stats['successful_expansions']
            print(f"   Avg parameters per success: {avg_params_per_success:.2f}")

    # Print expansion history
    if orchestrator.expansion_history:
        print(f"\n📜 Expansion History:")
        for i, exp in enumerate(orchestrator.expansion_history[-5:], 1):  # Last 5
            status = "✅" if exp.success else "❌"
            print(f"\n   {status} Expansion {i}")
            print(f"      Parameters: {len(exp.expansions_deployed)}")
            print(f"      Improvement: {exp.quality_improvement:+.3f}")
            print(f"      Time: {exp.total_time:.1f}s")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(
        description='Agent 16: Expansion Orchestrator Demo'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Run in mock mode (no API calls)'
    )
    parser.add_argument(
        '--real',
        action='store_true',
        help='Run with real API (requires ANTHROPIC_API_KEY)'
    )
    parser.add_argument(
        '--demo',
        type=int,
        choices=[1, 2, 3, 4, 5, 0],
        default=0,
        help='Run specific demo (1-5) or all (0)'
    )

    args = parser.parse_args()

    # Determine mode
    if args.real:
        mock_mode = False
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
            print("   Set it with: export ANTHROPIC_API_KEY='your-key'")
            sys.exit(1)
    else:
        mock_mode = True
        api_key = None

    # Print header
    print_banner("Agent 16: Expansion Orchestrator - Comprehensive Demo")

    print(f"⚙️  Configuration:")
    print(f"   Mode: {'🤖 Real API' if not mock_mode else '🎭 Mock'}")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Python: {sys.version.split()[0]}")

    # Initialize orchestrator
    print(f"\n📦 Initializing Expansion Orchestrator...")
    orchestrator = ExpansionOrchestrator(
        api_key=api_key,
        mock_mode=mock_mode
    )
    print(f"   ✅ Orchestrator initialized")

    # Run demos
    demo_num = args.demo

    try:
        if demo_num == 0 or demo_num == 1:
            demo_single_expansion(orchestrator, mock_mode)
            if demo_num != 0:
                return

        if demo_num == 0:
            print("\n" + "="*80)
            input("Press Enter to continue to Demo 2...")

        if demo_num == 0 or demo_num == 2:
            demo_batch_expansion(orchestrator)
            if demo_num != 0:
                return

        if demo_num == 0:
            print("\n" + "="*80)
            input("Press Enter to continue to Demo 3...")

        if demo_num == 0 or demo_num == 3:
            demo_progressive_expansion(orchestrator)
            if demo_num != 0:
                return

        if demo_num == 0:
            print("\n" + "="*80)
            input("Press Enter to continue to Demo 4...")

        if demo_num == 0 or demo_num == 4:
            demo_safety_mechanisms(orchestrator)
            if demo_num != 0:
                return

        if demo_num == 0:
            print("\n" + "="*80)
            input("Press Enter to continue to Demo 5...")

        if demo_num == 0 or demo_num == 5:
            demo_expansion_statistics(orchestrator)

        # Final message
        print_banner("ALL DEMOS COMPLETE", '=')
        print(f"\n✅ Agent 16 orchestration demo completed successfully!")
        print(f"\n📚 For more information, see:")
        print(f"   - midi_generator/AGENT_16_ORCHESTRATOR.md")
        print(f"   - midi_generator/orchestration/expansion_orchestrator.py")
        print(f"\n🚀 The self-expanding music generation system is ready!")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
