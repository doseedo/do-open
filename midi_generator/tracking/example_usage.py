#!/usr/bin/env python3
"""
Example Usage of Expansion History Tracker
==========================================

Demonstrates comprehensive usage of the expansion tracking system
for the self-expanding inverse music generation system.
"""

import hashlib
from datetime import datetime, timedelta
from expansion_history import (
    ExpansionHistoryTracker,
    ExpansionEvent,
    ExpansionTrigger,
    ExpansionStatus,
    ExpansionPhase,
    ParameterImpact,
    GapDetectionRecord,
    LLMProposalRecord,
    SystemSnapshot
)


def example_1_basic_expansion():
    """Example 1: Log a basic parameter expansion"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Parameter Expansion")
    print("="*80)

    tracker = ExpansionHistoryTracker(
        db_path="example_history.db",
        json_backup_path="example_history.json"
    )

    # Create an expansion event
    event = ExpansionEvent(
        parameters_added=[
            "harmony.jazz.voicing_cluster_density",
            "harmony.jazz.voicing_spread_factor"
        ],
        parameter_count=2,
        trigger=ExpansionTrigger.MANUAL_ADDITION,
        code_files_modified=["generators/advanced_harmony_generator.py"],
        code_lines_added=85,
        generator_enhanced=True,
        phase=ExpansionPhase.PHASE_1,
        agent_responsible="Agent 3 - Registry Builder",
        notes="Added parameters for close position jazz voicings"
    )

    event_id = tracker.log_expansion_event(event)
    print(f"✅ Logged expansion event: {event_id}")
    print(f"   Parameters added: {event.parameter_count}")
    print(f"   Trigger: {event.trigger.value}")

    return tracker


def example_2_reconstruction_driven():
    """Example 2: Reconstruction failure-driven expansion"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Reconstruction Failure-Driven Expansion")
    print("="*80)

    tracker = ExpansionHistoryTracker(
        db_path="example_history.db",
        json_backup_path="example_history.json"
    )

    # Simulate reconstruction failure
    event = ExpansionEvent(
        parameters_added=[
            "bass.stride_pattern.octave_jump",
            "bass.stride_pattern.alternation_rate",
            "bass.stride_pattern.walking_probability"
        ],
        parameter_count=3,
        trigger=ExpansionTrigger.RECONSTRUCTION_FAILURE,
        trigger_details={
            "midi_file": "examples/fats_waller_stride.mid",
            "failed_feature": "stride_bass_pattern",
            "reconstruction_error": 0.52
        },
        reconstruction_accuracy_before=0.65,
        reconstruction_accuracy_after=0.88,
        code_files_modified=[
            "generators/bass_generator.py",
            "analysis/deep_feature_extractor.py"
        ],
        code_lines_added=145,
        generator_enhanced=True,
        feature_extractor_enhanced=True,
        xgboost_models_trained=3,
        model_training_time_seconds=78.5,
        model_accuracy_metrics={
            "bass.stride_pattern.octave_jump": 0.89,
            "bass.stride_pattern.alternation_rate": 0.85,
            "bass.stride_pattern.walking_probability": 0.91
        },
        test_midi_files_passed=15,
        test_midi_files_failed=0,
        musical_validation_passed=True,
        phase=ExpansionPhase.PHASE_1,
        agent_responsible="Agent 5 - Reconstruction Engine",
        notes="Implemented stride piano bass pattern recognition and generation"
    )

    event_id = tracker.log_expansion_event(event)
    tracker.update_event_status(event_id, ExpansionStatus.DEPLOYED)
    tracker.update_event_impact(event_id, ParameterImpact.HIGH_VALUE, 0.23)

    print(f"✅ Logged reconstruction-driven expansion: {event_id}")
    print(f"   Accuracy improvement: {event.reconstruction_accuracy_after - event.reconstruction_accuracy_before:.2%}")
    print(f"   Models trained: {event.xgboost_models_trained}")
    print(f"   Training time: {event.model_training_time_seconds:.1f}s")

    return tracker


def example_3_gap_detection_workflow():
    """Example 3: Complete gap detection workflow"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Gap Detection → LLM Proposal → Resolution")
    print("="*80)

    tracker = ExpansionHistoryTracker(
        db_path="example_history.db",
        json_backup_path="example_history.json"
    )

    # Step 1: Detect gap
    gap = GapDetectionRecord(
        gap_type="missing_feature",
        description="Cannot reconstruct polyrhythmic patterns with odd subdivisions (5/4, 7/8)",
        severity="high",
        midi_file_triggering="examples/take_five.mid",
        genre_context="jazz",
        musical_characteristics={
            "time_signature": "5/4",
            "subdivision": "quintuplets",
            "pattern_type": "polyrhythmic",
            "complexity": "high"
        }
    )

    gap_id = tracker.log_gap_detection(gap)
    print(f"🔍 Detected gap: {gap_id}")
    print(f"   Type: {gap.gap_type}")
    print(f"   Severity: {gap.severity}")
    print(f"   Description: {gap.description}")

    # Step 2: LLM proposes solution
    proposal = LLMProposalRecord(
        llm_model="claude-sonnet-4-5",
        prompt_hash=hashlib.sha256(b"analyze polyrhythmic gap and propose parameters").hexdigest(),
        proposed_parameters=[
            {
                "name": "rhythm.polyrhythm.odd_subdivision_support",
                "type": "boolean",
                "default": True,
                "description": "Enable odd subdivision support (5, 7, 9, 11)"
            },
            {
                "name": "rhythm.polyrhythm.subdivision_ratio",
                "type": "array_int",
                "default": [3, 5, 7],
                "description": "Supported subdivision ratios"
            },
            {
                "name": "rhythm.polyrhythm.cross_rhythm_probability",
                "type": "probability",
                "default": 0.3,
                "description": "Probability of cross-rhythm patterns"
            }
        ],
        rationale="To support odd-time polyrhythmic patterns like those in Take Five, "
                 "we need parameters controlling subdivision ratios and cross-rhythms. "
                 "This enables reconstruction and generation of complex jazz compositions.",
        triggered_by_gap=gap_id
    )

    proposal_id = tracker.log_llm_proposal(proposal)
    print(f"🤖 LLM proposed solution: {proposal_id}")
    print(f"   Model: {proposal.llm_model}")
    print(f"   Parameters proposed: {len(proposal.proposed_parameters)}")

    # Step 3: Review proposal
    tracker.review_llm_proposal(
        proposal_id,
        decision="accepted",
        notes="Excellent proposal. Addresses gap comprehensively with minimal parameter count.",
        accepted_params=[
            "rhythm.polyrhythm.odd_subdivision_support",
            "rhythm.polyrhythm.subdivision_ratio",
            "rhythm.polyrhythm.cross_rhythm_probability"
        ]
    )
    print(f"✅ Proposal reviewed and accepted")

    # Step 4: Implement expansion
    event = ExpansionEvent(
        parameters_added=[
            "rhythm.polyrhythm.odd_subdivision_support",
            "rhythm.polyrhythm.subdivision_ratio",
            "rhythm.polyrhythm.cross_rhythm_probability"
        ],
        parameter_count=3,
        trigger=ExpansionTrigger.LLM_PROPOSAL,
        trigger_details={"proposal_id": proposal_id},
        gap_description=gap.description,
        llm_proposal=proposal_id,
        reconstruction_accuracy_before=0.58,
        reconstruction_accuracy_after=0.84,
        code_files_modified=[
            "algorithms/rhythm_engine.py",
            "generators/harmonic_rhythm.py",
            "analysis/deep_feature_extractor.py"
        ],
        code_lines_added=230,
        generator_enhanced=True,
        feature_extractor_enhanced=True,
        xgboost_models_trained=3,
        model_training_time_seconds=95.2,
        test_midi_files_passed=22,
        test_midi_files_failed=1,
        musical_validation_passed=True,
        phase=ExpansionPhase.PHASE_1,
        agent_responsible="Agent 5",
        notes="Implemented polyrhythmic support for odd subdivisions"
    )

    event_id = tracker.log_expansion_event(event)
    tracker.update_event_status(event_id, ExpansionStatus.DEPLOYED)
    tracker.update_event_impact(event_id, ParameterImpact.TRANSFORMATIVE, 0.26)
    print(f"🚀 Expansion deployed: {event_id}")

    # Step 5: Close gap
    tracker.update_gap_status(gap_id, "resolved", event_id)
    print(f"✓ Gap resolved")

    # Step 6: Update LLM proposal outcome
    llm_proposal = tracker.llm_proposals[proposal_id]
    llm_proposal.outcome_assessment = ParameterImpact.TRANSFORMATIVE
    llm_proposal.effectiveness_score = 0.95
    llm_proposal.expansion_event_id = event_id

    print(f"\n📊 Workflow Summary:")
    print(f"   Gap → Proposal → Review → Implementation → Deployment → Gap Closed")
    print(f"   Accuracy: {event.reconstruction_accuracy_before:.2%} → {event.reconstruction_accuracy_after:.2%}")
    print(f"   LLM Effectiveness: {llm_proposal.effectiveness_score:.2%}")

    return tracker


def example_4_system_snapshots():
    """Example 4: Capture system snapshots over time"""
    print("\n" + "="*80)
    print("EXAMPLE 4: System Growth Tracking via Snapshots")
    print("="*80)

    tracker = ExpansionHistoryTracker(
        db_path="example_history.db",
        json_backup_path="example_history.json"
    )

    # Simulate snapshots over time
    snapshots_data = [
        {
            "timestamp": datetime.now() - timedelta(days=30),
            "total_parameters": 165,
            "phase": ExpansionPhase.FOUNDATION,
            "accuracy": 0.72
        },
        {
            "timestamp": datetime.now() - timedelta(days=20),
            "total_parameters": 185,
            "phase": ExpansionPhase.PHASE_1,
            "accuracy": 0.76
        },
        {
            "timestamp": datetime.now() - timedelta(days=10),
            "total_parameters": 220,
            "phase": ExpansionPhase.PHASE_1,
            "accuracy": 0.81
        },
        {
            "timestamp": datetime.now(),
            "total_parameters": 267,
            "phase": ExpansionPhase.PHASE_1,
            "accuracy": 0.85
        }
    ]

    for data in snapshots_data:
        snapshot = SystemSnapshot(
            timestamp=data["timestamp"],
            total_parameters=data["total_parameters"],
            parameters_by_category={
                "harmony": int(data["total_parameters"] * 0.35),
                "melody": int(data["total_parameters"] * 0.25),
                "rhythm": int(data["total_parameters"] * 0.20),
                "bass": int(data["total_parameters"] * 0.10),
                "other": int(data["total_parameters"] * 0.10)
            },
            parameters_by_type={
                "continuous": int(data["total_parameters"] * 0.45),
                "categorical": int(data["total_parameters"] * 0.30),
                "boolean": int(data["total_parameters"] * 0.15),
                "probability": int(data["total_parameters"] * 0.10)
            },
            total_code_lines=100000 + (data["total_parameters"] * 50),
            generator_lines=85000 + (data["total_parameters"] * 40),
            feature_extractor_features=120 + int((data["total_parameters"] - 165) * 0.5),
            average_reconstruction_accuracy=data["accuracy"],
            reconstruction_accuracy_by_genre={
                "jazz": data["accuracy"] + 0.02,
                "classical": data["accuracy"] - 0.01,
                "blues": data["accuracy"] + 0.01,
                "rock": data["accuracy"],
            },
            xgboost_models_count=data["total_parameters"],
            average_model_accuracy=data["accuracy"] * 0.95,
            training_data_size=1200 + (data["total_parameters"] * 5),
            active_parameters=data["total_parameters"],
            deprecated_parameters=max(0, data["total_parameters"] - 165) // 20,
            redundant_parameters=max(0, data["total_parameters"] - 165) // 30,
            phase=data["phase"],
            progress_to_phase_goal=(data["total_parameters"] / 515) * 100
        )

        tracker.capture_system_snapshot(snapshot)

    print(f"📸 Captured {len(snapshots_data)} snapshots")
    print(f"\nGrowth over 30 days:")
    print(f"   Parameters: {snapshots_data[0]['total_parameters']} → {snapshots_data[-1]['total_parameters']} (+{snapshots_data[-1]['total_parameters'] - snapshots_data[0]['total_parameters']})")
    print(f"   Accuracy: {snapshots_data[0]['accuracy']:.2%} → {snapshots_data[-1]['accuracy']:.2%} (+{snapshots_data[-1]['accuracy'] - snapshots_data[0]['accuracy']:.2%})")
    print(f"   Progress to Phase 1: {(snapshots_data[-1]['total_parameters'] / 515) * 100:.1f}%")

    return tracker


def example_5_analytics():
    """Example 5: Generate comprehensive analytics"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Comprehensive Analytics")
    print("="*80)

    # First, populate with example data
    tracker = example_2_reconstruction_driven()
    tracker = example_3_gap_detection_workflow()

    # Generate analytics
    analytics = tracker.generate_analytics()

    print(f"\n📊 EXPANSION ANALYTICS")
    print(f"{'='*80}")
    print(f"\nExpansion Summary:")
    print(f"   Total Expansions: {analytics.total_expansions}")
    print(f"   Success Rate: {analytics.success_rate*100:.1f}%")
    print(f"   Parameters Added: {analytics.parameters_added}")
    print(f"   Net Growth: {analytics.net_parameter_growth}")
    print(f"   Growth Rate: {analytics.growth_rate_per_day:.2f} params/day")

    print(f"\nImpact Distribution:")
    print(f"   Transformative: {analytics.transformative_expansions}")
    print(f"   High Value: {analytics.high_value_expansions}")
    print(f"   Low Value: {analytics.low_value_expansions}")
    print(f"   Redundant: {analytics.redundant_expansions}")

    print(f"\nPerformance:")
    print(f"   Avg Accuracy Improvement: {analytics.average_accuracy_improvement*100:.2f}%")
    print(f"   Avg Params per Expansion: {analytics.average_parameters_per_expansion:.1f}")

    print(f"\nLLM Effectiveness:")
    print(f"   Total Proposals: {analytics.llm_proposals_total}")
    print(f"   Acceptance Rate: {analytics.llm_acceptance_rate*100:.1f}%")
    print(f"   Avg Effectiveness: {analytics.llm_average_effectiveness:.2f}")

    print(f"\nGap Closure:")
    print(f"   Gaps Detected: {analytics.gaps_detected}")
    print(f"   Gaps Resolved: {analytics.gaps_resolved}")
    print(f"   Resolution Rate: {analytics.gap_resolution_rate*100:.1f}%")

    return tracker


def example_6_effectiveness_report():
    """Example 6: Generate effectiveness report with recommendations"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Expansion Effectiveness Report")
    print("="*80)

    # Use tracker with data
    tracker = ExpansionHistoryTracker(
        db_path="example_history.db",
        json_backup_path="example_history.json"
    )

    # Generate comprehensive effectiveness report
    report = tracker.generate_expansion_effectiveness_report()

    print(f"\n📋 EFFECTIVENESS REPORT")
    print(f"{'='*80}")

    print(f"\nSystem Overview:")
    for key, value in report['system_overview'].items():
        print(f"   {key}: {value}")

    print(f"\nExpansion Summary:")
    for key, value in report['expansion_summary'].items():
        print(f"   {key}: {value}")

    print(f"\nImpact Distribution:")
    for key, value in report['impact_distribution'].items():
        print(f"   {key}: {value}")

    if report.get('recommendations'):
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")

    return tracker


def example_7_parameter_evolution():
    """Example 7: Track parameter evolution"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Parameter Evolution Report")
    print("="*80)

    tracker = ExpansionHistoryTracker(
        db_path="example_history.db",
        json_backup_path="example_history.json"
    )

    # Check if we have parameters to analyze
    if not tracker.parameter_history:
        print("   No parameters in history yet. Run other examples first.")
        return tracker

    # Get first parameter
    param_path = list(tracker.parameter_history.keys())[0]

    # Update usage statistics (simulate usage)
    for i in range(50):
        tracker.update_parameter_usage(
            param_path,
            used_in_generation=(i % 2 == 0),
            used_in_reconstruction=(i % 3 == 0),
            genre=["jazz", "blues", "classical"][i % 3]
        )

    # Generate evolution report
    report = tracker.generate_parameter_evolution_report(param_path)

    if "error" in report:
        print(f"   {report['error']}")
    else:
        print(f"\n🔬 PARAMETER EVOLUTION: {param_path}")
        print(f"{'='*80}")
        print(f"\nBasic Info:")
        print(f"   Name: {report['parameter_name']}")
        print(f"   Created: {report['created']}")
        print(f"   Age: {report['age_days']} days")
        print(f"   Type: {report['type']}")

        print(f"\nUsage:")
        print(f"   Total Usage: {report['usage']['total_usage']}")
        print(f"   In Generation: {report['usage']['times_used_in_generation']}")
        print(f"   In Reconstruction: {report['usage']['times_used_in_reconstruction']}")
        print(f"   Genres: {', '.join(report['usage']['genres_utilized'])}")

        print(f"\nPerformance:")
        print(f"   Prediction Accuracy: {report['performance']['average_prediction_accuracy']:.2%}")
        print(f"   Feature Importance: {report['performance']['feature_importance_score']:.2f}")
        print(f"   Musical Impact: {report['performance']['musical_impact_score']:.2f}")

    return tracker


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("EXPANSION HISTORY TRACKER - COMPREHENSIVE EXAMPLES")
    print("="*80)
    print("\nThis demonstrates all capabilities of the expansion tracking system")
    print("for the self-expanding inverse music generation system.\n")

    try:
        # Run examples sequentially
        example_1_basic_expansion()
        example_2_reconstruction_driven()
        example_3_gap_detection_workflow()
        example_4_system_snapshots()
        example_5_analytics()
        example_6_effectiveness_report()
        example_7_parameter_evolution()

        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nGenerated files:")
        print("   - example_history.db (SQLite database)")
        print("   - example_history.json (JSON backup)")
        print("\nYou can now query the tracker programmatically or inspect the database.")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
