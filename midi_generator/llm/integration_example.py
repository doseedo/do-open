"""
Integration Example: Agent 11 in Self-Expanding Music System
==============================================================

This demonstrates how Agent 11 (LLM Parameter Proposal Agent) integrates
with the broader self-expanding music generation system.

Workflow:
1. User provides MIDI file system cannot reproduce
2. Gap Detector (Agent 10) identifies missing capabilities
3. Agent 11 proposes new parameters
4. Parameters integrated into Universal Registry
5. XGBoost models trained for new parameters
6. System can now handle previously impossible music

Author: Agent 11 - Parameter Proposal Agent
License: MIT
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from llm.parameter_proposer import (
    LLMParameterProposalAgent,
    GapAnalysis,
    ProposalStatus
)
from parameters.universal_registry import REGISTRY


def example_1_basic_proposal():
    """
    Example 1: Basic parameter proposal from gap analysis
    """

    print("=" * 80)
    print("EXAMPLE 1: Basic Parameter Proposal")
    print("=" * 80)

    # Simulate gap analysis from Gap Detector
    gap = GapAnalysis(
        suggested_parameter='harmony.voicing.quartal_probability',
        affected_features=[
            'quartal_voicing_count',
            'fourth_interval_ratio',
            'open_voicing_ratio'
        ],
        avg_error=0.75,
        impact_score=0.88,
        rationale=(
            'Input MIDI (McCoy Tyner - Passion Dance) has extensive quartal harmony. '
            'System produces tertian voicings, resulting in wrong harmonic color.'
        ),
        confidence=0.92,
        priority='HIGH',
        parameter_info={
            'type': 'PROBABILITY',
            'musical_rationale': (
                'Quartal voicings build chords from stacked fourths (7 semitones) '
                'instead of thirds. Common in modal jazz (McCoy Tyner, Herbie Hancock), '
                'modern classical (Hindemith, Bartók), and impressionist music (Debussy).'
            ),
            'typical_usage': {
                'modal_jazz': 0.7,
                'bebop': 0.1,
                'impressionist': 0.6,
                'swing': 0.0,
                'cool_jazz': 0.4
            }
        },
        gap_id='GAP_20250120_001'
    )

    print(f"\n📋 Gap Detected:")
    print(f"   Parameter: {gap.suggested_parameter}")
    print(f"   Error: {gap.avg_error:.2f}")
    print(f"   Impact: {gap.impact_score:.2f}")
    print(f"   Rationale: {gap.rationale}")

    # Initialize agent
    agent = LLMParameterProposalAgent()

    # Note: This requires ANTHROPIC_API_KEY to actually call the API
    # For demonstration, we'll show the expected workflow

    print(f"\n🤖 Agent 11 would:")
    print(f"   1. Build comprehensive prompt with musical context")
    print(f"   2. Call Claude API with gap analysis")
    print(f"   3. Parse JSON response into ParameterProposal")
    print(f"   4. Validate against 5-stage validation system")
    print(f"   5. Track in proposal history")

    print(f"\n📤 Expected Output:")
    print(f"""
   ParameterProposal(
       name='harmony.voicing.quartal_probability',
       type='PROBABILITY',
       range=[0.0, 1.0],
       default=0.3,
       description='Probability of using quartal voicings instead of tertian harmony',
       musical_context='Quartal harmony common in modal jazz (McCoy Tyner, Herbie Hancock)...',
       implementation_strategy='In voicing generation, check probability. If triggered, build voicing with perfect fourths...',
       test_cases=[
           TestCase(value=0.0, expected='No quartal voicings', features={{'quartal_count': 0}}),
           TestCase(value=0.5, expected='Mixed voicings', features={{'quartal_count': '>10'}}),
           TestCase(value=1.0, expected='Pure quartal', features={{'quartal_count': '>20'}})
       ],
       status=ProposalStatus.VALIDATED
   )
    """)

    print(f"\n✅ After validation, parameter ready for integration!")


def example_2_batch_processing():
    """
    Example 2: Batch processing multiple gaps
    """

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Batch Processing Multiple Gaps")
    print("=" * 80)

    # Simulate multiple gaps from Gap Detector
    gaps = [
        {
            'suggested_parameter': 'melody.bebop.chromatic_approach_prob',
            'affected_features': ['chromatic_approach_count', 'bebop_scale_usage'],
            'avg_error': 0.82,
            'impact_score': 0.91,
            'priority': 'HIGH'
        },
        {
            'suggested_parameter': 'rhythm.clave.pattern_type',
            'affected_features': ['clave_pattern_detected', 'latin_rhythm_ratio'],
            'avg_error': 0.68,
            'impact_score': 0.85,
            'priority': 'MEDIUM'
        },
        {
            'suggested_parameter': 'instrumentation.drums.brush_technique',
            'affected_features': ['brush_texture', 'swish_count'],
            'avg_error': 0.71,
            'impact_score': 0.77,
            'priority': 'MEDIUM'
        }
    ]

    print(f"\n📋 Detected {len(gaps)} gaps requiring new parameters")

    agent = LLMParameterProposalAgent()

    print(f"\n🤖 Batch Processing Workflow:")
    print(f"   1. Sort gaps by priority and impact")
    print(f"   2. Process HIGH priority gaps first")
    print(f"   3. Auto-integrate high-confidence proposals (>0.9)")
    print(f"   4. Queue medium-confidence for human review")
    print(f"   5. Track all proposals in history")

    for gap in gaps:
        print(f"\n   Processing: {gap['suggested_parameter']}")
        print(f"      Priority: {gap['priority']}")
        print(f"      Impact: {gap['impact_score']:.2f}")
        print(f"      → Would generate proposal, validate, track")

    print(f"\n📈 Expected Results:")
    print(f"   Total proposals: {len(gaps)}")
    print(f"   Estimated success rate: ~90%")
    print(f"   Auto-integrated: ~2-3 parameters")
    print(f"   Pending review: ~0-1 parameters")


def example_3_full_integration():
    """
    Example 3: Full integration with existing system
    """

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Full System Integration")
    print("=" * 80)

    print(f"\n📊 Current System State:")
    print(f"   Registry parameters: {len(REGISTRY.get_all_parameters())}")

    print(f"\n🔄 Self-Expanding Workflow:")
    print(f"""
   1. USER ACTION:
      → Provides MIDI file: 'thelonious_monk_trinkle_tinkle.mid'

   2. SYNTHESIS ATTEMPT:
      → System generates MIDI from extracted features
      → Reconstruction error: 0.82 (high!)

   3. GAP DETECTOR (Agent 10):
      → Analyzes feature errors
      → Identifies: Monk's whole-tone runs not reproducible
      → Proposes: 'melody.scales.whole_tone_probability'

   4. PARAMETER PROPOSAL (Agent 11):
      → Calls Claude API with gap analysis
      → Claude proposes:
         {{
           name: 'melody.scales.whole_tone_probability',
           type: 'PROBABILITY',
           range: [0.0, 1.0],
           default: 0.2,
           description: 'Probability of whole-tone scale passages',
           musical_context: 'Whole-tone scales used by Debussy, Monk...',
           implementation_strategy: 'In scale selection, check probability...',
           test_cases: [...]
         }}
      → Validates (passes all checks)
      → Status: VALIDATED

   5. INTEGRATION:
      → Parameter added to Universal Registry
      → Now: {len(REGISTRY.get_all_parameters()) + 1} parameters

   6. MODEL TRAINING (Agent 9):
      → XGBoost model trained for new parameter
      → Training data: 1000 MIDI files
      → Model learns: whole_tone_probability from features

   7. CODE GENERATION (Future Agent 12):
      → Generates implementation in melody_generator.py
      → Adds whole-tone scale logic to scale selection

   8. VALIDATION:
      → Re-synthesize 'thelonious_monk_trinkle_tinkle.mid'
      → New reconstruction error: 0.31 (much better!)
      → System has expanded to handle Monk's style

   9. KNOWLEDGE RETENTION:
      → New parameter persists in registry
      → Model saved for future use
      → System permanently improved
    """)

    print(f"\n✅ System Self-Expanded Successfully!")
    print(f"   Before: Could not reproduce whole-tone runs")
    print(f"   After: Can generate Monk-style whole-tone passages")
    print(f"   Process: Fully automated via LLM reasoning")


def example_4_validation_details():
    """
    Example 4: Validation system in detail
    """

    print("\n" + "=" * 80)
    print("EXAMPLE 4: Validation System")
    print("=" * 80)

    print(f"\n🔍 5-Stage Validation Process:")

    print(f"\n   STAGE 1: Structural Validation")
    print(f"      ✅ Name format: domain.module.parameter")
    print(f"      ✅ No duplicates in registry")
    print(f"      ✅ Valid parameter type")
    print(f"      ✅ Range consistency")
    print(f"      Example: 'harmony.voicing.quartal_probability' → ✅ PASS")

    print(f"\n   STAGE 2: Semantic Validation")
    print(f"      ✅ Default within range")
    print(f"      ✅ Categorical options valid")
    print(f"      ✅ Test case coverage (min 2, preferably 3+)")
    print(f"      Example: default=0.3, range=[0.0,1.0] → ✅ PASS")

    print(f"\n   STAGE 3: Musical Validation")
    print(f"      ✅ Description quality (>20 chars)")
    print(f"      ✅ Musical context depth (>50 chars)")
    print(f"      ✅ Musical terminology present")
    print(f"      Example: 'quartal voicings', 'modal jazz' → ✅ PASS")

    print(f"\n   STAGE 4: Implementation Validation")
    print(f"      ✅ Integration points specified")
    print(f"      ✅ Implementation strategy detailed")
    print(f"      ✅ Affected features identified")
    print(f"      Example: 'generators/harmony_generator.py::generate_voicing()' → ✅ PASS")

    print(f"\n   STAGE 5: Relationship Validation")
    print(f"      ✅ Conflicts are valid parameters")
    print(f"      ✅ Dependencies exist in registry")
    print(f"      ✅ No circular dependencies")
    print(f"      Example: related=['harmony.voicing.quintal_probability'] → ✅ PASS")

    print(f"\n📊 Validation Results:")
    print(f"   ✅ All stages passed → ProposalStatus.VALIDATED")
    print(f"   ⚠️  Warnings present → Still valid, but flagged for review")
    print(f"   ❌ Any stage failed → ProposalStatus.REJECTED")


def example_5_metrics_tracking():
    """
    Example 5: Metrics and analytics
    """

    print("\n" + "=" * 80)
    print("EXAMPLE 5: Metrics & Analytics")
    print("=" * 80)

    agent = LLMParameterProposalAgent()

    print(f"\n📈 Agent Metrics (Example):")
    print(f"""
   {{
     'total_proposals': 47,
     'successful_proposals': 42,
     'failed_proposals': 5,
     'success_rate': 0.894,
     'api_calls': 47,
     'api_errors': 0,

     'history_stats': {{
       'total_proposals': 47,
       'by_status': {{
         'pending': 3,
         'validated': 12,
         'implemented': 30,
         'rejected': 2
       }},
       'acceptance_rate': 0.894,
       'pending_count': 15
     }}
   }}
    """)

    print(f"\n📊 Analytics Insights:")
    print(f"   • 89.4% of proposals pass validation")
    print(f"   • 63.8% already implemented in system")
    print(f"   • 25.5% validated, pending integration")
    print(f"   • 6.4% pending review")
    print(f"   • 4.3% rejected (validation failures)")

    print(f"\n🎯 Quality Metrics:")
    print(f"   • Average confidence score: 0.87")
    print(f"   • Average test case coverage: 3.2 per parameter")
    print(f"   • Average musical context length: 287 chars")
    print(f"   • Average implementation detail: 215 chars")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║          AGENT 11 INTEGRATION EXAMPLES                                     ║
║          LLM Parameter Proposal Agent                                      ║
║                                                                            ║
║          Self-Expanding Inverse Music Generation System                    ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Run examples
    example_1_basic_proposal()
    example_2_batch_processing()
    example_3_full_integration()
    example_4_validation_details()
    example_5_metrics_tracking()

    print("\n" + "=" * 80)
    print("✅ ALL INTEGRATION EXAMPLES COMPLETE")
    print("=" * 80)
    print(f"\nAgent 11 is ready to integrate with:")
    print(f"   • Gap Detector (Agent 10) - for gap analysis input")
    print(f"   • Universal Registry (Agent 3) - for parameter storage")
    print(f"   • XGBoost Synthesizer (Agent 9) - for model training")
    print(f"   • Code Generator (Agent 12) - for implementation (future)")
    print(f"\n✨ The self-expanding system is operational!")
    print("=" * 80)
