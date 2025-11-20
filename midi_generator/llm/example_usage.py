"""
Example Usage of LLM Code Generation Agent - Agent 12
=====================================================

This script demonstrates how to use the LLM Code Generation Agent
to generate implementation code for new parameters.

Author: Agent 12 - Code Generation Agent
Date: 2025-11-20
License: MIT
"""

import json
import os
from pathlib import Path

from code_generator import LLMCodeGenerationAgent, GeneratedCode
from advanced_utilities import (
    DependencyAnalyzer,
    DocumentationGenerator,
    ImpactAnalyzer,
    Dependency
)
from integration_helpers import (
    CodeIntegrator,
    DeploymentHelper
)


def example_1_simple_parameter():
    """
    Example 1: Generate code for a simple probability parameter
    """
    print("\n" + "=" * 70)
    print("Example 1: Simple Probability Parameter")
    print("=" * 70)

    # Define parameter proposal
    proposal = {
        'name': 'harmony.voicing.quartal_probability',
        'type': 'continuous',
        'range': (0.0, 1.0),
        'default': 0.15,
        'description': 'Probability of using quartal voicings (fourths) instead of tertian (thirds)',
        'musical_context': (
            'Jazz voicings built in fourths instead of thirds. '
            'Common in modern jazz (McCoy Tyner) and fusion. '
            'Creates open, modern sound.'
        ),
        'implementation_strategy': (
            'Add probability check in voicing generation. '
            'When triggered, build chords in stacks of perfect fourths '
            'instead of traditional tertian harmony.'
        ),
        'affected_features': [
            'chord_density',
            'voicing_openness',
            'harmonic_color',
            'voice_leading'
        ],
        'test_cases': [
            {
                'value': 0.0,
                'expected': 'No quartal voicings, only tertian harmony',
                'test': 'All chords should use thirds'
            },
            {
                'value': 0.5,
                'expected': 'Approximately 50% quartal voicings',
                'test': 'Random distribution should average 50%'
            },
            {
                'value': 1.0,
                'expected': 'All quartal voicings',
                'test': 'No tertian chords'
            }
        ],
        'example_values': {
            'traditional_jazz': 0.1,
            'bebop': 0.05,
            'modern_jazz': 0.3,
            'modal_jazz': 0.5,
            'fusion': 0.6,
            'avant_garde': 0.8
        },
        'generator_integration_points': [
            'generators/advanced_harmony_generator.py::generate_voicing()',
            'core/modal_harmony.py::create_chord_voicing()'
        ]
    }

    print("\nParameter Proposal:")
    print(json.dumps(proposal, indent=2))

    # Note: Requires ANTHROPIC_API_KEY environment variable
    if 'ANTHROPIC_API_KEY' not in os.environ:
        print("\n⚠️  Set ANTHROPIC_API_KEY environment variable to run this example")
        print("\nExample output would include:")
        print("  - Registry update code")
        print("  - Generator modifications")
        print("  - New helper methods")
        print("  - Comprehensive tests")
        print("  - Integration notes")
        return

    # Initialize agent
    agent = LLMCodeGenerationAgent()

    # Generate implementation
    print("\n🔄 Generating implementation code via Claude...")
    generated_code = agent.generate_implementation(proposal)

    # Display results
    print("\n✅ Code generation complete!")
    print(f"\nRegistry Update ({len(generated_code.registry_update)} chars):")
    print(generated_code.registry_update[:200] + "..." if len(generated_code.registry_update) > 200 else generated_code.registry_update)

    print(f"\nGenerator Modifications: {len(generated_code.generator_modifications)} files")
    for file_path in generated_code.generator_modifications.keys():
        print(f"  - {file_path}")

    print(f"\nNew Methods: {len(generated_code.new_methods)}")
    for method_name in generated_code.new_methods.keys():
        print(f"  - {method_name}()")

    print(f"\nTest Code: {len(generated_code.test_code)} chars")

    print(f"\nIntegration Notes:")
    print(generated_code.integration_notes[:300] + "..." if len(generated_code.integration_notes) > 300 else generated_code.integration_notes)


def example_2_dependency_analysis():
    """
    Example 2: Analyze parameter dependencies
    """
    print("\n" + "=" * 70)
    print("Example 2: Dependency Analysis")
    print("=" * 70)

    analyzer = DependencyAnalyzer()

    # Add some dependencies
    deps = [
        Dependency(
            source='harmony.voicing.quartal_probability',
            target='harmony.voicing.density',
            dependency_type='enhances',
            strength=0.7,
            description='Quartal voicings work well with higher density'
        ),
        Dependency(
            source='harmony.voicing.quartal_probability',
            target='harmony.extensions.use_11th',
            dependency_type='enhances',
            strength=0.9,
            description='Quartal voicings naturally include 11th extensions'
        ),
        Dependency(
            source='harmony.voicing.quartal_probability',
            target='harmony.voicing.close_position',
            dependency_type='conflicts',
            strength=0.8,
            description='Quartal voicings are typically open, not close position'
        )
    ]

    for dep in deps:
        analyzer.add_dependency(dep)

    print("\n📊 Dependency Graph:")
    for source, dep_list in analyzer.dependency_graph.items():
        print(f"\n{source}:")
        for dep in dep_list:
            icon = "🔗" if dep.dependency_type == 'enhances' else "⚠️"
            print(f"  {icon} {dep.dependency_type} {dep.target} (strength: {dep.strength})")
            print(f"     {dep.description}")

    # Check dependency depth
    depth = analyzer.compute_dependency_depth('harmony.voicing.quartal_probability')
    print(f"\n📏 Dependency depth: {depth}")


def example_3_impact_analysis():
    """
    Example 3: Analyze impact of adding a parameter
    """
    print("\n" + "=" * 70)
    print("Example 3: Impact Analysis")
    print("=" * 70)

    # Import codebase index (would use real one in practice)
    from code_generator import CodebaseIndex

    # Create index
    index = CodebaseIndex()

    # Create analyzer
    analyzer = ImpactAnalyzer(index)

    # Analyze impact
    impact = analyzer.analyze_impact(
        parameter_name='harmony.voicing.quartal_probability',
        affected_files=[
            'generators/advanced_harmony_generator.py',
            'core/modal_harmony.py',
            'parameters/universal_registry.py'
        ]
    )

    print(f"\n📈 Impact Analysis Results:")
    print(f"  Parameter: {impact.parameter_name}")
    print(f"  Complexity Score: {impact.complexity_score:.1f}/100")
    print(f"  Risk Level: {impact.risk_level.upper()}")

    print(f"\n📁 Affected Files ({len(impact.affected_files)}):")
    for file in impact.affected_files:
        print(f"  - {file}")

    print(f"\n🧪 Testing Requirements ({len(impact.testing_requirements)}):")
    for req in impact.testing_requirements:
        print(f"  - {req}")

    if impact.migration_steps:
        print(f"\n🚀 Migration Steps ({len(impact.migration_steps)}):")
        for step in impact.migration_steps:
            print(f"  {step}")


def example_4_documentation_generation():
    """
    Example 4: Generate documentation for a parameter
    """
    print("\n" + "=" * 70)
    print("Example 4: Documentation Generation")
    print("=" * 70)

    # Mock parameter definition
    from dataclasses import dataclass
    from enum import Enum

    class ParameterType(Enum):
        CONTINUOUS = "continuous"

    class MusicalImpact(Enum):
        MEDIUM = "medium"

    @dataclass
    class MockParamDef:
        description: str = "Probability of using quartal voicings"
        param_type: ParameterType = ParameterType.CONTINUOUS
        default_value: float = 0.15
        min_value: float = 0.0
        max_value: float = 1.0
        musical_impact: MusicalImpact = MusicalImpact.MEDIUM
        genre_relevance: list = None
        depends_on: list = None

        def __post_init__(self):
            if self.genre_relevance is None:
                self.genre_relevance = ['jazz', 'fusion']
            if self.depends_on is None:
                self.depends_on = []

    param_def = MockParamDef()

    # Generate documentation
    doc_gen = DocumentationGenerator()

    doc = doc_gen.generate_parameter_doc(
        parameter_name='harmony.voicing.quartal_probability',
        parameter_def=param_def,
        code_locations=[
            'generators/advanced_harmony_generator.py::generate_voicing()',
            'core/modal_harmony.py::create_chord_voicing()'
        ]
    )

    print("\n📄 Generated Documentation:")
    print(doc)


def example_5_deployment_workflow():
    """
    Example 5: Complete deployment workflow
    """
    print("\n" + "=" * 70)
    print("Example 5: Deployment Workflow")
    print("=" * 70)

    # Create mock impact analysis
    from advanced_utilities import ImpactAnalysis

    impact = ImpactAnalysis(
        parameter_name='harmony.voicing.quartal_probability',
        affected_files=[
            'generators/advanced_harmony_generator.py',
            'core/modal_harmony.py',
            'parameters/universal_registry.py'
        ],
        complexity_score=35.0,
        risk_level='medium',
        testing_requirements=[
            'Unit test for parameter validation',
            'Integration test with existing voicing parameters',
            'Test edge cases (0.0, 1.0)',
            'Test with jazz genre profile',
            'Test with fusion genre profile'
        ],
        migration_steps=[
            '1. Review all affected files',
            '2. Update parameter registry',
            '3. Run full test suite',
            '4. Deploy to staging',
            '5. Monitor for issues'
        ]
    )

    # Create deployment helper
    helper = DeploymentHelper()

    # Generate deployment checklist
    print("\n✅ Deployment Checklist:")
    checklist = helper.create_deployment_checklist(
        'harmony.voicing.quartal_probability',
        impact,
        impact.affected_files
    )
    print(checklist)

    # Generate git commit message
    print("\n📝 Git Commit Message:")
    commit_msg = helper.generate_git_commit_message(
        'harmony.voicing.quartal_probability',
        impact.affected_files,
        impact
    )
    print(commit_msg)

    # Generate rollback plan
    print("\n🔄 Rollback Plan:")
    rollback = helper.create_rollback_plan(
        'harmony.voicing.quartal_probability',
        impact.affected_files
    )
    print(rollback[:500] + "..." if len(rollback) > 500 else rollback)


def example_6_code_integration():
    """
    Example 6: Integrate generated code (dry run)
    """
    print("\n" + "=" * 70)
    print("Example 6: Code Integration (Dry Run)")
    print("=" * 70)

    # Create mock generated code
    from code_generator import GeneratedCode

    generated = GeneratedCode(
        registry_update="""
"harmony.voicing.quartal_probability": ParameterDefinition(
    name="quartal_probability",
    full_path="harmony.voicing.quartal_probability",
    description="Probability of using quartal voicings",
    param_type=ParameterType.CONTINUOUS,
    default_value=0.15,
    min_value=0.0,
    max_value=1.0
),
""",
        generator_modifications={
            'generators/advanced_harmony_generator.py': """
def generate_voicing(self, chord, params):
    # Get quartal probability
    quartal_prob = params.get('harmony.voicing.quartal_probability', 0.15)
    quartal_prob = max(0.0, min(1.0, quartal_prob))

    # Use quartal voicing?
    if random.random() < quartal_prob:
        return self._create_quartal_voicing(chord)
    else:
        return self._create_tertian_voicing(chord)
"""
        },
        test_code="""
def test_quartal_probability():
    generator = HarmonyGenerator()
    params = {'harmony.voicing.quartal_probability': 1.0}
    voicing = generator.generate_voicing('Cmaj7', params)
    assert is_quartal_voicing(voicing)
"""
    )

    # Create integrator
    integrator = CodeIntegrator()

    # Dry run integration
    print("\n🔄 Integrating code (dry run)...")
    results = integrator.integrate_generated_code(generated, dry_run=True)

    print(f"\n{'✅' if results['success'] else '❌'} Integration Results:")
    print(f"  Success: {results['success']}")
    print(f"  Files Modified: {len(results['files_modified'])}")
    for file in results['files_modified']:
        print(f"    - {file}")
    print(f"  Files Created: {len(results['files_created'])}")
    for file in results['files_created']:
        print(f"    - {file}")

    if results['errors']:
        print(f"\n  Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"    - {error}")

    if results['warnings']:
        print(f"\n  Warnings: {len(results['warnings'])}")
        for warning in results['warnings']:
            print(f"    - {warning}")


def main():
    """
    Run all examples
    """
    print("\n" + "=" * 70)
    print("LLM Code Generation Agent - Example Usage")
    print("Agent 12: Code Generation Agent")
    print("=" * 70)

    examples = [
        ("Simple Parameter", example_1_simple_parameter),
        ("Dependency Analysis", example_2_dependency_analysis),
        ("Impact Analysis", example_3_impact_analysis),
        ("Documentation Generation", example_4_documentation_generation),
        ("Deployment Workflow", example_5_deployment_workflow),
        ("Code Integration", example_6_code_integration),
    ]

    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\n")
    choice = input("Enter example number (or 'all' to run all): ").strip().lower()

    if choice == 'all':
        for name, example_func in examples:
            try:
                example_func()
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        try:
            examples[int(choice) - 1][1]()
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print("Invalid choice. Running all examples...")
        for name, example_func in examples:
            try:
                example_func()
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")

    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
