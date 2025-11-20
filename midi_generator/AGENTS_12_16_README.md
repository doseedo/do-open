# Agents 12-16: Self-Expanding Music Generation System

## Overview

This document describes the implementation of **Agents 12-16**, a sophisticated system of 5 specialized agents that enable **automated self-expansion** of the music generation system. These agents work together to detect gaps in the system's capabilities, propose new parameters, generate code, validate changes, and deploy improvements—all with minimal human intervention.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  AGENT 16: Orchestrator                      │
│              (Master Coordination & Safety)                  │
└─────────────┬───────────────────────────────────────────────┘
              │
    ┌─────────┴─────────┬─────────┬─────────┬─────────┐
    │                   │         │         │         │
    ▼                   ▼         ▼         ▼         ▼
┌─────────┐      ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Agent 12 │      │Agent 13 │ │Agent 14│ │Agent 15│ │Safety  │
│ Code    │      │Musical  │ │Training│ │Model   │ │Monitor │
│Generator│      │Validator│ │Data Gen│ │Trainer │ │        │
└─────────┘      └─────────┘ └────────┘ └────────┘ └────────┘
```

## The Five Agents

### Agent 12: LLM Code Generation Agent
**Location:** `midi_generator/llm/code_generator.py`
**Lines of Code:** 1,147
**Purpose:** Generate production-ready Python code to implement new parameters

**Key Features:**
- Uses Claude API to generate code implementations
- Analyzes existing codebase context
- Generates registry updates, generator modifications, new methods, and tests
- Validates syntax and patterns
- Ensures backward compatibility
- Mock mode available for offline development

**Example Usage:**
```python
from midi_generator.llm.code_generator import LLMCodeGenerationAgent

# Create agent
agent = LLMCodeGenerationAgent(api_key="your_key", mock_mode=False)

# Define parameter proposal
proposal = {
    'name': 'harmony.voicing.quartal_probability',
    'type': 'CONTINUOUS',
    'range': (0.0, 1.0),
    'default': 0.3,
    'description': 'Probability of using quartal voicings',
    'musical_context': 'Common in modal jazz',
    'implementation_strategy': 'Check probability before voicing generation',
    'test_cases': [...]
}

# Generate implementation
implementation = agent.generate_implementation(proposal)

print(f"Registry update: {implementation.registry_update}")
print(f"Modified files: {list(implementation.generator_modifications.keys())}")
print(f"New methods: {list(implementation.new_methods.keys())}")
```

### Agent 13: Musical Validator
**Location:** `midi_generator/validation/musical_validator.py`
**Lines of Code:** 1,172
**Purpose:** Validate parameters and code for musical correctness

**Key Features:**
- 7 comprehensive validation checks
- LLM-powered musical validity assessment
- Music theory consistency checking
- Duplicate detection
- Range validation
- Code pattern validation
- Backward compatibility verification

**Validation Checks:**
1. Naming convention (domain.module.parameter)
2. Musical validity (real musical concept)
3. No duplicates (unique parameter)
4. Range appropriateness (sensible values)
5. Implementation viability (clear strategy)
6. Test coverage (comprehensive tests)
7. Theory consistency (no contradictions)

**Example Usage:**
```python
from midi_generator.validation.musical_validator import MusicalValidator

# Create validator
validator = MusicalValidator(api_key="your_key", mock_mode=False)

# Validate parameter proposal
result = validator.validate_parameter(proposal)

if result.valid:
    print(f"✅ Valid! Score: {result.overall_score:.2f}")
else:
    print(f"❌ Invalid: {result.errors}")

# Validate generated code
code_result = validator.validate_code(implementation, proposal)
print(f"Code valid: {code_result.valid}")
```

### Agent 14: Synthetic Training Data Generator
**Location:** `midi_generator/training/synthetic_data_generator.py`
**Lines of Code:** 946
**Purpose:** Generate diverse, high-quality training data for new parameters

**Key Features:**
- Latin hypercube sampling for even parameter coverage
- Genre-balanced dataset generation
- Musical coherence validation
- Diverse parameter variation
- Metadata tracking
- Export for XGBoost training

**Example Usage:**
```python
from midi_generator.training.synthetic_data_generator import SyntheticTrainingDataGenerator

# Create generator
generator = SyntheticTrainingDataGenerator()

# Generate training data
dataset = generator.generate_training_data(
    param_name='harmony.voicing.quartal_probability',
    param_def=proposal,
    n_examples=1000,
    output_dir=Path('training_data')
)

print(f"Generated {dataset.n_examples} examples")
print(f"Avg coherence: {dataset.statistics['avg_coherence']:.3f}")

# Or generate genre-balanced dataset
balanced_dataset = generator.generate_balanced_dataset(
    param_name='harmony.voicing.quartal_probability',
    param_def=proposal,
    n_per_genre=100,
    genres=['swing', 'bebop', 'modal', 'fusion']
)
```

### Agent 15: Model Training Specialist
**Location:** `midi_generator/training/model_trainer.py`
**Lines of Code:** 943
**Purpose:** Train XGBoost models for parameter prediction

**Key Features:**
- Automatic objective selection (regression/classification)
- Train/val/test splitting with stratification
- Early stopping
- Comprehensive metrics (R², MAE, RMSE, accuracy, F1)
- Hyperparameter tuning with grid search
- Feature importance analysis
- Quality thresholds (R² > 0.5 minimum)
- Batch training support

**Example Usage:**
```python
from midi_generator.training.model_trainer import ModelTrainingSpecialist

# Create trainer
trainer = ModelTrainingSpecialist(
    models_dir=Path('models/pretrained'),
    quality_threshold=0.5
)

# Train model
result = trainer.train_parameter_model(
    param_name='harmony.voicing.quartal_probability',
    param_def=proposal,
    training_data=[
        {'features': np.array([...]), 'parameter_value': 0.7},
        ...
    ]
)

if result.success:
    print(f"✅ Model trained!")
    print(f"Test R²: {result.metrics.test_r2:.3f}")
    print(f"Top features: {result.metrics.top_features[:5]}")
else:
    print(f"❌ Training failed: {result.error}")
```

### Agent 16: Expansion Orchestrator
**Location:** `midi_generator/orchestration/expansion_orchestrator.py`
**Lines of Code:** 933
**Purpose:** Master controller coordinating all expansion agents

**Key Features:**
- Complete 10-stage workflow automation
- Safety checkpoints and rollback
- Human approval workflow (optional)
- Quality verification
- Batch processing
- Expansion history tracking
- Comprehensive error handling

**Expansion Workflow:**
1. **Inverse Analysis** - Analyze MIDI, compute quality
2. **Gap Detection** - Identify missing parameters
3. **Parameter Proposal** - LLM proposes new parameter
4. **Parameter Validation** - Validate musical correctness
5. **Code Generation** - Generate implementation code
6. **Code Validation** - Validate syntax and patterns
7. **Training Data Generation** - Generate 1000 examples
8. **Model Training** - Train XGBoost model
9. **Quality Verification** - Verify improvement
10. **Deployment** - Deploy or rollback

**Example Usage:**
```python
from midi_generator.orchestration.expansion_orchestrator import ExpansionOrchestrator
from pathlib import Path

# Create orchestrator
orchestrator = ExpansionOrchestrator(
    api_key="your_key",
    mock_mode=False
)

# Run expansion from problematic MIDI
result = orchestrator.expand_from_midi(
    input_midi=Path('problematic_song.mid'),
    auto_approve=False,  # Require human approval
    max_expansions=3,    # Add up to 3 parameters
    min_improvement=0.05  # Require 5% quality improvement
)

if result.success:
    print(f"✅ Expansion successful!")
    print(f"Deployed: {result.expansions_deployed}")
    print(f"Quality: {result.initial_quality:.3f} → {result.final_quality:.3f}")
    print(f"Improvement: +{result.quality_improvement:.3f}")
else:
    print(f"❌ Expansion failed: {result.failure_reason}")

# Batch processing
results = orchestrator.batch_expand_from_dataset(
    midi_files=[Path('song1.mid'), Path('song2.mid')],
    max_expansions_per_file=2
)

# Get statistics
stats = orchestrator.get_expansion_statistics()
print(f"Total expansions: {stats['total_expansions']}")
print(f"Parameters added: {stats['total_parameters_added']}")
print(f"Avg improvement: {stats['avg_quality_improvement']:.3f}")
```

## Integration Example

Complete workflow example:

```python
from midi_generator.orchestration.expansion_orchestrator import ExpansionOrchestrator
from pathlib import Path
import os

# Set up API key
os.environ['ANTHROPIC_API_KEY'] = 'your_key_here'

# Create orchestrator (coordinates all agents)
orchestrator = ExpansionOrchestrator(mock_mode=False)

# Process a MIDI file that the system can't reconstruct well
midi_file = Path('jazz_examples/complex_voicings.mid')

print("Starting automated expansion...")
result = orchestrator.expand_from_midi(
    input_midi=midi_file,
    auto_approve=True,
    max_expansions=3
)

if result.success:
    print("\n" + "="*80)
    print("EXPANSION SUCCESSFUL!")
    print("="*80)
    print(f"\nInitial quality: {result.initial_quality:.3f}")
    print(f"Final quality:   {result.final_quality:.3f}")
    print(f"Improvement:     +{result.quality_improvement:.3f} ({result.quality_improvement*100:+.1f}%)")

    print(f"\nNew parameters deployed ({len(result.expansions_deployed)}):")
    for param in result.expansions_deployed:
        print(f"  • {param}")

    print(f"\nExpansion details:")
    for expansion in result.expansion_details:
        if expansion.status.value == 'success':
            print(f"\n  {expansion.parameter_name}:")
            if expansion.training_result and expansion.training_result.metrics:
                metrics = expansion.training_result.metrics
                print(f"    Test R²: {metrics.test_r2:.3f}")
                print(f"    Training time: {metrics.training_time:.2f}s")
                print(f"    Top feature: {metrics.top_features[0][0]}")
else:
    print(f"\n❌ Expansion failed: {result.failure_reason}")
```

## File Structure

```
midi_generator/
├── llm/
│   ├── __init__.py
│   └── code_generator.py          # Agent 12 (1,147 lines)
│
├── validation/
│   ├── __init__.py
│   └── musical_validator.py       # Agent 13 (1,172 lines)
│
├── training/
│   ├── __init__.py
│   ├── synthetic_data_generator.py  # Agent 14 (946 lines)
│   └── model_trainer.py            # Agent 15 (943 lines)
│
├── orchestration/
│   ├── __init__.py
│   └── expansion_orchestrator.py   # Agent 16 (933 lines)
│
└── synthesis/
    └── __init__.py                # Placeholder for future agents
```

## Dependencies

Required packages:
```bash
pip install anthropic           # Claude API
pip install xgboost            # Model training
pip install scikit-learn       # ML utilities
pip install scipy              # Latin hypercube sampling
pip install numpy pandas       # Data processing
pip install joblib             # Model serialization
pip install tqdm               # Progress bars
```

## Configuration

### API Key Setup

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY='your_key_here'
```

Or pass it directly:
```python
orchestrator = ExpansionOrchestrator(api_key='your_key_here')
```

### Mock Mode

For development/testing without API:
```python
orchestrator = ExpansionOrchestrator(mock_mode=True)
```

## Safety Features

### Checkpoints
- System creates checkpoints before any changes
- Automatic rollback on failure
- Preserves parameter registry and models

### Validation
- 7-stage parameter validation
- Code syntax checking
- Music theory consistency
- Backward compatibility verification

### Quality Thresholds
- Minimum R² > 0.5 for models
- Minimum quality improvement threshold
- Automatic hyperparameter tuning if below threshold

## Performance Metrics

**Total Code:** ~5,141 lines across 5 agents

**Agent Breakdown:**
- Agent 12 (Code Generator): 1,147 lines
- Agent 13 (Validator): 1,172 lines
- Agent 14 (Data Generator): 946 lines
- Agent 15 (Model Trainer): 943 lines
- Agent 16 (Orchestrator): 933 lines

**Typical Workflow Timing:**
- Parameter proposal: ~2-5s
- Validation: ~1-3s
- Code generation: ~10-30s
- Training data (1000 examples): ~5-10 minutes
- Model training: ~30-120s
- **Total per parameter: ~8-15 minutes**

## Future Enhancements

Planned for next phase:

1. **Agent 11: LLM Parameter Proposal** - More sophisticated parameter proposals
2. **Agent 10: Intelligent Gap Detector** - Better gap analysis using ML
3. **Agent 9: Inverse Analysis Coordinator** - Full inverse synthesis
4. **Deep Feature Extractor** - Extract 1000+ features from MIDI
5. **XGBoost Synthesizer** - Predict all parameters from features

## Troubleshooting

### Common Issues

**1. API Key Not Found**
```
WARNING: No API key found, using mock mode
```
Solution: Set `ANTHROPIC_API_KEY` environment variable

**2. Import Errors**
```
ImportError: cannot import name 'LLMCodeGenerationAgent'
```
Solution: Ensure you're in the correct directory and packages are installed

**3. Training Data Generation Fails**
```
High failure rate (300/1000)
```
Solution: Check generator configuration and coherence thresholds

## Contributing

To extend the system:

1. **Add New Agents**: Create in appropriate module (llm/, validation/, etc.)
2. **Extend Validation**: Add checks to `MusicalValidator`
3. **Improve Generation**: Enhance prompts in `LLMCodeGenerationAgent`
4. **Add Metrics**: Extend `TrainingMetrics` in model_trainer.py

## License

MIT License - See LICENSE file for details

## Authors

- Musical Program Synthesis Team
- Agent Implementation: Claude & Development Team

## References

- Main project README: `midi_generator/README.md`
- Parameter registry: `midi_generator/parameters/universal_registry.py`
- Integration docs: `midi_generator/INTEGRATION_SUMMARY.md`

---

**Last Updated:** 2025-01-20
**Version:** 1.0.0
**Status:** Production Ready
