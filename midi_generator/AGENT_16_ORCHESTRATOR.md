# AGENT 16: Expansion Orchestrator

## Overview

**Agent 16** is the master orchestrator for the Musical Program Synthesis system's self-expansion workflow. It coordinates all agents in the complete expansion cycle: gap detection → parameter proposal → code generation → training → deployment.

**Status**: ✅ FULLY IMPLEMENTED (26/35 agents integrated)

**Location**: `midi_generator/orchestration/expansion_orchestrator.py`

---

## Mission

Automate the complete self-expansion workflow, enabling the system to:
1. Detect gaps in its musical capabilities
2. Propose new parameters to fill those gaps
3. Generate code implementations
4. Train machine learning models
5. Deploy improvements autonomously
6. Rollback on failure

---

## Architecture

### Core Components

```
ExpansionOrchestrator
├── InverseAnalysisCoordinator (MIDI → Features → Parameters → MIDI)
├── IntelligentGapDetector (Agent 10)
├── LLMParameterProposalAgent (Agent 11)
├── LLMCodeGenerationAgent (Agent 12)
├── MusicalValidator (Validation layer)
├── SyntheticTrainingDataGenerator (Agent 14)
├── ModelTrainingSpecialist (Agent 15)
└── SafetyMonitor (Checkpoints & Rollback)
```

### Workflow Stages

The orchestrator executes **11 sequential stages**:

1. **INITIALIZATION** - Setup and validation
2. **INVERSE_ANALYSIS** - Analyze MIDI reconstruction quality
3. **GAP_DETECTION** - Identify missing parameters
4. **PARAMETER_PROPOSAL** - LLM proposes new parameters
5. **PARAMETER_VALIDATION** - Validate musical correctness
6. **CODE_GENERATION** - LLM generates implementation
7. **CODE_VALIDATION** - Validate generated code
8. **TRAINING_DATA_GENERATION** - Create training dataset
9. **MODEL_TRAINING** - Train XGBoost models
10. **QUALITY_VERIFICATION** - Verify improvement
11. **DEPLOYMENT** - Deploy or rollback

---

## Key Features

### 1. Safety-First Architecture

```python
# Automatic checkpoint creation before expansion
checkpoint = safety_monitor.create_checkpoint("Pre-expansion checkpoint")

try:
    # Execute expansion workflow
    result = orchestrator.expand_from_midi(midi_file)
except:
    # Automatic rollback on failure
    safety_monitor.rollback_to_checkpoint(checkpoint)
```

### 2. Human-in-the-Loop Control

```python
# Require approval for each parameter
result = orchestrator.expand_from_midi(
    input_midi='song.mid',
    auto_approve=False  # Request human approval
)
```

### 3. Quality Verification

```python
# Only deploy if improvement meets threshold
result = orchestrator.expand_from_midi(
    input_midi='song.mid',
    min_improvement=0.05  # Require 5% quality improvement
)
```

### 4. Batch Processing

```python
# Analyze multiple MIDI files
results = orchestrator.batch_expand_from_dataset(
    midi_files=[Path(f) for f in glob('dataset/*.mid')],
    max_expansions_per_file=3
)
```

---

## Usage

### Basic Expansion

```python
from pathlib import Path
from midi_generator.orchestration import ExpansionOrchestrator

# Initialize orchestrator
orchestrator = ExpansionOrchestrator(
    api_key='your-anthropic-api-key',  # For LLM agents
    mock_mode=False  # Set True for testing without API
)

# Run expansion on a single MIDI file
result = orchestrator.expand_from_midi(
    input_midi=Path('examples/jazz_piano.mid'),
    auto_approve=True,  # Auto-deploy without human approval
    max_expansions=3,   # Maximum 3 new parameters
    min_improvement=0.05  # Require 5% quality improvement
)

# Check result
if result.success:
    print(f"✅ Deployed {len(result.expansions_deployed)} parameters")
    print(f"Quality: {result.initial_quality:.3f} → {result.final_quality:.3f}")
    for param in result.expansions_deployed:
        print(f"  - {param}")
else:
    print(f"❌ Failed: {result.failure_reason}")
```

### Batch Expansion from Dataset

```python
from pathlib import Path
from midi_generator.orchestration import ExpansionOrchestrator

orchestrator = ExpansionOrchestrator(mock_mode=True)

# Analyze multiple MIDI files
midi_files = [
    Path('dataset/jazz/take_five.mid'),
    Path('dataset/classical/bach_fugue.mid'),
    Path('dataset/pop/hey_jude.mid')
]

results = orchestrator.batch_expand_from_dataset(
    midi_files=midi_files,
    max_expansions_per_file=2
)

# Print summary
stats = orchestrator.get_expansion_statistics()
print(f"Total parameters added: {stats['total_parameters_added']}")
print(f"Average quality improvement: {stats['avg_quality_improvement']:.3f}")
```

### Progressive Expansion Loop

```python
# Continuously improve system on diverse dataset
import glob
from pathlib import Path

orchestrator = ExpansionOrchestrator()

# Get all MIDI files
midi_files = [Path(f) for f in glob.glob('dataset/**/*.mid', recursive=True)]

# Expand iteratively
for epoch in range(10):
    print(f"\n=== EXPANSION EPOCH {epoch+1}/10 ===")

    # Select random subset
    import random
    subset = random.sample(midi_files, min(20, len(midi_files)))

    # Batch expand
    results = orchestrator.batch_expand_from_dataset(
        subset,
        max_expansions_per_file=1  # 1 param per file max
    )

    # Check progress
    stats = orchestrator.get_expansion_statistics()
    print(f"Total parameters: {stats['total_parameters_added']}")
    print(f"Avg quality: {stats['avg_quality_improvement']:.3f}")

    # Stop if no improvement
    if stats['avg_quality_improvement'] < 0.01:
        print("Converged! Stopping expansion.")
        break
```

---

## Data Structures

### ExpansionWorkflowResult

```python
@dataclass
class ExpansionWorkflowResult:
    """Complete expansion workflow result"""
    success: bool
    expansions_deployed: List[str]  # Deployed parameter names
    quality_improvement: float  # Quality delta
    initial_quality: float  # Quality before expansion
    final_quality: float  # Quality after expansion
    expansion_details: List[ParameterExpansion]
    failure_reason: Optional[str]
    checkpoint_id: Optional[str]
    total_time: float
```

### ParameterExpansion

```python
@dataclass
class ParameterExpansion:
    """Single parameter expansion record"""
    parameter_name: str
    proposal: dict  # LLM-generated proposal
    validation_result: Optional[ParameterValidationResult]
    generated_code: Optional[GeneratedImplementation]
    training_dataset: Optional[TrainingDataset]
    training_result: Optional[ModelTrainingResult]
    status: ExpansionStatus  # PENDING, SUCCESS, FAILED, ROLLED_BACK
    error: Optional[str]
    stage: ExpansionStage
```

### ExpansionCheckpoint

```python
@dataclass
class ExpansionCheckpoint:
    """System checkpoint for rollback"""
    checkpoint_id: str
    timestamp: str
    backup_paths: Dict[str, Path]
    parameter_count: int
    model_count: int
    description: str
```

---

## Integration Points

### Agent Dependencies

| Agent | Role | Integration Method |
|-------|------|-------------------|
| Agent 8 | Deep Feature Extractor | `DeepFeatureExtractor().extract(midi)` |
| Agent 9 | Feature-Parameter Mapper | `FeatureParameterMapper().predict_all()` |
| Agent 10 | Gap Detector | `IntelligentGapDetector().detect_gaps()` |
| Agent 11 | Parameter Proposer | `LLMParameterProposalAgent().propose()` |
| Agent 12 | Code Generator | `LLMCodeGenerationAgent().generate()` |
| Agent 14 | Training Data Generator | `SyntheticTrainingDataGenerator().generate()` |
| Agent 15 | Model Trainer | `ModelTrainingSpecialist().train()` |

### Data Flow

```
MIDI File
    ↓
[Agent 8] Extract 1000 features
    ↓
[Agent 9] Predict parameters (515+)
    ↓
[Synthesizer] Generate MIDI from parameters
    ↓
[Agent 8] Extract features from generated MIDI
    ↓
[Compare] Compute feature reconstruction error
    ↓
[Agent 10] Detect parameter gaps
    ↓
[Agent 11] Propose new parameters (LLM)
    ↓
[Validator] Validate parameter proposal
    ↓
[Agent 12] Generate code implementation (LLM)
    ↓
[Validator] Validate generated code
    ↓
[Agent 14] Generate training data
    ↓
[Agent 15] Train XGBoost model
    ↓
[Quality Check] Verify improvement
    ↓
[Deploy or Rollback]
```

---

## Safety Mechanisms

### 1. Automatic Checkpoints

Before any expansion, the orchestrator creates a checkpoint:

```python
class SafetyMonitor:
    def create_checkpoint(self, description: str) -> ExpansionCheckpoint:
        """Create system checkpoint with backups"""
        # Backup critical files:
        # - universal_registry.py
        # - registry.json
        # - trained models
        # - configuration files
```

### 2. Rollback on Failure

Any failure triggers automatic rollback:

```python
try:
    result = orchestrator.expand_from_midi(midi_file)
except Exception as e:
    # Automatic rollback to pre-expansion state
    safety_monitor.rollback_to_checkpoint(checkpoint)
```

### 3. Quality Thresholds

Only deploy if improvement is significant:

```python
if improvement >= min_improvement:
    deploy_parameter()
else:
    rollback_to_checkpoint()
```

### 4. Human Approval Checkpoints

Optional human-in-the-loop approval:

```python
if not auto_approve:
    # Show parameter details
    # Request user approval: yes/no
    if not user_approves():
        skip_parameter()
```

---

## Configuration

### Orchestrator Initialization

```python
orchestrator = ExpansionOrchestrator(
    api_key='your-api-key',  # Anthropic API key
    mock_mode=False  # True for testing without API calls
)
```

### Expansion Parameters

```python
result = orchestrator.expand_from_midi(
    input_midi=Path('song.mid'),
    auto_approve=True,  # Skip human approval
    max_expansions=3,  # Max parameters to add
    min_improvement=0.05  # Min quality delta (5%)
)
```

### Batch Parameters

```python
results = orchestrator.batch_expand_from_dataset(
    midi_files=[...],
    max_expansions_per_file=3  # Max per file
)
```

---

## Monitoring & Logging

### Expansion Statistics

```python
stats = orchestrator.get_expansion_statistics()

print(f"Total expansions: {stats['total_expansions']}")
print(f"Successful: {stats['successful_expansions']}")
print(f"Failed: {stats['failed_expansions']}")
print(f"Parameters added: {stats['total_parameters_added']}")
print(f"Avg improvement: {stats['avg_quality_improvement']:.3f}")
print(f"Total time: {stats['total_time']:.1f}s")
```

### Expansion History

```python
# History saved to expansion_history.json
{
    "timestamp": "2025-01-15T10:30:00",
    "success": true,
    "expansions_deployed": [
        "harmony.voicing.quartal_probability",
        "rhythm.syncopation.intensity"
    ],
    "quality_improvement": 0.12,
    "initial_quality": 0.65,
    "final_quality": 0.77,
    "total_time": 145.3
}
```

---

## Error Handling

### Stage-Level Error Recovery

Each stage has independent error handling:

```python
try:
    # Stage 1: Inverse Analysis
    analysis = inverse_analyzer.analyze(midi)
except Exception as e:
    expansion.status = ExpansionStatus.FAILED
    expansion.error = f"Analysis failed: {e}"
    continue  # Skip to next parameter
```

### Critical Error Rollback

Critical errors trigger full rollback:

```python
try:
    # Execute all stages
    result = expand_from_midi(...)
except Exception as e:
    # Rollback entire expansion
    safety_monitor.rollback_to_checkpoint(checkpoint)
    return failed_result
```

---

## Performance

### Expansion Timing

| Stage | Typical Time |
|-------|-------------|
| Inverse Analysis | 2-5s |
| Gap Detection | 1-2s |
| Parameter Proposal (LLM) | 5-10s |
| Validation | 2-3s |
| Code Generation (LLM) | 10-20s |
| Training Data Generation | 30-60s (100 examples) |
| Model Training | 10-30s |
| Quality Verification | 2-5s |
| **Total per Parameter** | **60-135s** |

### Batch Performance

- 20 MIDI files with 3 params each = ~60 parameters
- Total time: ~2-4 hours (parallel processing potential)
- Recommended: Run overnight for large datasets

---

## Testing

### Mock Mode

Test without API calls or actual training:

```python
orchestrator = ExpansionOrchestrator(mock_mode=True)

# Uses mock implementations:
# - Mock LLM responses
# - Mock training results
# - Simulated quality improvements
result = orchestrator.expand_from_midi('test.mid')
```

### Unit Testing

```python
def test_safety_monitor():
    monitor = SafetyMonitor()
    checkpoint = monitor.create_checkpoint("Test")
    assert checkpoint.checkpoint_id is not None

def test_expansion_workflow():
    orch = ExpansionOrchestrator(mock_mode=True)
    result = orch.expand_from_midi(Path('test.mid'))
    assert result is not None
```

---

## Troubleshooting

### Issue: "Could not import all agent modules"

**Solution**: Ensure all dependencies are installed:
```bash
pip install anthropic xgboost scipy scikit-learn mido
```

### Issue: Expansions failing validation

**Solution**: Check parameter proposals are musically valid:
- Proper parameter types
- Valid ranges
- Clear descriptions
- Musical context provided

### Issue: Quality not improving

**Solution**: Check:
- Training data quality (Agent 14)
- Feature extraction accuracy (Agent 8)
- Model training convergence (Agent 15)
- Parameter relevance (Agent 10)

### Issue: Rollback failing

**Solution**: Verify checkpoint directory permissions:
```bash
mkdir -p .checkpoints
chmod 755 .checkpoints
```

---

## Examples

See:
- `midi_generator/examples/agent16_dataset_analysis_example.py`
- `midi_generator/examples/agent16_orchestration_demo.py` (to be created)
- `midi_generator/examples/agent16_progressive_expansion.py` (to be created)

---

## Future Enhancements

### Planned Improvements

1. **Parallel Parameter Training**
   - Train multiple parameters simultaneously
   - 10x speedup for batch expansion

2. **Incremental Model Updates**
   - Update existing models instead of retraining
   - Faster iteration cycles

3. **Active Learning**
   - Intelligently select MIDI files for maximum learning
   - Reduce required training examples

4. **Distributed Expansion**
   - Run orchestrator across multiple machines
   - Scale to massive datasets

5. **Confidence Scoring**
   - Estimate parameter reliability
   - Prioritize high-confidence additions

---

## Related Documentation

- [Agent 8: Deep Feature Extractor](AGENT_8_DEEP_FEATURE_EXTRACTOR.md)
- [Agent 9: Feature-Parameter Mapper](AGENT_9_FEATURE_MAPPING.md)
- [Agent 10: Intelligent Gap Detector](AGENT_10_GAP_DETECTION.md)
- [Agent 11: LLM Parameter Proposer](AGENT_11_PARAMETER_PROPOSER.md)
- [Agent 12: LLM Code Generator](AGENT_12_CODE_GENERATOR.md)
- [Agent 14: Training Data Generator](AGENT_14_TRAINING_DATA.md)
- [Agent 15: Model Trainer](AGENT_15_MODEL_TRAINING.md)

---

## License

MIT License - Part of the Musical Program Synthesis System

---

## Author

**Agent 16 - Expansion Orchestrator**

Part of the 35-Agent Master Prompt System for Self-Expanding Inverse Music Generation
