# Semantic Discovery System - Quick Reference

## Core Components (Absolute Paths)

### Neural Architecture & Training
1. **SemanticFeatureEncoder** (Agent 3)
   - `/home/user/Do/midi_generator/learning/semantic_encoder.py`
   - Generic 200D → N semantic features encoder
   - Architecture: encoder + decoder + locality predictor
   - Loss: reconstruction + sparsity + locality + orthogonality

2. **SemanticDiscoveryPipeline** (Agent 7)
   - `/home/user/Do/midi_generator/learning/semantic_discovery_pipeline.py`
   - 6-stage end-to-end pipeline orchestration
   - Coordinates all agents (1-6, 8-9)
   - Configuration-driven, supports resumption

3. **GapDiscoveryTrainer** (Agent 5)
   - `/home/user/Do/midi_generator/learning/gap_discovery_trainer.py`
   - Training loop with early stopping & checkpointing
   - LocalityTransformGenerator for synthetic transforms
   - TrainingMonitor for TensorBoard/W&B logging
   - Semantic feature bank extraction

4. **GapDataset** (Agent 4)
   - `/home/user/Do/midi_generator/learning/gap_dataset.py`
   - Reconstruction gap computation
   - Gap statistics aggregation
   - ParameterMIDIGenerator for approximate reconstruction
   - Caching system for 5-10 GB datasets

### Feature Representation & Interpretation
5. **SemanticFeature** (Agent 2)
   - `/home/user/Do/midi_generator/learning/semantic_features.py`
   - Dataclass: feature_id, weight_vector, modality, activation_threshold, locality_constraints
   - Methods: get_activation_strength(), matches_pattern()
   - SemanticFeatureBank for managing collections

6. **FeatureInterpreter** (Agent 6)
   - `/home/user/Do/midi_generator/learning/feature_interpreter.py`
   - 30+ musical test patterns (pitch, harmony, rhythm, articulation, dynamics)
   - ConceptMatcher: matches features to known concepts
   - ParameterNameGenerator: generates human-readable names
   - ExtractionFunctionGenerator: creates parameter extractors

7. **MusicalLocalityFunctions** (Agent 1)
   - `/home/user/Do/midi_generator/learning/musical_locality.py`
   - 12 musical transformation types
   - apply_transformation() and get_available_transformations() interface
   - Used for locality constraint enforcement

8. **SemanticConstraintValidator** (Agent 8)
   - `/home/user/Do/midi_generator/learning/semantic_constraints.py`
   - MusicalValidityRules: valid domains, invalid patterns, constraints
   - LocalityConsistencyChecker: validates transformation invariance
   - RedundancyDetector: correlation-based duplicate detection
   - Severity levels: CRITICAL, WARNING, INFO

### Parameter Management
9. **UniversalParameterRegistry**
   - `/home/user/Do/midi_generator/parameters/universal_registry.py`
   - ParameterDefinition: name, type, constraints, validation, metadata
   - ParameterType: CONTINUOUS, INTEGER, CATEGORICAL, BOOLEAN, ARRAY, etc.
   - ParameterCategory: HARMONY, MELODY, RHYTHM, BASS, etc.
   - MusicalImpact: CRITICAL, HIGH, MEDIUM, LOW
   - Registry JSON: `registry.json`, `registry_with_structure.json`

### Big Band Integration Points
10. **Specialist Agents** (`/home/user/Do/midi_generator/experts/`)
   - `harmony_specialist.py` (Agent 18)
   - `rhythm_specialist.py` (Agent 20)
   - `melody_specialist.py`
   - `dynamics_specialist.py`
   - `texture_specialist.py`
   - `structure_specialist.py` (Agent 23)

---

## What Exists vs. What's Missing

### EXISTS (25 files in learning/ directory)
- semantic_encoder.py ✓
- semantic_discovery_pipeline.py ✓
- gap_discovery_trainer.py ✓
- gap_dataset.py ✓
- feature_interpreter.py ✓
- semantic_constraints.py ✓
- semantic_features.py ✓
- musical_locality.py ✓
- hierarchical_trainer.py ✓
- hierarchical_mtl.py ✓
- hierarchical_predictor.py ✓
- feature_parameter_mapper.py ✓
- pattern_extractor.py ✓
- corpus_learner.py ✓
- motif_library.py ✓
- natural_language_predictor.py ✓
- auto_labeler.py ✓
- labeling_tool.py ✓
- dataset_utils.py ✓
- dataset_statistics.py ✓

### MISSING FOR MODULAR INTEGRATION
- HarmonySemanticEncoder ✗
- RhythmSemanticEncoder ✗
- FormSemanticEncoder ✗
- TextureSemanticEncoder ✗
- OrchestrationSemanticEncoder ✗
- ModularEncoderFactory ✗
- MultiModalSemanticDiscoveryPipeline ✗
- DomainAwareValidator ✗
- DomainAwareInterpreter ✗
- ModularParameterExtractors (harmony, rhythm, form, texture, orchestration) ✗
- Multi-task learning framework ✗

---

## Key Findings

### Architecture Summary
```
OptimizedFeatureExtractor (200D)
├── SemanticFeatureEncoder (20-30D)
├── GapDiscoveryTrainer (trains)
├── FeatureInterpreter (→ names)
├── SemanticConstraintValidator (validates)
└── UniversalParameterRegistry (stores)
```

### Integration Chain
MIDI → 200D Features → Semantic Encoder → Learned Features → Interpretation → Parameters

### Current Limitation
Generic encoder doesn't optimize for specific musical domains (harmony, rhythm, form, etc.)

### Next Step
Create modular encoders for each specialist agent domain

---

## Documentation Files
- `SEMANTIC_DISCOVERY_EXPLORATION.md` (This exploration)
- `README_AGENT03_SEMANTIC_ENCODER.md` (Encoder guide)
- `AGENT_7_INTEGRATION_GUIDE.md` (Pipeline integration)
- `AGENT_05_ARCHITECTURE_REPORT.md` (Trainer architecture)
- `AGENT_06_FEATURE_INTERPRETER_README.md` (Interpreter docs)
- `GAP_DATASET_README.md` (Dataset docs)
- `AGENT_02_SEMANTIC_FEATURES.md` (Feature representations)

---

## Configuration Highlights

### SemanticFeatureEncoder (EncoderConfig)
- input_dim: 200
- hidden_dim: 512
- num_semantic_features: 20-30
- num_locality_types: 12
- reconstruction_weight: 1.0
- sparsity_weight: 0.01
- locality_weight: 0.5

### GapDiscoveryTrainer (TrainingConfig)
- batch_size: 64
- num_epochs: 200
- learning_rate: 0.001
- weight_decay: 1e-5
- early_stopping_patience: 25
- locality_types: transpose, time_shift, octave_shift, augment, diminish
- device: auto-detect (cuda/mps/cpu)

### SemanticDiscoveryPipeline (PipelineConfig)
- num_semantic_features: 25-30
- max_epochs: 100
- batch_size: 64
- learning_rate: 0.001
- early_stopping_patience: 10
- redundancy_threshold: 0.9
- interpretation_threshold: 0.6

---

## Integration Opportunities

1. **Create HarmonySemanticEncoder**
   - Inherits from SemanticFeatureEncoder
   - Specializes for harmony/chord features
   - Uses HarmonySpecialist for validation

2. **Create RhythmSemanticEncoder**
   - Specializes for rhythm/groove features
   - Uses RhythmSpecialist for validation

3. **Create ModularDiscoveryPipeline**
   - Coordinates multiple domain encoders
   - Combines results into unified parameter set

4. **Extend Parameter Registry**
   - Add modular parameter definitions
   - Link discovered parameters to specialists

---

## File Summary (Absolute Paths)

### Learning Module (25 files)
- `/home/user/Do/midi_generator/learning/semantic_encoder.py`
- `/home/user/Do/midi_generator/learning/semantic_discovery_pipeline.py`
- `/home/user/Do/midi_generator/learning/gap_discovery_trainer.py`
- `/home/user/Do/midi_generator/learning/gap_dataset.py`
- `/home/user/Do/midi_generator/learning/feature_interpreter.py`
- `/home/user/Do/midi_generator/learning/semantic_constraints.py`
- `/home/user/Do/midi_generator/learning/semantic_features.py`
- `/home/user/Do/midi_generator/learning/musical_locality.py`
- [15 other supporting files]

### Parameters Module
- `/home/user/Do/midi_generator/parameters/universal_registry.py`
- `/home/user/Do/midi_generator/parameters/registry.json`
- `/home/user/Do/midi_generator/parameters/registry_with_structure.json`

### Experts Module (6 specialists)
- `/home/user/Do/midi_generator/experts/harmony_specialist.py`
- `/home/user/Do/midi_generator/experts/rhythm_specialist.py`
- `/home/user/Do/midi_generator/experts/melody_specialist.py`
- `/home/user/Do/midi_generator/experts/dynamics_specialist.py`
- `/home/user/Do/midi_generator/experts/texture_specialist.py`
- `/home/user/Do/midi_generator/experts/structure_specialist.py`

### Documentation
- `/home/user/Do/midi_generator/learning/SEMANTIC_DISCOVERY_EXPLORATION.md` [NEW]
- `/home/user/Do/midi_generator/learning/README_AGENT03_SEMANTIC_ENCODER.md`
- `/home/user/Do/midi_generator/learning/AGENT_7_INTEGRATION_GUIDE.md`
- `/home/user/Do/midi_generator/learning/AGENT_05_ARCHITECTURE_REPORT.md`
- `/home/user/Do/midi_generator/learning/AGENT_06_FEATURE_INTERPRETER_README.md`
- `/home/user/Do/midi_generator/learning/GAP_DATASET_README.md`

