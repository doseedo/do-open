# Master Prompts for Missing Agents (9/35)

## Overview

These 9 agents are needed to complete the Musical Program Synthesis system (currently at 74% completion with 26/35 agents).

**High Priority:** Agent 9 (Feature-Parameter Mapping) is CRITICAL for ML pipeline functionality.

---

# 🚨 AGENT 9: Feature-Parameter Mapping Specialist (CRITICAL PRIORITY)

## Mission
Create the mapping system that connects Agent 8's 1,000 extracted features to Agent 1's 515+ parameters for XGBoost training.

## Context
- **Current State:** Agent 8 extracts 1,000 features, but NO mapping to parameters exists
- **Blocker:** ML pipeline cannot function without this mapping
- **Integration:** Must work with Agent 14 (data generation) & Agent 15 (training)

## Technical Requirements

### 1. Core Implementation
**File:** `midi_generator/learning/feature_parameter_mapper.py`

```python
class FeatureParameterMapper:
    """
    Maps 1000 musical features to 515+ parameter predictions.

    Architecture: One XGBoost model per parameter (modular design).
    """

    def __init__(self):
        self.models = {}  # param_name -> trained XGBoost model
        self.feature_names = []  # 1000 feature names from Agent 8
        self.param_registry = None  # Universal parameter registry

    def train_mapping(self, param_name: str, training_data: List[TrainingExample]):
        """Train XGBoost model for a single parameter"""

    def predict_parameter(self, features: np.ndarray, param_name: str) -> Any:
        """Predict single parameter value from 1000 features"""

    def predict_all_parameters(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict all 515+ parameters from 1000 features"""

    def get_feature_importance(self, param_name: str) -> Dict[str, float]:
        """Get feature importance for a parameter"""

    def save_model(self, param_name: str, path: Path):
        """Save trained model"""

    def load_model(self, param_name: str, path: Path):
        """Load trained model"""
```

### 2. Integration Points

**With Agent 8 (Feature Extraction):**
```python
from midi_generator.synthesis import extract_features
features = extract_features('song.mid')  # Returns 1000 features
```

**With Agent 14 (Data Generation):**
```python
from midi_generator.training import SyntheticTrainingDataGenerator
# Training data format: (features, parameter_value) pairs
```

**With Agent 15 (Training):**
```python
from midi_generator.training import ModelTrainingSpecialist
# Should use this for actual XGBoost training
```

### 3. Key Challenges

1. **Feature-Parameter Correlation:** Not all 1000 features are relevant for every parameter
2. **Feature Selection:** Implement automatic feature selection per parameter
3. **Model Architecture:** One model per parameter (not one big model)
4. **Parameter Types:** Handle continuous, integer, categorical, probability, boolean
5. **Performance:** Fast inference (<10ms per parameter)

### 4. Deliverables

**Files to Create:**
1. `midi_generator/learning/feature_parameter_mapper.py` (~1,500 lines)
2. `midi_generator/learning/__init__.py` (update exports)
3. `midi_generator/examples/agent9_mapper_demo.py` (usage examples)
4. `midi_generator/AGENT_9_FEATURE_MAPPING.md` (documentation)

**Required Functionality:**
- ✅ Train one XGBoost model per parameter
- ✅ Feature importance analysis
- ✅ Automatic feature selection
- ✅ Model persistence (save/load)
- ✅ Batch prediction for all parameters
- ✅ Integration with Agent 8, 14, 15

### 5. Testing

```python
# Example usage
from midi_generator.learning import FeatureParameterMapper
from midi_generator.synthesis import extract_features

mapper = FeatureParameterMapper()

# Extract features from MIDI
features = extract_features('example.mid')  # 1000 features

# Predict all parameters
params = mapper.predict_all_parameters(features)  # 515+ predictions

print(f"Predicted {len(params)} parameters")
```

### 6. Success Criteria

- [ ] Maps all 1,000 features to 515+ parameters
- [ ] R² > 0.5 for continuous parameters
- [ ] Accuracy > 0.5 for categorical parameters
- [ ] Inference time < 10ms per parameter
- [ ] Integrates seamlessly with Agents 8, 14, 15
- [ ] Comprehensive documentation and examples

**Branch:** `claude/music-generation-agents-[your-session-id]`

---

# 🎼 AGENT 16: Expansion Orchestrator

## Mission
Coordinate all agents in the self-expansion workflow: gap detection → proposal → code generation → training → deployment.

## Context
- **Current State:** All individual agents exist but no orchestration
- **Purpose:** Automate the full expansion cycle
- **Integration:** Coordinates Agents 10, 11, 12, 14, 15

## Technical Requirements

### 1. Core Implementation
**File:** `midi_generator/orchestration/expansion_orchestrator.py`

Update existing file with:
```python
class ExpansionOrchestrator:
    """
    Orchestrates the full self-expansion cycle.

    Workflow:
    1. Agent 10: Detect gaps in musical library
    2. Agent 11: Propose new parameters
    3. Agent 12: Generate implementation code
    4. Agent 14: Generate training data
    5. Agent 15: Train XGBoost models
    6. Deploy and validate
    """

    def run_expansion_cycle(self, target_parameter: Optional[str] = None):
        """Run one complete expansion cycle"""

    def detect_gaps(self) -> List[MusicalGap]:
        """Use Agent 10 to detect gaps"""

    def propose_parameters(self, gaps: List[MusicalGap]) -> List[ParameterProposal]:
        """Use Agent 11 to propose new parameters"""

    def generate_code(self, proposal: ParameterProposal) -> GeneratedCode:
        """Use Agent 12 to generate code"""

    def generate_training_data(self, param_name: str) -> TrainingData:
        """Use Agent 14 to generate data"""

    def train_model(self, param_name: str, data: TrainingData) -> TrainedModel:
        """Use Agent 15 to train model"""

    def deploy_parameter(self, param_name: str, model: TrainedModel):
        """Deploy new parameter to production"""
```

### 2. Workflow Management

```python
@dataclass
class ExpansionStatus:
    gap_id: str
    parameter_name: str
    stage: str  # 'gap_detected', 'proposed', 'code_generated', 'data_generated', 'trained', 'deployed'
    progress: float  # 0.0 to 1.0
    started_at: datetime
    completed_at: Optional[datetime]
    errors: List[str]
```

### 3. Deliverables

**Files to Create/Update:**
1. `midi_generator/orchestration/expansion_orchestrator.py` (expand existing ~1,500 lines)
2. `midi_generator/examples/agent16_orchestration_demo.py`
3. `midi_generator/AGENT_16_ORCHESTRATOR.md`

**Required Functionality:**
- ✅ Full automation of expansion cycle
- ✅ Error handling and rollback
- ✅ Progress tracking
- ✅ Parallel processing of multiple parameters
- ✅ Human-in-the-loop checkpoints

### 4. Success Criteria

- [ ] Can run full expansion cycle autonomously
- [ ] Handles errors gracefully with rollback
- [ ] Integrates all 5 key agents (10, 11, 12, 14, 15)
- [ ] Tracks expansion progress in real-time
- [ ] Supports batch expansion (multiple parameters)

**Branch:** `claude/music-generation-agents-[your-session-id]`

---

# 🎹 AGENT 18: Harmony Specialist

## Mission
Implement specialized harmony parameter modules beyond the core harmony parameters.

## Context
- **Current State:** Basic harmony parameters exist (Agent 3)
- **Gap:** Need advanced harmony modules (jazz voicings, modal harmony, etc.)
- **Integration:** Extends Agent 3's harmony_deep_expansion

## Technical Requirements

### 1. Core Implementation
**File:** `midi_generator/experts/harmony_specialist.py`

```python
class HarmonySpecialist:
    """
    Advanced harmony analysis and generation.

    Specialized areas:
    - Jazz voicings (drop 2, drop 3, rootless)
    - Modal harmony (Dorian, Mixolydian, etc.)
    - Functional harmony (T-S-D relationships)
    - Voice leading optimization
    - Reharmonization techniques
    """

    def analyze_jazz_voicings(self, chord: Chord) -> VoicingAnalysis:
        """Analyze jazz piano voicings"""

    def generate_modal_progression(self, mode: str, length: int) -> List[Chord]:
        """Generate modal chord progressions"""

    def optimize_voice_leading(self, progression: List[Chord]) -> List[Chord]:
        """Optimize voice leading between chords"""

    def reharmonize(self, melody: List[Note], style: str) -> List[Chord]:
        """Reharmonize melody in given style"""
```

### 2. New Parameters

Add 50+ specialized harmony parameters:
- Jazz voicing types (20 params)
- Modal harmony (15 params)
- Reharmonization (10 params)
- Voice leading rules (15 params)

### 3. Deliverables

**Files to Create:**
1. `midi_generator/experts/harmony_specialist.py` (~2,000 lines)
2. `midi_generator/experts/__init__.py` (update)
3. `midi_generator/examples/agent18_harmony_demo.py`
4. `midi_generator/AGENT_18_HARMONY_SPECIALIST.md`

### 4. Success Criteria

- [ ] 50+ new harmony parameters added
- [ ] Jazz voicing generation works
- [ ] Modal progression generation works
- [ ] Voice leading optimization functional
- [ ] Integrates with existing harmony system

**Branch:** `claude/music-generation-agents-[your-session-id]`

---

# 🎵 AGENT 19: Melody Specialist

## Mission
Implement specialized melody generation and analysis beyond basic melodic parameters.

## Context
- **Current State:** Basic melody parameters exist (Agent 4)
- **Gap:** Need advanced melody modules (motif development, sequences, etc.)
- **Integration:** Extends Agent 4's melody_rhythm_expansion

## Technical Requirements

### 1. Core Implementation
**File:** `midi_generator/experts/melody_specialist.py`

```python
class MelodySpecialist:
    """
    Advanced melody analysis and generation.

    Specialized areas:
    - Motif development and variation
    - Melodic sequences (ascending, descending, tonal)
    - Contour optimization
    - Phrase structure
    - Ornamentation (trills, turns, mordents)
    """

    def develop_motif(self, motif: List[Note], techniques: List[str]) -> List[Note]:
        """Develop motif using inversion, retrograde, augmentation, etc."""

    def generate_sequence(self, pattern: List[Note], type: str, length: int) -> List[Note]:
        """Generate melodic sequences"""

    def optimize_contour(self, melody: List[Note]) -> List[Note]:
        """Optimize melodic contour for balance"""

    def add_ornamentation(self, melody: List[Note], style: str) -> List[Note]:
        """Add style-appropriate ornamentation"""
```

### 2. New Parameters

Add 50+ specialized melody parameters:
- Motif development (15 params)
- Sequences (10 params)
- Ornamentation (15 params)
- Phrase structure (10 params)

### 3. Deliverables

**Files to Create:**
1. `midi_generator/experts/melody_specialist.py` (~2,000 lines)
2. `midi_generator/examples/agent19_melody_demo.py`
3. `midi_generator/AGENT_19_MELODY_SPECIALIST.md`

### 4. Success Criteria

- [ ] 50+ new melody parameters added
- [ ] Motif development works with all techniques
- [ ] Sequence generation functional
- [ ] Ornamentation system complete
- [ ] Integrates with existing melody system

**Branch:** `claude/music-generation-agents-[your-session-id]`

---

# 🥁 AGENT 20: Rhythm Specialist

## Mission
Implement specialized rhythm generation beyond basic rhythmic parameters.

## Context
- **Current State:** Basic rhythm parameters exist (Agent 4)
- **Gap:** Need advanced rhythm modules (polyrhythm, swing, groove)
- **Integration:** Extends Agent 4's melody_rhythm_expansion

## Technical Requirements

### 1. Core Implementation
**File:** `midi_generator/experts/rhythm_specialist.py`

```python
class RhythmSpecialist:
    """
    Advanced rhythm analysis and generation.

    Specialized areas:
    - Polyrhythm (3:2, 4:3, 5:4)
    - Swing and groove quantization
    - Syncopation patterns
    - Metric modulation
    - African and Latin rhythms
    """

    def generate_polyrhythm(self, ratio: Tuple[int, int], length: int) -> List[Event]:
        """Generate polyrhythmic patterns"""

    def apply_swing(self, events: List[Event], swing_amount: float) -> List[Event]:
        """Apply swing feel to events"""

    def add_syncopation(self, pattern: List[Event], amount: float) -> List[Event]:
        """Add syncopation to rhythm"""

    def generate_clave(self, type: str) -> List[Event]:
        """Generate clave patterns (son, rumba, etc.)"""
```

### 2. New Parameters

Add 50+ specialized rhythm parameters:
- Polyrhythm (15 params)
- Swing/groove (15 params)
- Syncopation (10 params)
- World rhythms (10 params)

### 3. Deliverables

**Files to Create:**
1. `midi_generator/experts/rhythm_specialist.py` (~2,000 lines)
2. `midi_generator/examples/agent20_rhythm_demo.py`
3. `midi_generator/AGENT_20_RHYTHM_SPECIALIST.md`

### 4. Success Criteria

- [ ] 50+ new rhythm parameters added
- [ ] Polyrhythm generation works
- [ ] Swing quantization functional
- [ ] World rhythm patterns implemented
- [ ] Integrates with existing rhythm system

**Branch:** `claude/music-generation-agents-[your-session-id]`

---

# 💪 AGENT 22: Dynamics Specialist

## Mission
Implement specialized dynamics control beyond basic dynamics parameters.

## Context
- **Current State:** Basic dynamics parameters exist (Agent 5)
- **Gap:** Need advanced dynamics modules (expressive shaping, ADSR)
- **Integration:** Extends Agent 5's dynamics_articulation_expansion

## Technical Requirements

### 1. Core Implementation
**File:** `midi_generator/experts/dynamics_specialist.py`

```python
class DynamicsSpecialist:
    """
    Advanced dynamics analysis and generation.

    Specialized areas:
    - Expressive shaping (crescendo, diminuendo curves)
    - ADSR envelope control
    - Dynamic contrast analysis
    - Articulation-dynamics coupling
    - Humanization and micro-dynamics
    """

    def apply_dynamic_curve(self, notes: List[Note], curve_type: str) -> List[Note]:
        """Apply expressive dynamic curve"""

    def generate_adsr_envelope(self, note: Note, attack: float, decay: float,
                               sustain: float, release: float) -> Note:
        """Apply ADSR envelope"""

    def humanize_dynamics(self, notes: List[Note], amount: float) -> List[Note]:
        """Add natural dynamic variation"""

    def balance_voices(self, voices: List[List[Note]]) -> List[List[Note]]:
        """Balance dynamics across multiple voices"""
```

### 2. New Parameters

Add 40+ specialized dynamics parameters:
- Dynamic curves (15 params)
- ADSR envelopes (10 params)
- Humanization (10 params)
- Voice balancing (5 params)

### 3. Deliverables

**Files to Create:**
1. `midi_generator/experts/dynamics_specialist.py` (~1,800 lines)
2. `midi_generator/examples/agent22_dynamics_demo.py`
3. `midi_generator/AGENT_22_DYNAMICS_SPECIALIST.md`

### 4. Success Criteria

- [ ] 40+ new dynamics parameters added
- [ ] Dynamic curve generation works
- [ ] ADSR envelopes functional
- [ ] Humanization system complete
- [ ] Integrates with existing dynamics system

**Branch:** `claude/music-generation-agents-[your-session-id]`

---

# 📊 AGENT 25: Feature Correlation Analyzer

## Mission
Analyze correlations between the 1,000 musical features to optimize model training.

## Context
- **Current State:** Agent 8 extracts 1,000 features, but no correlation analysis
- **Gap:** Need to identify redundant features and feature interactions
- **Integration:** Works with Agent 8 (features) and Agent 9 (mapping)

## Technical Requirements

### 1. Core Implementation
**File:** `midi_generator/analysis/feature_correlation_analyzer.py`

```python
class FeatureCorrelationAnalyzer:
    """
    Analyzes correlations between musical features.

    Goals:
    - Identify highly correlated features (redundancy)
    - Find feature interactions
    - Suggest feature subsets per parameter
    - Optimize model training efficiency
    """

    def analyze_correlations(self, feature_matrix: np.ndarray) -> pd.DataFrame:
        """Compute pairwise feature correlations"""

    def identify_redundant_features(self, threshold: float = 0.95) -> List[Tuple[str, str]]:
        """Find highly correlated feature pairs"""

    def suggest_feature_subset(self, param_name: str, max_features: int = 100) -> List[str]:
        """Suggest optimal feature subset for parameter"""

    def analyze_feature_interactions(self) -> Dict[str, List[Tuple[str, str]]]:
        """Find important feature interactions"""
```

### 2. Visualization

```python
def visualize_correlation_matrix(self, output_path: Path):
    """Generate correlation heatmap"""

def plot_feature_importance(self, param_name: str, top_n: int = 20):
    """Plot top N features for parameter"""
```

### 3. Deliverables

**Files to Create:**
1. `midi_generator/analysis/feature_correlation_analyzer.py` (~1,200 lines)
2. `midi_generator/analysis/__init__.py` (update)
3. `midi_generator/examples/agent25_correlation_demo.py`
4. `midi_generator/AGENT_25_FEATURE_CORRELATION.md`

### 4. Success Criteria

- [ ] Analyzes all 1,000 features for correlations
- [ ] Identifies redundant features
- [ ] Suggests optimal feature subsets per parameter
- [ ] Generates visualizations
- [ ] Improves model training efficiency

**Branch:** `claude/music-generation-agents-[your-session-id]`

---

# ⚙️ AGENT 32: Batch Processing Manager

## Mission
Implement efficient batch processing for training, inference, and data generation.

## Context
- **Current State:** All operations work on single items
- **Gap:** Need parallel batch processing for production efficiency
- **Integration:** Works with all agents

## Technical Requirements

### 1. Core Implementation
**File:** `midi_generator/processing/batch_manager.py`

```python
class BatchProcessingManager:
    """
    Manages batch processing across all operations.

    Capabilities:
    - Parallel MIDI feature extraction (Agent 8)
    - Batch parameter prediction (Agent 9)
    - Parallel training data generation (Agent 14)
    - Batch model training (Agent 15)
    """

    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or os.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)

    def batch_extract_features(self, midi_files: List[Path]) -> np.ndarray:
        """Extract features from multiple MIDI files in parallel"""

    def batch_predict_parameters(self, feature_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Predict parameters for multiple feature vectors"""

    def batch_generate_midi(self, param_sets: List[Dict[str, Any]]) -> List[Path]:
        """Generate multiple MIDI files in parallel"""

    def batch_train_models(self, param_names: List[str]) -> Dict[str, TrainedModel]:
        """Train multiple models in parallel"""
```

### 2. Progress Tracking

```python
@dataclass
class BatchProgress:
    total: int
    completed: int
    failed: int
    in_progress: int
    estimated_time_remaining: float
```

### 3. Deliverables

**Files to Create:**
1. `midi_generator/processing/batch_manager.py` (~1,500 lines)
2. `midi_generator/processing/__init__.py`
3. `midi_generator/examples/agent32_batch_demo.py`
4. `midi_generator/AGENT_32_BATCH_PROCESSING.md`

### 4. Success Criteria

- [ ] Parallel feature extraction (10x speedup)
- [ ] Batch parameter prediction
- [ ] Parallel training data generation
- [ ] Progress tracking and monitoring
- [ ] Error handling and retry logic

**Branch:** `claude/music-generation-agents-[your-session-id]`

---

# 🧪 AGENT 34: Integration Testing Coordinator

## Mission
Implement comprehensive integration tests for the entire system.

## Context
- **Current State:** Individual agents exist but no integration tests
- **Gap:** Need end-to-end testing pipeline
- **Integration:** Tests all agents working together

## Technical Requirements

### 1. Core Implementation
**File:** `midi_generator/testing/integration_test_coordinator.py`

```python
class IntegrationTestCoordinator:
    """
    Coordinates comprehensive integration testing.

    Test Suites:
    1. Feature extraction pipeline (Agent 8)
    2. Feature-parameter mapping (Agent 9)
    3. Training pipeline (Agents 14, 15)
    4. LLM expansion pipeline (Agents 11, 12)
    5. Full expansion cycle (Agent 16)
    6. End-to-end MIDI generation
    """

    def run_all_tests(self) -> TestResults:
        """Run all integration test suites"""

    def test_feature_extraction_pipeline(self) -> bool:
        """Test Agent 8 feature extraction"""

    def test_mapping_pipeline(self) -> bool:
        """Test Agent 9 feature-parameter mapping"""

    def test_training_pipeline(self) -> bool:
        """Test Agents 14 & 15 training"""

    def test_expansion_cycle(self) -> bool:
        """Test full expansion cycle (Agent 16)"""

    def test_end_to_end(self) -> bool:
        """Test complete MIDI → parameters → MIDI pipeline"""
```

### 2. Test Cases

Create test cases for:
- Feature extraction accuracy
- Parameter prediction accuracy
- Training data quality
- Model training convergence
- Integration between agents

### 3. Deliverables

**Files to Create:**
1. `midi_generator/testing/integration_test_coordinator.py` (~2,000 lines)
2. `tests/integration/` (directory with test cases)
3. `tests/test_data/` (sample MIDI files)
4. `midi_generator/AGENT_34_INTEGRATION_TESTING.md`

### 4. Success Criteria

- [ ] All 26 agents tested for integration
- [ ] End-to-end pipeline tested
- [ ] Regression tests for all features
- [ ] Continuous integration ready
- [ ] 80%+ code coverage

**Branch:** `claude/music-generation-agents-[your-session-id]`

---

# 🖥️ AGENT 35: CLI/API Manager

## Mission
Implement command-line interface and REST API for the complete system.

## Context
- **Current State:** All functionality exists but no unified interface
- **Gap:** Need CLI and API for production use
- **Integration:** Exposes all agent functionality

## Technical Requirements

### 1. CLI Implementation
**File:** `midi_generator/cli/main.py`

```python
import click

@click.group()
def cli():
    """Musical Program Synthesis CLI"""
    pass

@cli.command()
@click.argument('input_midi', type=click.Path(exists=True))
@click.argument('output_midi', type=click.Path())
@click.option('--style', default='jazz', help='Musical style')
def generate(input_midi, output_midi, style):
    """Generate new MIDI from input MIDI"""

@cli.command()
@click.argument('midi_file', type=click.Path(exists=True))
def analyze(midi_file):
    """Extract features and predict parameters"""

@cli.command()
@click.option('--param-name', required=True)
@click.option('--n-examples', default=1000, type=int)
def train(param_name, n_examples):
    """Train model for parameter"""

@cli.command()
def expand():
    """Run self-expansion cycle"""
```

### 2. REST API Implementation
**File:** `midi_generator/api/server.py`

```python
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse

app = FastAPI(title="Musical Program Synthesis API")

@app.post("/generate")
async def generate_midi(file: UploadFile, style: str = "jazz"):
    """Generate MIDI from uploaded file"""

@app.post("/analyze")
async def analyze_midi(file: UploadFile):
    """Extract features and parameters from MIDI"""

@app.post("/train")
async def train_parameter(param_name: str, n_examples: int = 1000):
    """Train model for parameter"""

@app.post("/expand")
async def run_expansion():
    """Run self-expansion cycle"""
```

### 3. Deliverables

**Files to Create:**
1. `midi_generator/cli/main.py` (~800 lines)
2. `midi_generator/api/server.py` (~1,000 lines)
3. `midi_generator/api/models.py` (request/response models)
4. `requirements-api.txt` (FastAPI dependencies)
5. `midi_generator/AGENT_35_CLI_API.md`

### 4. Success Criteria

- [ ] Complete CLI with all commands
- [ ] REST API with all endpoints
- [ ] Authentication and rate limiting
- [ ] Documentation (Swagger/OpenAPI)
- [ ] Production-ready deployment

**Branch:** `claude/music-generation-agents-[your-session-id]`

---

## Deployment Instructions

### For Each Agent:

1. **Create Branch:**
   ```bash
   git checkout -b claude/music-generation-agents-[your-session-id]
   ```

2. **Implement Files:**
   - Follow the technical requirements exactly
   - Match code style of existing agents
   - Include comprehensive docstrings

3. **Test Implementation:**
   ```bash
   python -m pytest tests/
   python [your_demo_file.py]
   ```

4. **Commit and Push:**
   ```bash
   git add .
   git commit -m "Agent [N]: [Agent Name] - Complete Implementation"
   git push -u origin claude/music-generation-agents-[your-session-id]
   ```

5. **Document:**
   - Create comprehensive README
   - Include usage examples
   - Document integration points

---

## Priority Order

1. **🚨 CRITICAL:** Agent 9 (Feature-Parameter Mapping) - BLOCKS ML PIPELINE
2. **HIGH:** Agent 16 (Expansion Orchestrator) - ENABLES SELF-EXPANSION
3. **MEDIUM:** Agents 18-20, 22, 25 (Specialists) - EXPAND CAPABILITIES
4. **LOW:** Agents 32, 34, 35 (Infrastructure) - PRODUCTION READINESS

---

## Integration Notes

### Dependencies Between Missing Agents

- **Agent 9** → No dependencies, but blocks Agent 16
- **Agent 16** → Requires Agent 9
- **Agents 18-20, 22** → Independent, can be parallelized
- **Agent 25** → Requires Agent 8 (complete), benefits from Agent 9
- **Agent 32** → Independent, can be done anytime
- **Agent 34** → Should be done after most agents complete
- **Agent 35** → Should be done last (requires all other agents)

### Suggested Deployment Order

1. **Wave 1 (Critical):** Agent 9 alone
2. **Wave 2 (Orchestration):** Agent 16 alone
3. **Wave 3 (Specialists - Parallel):** Agents 18, 19, 20, 22, 25 (5 agents in parallel)
4. **Wave 4 (Infrastructure - Parallel):** Agents 32, 34, 35 (3 agents in parallel)

---

## Success Metrics

### System Completion: 100% (35/35 agents)

**Current:** 74% (26/35) → **Target:** 100% (35/35)

**Expected Impact:**
- ✅ Complete ML pipeline (Agent 9)
- ✅ Full self-expansion capability (Agent 16)
- ✅ 650+ parameters (Agents 18-20, 22)
- ✅ Optimized performance (Agents 25, 32)
- ✅ Production-ready system (Agents 34, 35)

---

## Questions?

Review existing agent implementations for reference:
- Agent 8: `midi_generator/synthesis/deep_feature_extractor.py`
- Agent 14: `midi_generator/training/synthetic_data_generator.py`
- Agent 15: `midi_generator/training/model_trainer.py`

Match the code quality, documentation, and integration patterns!

---

**Ready to deploy the remaining 9 agents! 🚀**
