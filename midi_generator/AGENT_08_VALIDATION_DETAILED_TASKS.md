# Agent 08: Validation Framework Builder - Detailed Task Breakdown

**Mission:** Create comprehensive testing and validation suite for the Dø MIDI Generator v2.0 training readiness system.

**Target Output:** ~7,000 lines of code

**Dependencies:**
- Agent 05: Hierarchical MTL Architecture (model to validate)
- Agent 06: Training Pipeline (training outputs to validate)

**Key Deliverables:**
1. Per-Parameter Validation Metrics (`validation/parameter_validators.py`)
2. Musical Quality Validators (`validation/musical_quality.py`)
3. Genre-Specific Validation Suite (`validation/genre_validators.py`)
4. Cross-Genre Generalization Tests (`validation/generalization_tests.py`)
5. Regression Test Framework (`validation/regression_framework.py`)
6. Human Evaluation Framework (`validation/human_evaluation.py`)
7. Comprehensive Test Suite (`tests/test_validation_framework.py`)

---

## Detailed Task List (40 tasks)

### Phase 1: Foundation & Infrastructure (Tasks 1-10)

#### 1. ☐ **Design validation framework architecture**
   - Define validation pipeline: Data → Model → Predictions → Validation
   - Create base classes: `ParameterValidator`, `MusicalValidator`, `GenreValidator`
   - Define validation result data structures
   - Integration points with Agent 05 (MTL model) and Agent 06 (training pipeline)

#### 2. ☐ **Create validation result data structures**
   ```python
   @dataclass
   class ParameterValidationResult:
       parameter_name: str
       predicted_value: float
       ground_truth: float
       error: float
       error_percentage: float
       passed: bool  # Within acceptable threshold

   @dataclass
   class MusicalValidationResult:
       validation_type: str  # 'intervals', 'harmony', 'rhythm'
       passed: bool
       score: float  # 0-1
       violations: List[str]
       metrics: Dict[str, float]
   ```

#### 3. ☐ **Set up validation metrics registry**
   - Define metrics for each of 50 hierarchical parameters
   - Level 1 (8 params): genre accuracy, tempo MAE, key detection F1
   - Level 2 (20 params): harmony R², melody contour accuracy, rhythm precision
   - Level 3 (22 params): genre-specific metric thresholds
   - Create `validation_metrics.json` with thresholds

#### 4. ☐ **Implement base ParameterValidator class**
   ```python
   class ParameterValidator:
       def validate_prediction(self, predicted, ground_truth, param_spec):
           # Calculate error metrics
           # Check against thresholds
           # Return validation result

       def validate_distribution(self, predictions, ground_truths):
           # Statistical distribution checks
           # KS test, chi-squared test

       def validate_range(self, value, param_spec):
           # Check value is within valid range
   ```

#### 5. ☐ **Create validation utilities module**
   - `utils/validation_utils.py`:
     - MIDI comparison functions
     - Statistical test helpers (KS test, t-test, chi-squared)
     - Musical distance metrics
     - Error aggregation functions

#### 6. ☐ **Design validation pipeline workflow**
   ```
   Input MIDI → Feature Extraction → Model Prediction →
   Parameter Validation → Musical Quality Validation →
   Genre Validation → Report Generation
   ```
   - Create `ValidationPipeline` class
   - Support batch validation
   - Parallel processing for large datasets

#### 7. ☐ **Set up test data fixtures**
   - Select 100 MIDI files from labeled dataset for validation
   - 20 files per genre (jazz, classical, rock, electronic, pop)
   - Include edge cases (very fast/slow tempo, unusual keys, complex harmony)
   - Create `validation_fixtures/` directory

#### 8. ☐ **Implement validation configuration system**
   ```yaml
   # validation_config.yaml
   parameter_thresholds:
     tempo.bpm:
       mae_threshold: 5.0  # BPM
       percentage_threshold: 0.05
     harmony.chord_density:
       mae_threshold: 0.3
     # ... for all 50 parameters

   musical_quality_thresholds:
     interval_validity: 0.95
     voice_leading_score: 0.80
     rhythm_consistency: 0.90
   ```

#### 9. ☐ **Create validation result storage**
   - Design JSON schema for validation results
   - Implement `ValidationResultsStore` class
   - Support versioning (compare v1 vs v2 model results)
   - Time-series storage for regression detection

#### 10. ☐ **Build validation logging system**
   - Structured logging for all validations
   - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
   - Integration with existing monitoring (Agent 31)
   - Validation event tracking

---

### Phase 2: Per-Parameter Validation (Tasks 11-18)

#### 11. ☐ **Implement Level 1 parameter validators**
   - **genre.primary**: Multi-class classification accuracy, F1 score per genre
   - **tempo.bpm**: MAE, RMSE, percentage error (target: MAE < 5 BPM)
   - **time_signature**: Exact match accuracy, confusion matrix
   - **key.tonic**: Pitch class accuracy (12-class)
   - **key.mode**: Binary accuracy (major/minor)
   - **energy.level**: MAE, correlation with ground truth
   - **complexity.overall**: MAE, Spearman correlation
   - **structure.form**: Categorical accuracy

#### 12. ☐ **Implement Level 2 Harmony validators**
   - **harmony.chord_density**: MAE, distribution similarity (KS test)
   - **harmony.complexity**: Correlation with chord extension usage
   - **harmony.chromaticism**: MAE, chromatic note ratio validation
   - **harmony.tension**: Subjective metric correlation
   - **harmony.voicing_spread**: MAE in semitones
   - **harmony.progression_predictability**: Entropy comparison

#### 13. ☐ **Implement Level 2 Melody validators**
   - **melody.note_density**: Notes per measure MAE
   - **melody.range_semitones**: Exact semitone range comparison
   - **melody.contour_smoothness**: Correlation, leap analysis
   - **melody.rhythmic_complexity**: Entropy/variance comparison
   - **melody.repetition**: Motif detection accuracy

#### 14. ☐ **Implement Level 2 Rhythm validators**
   - **rhythm.subdivision**: Categorical accuracy (8th, 16th, triplet)
   - **rhythm.syncopation**: Syncopation ratio MAE
   - **rhythm.groove_consistency**: Timing deviation comparison
   - **rhythm.polyrhythm**: Polyrhythm detection accuracy
   - **rhythm.swing_amount**: Swing ratio MAE

#### 15. ☐ **Implement Level 2 Dynamics/Texture validators**
   - **dynamics.overall_level**: Velocity MAE
   - **dynamics.range**: Velocity std dev comparison
   - **texture.polyphony**: Max simultaneous notes accuracy
   - **texture.density**: Notes per second MAE

#### 16. ☐ **Implement Level 3 genre-specific validators**
   - **Jazz validators**: swing_feel, walking_bass, improvisation_ratio, bebop_vocabulary
   - **Classical validators**: counterpoint, development_density, voice_leading_quality
   - **Rock validators**: power_chord_ratio, riff_repetition, distortion_level
   - **Electronic validators**: quantization, filter_movement, arpeggio_density
   - **Hip-Hop validators**: sample_based, boom_bap_feel
   - **Latin validators**: clave_pattern, montuno_complexity

#### 17. ☐ **Create parameter validation test suite**
   ```python
   class TestParameterValidation(unittest.TestCase):
       def test_tempo_prediction_accuracy(self):
           # Test tempo MAE < 5 BPM on validation set

       def test_key_detection_f1_score(self):
           # Test key detection F1 > 0.85

       def test_harmony_chord_density_correlation(self):
           # Test correlation > 0.80

       # ... for all 50 parameters
   ```

#### 18. ☐ **Build parameter validation dashboard**
   - Per-parameter accuracy visualization
   - Error distribution plots
   - Confusion matrices for categorical params
   - Regression plots for continuous params
   - Integration with Agent 31 quality dashboard

---

### Phase 3: Musical Quality Validation (Tasks 19-26)

#### 19. ☐ **Implement interval validity checker**
   ```python
   class IntervalValidator:
       def validate_melodic_intervals(self, generated_midi, reference_midi):
           # Extract intervals from both
           # Compare interval distributions
           # Flag unnatural leaps (>12 semitones)
           # Calculate similarity score

       def validate_interval_distribution(self, intervals, genre):
           # Genre-specific interval expectations
           # Bebop: favors m2, M2, m3, M3
           # Classical: favors stepwise motion
   ```
   - Target: >95% valid intervals
   - Genre-specific interval norms

#### 20. ☐ **Implement harmony correctness validator**
   ```python
   class HarmonyValidator:
       def validate_chord_progressions(self, chords, genre, key):
           # Check functional harmony (V→I, ii→V→I)
           # Validate chord types for genre
           # Check voice leading between chords
           # Flag awkward progressions

       def validate_voice_leading(self, chord_sequence):
           # Parallel 5ths/octaves (warn vs error by genre)
           # Voice crossing frequency
           # Average voice movement
   ```
   - Use existing `midi_generator/validation/musical_validator.py` as reference
   - Extend with model-specific checks

#### 21. ☐ **Implement rhythm consistency validator**
   ```python
   class RhythmValidator:
       def validate_rhythm_consistency(self, generated_midi):
           # Check for consistent subdivision
           # Detect rhythm pattern repetition
           # Validate swing ratio consistency
           # Flag timing anomalies

       def validate_groove(self, midi, expected_groove):
           # Microtiming analysis
           # Swing ratio accuracy
           # Syncopation pattern matching
   ```

#### 22. ☐ **Create voice range validator**
   - Check all generated notes are within instrument ranges
   - Piano: A0 (21) to C8 (108)
   - Guitar: E2 (40) to E6 (88)
   - Saxophone: Bb3 (58) to F#6 (90)
   - Validate by instrument type in multi-track MIDI

#### 23. ☐ **Implement musical coherence checker**
   ```python
   def validate_musical_coherence(generated_midi, parameters):
       # Check parameter consistency with output
       # If tempo=180, check actual note timing
       # If key=C major, check pitch class distribution
       # If harmony.complexity=high, check chord extensions
   ```

#### 24. ☐ **Create comparison metrics vs. reference**
   - **MIDI-level comparison**:
     - Note accuracy: % of notes in correct position (±10ms)
     - Rhythm accuracy: IOI (inter-onset interval) correlation
     - Pitch accuracy: Pitch class distribution similarity
   - **Feature-level comparison**:
     - Parameter recovery rate: Can we extract same params from generated MIDI?
   - Use existing `midi_generator/analysis/midi_analyzer.py`

#### 25. ☐ **Build musical quality test suite**
   ```python
   class TestMusicalQuality(unittest.TestCase):
       def test_no_extreme_leaps(self):
           # Generated melody should have <5% leaps >12 semitones

       def test_harmony_functional(self):
           # Bebop: >80% functional progressions

       def test_rhythm_consistency(self):
           # Swing ratio std dev < 0.1

       def test_voice_ranges(self):
           # All notes within instrument ranges
   ```

#### 26. ☐ **Create musical quality scoring system**
   ```python
   def calculate_musical_quality_score(midi, parameters):
       scores = {
           'interval_validity': validate_intervals(),
           'harmony_correctness': validate_harmony(),
           'rhythm_consistency': validate_rhythm(),
           'voice_ranges': validate_ranges(),
           'coherence': validate_coherence()
       }

       # Weighted average
       overall = (
           scores['interval_validity'] * 0.25 +
           scores['harmony_correctness'] * 0.30 +
           scores['rhythm_consistency'] * 0.25 +
           scores['voice_ranges'] * 0.10 +
           scores['coherence'] * 0.10
       )

       return overall, scores
   ```
   - Target: Overall score > 0.85 for production

---

### Phase 4: Genre-Specific Validation (Tasks 27-32)

#### 27. ☐ **Create GenreValidator base class**
   ```python
   class GenreValidator:
       def __init__(self, genre_name, genre_characteristics):
           self.genre = genre_name
           self.characteristics = genre_characteristics

       def validate_style_adherence(self, midi, parameters):
           # Genre-specific checks
           pass

       def calculate_authenticity_score(self, midi):
           # Compare to genre reference statistics
           pass
   ```

#### 28. ☐ **Implement Jazz validator**
   - **Swing feel validation**:
     - Check swing ratio (2:1 to 3:1 for bebop)
     - Validate triplet feel vs straight 8ths
   - **Walking bass validation**:
     - Detect quarter note bass pattern
     - Check chromatic approaches
   - **Bebop vocabulary**:
     - Detect bebop scales usage
     - Enclosures, chromatic approaches
   - **Improvisation patterns**:
     - Motif development
     - Call-response phrases

#### 29. ☐ **Implement Classical validator**
   - **Counterpoint validation**:
     - Voice independence
     - Parallel motion limits
     - Contrary motion frequency
   - **Voice leading quality**:
     - Use existing voice_leading_quality metrics
     - Minimal voice movement
   - **Development techniques**:
     - Thematic transformation detection
     - Sequence usage

#### 30. ☐ **Implement Rock/Metal validator**
   - **Power chord detection**:
     - Root + 5th interval patterns
     - Distortion articulation markers
   - **Riff repetition**:
     - Pattern repetition frequency
     - Riff length (typically 1-2 bars)
   - **Guitar techniques**:
     - Bends, slides, hammer-ons (in MIDI articulation)

#### 31. ☐ **Implement Electronic/Hip-Hop validators**
   - **Quantization strictness**:
     - Grid alignment (16th/32nd note)
     - Timing deviation < 5ms
   - **Loop-based structure**:
     - Detect 4/8/16 bar loops
     - Sample repetition
   - **Drum pattern validation**:
     - Boom-bap (kick-snare-kick-snare)
     - Hi-hat patterns

#### 32. ☐ **Create genre validation test suite**
   ```python
   class TestGenreValidation(unittest.TestCase):
       def test_bebop_swing_ratio(self):
           # Swing ratio 2.5:1 ± 0.3

       def test_classical_voice_leading(self):
           # Voice leading score > 0.85

       def test_rock_power_chord_usage(self):
           # If power_chord_ratio > 0.7, detect >70% power chords

       def test_electronic_quantization(self):
           # Timing deviation < 5ms
   ```

---

### Phase 5: Cross-Genre Generalization Testing (Tasks 33-36)

#### 33. ☐ **Design cross-genre test methodology**
   - **Train on 4 genres, test on 5th**:
     - Train: Jazz, Classical, Rock, Pop → Test: Electronic
     - Rotate for all genres
   - **Measure generalization**:
     - Parameter prediction accuracy drop
     - Musical quality score drop
     - Genre misclassification rate

#### 34. ☐ **Implement cross-genre validation pipeline**
   ```python
   class CrossGenreValidator:
       def train_without_genre(self, excluded_genre):
           # Train model on 4/5 genres

       def test_on_held_out_genre(self, held_out_genre):
           # Test on excluded genre
           # Measure accuracy degradation

       def calculate_generalization_score(self):
           # Compare to in-distribution performance
   ```

#### 35. ☐ **Test genre boundary cases**
   - Fusion genres: jazz-rock, classical-electronic
   - Edge cases: very fast metal (>200 BPM), very slow ambient (<60 BPM)
   - Unusual parameters: 7/8 time, atonal music, microtonal

#### 36. ☐ **Create generalization test suite**
   ```python
   class TestGeneralization(unittest.TestCase):
       def test_cross_genre_accuracy_drop(self):
           # Accuracy drop < 15% on held-out genre

       def test_genre_boundary_cases(self):
           # Fusion genres handled gracefully

       def test_parameter_transfer(self):
           # Universal params (Level 2) transfer well
   ```

---

### Phase 6: Regression Testing Framework (Tasks 37-38)

#### 37. ☐ **Create regression test suite**
   ```python
   class RegressionTestFramework:
       def __init__(self, baseline_model_path, baseline_results_path):
           # Load baseline model and results

       def run_regression_tests(self, new_model):
           # Test new model on same validation set
           # Compare results to baseline
           # Flag degradations

       def detect_performance_regression(self):
           # Per-parameter regression detection
           # Musical quality regression
           # Genre accuracy regression
   ```
   - Store baseline results in `validation/baselines/`
   - Version control validation results

#### 38. ☐ **Implement continuous validation**
   - Integration with training pipeline (Agent 06)
   - Automatic validation after each training epoch
   - Alert on regression (>5% accuracy drop)
   - CI/CD integration

---

### Phase 7: Human Evaluation Framework (Tasks 39-40)

#### 39. ☐ **Design human evaluation protocol**
   ```python
   class HumanEvaluationFramework:
       def generate_listening_test(self, generated_midis, reference_midis):
           # Create A/B comparison pairs
           # Randomize order
           # Generate evaluation form

       def create_evaluation_criteria(self):
           # Musicality (1-5)
           # Style accuracy (1-5)
           # Technical quality (1-5)
           # Overall preference
   ```
   - Based on existing `tests/listening_test_framework.py`
   - Adapt for parameter learning validation

#### 40. ☐ **Build evaluation analysis tools**
   ```python
   def analyze_human_evaluations(responses):
       # Calculate inter-rater agreement (Krippendorff's alpha)
       # Average scores per model
       # Correlation with automatic metrics
       # Identify failure modes
   ```
   - Export results to CSV/JSON
   - Generate human evaluation report

---

## Integration Requirements

### With Agent 05 (Hierarchical MTL Architecture)
```python
# Load trained model
model = HierarchicalMTLModel.load('models/hierarchical_mtl_v1.pth')

# Validate predictions
validator = ValidationPipeline(config='validation_config.yaml')
results = validator.validate_model(model, validation_dataset)
```

### With Agent 06 (Training Pipeline)
```python
# Hook into training loop
def on_epoch_end(epoch, model):
    validation_results = validator.validate_model(model, val_set)
    if validation_results.overall_score < 0.85:
        logger.warning(f"Epoch {epoch}: Validation score below threshold")
```

### With Agent 31 (Quality Metrics Dashboard)
```python
# Report validation metrics to dashboard
dashboard.update_validation_metrics(
    model_version='v1.0',
    validation_score=results.overall_score,
    per_parameter_scores=results.parameter_scores,
    musical_quality_score=results.musical_quality_score
)
```

---

## Success Criteria

✅ **Per-Parameter Validation:**
- All 50 parameters have defined validation metrics
- Automated validation pipeline functional
- Target: >85% parameters meet accuracy thresholds

✅ **Musical Quality Validation:**
- Interval validity >95%
- Harmony correctness >80%
- Rhythm consistency >90%
- Overall musical quality score >0.85

✅ **Genre-Specific Validation:**
- Each genre has custom validation functions
- Genre-specific characteristics validated
- Authenticity scores >0.85 per genre

✅ **Cross-Genre Generalization:**
- Accuracy drop <15% on held-out genres
- Universal parameters transfer well

✅ **Regression Testing:**
- Baseline results stored and versioned
- Automatic regression detection
- No regressions in production releases

✅ **Test Coverage:**
- >90% code coverage
- All validation functions unit tested
- Integration tests with mock models

---

## File Structure

```
midi_generator/
├── validation/
│   ├── __init__.py
│   ├── parameter_validators.py       # Tasks 11-18 (800 lines)
│   ├── musical_quality.py            # Tasks 19-26 (1200 lines)
│   ├── genre_validators.py           # Tasks 27-32 (1000 lines)
│   ├── generalization_tests.py       # Tasks 33-36 (600 lines)
│   ├── regression_framework.py       # Tasks 37-38 (500 lines)
│   ├── human_evaluation.py           # Tasks 39-40 (400 lines)
│   ├── validation_pipeline.py        # Tasks 1-10 (800 lines)
│   ├── validation_utils.py           # Task 5 (400 lines)
│   └── validation_config.yaml        # Task 8
├── tests/
│   ├── test_parameter_validation.py  # Task 17 (600 lines)
│   ├── test_musical_quality.py       # Task 25 (500 lines)
│   ├── test_genre_validation.py      # Task 32 (400 lines)
│   └── test_generalization.py        # Task 36 (300 lines)
└── validation_fixtures/              # Task 7
    ├── jazz/ (20 files)
    ├── classical/ (20 files)
    ├── rock/ (20 files)
    ├── electronic/ (20 files)
    └── pop/ (20 files)

Total: ~7,500 lines
```

---

## Timeline Estimate

**Phase 1 (Foundation):** 2-3 days
**Phase 2 (Per-Parameter):** 3-4 days
**Phase 3 (Musical Quality):** 3-4 days
**Phase 4 (Genre-Specific):** 2-3 days
**Phase 5 (Generalization):** 2 days
**Phase 6 (Regression):** 1-2 days
**Phase 7 (Human Eval):** 1-2 days

**Total:** 14-20 days (with Agent 05/06 dependencies met)

---

## Dependencies on Other Agents

**Requires completion of:**
- ✅ Agent 01: Hierarchical parameters defined
- ✅ Agent 02: MIDI corpus acquired
- ✅ Agent 03: Dataset labeled
- ⏳ Agent 05: MTL model architecture (BLOCKING)
- ⏳ Agent 06: Training pipeline (BLOCKING)

**Can be done in parallel:**
- Tasks 1-10, 19-26 can start before Agent 05/06 complete
- Framework design and musical quality validators are model-agnostic
- Tasks 11-18, 27-40 require trained model

---

**Agent 08 Status:** READY TO START (Foundation & Musical Quality)
**Blocked Tasks:** Parameter validation (need trained model from Agent 05/06)
**Current Focus:** Build framework and model-agnostic validators
