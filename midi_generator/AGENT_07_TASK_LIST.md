# Agent 07: Multi-Genre Data Specialist - Detailed Task List

**Mission:** Handle genre-specific data requirements and balancing for multi-genre MIDI corpus training.

**System:** Dø MIDI Generator v2.0 (159,683 LOC)
**Estimated LOC:** ~6,000 lines
**Timeline:** Week 4 (depends on Agent 03 & 05)
**Date:** November 20, 2025

---

## Deliverables

1. **Genre Stratification Framework** (`genre_stratifier.py`)
2. **Multi-Genre Data Augmentation Suite** (`genre_augmentation.py`)
3. **Genre Balance Handler** (`genre_balancer.py`)
4. **Cross-Genre Transfer System** (`cross_genre_transfer.py`)
5. **Genre-Specific Validation Toolkit** (`genre_validation.py`)
6. **Multi-Genre Strategy Documentation** (`MULTI_GENRE_STRATEGY.md`)

---

## Detailed Task List (45 tasks)

### Phase 1: Analysis & Strategy Design (Tasks 1-12)

#### 1. ☐ **Analyze genre distribution requirements**
   - Target distribution: Jazz 150, Classical 200, Rock 100, Electronic 120, Pop 180
   - Calculate proportions: Jazz 20%, Classical 26.7%, Rock 13.3%, Electronic 16%, Pop 24%
   - Identify potential imbalance issues
   - Design target distribution for training

#### 2. ☐ **Design genre stratification strategy**
   - Stratified sampling approach
   - Ensure proportional representation in train/val/test
   - Handle subgenre diversity within each genre
   - Preserve parameter distribution characteristics per genre

#### 3. ☐ **Research MIDI-specific data augmentation**
   - Pitch transposition (key changes)
   - Tempo scaling (time stretching)
   - Velocity perturbation
   - Timing jitter (humanization)
   - Voice permutation
   - Harmonic substitution
   - Rhythmic variation

#### 4. ☐ **Design genre-specific augmentation rules**
   - **Jazz:**
     - Allow wide pitch transposition (-5 to +7 semitones)
     - Moderate tempo variation (0.85x to 1.15x)
     - High velocity variation (swing dynamics)
     - Timing jitter for swing feel
     - Chord substitution (tritone subs, extensions)

   - **Classical:**
     - Conservative pitch transposition (-3 to +3)
     - Minimal tempo variation (0.95x to 1.05x)
     - Preserve dynamic contours
     - Minimal timing jitter
     - Preserve voice leading

   - **Rock:**
     - Moderate transposition (-3 to +5)
     - Moderate tempo (0.9x to 1.1x)
     - Power chord preservation
     - Riff repetition intact

   - **Electronic:**
     - Free transposition (-7 to +7)
     - Wide tempo variation (0.8x to 1.2x)
     - Quantization preservation
     - Pattern repetition intact

   - **Pop:**
     - Moderate transposition (-4 to +4)
     - Moderate tempo (0.9x to 1.1x)
     - Preserve hook/chorus structure

#### 5. ☐ **Design validation split strategy**
   - 70% train, 15% validation, 15% test
   - Stratify by genre
   - Stratify by subgenre within each genre
   - Ensure tempo/key diversity in each split
   - Ensure complexity range in each split

#### 6. ☐ **Design genre imbalance handling**
   - **Over-sampling minority classes:** Rock (100 files)
   - **Under-sampling majority classes:** Classical (200 files)
   - **SMOTE-like synthesis** for rare genre/parameter combinations
   - **Weighted loss functions** to penalize majority class errors less
   - **Augmentation multipliers** per genre

#### 7. ☐ **Design cross-genre transfer learning strategy**
   - **Pre-training:** Train on all genres, then fine-tune per genre
   - **Genre embeddings:** Learn shared representations
   - **Genre-specific heads:** Specialized outputs per genre
   - **Transfer matrix:** Which genres help predict others?
   - **Domain adaptation:** Reduce genre-specific biases

#### 8. ☐ **Calculate augmentation multipliers**
   ```python
   # Target: balanced 500 files per genre for training
   augmentation_multipliers = {
       'jazz': 500 / (150 * 0.7) = 4.76x  # 105 train → 500
       'classical': 500 / (200 * 0.7) = 3.57x  # 140 train → 500
       'rock': 500 / (100 * 0.7) = 7.14x  # 70 train → 500
       'electronic': 500 / (120 * 0.7) = 5.95x  # 84 train → 500
       'pop': 500 / (180 * 0.7) = 3.97x  # 126 train → 500
   }
   ```

#### 9. ☐ **Design parameter preservation constraints**
   - Which parameters must be preserved during augmentation?
   - Level 1 (Global): `structure.form` → preserve
   - Level 2 (Universal): Most should scale appropriately
   - Level 3 (Genre-specific): Must be preserved
   - Create constraint matrix: [Augmentation Type × Parameter]

#### 10. ☐ **Design quality validation for augmented data**
    - Check musical validity after augmentation
    - Ensure harmony remains coherent after transposition
    - Verify rhythm integrity after tempo scaling
    - Check parameter drift (augmented vs original)
    - Flag invalid augmentations

#### 11. ☐ **Design genre similarity matrix**
    ```python
    # For transfer learning: which genres are similar?
    genre_similarity = {
        'jazz': {'jazz': 1.0, 'classical': 0.4, 'rock': 0.3, 'electronic': 0.2, 'pop': 0.5},
        'classical': {'jazz': 0.4, 'classical': 1.0, 'rock': 0.3, 'electronic': 0.2, 'pop': 0.4},
        'rock': {'jazz': 0.3, 'classical': 0.3, 'rock': 1.0, 'electronic': 0.4, 'pop': 0.7},
        'electronic': {'jazz': 0.2, 'classical': 0.2, 'rock': 0.4, 'electronic': 1.0, 'pop': 0.6},
        'pop': {'jazz': 0.5, 'classical': 0.4, 'rock': 0.7, 'electronic': 0.6, 'pop': 1.0}
    }
    ```

#### 12. ☐ **Document genre-specific considerations**
    - Musical theory constraints per genre
    - Cultural considerations
    - Historical authenticity requirements
    - Common pitfalls and how to avoid them

---

### Phase 2: Implementation - Core Infrastructure (Tasks 13-25)

#### 13. ☐ **Create genre stratifier class**
    ```python
    class GenreStratifier:
        """
        Handles stratified splitting of multi-genre dataset
        """
        def __init__(self, stratify_by=['genre', 'subgenre', 'tempo_range', 'complexity']):
            pass

        def split(self, dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
            """Returns train, val, test splits with genre balance"""
            pass

        def validate_split(self, train, val, test):
            """Check if split maintains genre distribution"""
            pass
    ```

#### 14. ☐ **Implement base augmentation class**
    ```python
    class MIDIAugmentation:
        """Base class for MIDI augmentations"""

        def __init__(self, probability=1.0, preserve_parameters=None):
            self.probability = probability
            self.preserve_parameters = preserve_parameters or []

        def augment(self, midi_data):
            """Apply augmentation to MIDI data"""
            raise NotImplementedError

        def validate(self, original, augmented):
            """Validate augmentation quality"""
            raise NotImplementedError
    ```

#### 15. ☐ **Implement pitch transposition augmentation**
    ```python
    class PitchTransposition(MIDIAugmentation):
        """Transpose MIDI by semitones"""

        def __init__(self, semitone_range=(-3, 3), genre_specific=True):
            pass

        def augment(self, midi_data):
            # Transpose all notes
            # Update key signature
            # Ensure harmony remains valid
            pass
    ```

#### 16. ☐ **Implement tempo scaling augmentation**
    ```python
    class TempoScaling(MIDIAugmentation):
        """Scale tempo while preserving pitch"""

        def __init__(self, tempo_range=(0.9, 1.1), genre_specific=True):
            pass

        def augment(self, midi_data):
            # Scale all timing events
            # Update tempo meta events
            # Preserve relative timing
            pass
    ```

#### 17. ☐ **Implement velocity perturbation augmentation**
    ```python
    class VelocityPerturbation(MIDIAugmentation):
        """Add controlled randomness to velocities"""

        def __init__(self, variance=5, preserve_dynamics=True):
            pass

        def augment(self, midi_data):
            # Add Gaussian noise to velocities
            # Clip to valid MIDI range [1, 127]
            # Preserve overall dynamic contour
            pass
    ```

#### 18. ☐ **Implement timing jitter augmentation**
    ```python
    class TimingJitter(MIDIAugmentation):
        """Add humanization through timing variation"""

        def __init__(self, jitter_ms=10, swing_preservation=True):
            pass

        def augment(self, midi_data):
            # Add small random timing offsets
            # Preserve swing feel if present
            # Maintain rhythmic structure
            pass
    ```

#### 19. ☐ **Implement harmonic substitution augmentation**
    ```python
    class HarmonicSubstitution(MIDIAugmentation):
        """Apply jazz-style chord substitutions"""

        def __init__(self, substitution_prob=0.2, jazz_only=True):
            pass

        def augment(self, midi_data):
            # Detect chords
            # Apply substitutions (tritone, dim, extensions)
            # Maintain harmonic function
            pass
    ```

#### 20. ☐ **Create genre-specific augmentation pipeline**
    ```python
    class GenreAugmentationPipeline:
        """Applies genre-appropriate augmentations"""

        def __init__(self, genre):
            self.genre = genre
            self.pipeline = self._build_pipeline(genre)

        def _build_pipeline(self, genre):
            """Construct augmentation chain for genre"""
            if genre == 'jazz':
                return [
                    PitchTransposition((-5, 7)),
                    TempoScaling((0.85, 1.15)),
                    VelocityPerturbation(variance=10),
                    TimingJitter(jitter_ms=15),
                    HarmonicSubstitution(prob=0.3)
                ]
            elif genre == 'classical':
                return [
                    PitchTransposition((-3, 3)),
                    TempoScaling((0.95, 1.05)),
                    VelocityPerturbation(variance=3)
                ]
            # ... other genres

        def augment(self, midi_data, num_variations=4):
            """Generate N augmented versions"""
            pass
    ```

#### 21. ☐ **Implement genre balancer**
    ```python
    class GenreBalancer:
        """Balance dataset across genres"""

        def __init__(self, target_samples_per_genre=500):
            self.target = target_samples_per_genre

        def balance(self, dataset):
            """
            Apply over-sampling and augmentation to balance genres
            """
            pass

        def compute_class_weights(self, dataset):
            """
            Calculate loss weights for imbalanced training
            """
            pass
    ```

#### 22. ☐ **Implement cross-genre transfer utilities**
    ```python
    class CrossGenreTransfer:
        """Utilities for cross-genre learning"""

        def __init__(self, similarity_matrix):
            self.similarity = similarity_matrix

        def get_transfer_genres(self, target_genre, top_k=2):
            """Get most similar genres for transfer learning"""
            pass

        def create_mixed_batch(self, batch_size, genre_ratios):
            """Create training batch with specific genre mix"""
            pass
    ```

#### 23. ☐ **Implement augmentation validator**
    ```python
    class AugmentationValidator:
        """Validate quality of augmented MIDI"""

        def validate_harmony(self, midi_data):
            """Check harmonic validity"""
            pass

        def validate_rhythm(self, midi_data):
            """Check rhythmic coherence"""
            pass

        def check_parameter_drift(self, original_params, augmented_params):
            """Ensure parameters haven't drifted too much"""
            pass
    ```

#### 24. ☐ **Create genre-specific validation splits**
    ```python
    class GenreValidationSplitter:
        """Create genre-stratified validation sets"""

        def create_splits(self, dataset, stratify_cols):
            """Stratified K-fold splits"""
            pass

        def create_genre_specific_test_set(self, dataset):
            """Separate test sets per genre for analysis"""
            pass
    ```

#### 25. ☐ **Implement data statistics tracker**
    ```python
    class GenreDataStatistics:
        """Track dataset statistics across genres"""

        def compute_genre_distribution(self, dataset):
            pass

        def compute_parameter_distributions(self, dataset, by_genre=True):
            pass

        def visualize_genre_balance(self):
            pass
    ```

---

### Phase 3: Genre-Specific Implementations (Tasks 26-35)

#### 26. ☐ **Implement Jazz-specific augmentation**
    - Swing timing variations
    - Bebop vocabulary preservation
    - Walking bass pattern integrity
    - Chord extension variations
    - Improvisation section handling

#### 27. ☐ **Implement Classical-specific augmentation**
    - Voice leading preservation
    - Counterpoint integrity
    - Dynamic contour preservation
    - Minimal timing variation
    - Period-appropriate constraints

#### 28. ☐ **Implement Rock-specific augmentation**
    - Power chord detection and preservation
    - Riff repetition integrity
    - Distortion-compatible transpositions
    - Drum pattern preservation

#### 29. ☐ **Implement Electronic-specific augmentation**
    - Quantization grid preservation
    - Arpeggio pattern integrity
    - Filter envelope preservation
    - Loop/pattern structure maintenance

#### 30. ☐ **Implement Pop-specific augmentation**
    - Hook/chorus preservation
    - Verse-chorus structure integrity
    - Moderate augmentation ranges
    - Mainstream palatability checks

#### 31. ☐ **Create genre-specific test cases**
    - Unit tests for each genre augmentation
    - Musical validity checks
    - Parameter preservation tests
    - Edge case handling

#### 32. ☐ **Implement genre confusion matrix**
    ```python
    class GenreConfusionAnalyzer:
        """Analyze genre misclassification patterns"""

        def compute_confusion_matrix(self, predictions, labels):
            pass

        def identify_confusable_genres(self):
            """Which genres are often confused?"""
            pass

        def suggest_augmentation_strategy(self, confusable_pairs):
            """Suggest augmentations to improve discrimination"""
            pass
    ```

#### 33. ☐ **Implement subgenre handling**
    ```python
    class SubgenreHandler:
        """Handle subgenre diversity within genres"""

        def ensure_subgenre_diversity(self, dataset, genre):
            """Ensure all subgenres represented"""
            pass

        def balance_subgenres(self, dataset, genre):
            """Balance subgenres within genre"""
            pass
    ```

#### 34. ☐ **Create genre-specific feature importance**
    ```python
    class GenreFeatureAnalyzer:
        """Analyze which features are most important per genre"""

        def compute_genre_feature_importance(self, model, genre):
            pass

        def identify_genre_discriminative_features(self):
            """Features that distinguish genres"""
            pass
    ```

#### 35. ☐ **Implement adaptive augmentation**
    ```python
    class AdaptiveAugmentation:
        """Adjust augmentation based on training progress"""

        def adjust_augmentation_strength(self, epoch, performance):
            """Reduce augmentation as model improves"""
            pass

        def focus_on_hard_examples(self, error_analysis):
            """Augment more in areas of poor performance"""
            pass
    ```

---

### Phase 4: Testing & Documentation (Tasks 36-45)

#### 36. ☐ **Create comprehensive unit tests**
    - Test each augmentation type
    - Test genre stratification
    - Test balancing strategies
    - Test validation split quality

#### 37. ☐ **Create integration tests**
    - Test full augmentation pipeline
    - Test with dummy dataset
    - Test genre-specific paths
    - Test error handling

#### 38. ☐ **Performance benchmarking**
    - Augmentation speed per technique
    - Memory usage
    - Batch augmentation performance
    - Optimization opportunities

#### 39. ☐ **Create visualization tools**
    - Genre distribution plots
    - Augmentation effect visualizations
    - Parameter drift plots
    - Confusion matrices

#### 40. ☐ **Write API documentation**
    - Class documentation
    - Method signatures
    - Usage examples
    - Best practices

#### 41. ☐ **Create usage tutorials**
    - Quick start guide
    - Genre-specific examples
    - Advanced usage patterns
    - Troubleshooting guide

#### 42. ☐ **Document genre-specific strategies**
    - Why each genre needs different augmentation
    - Musical theory background
    - Design decisions and trade-offs
    - Future improvements

#### 43. ☐ **Create validation report template**
    - Genre balance metrics
    - Augmentation quality scores
    - Parameter preservation rates
    - Recommendations

#### 44. ☐ **Integration preparation**
    - Interface documentation for Agent 05 (MTL)
    - Interface documentation for Agent 06 (Training)
    - Data format specifications
    - API contracts

#### 45. ☐ **Final validation and handoff**
    - All tests passing
    - Documentation complete
    - Examples working
    - Ready for integration with Agent 03's dataset

---

## Success Criteria

- ✅ Genre stratification maintains proportions in all splits
- ✅ Augmentation increases minority classes to balanced levels
- ✅ Augmented data passes musical validity checks
- ✅ Parameters preserved within acceptable drift (<5%)
- ✅ Cross-genre transfer utilities ready for Agent 05 integration
- ✅ All unit tests passing (>95% coverage)
- ✅ Comprehensive documentation complete
- ✅ Performance benchmarks meet targets (<0.5s per augmentation)

---

## Dependencies

### Input Dependencies (cannot proceed without):
- **Agent 03:** Labeled dataset with 750 files
- **Agent 05:** Hierarchical MTL architecture (for integration)

### Output Dependencies (who needs my work):
- **Agent 06:** Training Pipeline Engineer (uses augmented data)
- **Agent 08:** Validation Framework Builder (uses genre-specific validation)

---

## Implementation Priority

**Can Do Now (Independent):**
- Tasks 1-12: All strategy and design work
- Tasks 13-25: Core infrastructure implementation
- Tasks 26-35: Genre-specific implementations
- Tasks 36-45: Testing and documentation (with dummy data)

**Need Agent 03 (Labeled Dataset):**
- Actual data augmentation execution
- Real genre distribution analysis
- Parameter drift validation on real data

**Need Agent 05 (MTL Architecture):**
- Integration with training pipeline
- Cross-genre transfer learning implementation
- Genre-specific head training

---

**Status:** Ready to begin independent tasks (1-45 with dummy data)
**Next Steps:** Implement all core infrastructure and test with synthetic data
