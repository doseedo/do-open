# Multi-Genre Data Strategy Document
## Agent 07: Multi-Genre Data Specialist

**Date:** November 20, 2025
**Version:** 1.0
**Status:** Strategy Design Complete

---

## Table of Contents
1. [Genre Distribution Analysis](#genre-distribution-analysis)
2. [Stratification Strategy](#stratification-strategy)
3. [Data Augmentation Design](#data-augmentation-design)
4. [Validation Split Strategy](#validation-split-strategy)
5. [Genre Imbalance Handling](#genre-imbalance-handling)
6. [Cross-Genre Transfer Learning](#cross-genre-transfer-learning)
7. [Implementation Guidelines](#implementation-guidelines)

---

## 1. Genre Distribution Analysis

### Current Corpus Distribution
```
Genre       | Files | Percentage | Train (70%) | Val (15%) | Test (15%)
------------|-------|------------|-------------|-----------|------------
Jazz        | 150   | 20.0%      | 105         | 23        | 22
Classical   | 200   | 26.7%      | 140         | 30        | 30
Rock        | 100   | 13.3%      | 70          | 15        | 15
Electronic  | 120   | 16.0%      | 84          | 18        | 18
Pop         | 180   | 24.0%      | 126         | 27        | 27
------------|-------|------------|-------------|-----------|------------
TOTAL       | 750   | 100%       | 525         | 113       | 112
```

### Imbalance Analysis
- **Most represented:** Classical (200 files, 26.7%)
- **Least represented:** Rock (100 files, 13.3%)
- **Imbalance ratio:** 2:1 (Classical vs Rock)
- **Assessment:** MODERATE imbalance - requires balancing strategy

### Target Distribution (After Augmentation)
```
Genre       | Target Train Files | Augmentation Multiplier | Strategy
------------|-------------------|------------------------|------------------
Jazz        | 500               | 4.76x                  | Augmentation
Classical   | 500               | 3.57x                  | Augmentation
Rock        | 500               | 7.14x                  | Heavy Aug.
Electronic  | 500               | 5.95x                  | Augmentation
Pop         | 500               | 3.97x                  | Augmentation
------------|-------------------|------------------------|------------------
TOTAL       | 2,500             | 4.76x average          |
```

**Rationale:** Balanced training set of 500 files per genre allows model to learn each genre equally well without bias toward over-represented genres.

---

## 2. Stratification Strategy

### Multi-Level Stratification

#### Level 1: Primary Genre
- Ensure proportional representation in train/val/test
- Maintain 70/15/15 split within each genre

#### Level 2: Subgenre
- Within Jazz: bebop, swing, modal, fusion
- Within Classical: baroque, classical period, romantic, contemporary
- Within Rock: classic rock, progressive, metal
- Within Electronic: ambient, techno, IDM, dnb
- Ensure all subgenres represented in each split

#### Level 3: Musical Properties
Stratify by:
- **Tempo range:** Slow (40-80), Medium (80-140), Fast (140-200+)
- **Key:** All 12 keys represented
- **Time signature:** Common (4/4, 3/4) and complex (5/4, 7/8, etc.)
- **Complexity:** Simple, Medium, Complex (based on `complexity.overall` parameter)

### Stratification Algorithm
```python
def stratify(dataset):
    # 1. Group by genre
    # 2. Within genre, group by subgenre
    # 3. Within subgenre, group by tempo range
    # 4. Within tempo range, shuffle and split 70/15/15
    # 5. Verify all properties are balanced across splits
    return train, val, test
```

### Validation Criteria
- ✅ Each split has all 5 genres
- ✅ Genre proportions within ±2% of target
- ✅ Subgenre diversity maintained
- ✅ Tempo/key distributions similar across splits
- ✅ No data leakage between splits

---

## 3. Data Augmentation Design

### MIDI-Specific Augmentation Techniques

#### 3.1 Pitch Transposition
**Description:** Shift all pitches by N semitones
**Range:** Genre-dependent (see below)
**Preserves:** Intervals, contours, rhythms
**Changes:** Absolute pitch, key
**Complexity:** LOW

**Implementation:**
```python
def transpose(midi, semitones):
    for note in midi.notes:
        note.pitch += semitones
        note.pitch = clip(note.pitch, 0, 127)
    update_key_signature(midi, semitones)
```

#### 3.2 Tempo Scaling
**Description:** Multiply all timing by scale factor
**Range:** Genre-dependent (see below)
**Preserves:** Pitch, relative timing
**Changes:** Absolute tempo, duration
**Complexity:** LOW

**Implementation:**
```python
def scale_tempo(midi, factor):
    for event in midi.events:
        event.time *= factor
    midi.tempo *= factor
```

#### 3.3 Velocity Perturbation
**Description:** Add controlled noise to velocities
**Range:** Gaussian noise, σ = genre-dependent
**Preserves:** General dynamics, articulation
**Changes:** Exact velocities
**Complexity:** LOW

**Implementation:**
```python
def perturb_velocity(midi, variance=5):
    for note in midi.notes:
        noise = np.random.normal(0, variance)
        note.velocity += int(noise)
        note.velocity = clip(note.velocity, 1, 127)
```

#### 3.4 Timing Jitter (Humanization)
**Description:** Add small random timing offsets
**Range:** ±10-20ms
**Preserves:** Rhythm structure, swing
**Changes:** Exact timing (makes more human)
**Complexity:** MEDIUM

**Implementation:**
```python
def add_jitter(midi, jitter_ms=10):
    for note in midi.notes:
        offset = np.random.normal(0, jitter_ms)
        note.start_time += offset
        note.end_time += offset
```

#### 3.5 Harmonic Substitution
**Description:** Replace chords with musical substitutes
**Types:** Tritone sub, dim7, extensions (9th, 11th, 13th)
**Preserves:** Harmonic function
**Changes:** Exact chord voicings
**Complexity:** HIGH (requires music theory)

**Implementation:**
```python
def substitute_chords(midi, prob=0.2):
    chords = detect_chords(midi)
    for chord in chords:
        if random() < prob:
            substitute = get_substitution(chord)
            replace_chord(midi, chord, substitute)
```

#### 3.6 Voice Permutation (Orchestration)
**Description:** Swap instruments while preserving parts
**Preserves:** Notes, timing, structure
**Changes:** Instrumentation
**Complexity:** MEDIUM

**Implementation:**
```python
def permute_voices(midi):
    tracks = midi.tracks
    instrument_map = random_permutation(len(tracks))
    for i, track in enumerate(tracks):
        track.instrument = instrument_map[i]
```

---

### Genre-Specific Augmentation Rules

#### Jazz Augmentation
```python
JAZZ_AUGMENTATION = {
    'pitch_transposition': {
        'range': (-5, 7),  # Wide range for jazz standards
        'probability': 0.8
    },
    'tempo_scaling': {
        'range': (0.85, 1.15),  # Moderate variation
        'probability': 0.7
    },
    'velocity_perturbation': {
        'variance': 10,  # High for swing dynamics
        'probability': 0.9
    },
    'timing_jitter': {
        'jitter_ms': 15,  # Swing feel
        'preserve_swing': True,
        'probability': 0.8
    },
    'harmonic_substitution': {
        'types': ['tritone_sub', 'dim7', 'extensions'],
        'probability': 0.3
    }
}
```

**Rationale:**
- Jazz frequently modulates, so wide transposition OK
- Swing timing is essential, preserve with jitter
- Chord substitutions are idiomatic to jazz
- High dynamic variation is characteristic

#### Classical Augmentation
```python
CLASSICAL_AUGMENTATION = {
    'pitch_transposition': {
        'range': (-3, 3),  # Conservative - period authenticity
        'probability': 0.6
    },
    'tempo_scaling': {
        'range': (0.95, 1.05),  # Minimal - rubato is deliberate
        'probability': 0.5
    },
    'velocity_perturbation': {
        'variance': 3,  # Preserve carefully crafted dynamics
        'probability': 0.6
    },
    'timing_jitter': {
        'jitter_ms': 5,  # Minimal - precise timing
        'probability': 0.3
    },
    'harmonic_substitution': {
        'probability': 0.0  # DON'T alter harmony
    },
    'voice_permutation': {
        'probability': 0.0  # DON'T alter orchestration
    }
}
```

**Rationale:**
- Classical music has specific period constraints
- Harmony and voice leading are carefully composed
- Minimal augmentation preserves authenticity
- Focus on subtle variations only

#### Rock Augmentation
```python
ROCK_AUGMENTATION = {
    'pitch_transposition': {
        'range': (-3, 5),  # Common guitar tunings
        'probability': 0.7,
        'preserve_power_chords': True
    },
    'tempo_scaling': {
        'range': (0.9, 1.1),
        'probability': 0.7
    },
    'velocity_perturbation': {
        'variance': 8,  # Aggressive dynamics
        'probability': 0.8
    },
    'timing_jitter': {
        'jitter_ms': 12,  # Human drumming
        'probability': 0.7
    },
    'riff_preservation': {
        'detect_riffs': True,
        'keep_intact': True
    }
}
```

**Rationale:**
- Guitar-based music benefits from guitar-friendly transpositions
- Power chords and riffs are defining features - preserve
- Moderate augmentation range
- Human timing variation important

#### Electronic Augmentation
```python
ELECTRONIC_AUGMENTATION = {
    'pitch_transposition': {
        'range': (-7, 7),  # Wide range - electronic is flexible
        'probability': 0.9
    },
    'tempo_scaling': {
        'range': (0.8, 1.2),  # Wide tempo range
        'probability': 0.8
    },
    'velocity_perturbation': {
        'variance': 5,
        'probability': 0.6
    },
    'timing_jitter': {
        'jitter_ms': 2,  # MINIMAL - quantized
        'preserve_quantization': True,
        'probability': 0.3
    },
    'pattern_preservation': {
        'detect_loops': True,
        'keep_structure': True
    }
}
```

**Rationale:**
- Electronic music has no physical instrument constraints
- Wide pitch/tempo augmentation acceptable
- Quantization is defining feature - preserve
- Loop structures are essential - preserve

#### Pop Augmentation
```python
POP_AUGMENTATION = {
    'pitch_transposition': {
        'range': (-4, 4),  # Vocal range considerations
        'probability': 0.8
    },
    'tempo_scaling': {
        'range': (0.9, 1.1),
        'probability': 0.7
    },
    'velocity_perturbation': {
        'variance': 6,
        'probability': 0.7
    },
    'timing_jitter': {
        'jitter_ms': 10,
        'probability': 0.6
    },
    'structure_preservation': {
        'preserve_verse_chorus': True,
        'preserve_hooks': True
    }
}
```

**Rationale:**
- Pop is vocally-oriented, moderate transposition
- Verse-chorus structure is defining - preserve
- Hooks are critical - preserve
- Moderate augmentation across the board

---

### Augmentation Multipliers

To achieve balanced 500 files per genre in training:

```python
AUGMENTATION_MULTIPLIERS = {
    'jazz': 4.76,      # 105 → 500 (generate 395 augmented)
    'classical': 3.57,  # 140 → 500 (generate 360 augmented)
    'rock': 7.14,      # 70 → 500 (generate 430 augmented)  # HIGHEST
    'electronic': 5.95, # 84 → 500 (generate 416 augmented)
    'pop': 3.97        # 126 → 500 (generate 374 augmented)
}
```

**Strategy:** For each original file, generate N augmented versions:
- Rock: 6-7 augmented per original (highest imbalance)
- Electronic: 5-6 per original
- Jazz: 4-5 per original
- Pop: 3-4 per original
- Classical: 3-4 per original

---

## 4. Validation Split Strategy

### Split Ratios
- **Training:** 70% (525 original → 2,500 after augmentation)
- **Validation:** 15% (113 original, NO augmentation)
- **Test:** 15% (112 original, NO augmentation)

**Critical:** Validation and test sets use ONLY original files, not augmented. This ensures true generalization testing.

### Stratification Checklist
For each split, verify:

1. **Genre Balance**
   ```python
   for genre in ['jazz', 'classical', 'rock', 'electronic', 'pop']:
       split_ratio = count(split, genre) / count(split)
       target_ratio = count(dataset, genre) / count(dataset)
       assert abs(split_ratio - target_ratio) < 0.02  # Within 2%
   ```

2. **Subgenre Coverage**
   ```python
   for genre in genres:
       subgenres_in_split = get_subgenres(split, genre)
       subgenres_in_dataset = get_subgenres(dataset, genre)
       assert subgenres_in_split == subgenres_in_dataset
   ```

3. **Tempo Distribution**
   ```python
   tempo_bins = [0-80, 80-120, 120-160, 160-200, 200+]
   for bin in tempo_bins:
       check_proportion(split, bin)
   ```

4. **Key Distribution**
   ```python
   for key in ['C', 'C#', 'D', ..., 'B']:
       check_coverage(split, key)
   ```

5. **Complexity Range**
   ```python
   complexity_bins = [0-0.33, 0.33-0.67, 0.67-1.0]
   for bin in complexity_bins:
       check_proportion(split, bin)
   ```

---

## 5. Genre Imbalance Handling

### Multi-Strategy Approach

#### Strategy 1: Over-Sampling (Data Augmentation)
- **Applied to:** All genres, especially Rock (7.14x multiplier)
- **Method:** Generate augmented versions using genre-specific rules
- **Target:** 500 training files per genre
- **Pros:** Increases data, improves generalization
- **Cons:** Computation time, storage

#### Strategy 2: Weighted Loss Functions
```python
def compute_class_weights(dataset):
    """
    Inverse frequency weighting for loss function
    """
    genre_counts = count_by_genre(dataset)
    max_count = max(genre_counts.values())

    weights = {
        genre: max_count / count
        for genre, count in genre_counts.items()
    }

    # Example weights (before augmentation):
    # rock: 200/100 = 2.0 (double weight)
    # classical: 200/200 = 1.0 (base weight)

    return weights
```

**Usage in training:**
```python
loss_weights = compute_class_weights(train_dataset)
loss = weighted_cross_entropy(predictions, labels, weights=loss_weights)
```

#### Strategy 3: Genre-Specific Batch Sampling
```python
class BalancedGenreSampler:
    """
    Ensure each batch has equal representation from all genres
    """
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_genre = batch_size // 5  # 5 genres

    def sample_batch(self):
        batch = []
        for genre in ['jazz', 'classical', 'rock', 'electronic', 'pop']:
            samples = random_sample(self.dataset, genre, self.samples_per_genre)
            batch.extend(samples)
        shuffle(batch)
        return batch
```

#### Strategy 4: SMOTE-like Synthesis
For rare parameter combinations within genres:
```python
def synthesize_rare_examples(dataset, genre, param_combo):
    """
    Interpolate between similar examples to create new training samples
    """
    # Find K nearest neighbors with rare parameter combo
    neighbors = find_knn(dataset, genre, param_combo, k=5)

    # Interpolate in feature space
    new_features = interpolate(neighbors)

    # Generate MIDI from features (if possible)
    new_midi = generate_from_features(new_features)

    return new_midi
```

**Use case:** If "fast bebop jazz with high complexity" is rare, synthesize more examples.

---

## 6. Cross-Genre Transfer Learning

### Genre Similarity Matrix

Based on musical properties (harmony, rhythm, structure):

```python
GENRE_SIMILARITY = {
    'jazz': {
        'jazz': 1.00,
        'classical': 0.40,  # Harmonic sophistication
        'rock': 0.30,       # Some shared instruments
        'electronic': 0.20, # Less common
        'pop': 0.50        # Jazz influences in pop
    },
    'classical': {
        'jazz': 0.40,
        'classical': 1.00,
        'rock': 0.30,       # Progressive rock connections
        'electronic': 0.20, # Orchestral electronic
        'pop': 0.40        # Classical training common
    },
    'rock': {
        'jazz': 0.30,
        'classical': 0.30,
        'rock': 1.00,
        'electronic': 0.40,  # Industrial, synth-rock
        'pop': 0.70         # Strong overlap
    },
    'electronic': {
        'jazz': 0.20,
        'classical': 0.20,
        'rock': 0.40,
        'electronic': 1.00,
        'pop': 0.60         # Electronic pop
    },
    'pop': {
        'jazz': 0.50,
        'classical': 0.40,
        'rock': 0.70,
        'electronic': 0.60,
        'pop': 1.00
    }
}
```

### Transfer Learning Strategies

#### Strategy 1: Pre-train on All, Fine-tune per Genre
```python
# Phase 1: Pre-train on all genres
model = HierarchicalMTL()
model.train(all_genres_data, epochs=50)

# Phase 2: Fine-tune per genre
for genre in ['jazz', 'classical', 'rock', 'electronic', 'pop']:
    genre_model = clone(model)
    genre_model.train(genre_specific_data, epochs=20, lr=0.0001)
    save(genre_model, f"model_{genre}.pth")
```

#### Strategy 2: Multi-Task with Genre Embeddings
```python
class GenreEmbeddingMTL(nn.Module):
    def __init__(self):
        self.genre_embedding = nn.Embedding(5, 64)  # 5 genres → 64D
        self.shared_encoder = nn.Linear(200, 512)
        # Concatenate genre embedding with features

    def forward(self, features, genre_id):
        genre_emb = self.genre_embedding(genre_id)
        combined = torch.cat([features, genre_emb], dim=1)
        return self.shared_encoder(combined)
```

#### Strategy 3: Domain Adaptation
Use similar genres to improve rare genres:
```python
def train_with_transfer(target_genre, source_genres):
    """
    Use source genres to improve target genre performance
    """
    # 1. Pre-train on source genres
    model.train(source_genres_data)

    # 2. Fine-tune on target genre with small LR
    model.train(target_genre_data, lr=0.0001)

    # 3. Use domain confusion loss to reduce genre bias
    domain_loss = compute_domain_confusion(model)
    total_loss = task_loss + 0.1 * domain_loss
```

**Example:** Rock is minority class. Use Pop (similarity 0.70) and Electronic (similarity 0.40) to improve Rock model.

#### Strategy 4: Ensemble Learning
```python
class GenreEnsemble:
    def __init__(self):
        self.models = {
            'jazz': load_model('jazz'),
            'classical': load_model('classical'),
            # ... etc
        }

    def predict(self, features, genre):
        # Primary prediction from genre-specific model
        primary = self.models[genre].predict(features)

        # Get predictions from similar genres
        similar_genres = get_similar_genres(genre, top_k=2)
        secondary = [self.models[g].predict(features) for g in similar_genres]

        # Weighted ensemble
        weights = [0.7] + [0.15, 0.15]  # Primary + two similar
        final = weighted_average([primary] + secondary, weights)

        return final
```

---

## 7. Implementation Guidelines

### Parameter Preservation Constraints

During augmentation, ensure these parameters are preserved:

| Augmentation       | Must Preserve                          | Can Change                    |
|--------------------|---------------------------------------|-------------------------------|
| Pitch Trans.       | Intervals, contour, rhythm            | Absolute pitch, key           |
| Tempo Scaling      | Pitch, relative timing                | Absolute tempo, duration      |
| Velocity Perturb.  | Articulation, general dynamics        | Exact velocities              |
| Timing Jitter      | Rhythm structure, swing               | Exact note timing             |
| Harmonic Sub.      | Harmonic function, progression        | Chord voicings                |
| Voice Permute      | Notes, timing, structure              | Instrumentation               |

### Parameter Drift Thresholds

After augmentation, check parameter drift:

```python
def check_parameter_drift(original_params, augmented_params):
    """
    Ensure augmented parameters haven't drifted too much
    """
    thresholds = {
        # Level 1 (Global)
        'genre.primary': 0.0,  # MUST match exactly
        'tempo.bpm': 0.15,     # Within 15% change
        'key.tonic': 'ANY',    # Can change (transposition)
        'key.mode': 0.0,       # Must preserve major/minor
        'complexity.overall': 0.10,  # Within 10%

        # Level 2 (Universal)
        'harmony.chord_density': 0.10,
        'melody.contour_smoothness': 0.05,
        'rhythm.syncopation': 0.05,

        # Level 3 (Genre-specific)
        'jazz.walking_bass': 0.05,
        'classical.voice_leading_quality': 0.05,
        # ... etc
    }

    for param, threshold in thresholds.items():
        if threshold == 0.0:
            assert original_params[param] == augmented_params[param]
        elif threshold == 'ANY':
            pass  # No constraint
        else:
            original = original_params[param]
            augmented = augmented_params[param]
            drift = abs(augmented - original) / original
            assert drift < threshold, f"{param} drifted by {drift:.2%}"
```

### Quality Validation Checklist

After augmentation, validate:

1. **File Integrity**
   - [ ] MIDI file loads without errors
   - [ ] All tracks present
   - [ ] No corrupted events

2. **Musical Validity**
   - [ ] Pitches in valid range (0-127)
   - [ ] Velocities in valid range (1-127)
   - [ ] No negative timings
   - [ ] Tempo reasonable (40-200 BPM)

3. **Harmonic Validity**
   - [ ] Chords are musically valid
   - [ ] No impossible intervals (> 2 octaves in melody)
   - [ ] Key signature coherent

4. **Rhythmic Validity**
   - [ ] Time signatures valid
   - [ ] Note durations positive
   - [ ] Rhythmic patterns coherent

5. **Genre Characteristics**
   - [ ] Genre-specific features preserved
   - [ ] Style-appropriate transformations
   - [ ] No cross-genre contamination

### Performance Targets

```python
PERFORMANCE_TARGETS = {
    'augmentation_speed': {
        'pitch_transposition': 0.05,  # seconds per file
        'tempo_scaling': 0.05,
        'velocity_perturbation': 0.03,
        'timing_jitter': 0.10,
        'harmonic_substitution': 0.50,  # slower (complex)
        'voice_permutation': 0.10
    },
    'total_pipeline': 2.0,  # seconds per augmented file
    'batch_processing': {
        'files_per_minute': 30,
        'total_time_750_files': 25  # minutes
    },
    'memory_usage': {
        'per_file': 50,  # MB
        'peak_batch': 2000  # MB (2 GB)
    }
}
```

---

## Conclusion

This multi-genre data strategy provides a comprehensive framework for:
1. **Balanced training** across 5 diverse musical genres
2. **Genre-appropriate augmentation** that respects musical constraints
3. **Robust validation** through stratified splitting
4. **Cross-genre learning** to improve performance on minority classes
5. **Quality assurance** to maintain musical validity

The strategy is designed to be implemented independently of the labeled dataset, allowing parallel development. When Agent 03 completes the dataset labeling, these strategies can be immediately applied.

---

**Next Steps:**
1. Implement core infrastructure (Phase 2)
2. Implement genre-specific augmentation (Phase 3)
3. Test with dummy data
4. Integrate with Agent 03's labeled dataset when ready

**Status:** ✅ Phase 1 Complete (Tasks 1-12)
