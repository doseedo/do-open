# Agent 9: Machine Learning Integration & Pattern Discovery

## Overview

Agent 9 implements **machine learning and pattern discovery** capabilities for the MIDI generator library. This module provides sophisticated tools for learning from existing music, discovering patterns, and using learned knowledge to improve generation.

## Modules Created

### 1. Pattern Extractor (`learning/pattern_extractor.py`)
**850+ lines** - Comprehensive pattern mining and analysis

#### Features:
- **N-gram Analysis**: Extract and analyze pitch, interval, and rhythm n-grams
- **Pattern Mining**: Discover recurring melodic, harmonic, and rhythmic patterns
- **Melodic Clustering**: Group similar patterns using multiple similarity metrics
- **Statistical Analysis**: Compute entropy, complexity, and diversity metrics

#### Key Classes:

```python
from midi_generator.learning import PatternExtractor, NGramExtractor, MelodicClusterer

# Extract patterns from sequences
extractor = PatternExtractor()
patterns = extractor.extract_melodic_patterns(
    pitch_sequences,
    duration_sequences,
    min_frequency=2
)

# N-gram analysis
ngram = NGramExtractor()
ngram.build_ngram_model(sequences, 'interval')
predictions = ngram.predict_next(context=(2, 2), ngram_type='interval')

# Cluster similar melodies
clusterer = MelodicClusterer()
clusters = clusterer.cluster_hierarchical(n_clusters=5)
```

#### Algorithms Implemented:
- **Edit Distance**: Levenshtein distance for sequence similarity
- **LCS**: Longest Common Subsequence for contour matching
- **Hierarchical Clustering**: Agglomerative clustering with custom metrics
- **K-means**: Feature-based pattern grouping
- **Shannon Entropy**: Measure of melodic complexity

---

### 2. Corpus Learner (`learning/corpus_learner.py`)
**750+ lines** - Statistical style learning from MIDI corpora

#### Features:
- **Style Modeling**: Learn statistical characteristics of composers/genres
- **Style Generation**: Generate melodies in learned styles
- **Style Interpolation**: Create hybrid styles by interpolation
- **Classification**: Classify melodies by composer/genre/era

#### Key Classes:

```python
from midi_generator.learning import StyleLearner, StyleClassifier, CorpusAnalyzer

# Learn style from corpus
learner = StyleLearner()
bach_model = learner.learn_style("Bach", bach_sequences)
mozart_model = learner.learn_style("Mozart", mozart_sequences)

# Generate in style
melody = learner.generate_in_style("Bach", length=16, temperature=0.8)

# Interpolate styles
hybrid = learner.interpolate_styles("Bach", "Mozart", alpha=0.5)

# Classify melodies
classifier = StyleClassifier()
classifier.train(training_data)
label, confidence = classifier.predict(test_melody)
```

#### Statistical Models:
- **Pitch Class Distribution**: Tonality and key preferences
- **Interval Distribution**: Melodic motion characteristics
- **Rhythm Distribution**: Temporal patterns
- **N-gram Transition Probabilities**: Context-dependent prediction
- **KL Divergence**: Style comparison metric

---

### 3. Motif Library (`learning/motif_library.py`)
**650+ lines** - Motif extraction and searchable database

#### Features:
- **Motif Extraction**: Automatic discovery of salient melodic fragments
- **Multi-dimensional Tagging**: Emotion, genre, composer, era
- **Similarity Search**: Find similar motifs using multiple metrics
- **Transformations**: Transpose, retrograde, inversion, augmentation
- **Database Persistence**: JSON-based storage

#### Key Classes:

```python
from midi_generator.learning import Motif, MotifExtractor, MotifDatabase

# Create and store motifs
motif = Motif(
    id="beethoven_5th",
    notes=[67, 67, 67, 63],
    intervals=[0, 0, -4],
    rhythm=[0.5, 0.5, 0.5, 2.0],
    composer="Beethoven",
    emotion_tags=["dramatic", "heroic"]
)

# Search database
db = MotifDatabase("motif_library.json")
db.add_motif(motif)
results = db.search_by_tags(emotion="dramatic", genre="classical")

# Find similar motifs
similar = db.find_similar(query_motif, top_k=5, metric='combined')

# Transformations
transposed = motif.transpose(2)
retrograde = motif.retrograde()
inverted = motif.inversion()
```

#### Motif Transformations:
- **Transposition**: Shift pitch while preserving intervals
- **Retrograde**: Reverse note sequence
- **Inversion**: Flip intervals
- **Augmentation/Diminution**: Rhythmic scaling
- **Combination**: Intelligent motif chaining

---

### 4. Fitness Learning (`optimization/fitness_learning.py`)
**550+ lines** - Machine learning-based fitness functions

#### Features:
- **Supervised Learning**: Train from labeled examples
- **Preference Learning**: Learn from user ratings and comparisons
- **Multi-objective Optimization**: Combine multiple criteria
- **Feature Extraction**: 25+ musical features
- **Active Learning**: Efficient training data collection

#### Key Classes:

```python
from midi_generator.optimization import (
    LearnedFitnessFunction,
    PreferenceLearner,
    MultiObjectiveFitness,
    melodic_smoothness,
    climax_placement
)

# Train fitness function
fitness = LearnedFitnessFunction(model_type='random_forest')
fitness.train(labeled_examples)
score = fitness.evaluate(pitches, durations)

# Learn user preferences
pref_learner = PreferenceLearner()
pref_learner.add_rating(user_id, pitches, durations, rating=0.9)
user_model = pref_learner.train_user_model(user_id)

# Multi-objective optimization
multi_fit = MultiObjectiveFitness()
multi_fit.add_objective("smoothness", melodic_smoothness, weight=1.0)
multi_fit.add_objective("climax", climax_placement, weight=0.5)
overall_score = multi_fit.evaluate(melody)
```

#### Machine Learning Models:
- **Random Forest**: Robust classification/regression
- **SVM**: Support Vector Machines with RBF kernel
- **Gradient Boosting**: High-performance regression
- **Cross-validation**: 5-fold CV for evaluation

---

## Integration with Existing Code

### Pattern-Based Generation

```python
from midi_generator.learning import PatternExtractor
from midi_generator.generators import MelodyGenerator  # From other agents

# Extract patterns from corpus
extractor = PatternExtractor()
patterns = extractor.extract_melodic_patterns(corpus_sequences)

# Use top patterns as seed material
top_pattern = patterns[0]
generator = MelodyGenerator()
extended_melody = generator.extend_pattern(top_pattern.notes)
```

### Style-Aware Generation

```python
from midi_generator.learning import StyleLearner
from midi_generator.algorithms import MarkovChain  # From Agent 2

# Learn style
learner = StyleLearner()
style = learner.learn_style("Jazz", jazz_corpus)

# Generate using learned probabilities
markov = MarkovChain(transition_probs=style.n_gram_models[2])
melody = markov.generate(length=32)
```

### Fitness-Guided Evolution

```python
from midi_generator.optimization import LearnedFitnessFunction
from midi_generator.algorithms import GeneticAlgorithm  # From existing code

# Train fitness
fitness = LearnedFitnessFunction()
fitness.train(good_vs_bad_examples)

# Use in genetic algorithm
ga = GeneticAlgorithm(fitness_function=fitness.evaluate)
best_melody = ga.evolve(generations=100)
```

---

## Research Foundations

### Academic References

1. **Music Information Retrieval**
   - Müller, M. (2015). *Fundamentals of Music Processing*
   - Typke, R. et al. (2005). "A Survey of Music Information Retrieval Systems"

2. **Pattern Discovery**
   - Conklin, D. & Witten, I. (1995). "Multiple Viewpoint Systems for Music Prediction"
   - Pearce, M. & Wiggins, G. (2012). "Auditory Expectation"

3. **Style Modeling**
   - Cope, D. (1996). *Experiments in Musical Intelligence*
   - Pachet, F. (2003). "The Continuator: Musical Interaction with Style"

4. **Preference Learning**
   - Furnkranz, J. & Hüllermeier, E. (2010). *Preference Learning*
   - Miranda, E. & Biles, J. (2007). *Evolutionary Computer Music*

5. **Melodic Similarity**
   - Typke, R. et al. (2005). "Using Transportation Distances for Measuring Melodic Similarity"
   - Mongeau, M. & Sankoff, D. (1990). "Comparison of Musical Sequences"

---

## Usage Examples

### Complete Workflow Example

```python
"""
Complete machine learning workflow for MIDI generation
"""

# 1. Analyze existing corpus
from midi_generator.learning import CorpusAnalyzer, PatternExtractor, StyleLearner

analyzer = CorpusAnalyzer()
# Add MIDI files to corpus
for filepath in corpus_files:
    analyzer.add_file(filepath, composer="Bach", genre="baroque")

# 2. Extract patterns
extractor = PatternExtractor()
patterns = extractor.extract_melodic_patterns(
    analyzer.pitch_sequences,
    analyzer.duration_sequences
)

print(f"Found {len(patterns)} patterns")

# 3. Learn style
learner = StyleLearner()
bach_style = learner.learn_style("Bach", analyzer.pitch_sequences)

# 4. Build motif library
from midi_generator.learning import MotifExtractor, MotifDatabase

motif_extractor = MotifExtractor()
db = MotifDatabase("baroque_motifs.json")

for seq, dur in zip(analyzer.pitch_sequences, analyzer.duration_sequences):
    motifs = motif_extractor.extract_from_sequence(
        seq, dur, composer="Bach", genre="baroque"
    )
    for motif in motifs:
        db.add_motif(motif)

# 5. Train fitness function
from midi_generator.optimization import LearnedFitnessFunction, MelodyExample

examples = []
# Add good examples
for seq in good_melodies:
    examples.append(MelodyExample(seq, durations, quality_label=3))

# Add bad examples
for seq in bad_melodies:
    examples.append(MelodyExample(seq, durations, quality_label=0))

fitness = LearnedFitnessFunction()
fitness.train(examples)

# 6. Generate with learned knowledge
melody = learner.generate_in_style("Bach", length=32)

# Evaluate quality
quality_score = fitness.evaluate(melody)
print(f"Generated melody quality: {quality_score:.2f}")

# Find similar motifs
similar_motifs = db.find_similar(
    Motif(notes=melody[:8], ...),
    top_k=5
)
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| N-gram extraction | O(n*L) | O(n) |
| Pattern clustering | O(n²*L) | O(n²) |
| Style learning | O(m*L) | O(V) |
| Motif search | O(n*L²) | O(n) |
| Fitness evaluation | O(F) | O(F) |

Where:
- n = number of sequences
- L = average sequence length
- m = corpus size
- V = vocabulary size
- F = number of features

### Scalability

- **Small corpus** (< 100 files): All operations < 1 second
- **Medium corpus** (100-1000 files): Pattern extraction < 10 seconds
- **Large corpus** (> 1000 files): Use sampling or distributed processing

---

## Dependencies

```python
# Required
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Optional (for enhanced features)
matplotlib>=3.4.0  # Visualization
pandas>=1.3.0      # Data analysis
```

Install with:
```bash
pip install numpy scipy scikit-learn
```

---

## Testing

Run the test suite:

```bash
cd /home/user/Do/midi_generator
python tests/test_learning.py
```

Run the demonstration:

```bash
python examples/pattern_learning_demo.py
```

---

## File Structure

```
midi_generator/
├── learning/
│   ├── __init__.py                 # Package exports
│   ├── pattern_extractor.py        # 850+ lines - Pattern mining
│   ├── corpus_learner.py           # 750+ lines - Style learning
│   └── motif_library.py            # 650+ lines - Motif database
│
├── optimization/
│   ├── __init__.py                 # Package exports
│   └── fitness_learning.py         # 550+ lines - Learned fitness
│
├── examples/
│   └── pattern_learning_demo.py    # 400+ lines - Comprehensive demo
│
├── tests/
│   └── test_learning.py            # 300+ lines - Test suite
│
└── docs/
    └── AGENT_9_ML_PATTERN_DISCOVERY.md  # This file
```

**Total Lines of Code: 3,500+**

---

## Future Enhancements

### Potential Additions

1. **Deep Learning Integration**
   - LSTM/Transformer models for sequence prediction
   - VAE for latent space interpolation
   - GAN for adversarial generation

2. **Advanced MIR**
   - Audio feature extraction (MFCCs, chroma)
   - Onset detection and beat tracking
   - Automatic transcription integration

3. **Graph-Based Methods**
   - Musical knowledge graphs
   - Harmonic space navigation
   - Constraint networks

4. **Real-time Learning**
   - Online learning from user feedback
   - Reinforcement learning for generation
   - Transfer learning across styles

---

## Contributing

When extending these modules:

1. **Maintain type hints** for all functions
2. **Write comprehensive docstrings** (Google style)
3. **Add unit tests** for new features
4. **Update examples** to demonstrate new capabilities
5. **Document research foundations** in code comments

---

## License

MIT License - See main repository LICENSE file

---

## Contact

**Agent 9 - Machine Learning Integration & Pattern Discovery**

For questions about these modules or integration with other agents' work, refer to the main repository documentation.

---

*Built with ❤️ for the ultimate MIDI generation library*
