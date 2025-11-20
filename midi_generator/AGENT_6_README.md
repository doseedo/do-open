# Agent 6: Natural Language Parameter Predictor

## Overview

Agent 6 implements a comprehensive **Natural Language Parameter Predictor** system that converts natural language descriptions into precise parameter values for music generation. This enables users to type intuitive descriptions like "Generate a Sinatra-style ballad" and receive 515+ parameter values ready for MIDI generation.

## 🎯 Core Innovation

**Hybrid LLM + Knowledge-Based System:**
- Converts text descriptions to 515+ parameters via Claude LLM
- Falls back to keyword-based extraction when LLM unavailable
- Uses style database with 100+ examples for few-shot learning
- Validates and fills defaults for all parameters
- Integrates seamlessly with UniversalParameterRegistry

## 📁 Files Created

### Main Implementation
- **`learning/natural_language_predictor.py`** (2,200+ lines)
  - `NaturalLanguageParameterPredictor` - Main predictor class
  - `StyleDatabase` - Storage for text→parameter mappings
  - `ConceptExtractor` - Extracts musical concepts from text
  - `ParameterValidator` - Validates and fills defaults
  - `MusicalConcepts` - Data class for extracted concepts
  - `StyleExample` - Data class for example mappings

### Demonstration
- **`examples/agent6_nlp_demo.py`** (400+ lines)
  - 6 comprehensive demonstrations
  - Usage examples
  - Integration workflow

### Documentation
- **`AGENT_6_README.md`** (this file)

## 🎵 Architecture

```
User Input: "Generate a Sinatra-style ballad"
           ↓
    [Concept Extraction]
    - Genre: jazz
    - Era: 1950s
    - Mood: sultry
    - Artists: [Sinatra]
           ↓
   [Style Database Lookup]
    - Find similar examples
    - Retrieve reference parameters
           ↓
    [LLM Parameter Prediction]
    - Build comprehensive prompt
    - Predict ALL 515 parameters
    - Return JSON response
           ↓
   [Validation & Defaults]
    - Validate each parameter
    - Fill missing with defaults
    - Ensure coherence
           ↓
Output: {harmony.voicing.type: "close", rhythm.swing.amount: 0.54, ...}
        (515 parameters ready for generation)
```

## 📊 Style Database

The system includes **13 pre-defined style examples:**

### Jazz Styles
1. **sinatra_ballad** - Sultry Sinatra-style ballad with lush orchestration
2. **sinatra_uptempo** - Swinging uptempo Sinatra arrangement with punchy brass
3. **bebop_fast** - Fast bebop piano with walking bass and brushed drums
4. **basie** - Count Basie style: simple, punchy, riff-based
5. **ellington** - Duke Ellington style: complex, exotic harmonies
6. **thad_jones** - Thad Jones style: modern, angular voicings

### Classical Styles
7. **mozart** - Wolfgang Amadeus Mozart style: elegant, balanced
8. **beethoven** - Ludwig van Beethoven style: dramatic, powerful
9. **romantic_orchestra** - Lush romantic orchestral arrangement

### Other Styles
10. **minimalist_ambient** - Sparse minimalist soundscape
11. **bossa_nova** - Smooth bossa nova with gentle syncopation
12. **chicago_blues** - Electric Chicago blues with shuffle feel
13. **ravi_shankar** - Ravi Shankar style: raga-based, improvisational

## 🚀 Usage

### Basic Usage (with API key)

```python
from midi_generator.learning.natural_language_predictor import predict_from_text

# Simple text-to-parameters conversion
params = predict_from_text("Generate a Sinatra-style ballad")
# Returns dictionary with 515+ parameters
```

### Advanced Usage

```python
from midi_generator.learning.natural_language_predictor import (
    NaturalLanguageParameterPredictor
)

# Initialize with API key
predictor = NaturalLanguageParameterPredictor(api_key="your_key")

# Predict parameters
params = predictor.predict_parameters("Fast bebop piano solo with walking bass")

# Access individual parameters
print(params["rhythm.swing.amount"])  # 0.62
print(params["harmony.voicing.type"])  # "rootless_a"
print(params["bass.style.walking_probability"])  # 0.95
```

### Using Pre-defined Styles

```python
from midi_generator.learning.natural_language_predictor import get_style_database

# Access style database
db = get_style_database()

# Get a specific style
example = db.get_example("sinatra_ballad")
params = example.parameters  # Pre-defined parameter set

# List all available styles
styles = list(db.examples.keys())
print(styles)  # ['basie', 'ellington', 'sinatra_ballad', ...]
```

### Using Musical Concepts

```python
from midi_generator.learning.natural_language_predictor import (
    NaturalLanguageParameterPredictor,
    MusicalConcepts
)

predictor = NaturalLanguageParameterPredictor(api_key="your_key")

# Create concepts manually
concepts = MusicalConcepts(
    genre="jazz",
    era="1950s",
    mood="sultry",
    tempo_descriptor="ballad",
    reference_artists=["Frank Sinatra"],
    instrumentation=["piano", "strings", "brass"]
)

# Predict parameters from concepts
params = predictor.predict_with_concepts(concepts)
```

### Without API Key (Fallback Mode)

```python
from midi_generator.learning.natural_language_predictor import (
    NaturalLanguageParameterPredictor
)

# Initialize without API key (uses keyword-based extraction)
predictor = NaturalLanguageParameterPredictor()

# Basic keyword-based concept extraction
params = predictor.predict_parameters("Sinatra jazz ballad")
# Returns parameters from most similar example in database
```

## 🔧 Features

### 1. Concept Extraction
- **LLM-based extraction**: Uses Claude to extract detailed musical concepts
- **Keyword-based fallback**: Works without API key using pattern matching
- **Extracted concepts**:
  - Genre (jazz, classical, rock, electronic, etc.)
  - Subgenre (bebop, bossa nova, symphony, etc.)
  - Era (1950s, baroque, modern, etc.)
  - Mood (sultry, aggressive, mellow, upbeat)
  - Tempo descriptor (slow, fast, ballad, uptempo)
  - Instrumentation (piano, bass, drums, strings, etc.)
  - Technical terms (walking bass, quartal voicings, brushes)
  - Reference artists (Sinatra, Coltrane, Mozart, etc.)
  - Rhythmic feel (swing, straight, shuffle)
  - Harmonic complexity (simple, moderate, complex)

### 2. Similar Style Matching
- Finds 3 most similar examples from database
- Calculates similarity scores based on:
  - Genre match (weight: 3.0)
  - Era match (weight: 2.0)
  - Mood match (weight: 2.0)
  - Reference artist match (weight: 3.0)
  - Technical term overlap (weight: 1.0 per match)

### 3. LLM Parameter Prediction
- Comprehensive system prompt with:
  - Parameter registry (515+ parameters)
  - Genre knowledge (profile summaries)
  - Few-shot examples (3-5 examples)
  - Prediction principles (specificity, coherence, musicality)
- Temperature 0.3 for consistency
- JSON-only output format
- Error handling and fallback to examples

### 4. Parameter Validation
- Validates ALL parameter values against registry
- Checks type constraints (continuous, categorical, boolean, etc.)
- Validates ranges (min/max for continuous parameters)
- Validates options (for categorical parameters)
- Fills missing parameters with defaults
- Returns comprehensive warnings list

### 5. Logging & Error Handling
- Comprehensive logging at INFO level
- Error recovery with fallback strategies
- Warning messages for:
  - Missing API key
  - Failed concept extraction
  - Invalid parameters
  - Missing parameters

## 🧪 Testing

Run the comprehensive demonstration:

```bash
# Without API key (partial demo)
python3 midi_generator/examples/agent6_nlp_demo.py

# With API key (full demo)
export ANTHROPIC_API_KEY='your_api_key_here'
python3 midi_generator/examples/agent6_nlp_demo.py
```

### Demonstration Includes:
1. **Style Database** - Browse 13 pre-defined styles
2. **Concept Extraction** - Extract concepts from text (requires API key)
3. **Similar Style Matching** - Find similar styles
4. **Full Parameter Prediction** - Complete text-to-parameters (requires API key)
5. **Parameter Validation** - Validate and fill defaults
6. **Integration Workflow** - See how it fits in the system

## 📈 Statistics

- **Module Size**: 2,200+ lines
- **Style Examples**: 13 pre-defined
- **Parameters Predicted**: 515+
- **Concept Fields**: 12
- **Example Parameters per Style**: 8-15
- **Validation Warnings**: Comprehensive error messages
- **Fallback Modes**: 2 (LLM + keyword-based)

## 🔗 Integration with System

### Input to Generation Pipeline
```python
# 1. User provides natural language
description = "Generate a sultry Sinatra ballad"

# 2. Predict parameters
predictor = NaturalLanguageParameterPredictor(api_key="key")
params = predictor.predict_parameters(description)

# 3. Pass to HarmonyModule (or other generator)
from midi_generator.generators.harmony_module import HarmonyModule

generator = HarmonyModule(**params)
midi = generator.generate()
midi.save("output.mid")
```

### Part of Self-Expanding System
```
User Input → Natural Language Predictor → Parameters → Generator → MIDI
                                                                      ↓
                                                           [If unsatisfactory]
                                                                      ↓
                                          Feature Extractor → Gap Detection
                                                                      ↓
                                          LLM Proposes New Parameters
                                                                      ↓
                                          Code Generation → Training
                                                                      ↓
                                          System Expands Automatically
```

## 🎯 Key Innovations

1. **Hybrid Approach**: Combines LLM intelligence with knowledge-based fallback
2. **Comprehensive**: Predicts ALL 515+ parameters from single text description
3. **Musical Understanding**: Knows relationships between concepts (e.g., Sinatra → swing + close voicings)
4. **Self-Contained**: Works independently but integrates seamlessly
5. **Robust**: Multiple fallback strategies ensure it always returns valid parameters
6. **Extensible**: Easy to add new styles to database

## 📝 Future Enhancements

Potential improvements for future agents:

1. **Style Learning**: Learn new styles from MIDI analysis
2. **User Feedback**: Incorporate user preferences over time
3. **Multi-Style Fusion**: Combine multiple styles (e.g., "Sinatra meets Coltrane")
4. **Dynamic Expansion**: Automatically add successful parameter sets to database
5. **Context-Aware**: Remember previous generations in session
6. **Interactive Refinement**: Allow user to tweak concepts iteratively

## 🏆 Success Metrics

- ✅ **13 style examples** loaded successfully
- ✅ **2,200+ lines** of production-quality code
- ✅ **Comprehensive validation** with detailed warnings
- ✅ **Dual-mode operation** (LLM + keyword-based)
- ✅ **Full integration** with UniversalParameterRegistry
- ✅ **Extensive documentation** and examples
- ✅ **6 demonstrations** covering all features

## 👨‍💻 Author

**Agent 6 - Natural Language Parameter Predictor**
Part of: 35-Agent Musical Program Synthesis System
Date: 2025-11-20

## 📄 License

MIT License (inherits from project)

---

**Status**: ✅ **COMPLETE** - Agent 6 successfully implements comprehensive natural language parameter prediction with 2,200+ lines of code, 13 style examples, and full integration with the parameter registry.
