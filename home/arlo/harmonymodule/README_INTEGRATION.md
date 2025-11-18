# Integration Guide - HarmonyModule Library

## 📍 Directory Structure

```
home/arlo/harmonymodule/
├── README.md                    # Main documentation
├── inference/                   # Existing production pipeline
├── midi_generator/              # 10-agent MIDI generation system (71 files)
├── scripts/                     # Production utilities (19 files)
├── advanced_modules/            # Graduate-level modules (8 files)
└── docs/                        # Documentation (3 files)
```

---

## 🔧 Setting Up Import Paths

### **Option 1: Using sys.path (Recommended for Scripts)**

Add this to the beginning of your Python scripts:

```python
import sys
import os

# Get absolute path to harmonymodule directory
HARMONYMODULE_PATH = os.path.abspath('home/arlo/harmonymodule')

# Add all module paths
sys.path.insert(0, os.path.join(HARMONYMODULE_PATH, 'midi_generator'))
sys.path.insert(0, os.path.join(HARMONYMODULE_PATH, 'advanced_modules'))
sys.path.insert(0, os.path.join(HARMONYMODULE_PATH, 'scripts'))
```

### **Option 2: Environment Variable (Recommended for Production)**

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export PYTHONPATH="${PYTHONPATH}:/full/path/to/home/arlo/harmonymodule/midi_generator"
export PYTHONPATH="${PYTHONPATH}:/full/path/to/home/arlo/harmonymodule/advanced_modules"
export PYTHONPATH="${PYTHONPATH}:/full/path/to/home/arlo/harmonymodule/scripts"
```

Then in Python:

```python
# No sys.path needed - environment handles it!
from harmony_advanced import NeoRiemannianTransformer
from genres.blues import BluesGenerator
from arrange import AdvancedArranger
```

### **Option 3: Working Directory (Simplest)**

Change to the module directory before importing:

```bash
cd home/arlo/harmonymodule/advanced_modules
python
```

```python
# Now you can import directly
from melody_advanced import ContourTheory
```

---

## 🎹 Module Import Patterns

### **1. Advanced Modules** (Graduate-Level Theory)

**Location:** `home/arlo/harmonymodule/advanced_modules/`

```python
import sys
sys.path.insert(0, 'home/arlo/harmonymodule/advanced_modules')

# Harmony
from harmony_advanced import (
    VoiceLeadingAnalyzer,
    NeoRiemannianTransformer,
    ModalInterchangeGenerator,
    AdvancedSubstitutions,
    QuartalQuintalGenerator,
    FunctionalHarmonyAnalyzer,
    ConstraintBasedHarmonicGenerator
)

# Melody
from melody_advanced import (
    ContourTheory,
    ContourType,
    MotifDevelopment,
    Motif,
    PhraseStructure,
    IntervallicControl,
    Ornamentation,
    MusicalNarrative
)

# Film Scoring
from film_scoring_engine import (
    VideoAnalyzer,
    FilmScoringTechniques,
    LeitmotifEngine,
    TensionArc
)
```

---

### **2. MIDI Generator System** (10-Agent System)

**Location:** `home/arlo/harmonymodule/midi_generator/`

```python
import sys
sys.path.insert(0, 'home/arlo/harmonymodule/midi_generator')

# Rhythm & Algorithms
from algorithms.rhythm_engine import RhythmEngine
from algorithms.lsystem import LSystem
from algorithms.cellular_automata import CellularAutomata
from algorithms.constraint_solver import ConstraintSolver

# Core Theory
from core.neo_riemannian import NeoRiemannian
from core.modal_harmony import ModalHarmony
from core.microtonality import Microtonality

# Generators
from generators.orchestrator import Orchestrator
from generators.form_generator import FormGenerator
from generators.development_engine import DevelopmentEngine

# Genres
from genres.blues import BluesGenerator
from genres.gospel import GospelGenerator
from genres.reggae import ReggaeGenerator
from genres.electronic import EDMGenerator
from genres.world.african import AfricanRhythms
from genres.world.indian import IndianMusic
from genres.world.arabic import ArabicMusic

# Learning & Analysis
from learning.pattern_extractor import PatternExtractor
from learning.corpus_learner import CorpusLearner
from transformation.style_transfer import StyleTransfer
```

---

### **3. Production Scripts** (Utilities)

**Location:** `home/arlo/harmonymodule/scripts/`

```python
import sys
sys.path.insert(0, 'home/arlo/harmonymodule/scripts')

from arrange import AdvancedArranger
from chord_progression_generator import ChordProgressionGenerator
from melody_harmonizer_improved import MelodyHarmonizer
from chord_audio_extractor import extract_chords_from_audio
from midi_chord_extractor import extract_chords_from_midi
```

---

## 🎯 Integration Examples

### **Example 1: Complete Film Score Workflow**

Combine advanced modules + MIDI generator for a film score:

```python
import sys
sys.path.insert(0, 'home/arlo/harmonymodule/advanced_modules')
sys.path.insert(0, 'home/arlo/harmonymodule/midi_generator')

import mido
from mido import Message, MidiFile, MidiTrack

# Import film scoring engine
from film_scoring_engine import (
    FilmScoringTechniques,
    LeitmotifEngine,
    TensionArc
)

# Import harmony tools
from harmony_advanced import NeoRiemannianTransformer

# Import MIDI generator
from generators.orchestrator import Orchestrator
from core.neo_riemannian import NeoRiemannian

# Step 1: Create leitmotif
leitmotif_engine = LeitmotifEngine()
hero_theme = leitmotif_engine.add_motif(
    name="hero_theme",
    notes=[60, 64, 67, 72],
    durations=[1.0, 1.0, 1.0, 2.0]
)

# Step 2: Create harmonic progression with Neo-Riemannian
neo = NeoRiemannianTransformer()
progression = neo.generate_plr_sequence("C", ['P', 'L', 'R', 'P'])
# Result: ['C', 'Cm', 'Em', 'C', 'Cm']

# Step 3: Orchestrate
orchestrator = Orchestrator()
orchestrated = orchestrator.orchestrate(progression, ensemble="full_orchestra")

# Step 4: Export to MIDI
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

# Add orchestrated notes
for note_event in orchestrated:
    track.append(Message('note_on', note=note_event['pitch'],
                        velocity=note_event['velocity'], time=0))
    track.append(Message('note_off', note=note_event['pitch'],
                        velocity=0, time=note_event['duration']))

mid.save('film_score_complete.mid')
print("✅ Film score generated: film_score_complete.mid")
```

---

### **Example 2: Jazz Composition with Advanced Harmony**

```python
import sys
sys.path.insert(0, 'home/arlo/harmonymodule/advanced_modules')
sys.path.insert(0, 'home/arlo/harmonymodule/midi_generator')

from harmony_advanced import (
    AdvancedSubstitutions,
    QuartalQuintalGenerator
)
from genres.blues import BluesGenerator
from melody_advanced import ContourTheory, ContourType

# Generate 12-bar blues progression
blues = BluesGenerator(key='Bb', tempo=120)
progression = blues.generate_12_bar_blues()

# Add tritone substitutions
enhanced_progression = []
for chord in progression:
    if chord.endswith('7'):  # Dominant chord
        # Randomly substitute with tritone
        sub = AdvancedSubstitutions.tritone_substitute(chord)
        enhanced_progression.append(sub)
    else:
        enhanced_progression.append(chord)

# Generate melody with arch contour
melody = ContourTheory.generate_contour(
    length=32,
    target_contour=ContourType.ARCH,
    pitch_range=(65, 84),  # Bb3 to C6
    climax_position=0.618
)

print("Jazz progression:", enhanced_progression)
print("Melody:", melody)
```

---

### **Example 3: Algorithmic Composition with Constraints**

```python
import sys
sys.path.insert(0, 'home/arlo/harmonymodule/advanced_modules')
sys.path.insert(0, 'home/arlo/harmonymodule/midi_generator')

from harmony_advanced import (
    ConstraintBasedHarmonicGenerator,
    no_parallel_chord_motion,
    prefer_strong_cadence
)
from algorithms.lsystem import LSystem
from melody_advanced import MotifDevelopment, Motif

# Generate harmonic progression with constraints
generator = ConstraintBasedHarmonicGenerator(key="G", mode="major")
generator.add_constraint(no_parallel_chord_motion, "no_repeats")
generator.add_constraint(prefer_strong_cadence, "strong_cadence")

progression = generator.generate_progression(length=8, end_chord="G")

# Generate melodic material using L-systems
lsystem = LSystem(axiom="A", rules={"A": "AB", "B": "A"}, iterations=4)
pattern = lsystem.generate()

# Create motif and develop it
motif = Motif(pitches=[67, 71, 74], durations=[1.0, 1.0, 1.0])
inverted = MotifDevelopment.inversion(motif)
retrograde = MotifDevelopment.retrograde(motif)

print("Progression:", progression)
print("L-system pattern:", pattern)
print("Motif variations: original, inverted, retrograde")
```

---

### **Example 4: World Music with Modal Interchange**

```python
import sys
sys.path.insert(0, 'home/arlo/harmonymodule/advanced_modules')
sys.path.insert(0, 'home/arlo/harmonymodule/midi_generator')

from harmony_advanced import (
    ModalInterchangeGenerator,
    ModalInterchangeSource
)
from genres.world.indian import IndianMusic
from genres.world.african import AfricanRhythms

# Generate Indian raga melody
indian = IndianMusic()
raga_melody = indian.generate_raga_phrase('bhairav')

# Add borrowed chords from parallel Dorian
modal_gen = ModalInterchangeGenerator(key='D', mode='major')
borrowed = modal_gen.get_borrowed_chords(ModalInterchangeSource.DORIAN)

# Generate African polyrhythm
african = AfricanRhythms()
polyrhythm = african.generate_polyrhythm(beats=16)

print("Raga melody:", raga_melody)
print("Borrowed chords from Dorian:", borrowed)
print("African polyrhythm ready")
```

---

## 📋 Common Import Patterns Cheatsheet

### **Quick Setup (All Modules)**

```python
import sys
import os

# One-time setup for all imports
HARMONYMODULE = os.path.abspath('home/arlo/harmonymodule')
sys.path.extend([
    os.path.join(HARMONYMODULE, 'midi_generator'),
    os.path.join(HARMONYMODULE, 'advanced_modules'),
    os.path.join(HARMONYMODULE, 'scripts')
])

# Now import from any module
from harmony_advanced import NeoRiemannianTransformer
from genres.blues import BluesGenerator
from arrange import AdvancedArranger
```

### **Import Only What You Need**

```python
# Minimal imports for specific task
import sys
sys.path.insert(0, 'home/arlo/harmonymodule/advanced_modules')

from melody_advanced import ContourTheory, ContourType

melody = ContourTheory.generate_contour(8, ContourType.ARCH)
```

---

## 🚨 Common Issues & Solutions

### **Issue 1: ModuleNotFoundError**

**Error:** `ModuleNotFoundError: No module named 'harmony_advanced'`

**Solution:** Add the correct path to sys.path:

```python
import sys
sys.path.insert(0, 'home/arlo/harmonymodule/advanced_modules')
from harmony_advanced import NeoRiemannianTransformer
```

---

### **Issue 2: Relative Import Errors**

**Error:** `ImportError: attempted relative import with no known parent package`

**Solution:** Use absolute imports and set sys.path:

```python
# ❌ DON'T DO THIS:
from ..harmony_advanced import NeoRiemannianTransformer

# ✅ DO THIS:
import sys
sys.path.insert(0, 'home/arlo/harmonymodule/advanced_modules')
from harmony_advanced import NeoRiemannianTransformer
```

---

### **Issue 3: Conflicting Module Names**

**Error:** Importing the wrong module (e.g., old vs. new version)

**Solution:** Use absolute paths and check sys.path order:

```python
import sys

# Insert new paths FIRST (at index 0)
sys.path.insert(0, 'home/arlo/harmonymodule/advanced_modules')

# Verify which module is loaded
import harmony_advanced
print(harmony_advanced.__file__)  # Should show correct path
```

---

## 📚 Testing Your Setup

Run this diagnostic script to verify all imports work:

```python
#!/usr/bin/env python3
"""Test all HarmonyModule imports"""

import sys
import os

# Setup paths
HARMONYMODULE = os.path.abspath('home/arlo/harmonymodule')
sys.path.extend([
    os.path.join(HARMONYMODULE, 'midi_generator'),
    os.path.join(HARMONYMODULE, 'advanced_modules'),
    os.path.join(HARMONYMODULE, 'scripts')
])

print("Testing imports...\n")

# Test advanced modules
try:
    from harmony_advanced import NeoRiemannianTransformer
    print("✅ harmony_advanced")
except ImportError as e:
    print(f"❌ harmony_advanced: {e}")

try:
    from melody_advanced import ContourTheory
    print("✅ melody_advanced")
except ImportError as e:
    print(f"❌ melody_advanced: {e}")

try:
    from film_scoring_engine import LeitmotifEngine
    print("✅ film_scoring_engine")
except ImportError as e:
    print(f"❌ film_scoring_engine: {e}")

# Test MIDI generator
try:
    from genres.blues import BluesGenerator
    print("✅ midi_generator.genres.blues")
except ImportError as e:
    print(f"❌ midi_generator.genres.blues: {e}")

try:
    from algorithms.rhythm_engine import RhythmEngine
    print("✅ midi_generator.algorithms.rhythm_engine")
except ImportError as e:
    print(f"❌ midi_generator.algorithms.rhythm_engine: {e}")

# Test scripts
try:
    from arrange import AdvancedArranger
    print("✅ scripts.arrange")
except ImportError as e:
    print(f"❌ scripts.arrange: {e}")

print("\n✅ Import test complete!")
```

Save as `test_imports.py` and run:

```bash
python test_imports.py
```

---

## 🎵 Next Steps

1. **Read the documentation:**
   - `docs/QUICK_START_TESTING_GUIDE.md` - Generate MIDI files
   - `docs/HARMONY_MELODY_10X_ENHANCEMENT_SUMMARY.md` - Advanced theory
   - `docs/COMPLETE_LIBRARY_SUMMARY.md` - Full library overview

2. **Run the examples:**
   ```bash
   cd home/arlo/harmonymodule/midi_generator/examples
   python 01_neo_riemannian_film_score.py
   ```

3. **Run the tests:**
   ```bash
   cd home/arlo/harmonymodule/advanced_modules
   python test_melody_advanced.py  # 37 tests
   ```

4. **Create your own compositions:**
   - Use the integration examples above as templates
   - Mix and match modules for your specific needs
   - Experiment with different combinations!

---

## ✅ Summary

**Import Paths:**
- Advanced Modules: `home/arlo/harmonymodule/advanced_modules/`
- MIDI Generator: `home/arlo/harmonymodule/midi_generator/`
- Production Scripts: `home/arlo/harmonymodule/scripts/`

**Setup Methods:**
- sys.path.insert() - Best for scripts
- PYTHONPATH - Best for production
- Working directory - Simplest for interactive use

**Integration:**
- All modules work together seamlessly
- Combine advanced theory + MIDI generation + production tools
- See examples above for common workflows

**Happy composing! 🎹🎸🎺🎻**
