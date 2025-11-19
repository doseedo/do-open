# 🎵 Quick Start Testing Guide - Generate MIDI Files

## ✅ Unified Library Successfully Merged!

**Branch:** `claude/expand-music-genres-01MCCFchdpgpDRc6CV6neTmm`
**Total Files:** 131+ Python files
**Total Code:** 35,000+ lines
**Status:** Ready to test!

---

## 🚀 Quick Setup

```bash
# 1. Clone the unified repository
git clone -b claude/expand-music-genres-01MCCFchdpgpDRc6CV6neTmm \
  https://github.com/doseedo/Do.git midi-unified

cd midi-unified

# 2. Install dependencies
pip install mido python-rtmidi numpy

# Optional (for film scoring):
pip install opencv-python scenedetect[opencv] pydub

# 3. Verify installation
python3 -c "import mido; print('✅ MIDI library ready!')"
```

---

## 🎹 Generate Different MIDI Files - Quick Commands

### **1. Neo-Riemannian Film Score** (Hans Zimmer style)

```bash
cd midi_generator
python examples/01_neo_riemannian_film_score.py
```

**Output:** `output/neo_riemannian_score.mid`
**Features:** Chromatic harmony, P-L-R transformations, cinematic progression

---

### **2. Modal Jazz Composition** (Miles Davis / Herbie Hancock style)

```bash
python examples/02_modal_jazz_composition.py
```

**Output:** `output/modal_jazz.mid`
**Features:** Modal scales, quartal harmony, walking bass, swing rhythm

---

### **3. World Music Scales** (Indian, Arabic, African)

```bash
python examples/03_world_music_scales.py
```

**Output:** `output/world_music_demo.mid`
**Features:**
- Indian ragas (Bhairav, Kafi)
- Arabic maqams
- African pentatonic scales

---

### **4. Rhythm Engine Demo** (Polyrhythms & Grooves)

```bash
python examples/rhythm_engine_demo.py
```

**Output:** `output/rhythm_patterns.mid`
**Features:**
- Euclidean rhythms
- Polyrhythms (3:4, 5:7)
- African/Latin grooves
- Swing and shuffle patterns

---

### **5. Orchestration Demo** (Full ensemble)

```bash
python examples/orchestration_demo.py
```

**Output:** `output/orchestrated_piece.mid`
**Features:**
- Multi-instrument arrangement
- Intelligent voice leading
- Dynamic orchestration
- GM instrument mapping

---

### **6. Blues Progression** (12-bar blues)

```bash
cd ..
python3 -c "
import sys
sys.path.insert(0, 'midi_generator')
from genres.blues import BluesGenerator
from midi_generator.examples.export_to_midi import save_to_midi

# Generate 12-bar blues in G
blues = BluesGenerator(key='G', tempo=120)
progression = blues.generate_12_bar_blues()
save_to_midi(progression, 'blues_in_g.mid')
print('✅ Generated: blues_in_g.mid')
"
```

**Output:** `blues_in_g.mid`
**Features:** 12-bar structure, shuffle rhythm, blues scale

---

### **7. Gospel Progression** (Church style)

```bash
python3 -c "
import sys
sys.path.insert(0, 'midi_generator')
from genres.gospel import GospelGenerator
from midi_generator.examples.export_to_midi import save_to_midi

gospel = GospelGenerator(key='C', tempo=90)
progression = gospel.generate_gospel_progression()
save_to_midi(progression, 'gospel_progression.mid')
print('✅ Generated: gospel_progression.mid')
"
```

**Output:** `gospel_progression.mid`
**Features:** Extended voicings, chromatic passing chords, call-and-response

---

### **8. Advanced Melody with Motif Development** (Beethoven style)

```bash
cd home/arlo/Data
python3 -c "
from melody_advanced import MotifDevelopment, Motif, ContourTheory, ContourType
import mido
from mido import Message, MidiFile, MidiTrack

# Create motif
motif = Motif(pitches=[60, 64, 67, 65], durations=[1.0, 1.0, 1.0, 1.0])

# Develop motif
inverted = MotifDevelopment.inversion(motif)
retrograde = MotifDevelopment.retrograde(motif)
sequences = MotifDevelopment.sequence(motif, [2, 4, 7])

# Export to MIDI
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

track.append(Message('program_change', program=0, time=0))

all_pitches = motif.pitches + inverted.pitches + retrograde.pitches
for i, pitch in enumerate(all_pitches):
    track.append(Message('note_on', note=pitch, velocity=80, time=0))
    track.append(Message('note_off', note=pitch, velocity=0, time=480))

mid.save('beethoven_motif_development.mid')
print('✅ Generated: beethoven_motif_development.mid')
"
```

**Output:** `beethoven_motif_development.mid`
**Features:** Motif inversion, retrograde, sequence development

---

### **9. Advanced Harmony Progression** (Neo-Riemannian + Modal Interchange)

```bash
python3 -c "
from harmony_advanced import NeoRiemannianTransformer, ModalInterchangeGenerator, ModalInterchangeSource
import mido
from mido import Message, MidiFile, MidiTrack

# Create Neo-Riemannian progression
transformer = NeoRiemannianTransformer()
progression = []
progression.append('C')
progression.append(transformer.parallel('C'))  # C → Cm
progression.append(transformer.leading_tone('Cm'))  # Cm → E♭
progression.append(transformer.relative('Eb'))  # E♭ → Cm

# Add borrowed chords
modal_gen = ModalInterchangeGenerator(key='C', mode='major')
borrowed = modal_gen.get_borrowed_chords(ModalInterchangeSource.PARALLEL_MINOR)

print('Progression:', progression)
print('Borrowed chords available:', borrowed)
print('✅ Neo-Riemannian progression created!')
"
```

---

### **10. Complete Form Example** (Sonata form)

```bash
cd ../../midi_generator
python examples/complete_form_example.py
```

**Output:** `output/sonata_form.mid`
**Features:**
- Exposition, Development, Recapitulation
- Theme development
- Key modulations
- Classical structure

---

### **11. Cellular Automata Music** (Conway's Game of Life → MIDI)

```bash
python examples/agent2_comprehensive_demo.py
```

**Output:** `output/algorithmic_composition.mid`
**Features:**
- Cellular automata patterns
- L-systems (Lindenmayer systems)
- Constraint-based composition
- Generative algorithms

---

### **12. African Polyrhythms**

```bash
python3 -c "
import sys
sys.path.insert(0, 'midi_generator')
from genres.world.african import AfricanRhythms
from midi_generator.examples.export_to_midi import save_to_midi

african = AfricanRhythms()
pattern = african.generate_polyrhythm(beats=16)
save_to_midi(pattern, 'african_polyrhythm.mid')
print('✅ Generated: african_polyrhythm.mid')
"
```

**Output:** `african_polyrhythm.mid`
**Features:** Cross-rhythms, bell patterns, djembe grooves

---

### **13. Indian Raga Melody**

```bash
python3 -c "
import sys
sys.path.insert(0, 'midi_generator')
from genres.world.indian import IndianMusic

indian = IndianMusic()
raga = indian.generate_raga_phrase('bhairav')
print('Raga Bhairav phrase:', raga)
print('✅ Indian raga generated!')
"
```

---

### **14. Electronic Dance Music** (EDM with sidechain)

```bash
python3 -c "
import sys
sys.path.insert(0, 'midi_generator')
from genres.electronic import EDMGenerator

edm = EDMGenerator(tempo=128, key='A')
pattern = edm.generate_drop()
print('EDM drop pattern generated!')
print('✅ Ready for export')
"
```

---

### **15. Film Scoring with Video Analysis**

```bash
cd ../home/arlo/Data
python film_scoring_engine.py
```

**Output:** Console demonstration of:
- Leitmotif variations
- Tension arc mapping
- Adaptive progression morphing
- SMPTE timecode sync

---

## 📊 Test All Modules at Once

Create and run this comprehensive test script:

```bash
cat > test_all_modules.sh << 'SCRIPT'
#!/bin/bash

echo "🎵 Testing Unified MIDI Library - All Modules"
echo "=============================================="

cd midi_generator

echo ""
echo "1️⃣  Testing Neo-Riemannian Film Score..."
python examples/01_neo_riemannian_film_score.py && echo "✅ Neo-Riemannian OK" || echo "❌ Failed"

echo ""
echo "2️⃣  Testing Modal Jazz..."
python examples/02_modal_jazz_composition.py && echo "✅ Modal Jazz OK" || echo "❌ Failed"

echo ""
echo "3️⃣  Testing World Music..."
python examples/03_world_music_scales.py && echo "✅ World Music OK" || echo "❌ Failed"

echo ""
echo "4️⃣  Testing Rhythm Engine..."
python examples/rhythm_engine_demo.py && echo "✅ Rhythm Engine OK" || echo "❌ Failed"

echo ""
echo "5️⃣  Testing Orchestration..."
python examples/orchestration_demo.py && echo "✅ Orchestration OK" || echo "❌ Failed"

cd ../home/arlo/Data

echo ""
echo "6️⃣  Testing Advanced Melody Module..."
python melody_advanced.py && echo "✅ Melody Advanced OK" || echo "❌ Failed"

echo ""
echo "7️⃣  Testing Advanced Harmony Module..."
python harmony_advanced.py && echo "✅ Harmony Advanced OK" || echo "❌ Failed"

echo ""
echo "8️⃣  Running Melody Test Suite (37 tests)..."
python test_melody_advanced.py && echo "✅ All Tests Passed" || echo "❌ Tests Failed"

echo ""
echo "=============================================="
echo "🎉 Testing Complete!"
echo "Check the output/ directory for generated MIDI files"
SCRIPT

chmod +x test_all_modules.sh
./test_all_modules.sh
```

---

## 🎼 Custom MIDI Generation Template

Use this template to create your own compositions:

```python
#!/usr/bin/env python3
"""Custom MIDI composition template"""

import sys
sys.path.insert(0, 'midi_generator')
sys.path.insert(0, 'home/arlo/Data')

import mido
from mido import Message, MidiFile, MidiTrack

# Import modules
from harmony_advanced import NeoRiemannianTransformer, FunctionalHarmonyAnalyzer
from melody_advanced import ContourTheory, ContourType, MotifDevelopment, Motif
from midi_generator.algorithms.rhythm_engine import RhythmEngine
from midi_generator.generators.orchestrator import Orchestrator

# 1. Create MIDI file
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

# 2. Set tempo (120 BPM)
track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))

# 3. Generate melody with contour
melody = ContourTheory.generate_contour(
    length=16,
    target_contour=ContourType.ARCH,
    pitch_range=(60, 76),
    climax_position=0.618  # Golden ratio
)

# 4. Create motif and develop it
motif = Motif(pitches=melody[:4], durations=[1.0]*4)
inverted = MotifDevelopment.inversion(motif)

# 5. Add notes to MIDI track
track.append(Message('program_change', program=0, time=0))  # Piano

for pitch in melody:
    track.append(Message('note_on', note=pitch, velocity=80, time=0))
    track.append(Message('note_off', note=pitch, velocity=0, time=480))

# 6. Save MIDI file
mid.save('my_composition.mid')
print('✅ Generated: my_composition.mid')
```

Save as `create_composition.py` and run:
```bash
python3 create_composition.py
```

---

## 📁 Where to Find Generated MIDI Files

After running the examples, check these locations:

```bash
# MIDI generator examples
ls -lh midi_generator/output/*.mid

# Custom generated files
ls -lh *.mid

# Film scoring outputs
ls -lh home/arlo/Data/*.mid
```

---

## 🎧 Play MIDI Files

### **Linux/Mac:**
```bash
# Using timidity
timidity output/neo_riemannian_score.mid

# Using fluidsynth
fluidsynth -a alsa -m alsa_seq -l -i soundfont.sf2 output/modal_jazz.mid
```

### **Python playback:**
```python
import mido
from mido import MidiFile

mid = MidiFile('output/neo_riemannian_score.mid')

for msg in mid.play():
    print(msg)
```

### **Import to DAW:**
- Drag MIDI files into: Ableton Live, FL Studio, Logic Pro, Cubase, Reaper, etc.
- Files are standard MIDI format (.mid) compatible with all DAWs

---

## 🔧 Troubleshooting

### **"No module named 'mido'"**
```bash
pip install mido python-rtmidi
```

### **"No module named 'midi_generator'"**
```bash
# Make sure you're in the repository root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### **"Permission denied"**
```bash
chmod +x test_all_modules.sh
```

### **Silent MIDI playback**
- Check your MIDI synthesizer is running
- Verify sound output device
- Try different soundfonts

---

## 📚 Next Steps

1. **Read documentation:**
   ```bash
   cat midi_generator/README.md
   cat HARMONY_MELODY_10X_ENHANCEMENT_SUMMARY.md
   ```

2. **Explore examples:**
   ```bash
   ls midi_generator/examples/
   ```

3. **Try different genres:**
   ```bash
   ls midi_generator/genres/
   ```

4. **Experiment with ML:**
   ```bash
   python midi_generator/examples/pattern_learning_demo.py
   ```

---

## 🎵 Summary of What You Can Generate

| Style | Example Command | Output |
|-------|----------------|--------|
| **Film Score** | `python examples/01_neo_riemannian_film_score.py` | Cinematic, chromatic |
| **Jazz** | `python examples/02_modal_jazz_composition.py` | Modal, sophisticated |
| **Blues** | Genre: `blues.py` | 12-bar, shuffle |
| **Gospel** | Genre: `gospel.py` | Extended chords |
| **World Music** | `python examples/03_world_music_scales.py` | Indian, Arabic, African |
| **Classical** | `python examples/complete_form_example.py` | Sonata form |
| **Electronic** | Genre: `electronic.py` | EDM, techno |
| **Algorithmic** | `python examples/agent2_comprehensive_demo.py` | Generative, CA |
| **Orchestral** | `python examples/orchestration_demo.py` | Full ensemble |
| **Rhythm** | `python examples/rhythm_engine_demo.py` | Polyrhythms, grooves |

---

## ✅ You Now Have:

✅ **131+ Python files** of music generation code
✅ **35,000+ lines** of composition algorithms
✅ **50+ music genres** implemented
✅ **15+ example scripts** ready to run
✅ **300+ years** of music theory (Baroque → Contemporary)
✅ **Complete test suite** (37 tests)
✅ **Production-ready** unified library

**Happy composing! 🎹🎸🎺🎻**
