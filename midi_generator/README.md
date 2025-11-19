# MIDI Generator Library

**The Ultimate MIDI Generation and Manipulation Framework**

A comprehensive, professional-grade Python library for algorithmic MIDI composition, analysis, and transformation. Built by the Dø (Doseedo) AI Music Platform.

## Features

### 🎵 Core Capabilities
- **Complete Music Theory** - Scales, chords, voice leading, Neo-Riemannian transformations
- **Advanced Algorithms** - Markov chains, genetic algorithms, L-systems, cellular automata
- **Genre Templates** - Jazz, Classical, Rock, EDM, World Music, and more
- **Professional Tools** - Orchestration, arrangement, harmonic analysis

### 🛠️ Utilities
- **MIDI I/O** - Robust reading/writing with multi-library support
- **Visualization** - Piano rolls, histograms, rhythm grids, ASCII art
- **Export** - JSON, MusicXML, ABC notation, batch processing
- **Analysis** - Comprehensive musical feature extraction

### 🎹 Integration
- Seamlessly integrates with existing `harmonymodule` code
- Compatible with DAWs, notation software, and music applications
- Extensible architecture for custom generators and algorithms

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/doseedo/Do.git
cd Do/midi_generator

# Install dependencies
pip install mido matplotlib

# Optional dependencies
pip install music21  # For advanced analysis
```

### Basic Usage

```python
from midi_generator.utils.midi_io import MIDINote, save_midi

# Create a simple melody
notes = [
    MIDINote(start=0.0, duration=1.0, pitch=60, velocity=80),  # C
    MIDINote(start=1.0, duration=1.0, pitch=64, velocity=80),  # E
    MIDINote(start=2.0, duration=1.0, pitch=67, velocity=80),  # G
]

# Save to MIDI file
save_midi(notes, 'melody.mid', tempo=120)
```

### Command Line Interface

```bash
# Get info about a MIDI file
python -m midi_generator.cli info song.mid

# Visualize as piano roll
python -m midi_generator.cli visualize song.mid --type piano-roll

# Analyze MIDI file
python -m midi_generator.cli analyze song.mid --visualize

# Export to MusicXML
python -m midi_generator.cli export song.mid --format musicxml

# Generate simple melody
python -m midi_generator.cli generate --output melody.mid --tempo 140
```

## Architecture

```
midi_generator/
├── core/              # Music theory (scales, chords, voice leading)
├── algorithms/        # Composition algorithms (Markov, genetic, L-systems)
├── generators/        # Content generators (melody, harmony, bass, drums)
├── genres/            # Genre-specific implementations
├── midi/              # MIDI utilities (CC automation, articulation, MPE)
├── learning/          # Pattern extraction and ML
├── transformation/    # Style transfer and variation
├── analysis/          # MIDI analysis tools
├── utils/             # I/O, visualization, export
├── tests/             # Comprehensive test suite
├── examples/          # 25+ working examples
├── gui/               # Optional GUI interface
├── docs/              # Complete documentation
└── cli.py             # Command-line interface
```

## Examples

The library includes 25+ comprehensive examples:

### Beginner (5 examples)
- Simple melody generation
- Chord progressions
- Rhythm patterns
- Visualization demos
- Export formats

### Genre-Specific (10 examples)
- Jazz progressions
- Blues patterns
- Rock arrangements
- Classical counterpoint
- EDM productions
- And more...

### Advanced (10 examples)
- Algorithmic composition
- Style transfer
- MIDI analysis
- Multi-track arrangements
- Microtonal music

See `examples/README.md` for the complete list.

## Documentation

- **[Tutorial](docs/TUTORIAL.md)** - Step-by-step learning guide
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[Music Theory Guide](docs/MUSIC_THEORY.md)** - Music theory concepts
- **[Examples Guide](examples/README.md)** - All example files

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python midi_generator/tests/test_all.py

# Or with pytest
pip install pytest
pytest midi_generator/tests/test_all.py -v
```

Test coverage includes:
- ✅ MIDI I/O operations
- ✅ Visualization functions
- ✅ Export to all formats
- ✅ Integration workflows
- ✅ Edge cases and error handling

## Integration with Harmony Module

The library seamlessly integrates with the existing `harmonymodule`:

```python
# Use existing chord progression generator
from harmonymodule.chord_progression_generator import (
    generate_chord_progression_midi, ScaleContext
)

# Generate jazz progression
chord_map = {0: 'Dm9', 4: 'G9', 8: 'Cmaj9'}
midi_path = generate_chord_progression_midi(
    chord_beat_map=chord_map,
    bpm=140,
    voicing='drop2',
    output_path='jazz.mid'
)
```

## Key Components

### MIDI I/O
```python
from midi_generator.utils.midi_io import load_midi, save_midi, MIDINote

# Load MIDI
midi = load_midi('song.mid')
print(f"Found {len(midi.notes)} notes at {midi.tempo} BPM")

# Create and save
notes = [MIDINote(start=0, duration=1, pitch=60, velocity=80)]
save_midi(notes, 'output.mid', tempo=120)
```

### Visualization
```python
from midi_generator.utils.visualization import visualize_piano_roll

# Create piano roll
visualize_piano_roll('song.mid', 'piano_roll.png')
```

### Export
```python
from midi_generator.utils.export import batch_export

# Export to all formats
batch_export('song.mid', 'exports/', formats=['json', 'musicxml', 'abc'])
```

## Requirements

### Core Dependencies
- Python 3.7+
- mido - MIDI file I/O
- matplotlib - Visualization

### Optional Dependencies
- music21 - Advanced music analysis
- numpy - Numerical operations
- pytest - Testing

## Performance

- **Fast I/O**: Efficient MIDI reading/writing with minimal overhead
- **Optimized**: Smart caching and batch processing
- **Scalable**: Handles files with thousands of notes
- **Memory-efficient**: Stream processing for large files

## Contributing

This library is part of the Dø AI Music Platform. Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Credits

**Developer**: Dø (Doseedo) AI Music Platform
**Agent 10**: Integration, Testing & Examples
**Built on**: Existing harmonymodule foundation

## Links

- **Repository**: https://github.com/doseedo/Do
- **Documentation**: docs/
- **Examples**: examples/
- **Issues**: https://github.com/doseedo/Do/issues

## Version

**v1.0.0** - Initial release with comprehensive features

---

**Make Beautiful Music with Code** 🎵

