# MIDI Generator - Unified Music Generation Library

A comprehensive, production-ready music generation framework combining advanced music theory, algorithmic composition, and genre-specific implementations.

## Features

### Core Capabilities
- **35+ Music Genres**: Jazz, blues, funk, electronic, rock, world music, and more
- **Advanced Harmony**: Modal systems, neo-Riemannian theory, microtonal scales
- **Algorithmic Composition**: L-systems, cellular automata, constraint solving
- **Professional Orchestration**: Intelligent instrument selection and voice leading
- **Style Fusion**: Blend any components from any genre
- **Context-Aware Generation**: Add tracks to existing MIDI files
- **Genre Detection**: Automatic MIDI genre classification
- **Pattern Learning**: Extract patterns from MIDI corpus

### Music Theory
- **Modal Harmony**: 21 modal scales with progression generators
- **Neo-Riemannian**: PLR transformations, hexatonic cycles, voice leading
- **Microtonality**: 24-TET, 53-TET, just intonation, world music scales
- **Form Generation**: Sonata, rondo, fugue, AABA, blues, verse-chorus
- **Voice Leading**: Optimization and smooth transitions

### Genre Implementations
Western: blues, jazz, pop, rock, country, reggae, gospel, electronic, funk, hip-hop, metal, R&B, singer-songwriter

World: African, Arabic (maqam), Indian (raga), Turkish (makam), Persian (dastgah)

## Installation

```bash
# Install dependencies
pip install mido numpy

# Add to Python path
export PYTHONPATH="/path/to/Do:$PYTHONPATH"
```

## Quick Start

### Basic Usage

```python
from midi_generator.generators import AdvancedHarmonyGenerator
from midi_generator.core import Mode

# Generate modal progression
gen = AdvancedHarmonyGenerator(root=0, octave=4)
progression = gen.generate_modal_progression(
    mode=Mode.DORIAN,
    progression_type="vamp",
    length=8
)
```

### Big Band Generator

```bash
# Production-ready big band arrangement
python tools/big_band/generate_final.py swing 140 0 jazz_blues

# With advanced harmony (31+ progression types)
python tools/big_band/generate_comprehensive.py modal 140 0 dorian_vamp
python tools/big_band/generate_comprehensive.py coltrane 180 0 coltrane_changes
```

### High-Level API

```python
from midi_generator.api import UnifiedMusicGenerator

# Context-aware generation
generator = UnifiedMusicGenerator()
new_track = generator.add_contextual_track(
    existing_midi="input.mid",
    genre="jazz",
    role="bass"
)
```

### Style Fusion

```python
from midi_generator.generators import StyleFusion

# Blend jazz harmony + funk rhythm + electronic instrumentation
fusion = StyleFusion()
result = fusion.blend_genres({
    'jazz': 0.5,      # 50% jazz characteristics
    'funk': 0.3,      # 30% funk
    'electronic': 0.2 # 20% electronic
})
```

## Directory Structure

```
midi_generator/
├── core/                   # Music theory foundations
├── algorithms/             # Composition algorithms
├── generators/             # Content generators
├── genres/                 # Genre implementations
├── analysis/               # MIDI analysis
├── transformation/         # Style transfer
├── tools/                  # Production tools
│   ├── big_band/          # Big band generators
│   └── examples/          # Working examples
├── docs/                   # Documentation
└── tests/                  # Test suite
```

## Documentation

- [Harmony System Guide](docs/HARMONY_ANALYSIS.md) - 31+ progression types
- [Big Band Generator Guide](tools/big_band/README.md) - Big band tools
- [Form Integration](docs/FORM_MODULE_INTEGRATION.md) - Musical forms
- [API Reference](docs/) - Complete API documentation

## Advanced Features

### Neo-Riemannian Transformations

```python
from midi_generator.core import Triad, TriadQuality
from midi_generator.generators import AdvancedHarmonyGenerator

gen = AdvancedHarmonyGenerator(root=0)
# PLR transformation progression (film scoring)
progression = gen.generate_neo_riemannian("P L R P", voice_lead=True)
```

### Microtonal Systems

```python
# Arabic maqam with quarter tones
maqam = gen.generate_arabic_maqam(ArabicMaqam.RAST)

# Indian raga
raga = gen.generate_indian_raga("Bhairav", ascending=True)
```

### Pattern Learning

```python
from midi_generator.learning import CorpusLearner

learner = CorpusLearner()
learner.learn_from_directory("midi_corpus/")
new_melody = learner.generate_similar_melody()
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Test specific module
python -m pytest tests/test_core/
```

## Examples

See `tools/examples/` for comprehensive demonstrations:
- Context-aware generation
- Style fusion
- Genre detection
- Modal jazz composition
- Film scoring
- And more...

## Version History

- **2.0.0** - Unified library (merged harmonymodule + standalone)
  - Combined all features into one comprehensive system
  - Added big band generators
  - Unified API and documentation

## License

MIT License - See LICENSE file for details

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for development guidelines.
