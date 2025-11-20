# Migration Guide - MIDI Generator v2.0

## Overview

Version 2.0 unifies two previously separate libraries:
- `harmonymodule/midi_generator` (advanced features)
- `midi_generator` (standalone version)

## For Standalone Users

### Import Changes

Most imports remain the same:

```python
# Still works:
from midi_generator.core import modal_harmony
from midi_generator.generators import AdvancedHarmonyGenerator

# New features now available:
from midi_generator.api import UnifiedMusicGenerator
from midi_generator.generators import StyleFusion
from midi_generator.analysis import GenreDetector
```

### Big Band Generators

Scripts moved to `tools/big_band/`:

```bash
# OLD
python generate_big_band_final.py swing 140 0 jazz_blues

# NEW
python tools/big_band/generate_final.py swing 140 0 jazz_blues
```

### New Features

You now have access to:
- Component-based composition system
- Style fusion (blend genres)
- Context-aware generation
- Genre detection
- MIDI inpainting
- 20+ specialized advanced modules

## For Harmonymodule Users

### Import Path Changes

```python
# OLD
from harmonymodule.midi_generator.core import modal_harmony

# NEW
from midi_generator.core import modal_harmony
```

### New Features

Big band generators are now included:

```python
# Now available:
from midi_generator.tools.big_band import generate_final
```

## Breaking Changes

### None!

All existing code should work with minimal changes. The unification is designed to be backwards compatible.

## New Capabilities

### 1. Unified API

```python
from midi_generator.api import UnifiedMusicGenerator

gen = UnifiedMusicGenerator()
result = gen.generate(genre="jazz", style="bebop", length=32)
```

### 2. Style Fusion

```python
from midi_generator.generators import StyleFusion

fusion = StyleFusion()
result = fusion.blend_genres({
    'jazz': 0.6,
    'funk': 0.4
})
```

### 3. Context-Aware Generation

```python
# Add a bass line that fits existing arrangement
new_bass = gen.add_contextual_track(
    existing_midi="arrangement.mid",
    genre="jazz",
    role="bass"
)
```

### 4. Genre Detection

```python
from midi_generator.analysis import GenreDetector

detector = GenreDetector()
genre, confidence = detector.detect("unknown.mid")
```

## File Locations

### Common Files
All 65 common files are in the same locations.

### Harmonymodule Unique Features
Now integrated at root level:
- `core/component_system.py`
- `generators/style_fusion.py`
- `generators/context_aware_generator.py`
- `analysis/genre_detector.py`
- And 23 more...

### Standalone Unique Features
- `tools/big_band/*.py` - Big band generators
- `tools/examples/` - Additional examples

## Recommendations

1. **Update imports** - Change harmonymodule paths to midi_generator
2. **Update scripts** - Big band generators moved to tools/big_band/
3. **Explore new features** - Try style fusion and context-aware generation
4. **Read documentation** - Check docs/ for comprehensive guides

## Support

If you encounter issues during migration:
1. Check this guide
2. Review documentation in docs/
3. See examples in tools/examples/
4. File an issue with details

## Timeline

- v1.x: Separate libraries
- v2.0: Unified library (current)
- v2.1+: Continued enhancements
