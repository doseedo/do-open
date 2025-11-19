# Meter Conversion Engine

**Agent 7 - Modular Fusion Enhancement Project**

A comprehensive system for converting MIDI files between different time signatures while preserving musical content, phrase structure, and artistic intent.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Concepts](#core-concepts)
6. [Conversion Strategies](#conversion-strategies)
7. [API Reference](#api-reference)
8. [Examples](#examples)
9. [Advanced Usage](#advanced-usage)
10. [Technical Details](#technical-details)
11. [Research & References](#research--references)

---

## Overview

The Meter Conversion Engine transforms MIDI files between different time signatures (4/4 ↔ 3/4, 5/4, 7/8, etc.) using intelligent algorithms that preserve:

- **Musical phrases** - Maintains phrase boundaries and structure
- **Harmonic content** - Preserves chord progressions
- **Melodic contour** - Keeps melodic shapes intact
- **Rhythmic feel** - Adapts rhythms appropriately to new meter

### What Makes This Special?

Unlike simple metadata changes, this engine actually **restructures the music** to make sense in the new meter, using techniques from:

- Elliott Carter's metric modulation
- Jazz reharmonization principles
- Classical phrase structure theory
- Progressive rock polymetric techniques

---

## Features

### ✅ Time Signature Conversions

- **Simple meters**: 2/4, 3/4, 4/4
- **Compound meters**: 6/8, 9/8, 12/8
- **Odd meters**: 5/4, 7/8, 11/8, 13/8
- **Custom groupings**: 7/8 as [2+2+3] or [3+2+2]

### ✅ Conversion Strategies

1. **Stretch/Compress** - Proportional time scaling
2. **Redistribute** - Remap notes to new measure structure
3. **Metric Modulation** - Elliott Carter-style pivot rhythms
4. **Phrase-Aware** - Preserves phrase boundaries (default)

### ✅ Advanced Features

- Automatic phrase boundary detection
- Tempo adjustment calculations
- Rhythm quantization options
- Voice leading preservation
- Hypermeter awareness (4/8/16 bar phrases)

---

## Installation

```python
# The module is part of the HarmonyModule library
from midi_generator.transformation.meter_converter import (
    MeterConverter,
    ConversionStrategy,
    MeterConversionParams
)
```

### Dependencies

- `mido` - MIDI file I/O
- `numpy` - Numerical calculations
- Standard library: `pathlib`, `dataclasses`, `fractions`

---

## Quick Start

### Basic Conversion

```python
from meter_converter import MeterConverter

# Load MIDI file
converter = MeterConverter("input.mid")

# Convert 4/4 to 3/4
result = converter.convert_meter(
    new_numerator=3,
    new_denominator=4
)

# Save result
if result.success:
    result.new_midi.save("output.mid")
    print(f"Conversion successful!")
    print(f"Tempo adjustment: {result.tempo_change_factor}x")
```

### Quick Function

```python
from meter_converter import convert_midi_meter

# One-line conversion
convert_midi_meter(
    "input.mid",
    "output.mid",
    new_numerator=7,
    new_denominator=8,
    strategy="phrase_aware"
)
```

---

## Core Concepts

### Time Signature Classification

The engine classifies meters into families:

```python
from meter_converter import TimeSignatureInfo, MeterFamily

# Create time signature
ts = TimeSignatureInfo(7, 8, grouping=[2, 2, 3])

print(ts.family)  # MeterFamily.ODD_METER
print(ts.beats_per_measure)  # 3.5
print(ts.grouping)  # [2, 2, 3]
```

**Meter Families:**

| Family | Examples | Characteristics |
|--------|----------|-----------------|
| Simple Duple | 2/4, 2/2 | Two beats per measure |
| Simple Triple | 3/4, 3/8 | Three beats per measure |
| Simple Quadruple | 4/4, 4/8 | Four beats per measure |
| Compound Duple | 6/8, 6/4 | Two dotted beats |
| Compound Triple | 9/8, 9/4 | Three dotted beats |
| Odd Meter | 5/4, 7/8, 11/8 | Irregular groupings |

### Phrase Boundaries

The engine detects phrase boundaries using:

1. **Long rests** - Silences likely indicate phrase endings
2. **Melodic contour** - Changes in melodic direction
3. **Harmonic rhythm** - Chord change patterns
4. **Hypermeter** - Regular 4/8/16 measure phrases

```python
# Phrase boundaries are detected automatically
converter = MeterConverter("song.mid")
converter._detect_phrase_boundaries()

for boundary in converter.phrase_boundaries:
    print(f"Phrase boundary at measure {boundary.measure_number}")
    print(f"  Type: {boundary.boundary_type}")
    print(f"  Confidence: {boundary.confidence}")
```

### Metric Modulation

Based on Elliott Carter's technique, metric modulation creates smooth tempo transitions by finding a "pivot rhythm" that exists in both meters.

**Formula:**
```
new_tempo / old_tempo = pivot_notes_new / pivot_notes_old
```

**Example:** 4/4 @ 120 BPM → 3/4 @ 90 BPM
- Pivot: Quarter note
- In 4/4: 4 quarter notes per measure
- In 3/4: 3 quarter notes per measure
- Tempo ratio: 3/4 = 0.75
- New tempo: 120 * 0.75 = 90 BPM

---

## Conversion Strategies

### 1. Stretch Strategy

**Proportionally stretches or compresses time.**

```python
params = MeterConversionParams(
    strategy=ConversionStrategy.STRETCH,
    adjust_tempo=True  # Compensate tempo to maintain speed
)

result = converter.convert_meter(3, 4, params=params)
```

**Best for:**
- Maintaining relative timing
- When absolute durations don't matter
- Quick conversions

**Example:** 4/4 → 3/4
- Stretch factor: 3/4 = 0.75
- All notes compressed to 75% duration
- Measure 1 (4 beats) → Measure 1 (3 beats)

### 2. Redistribute Strategy

**Remaps notes to new measure boundaries.**

```python
params = MeterConversionParams(
    strategy=ConversionStrategy.REDISTRIBUTE
)

result = converter.convert_meter(5, 4, params=params)
```

**Best for:**
- Maintaining note density
- When measures need different content
- Complex rhythmic transformations

**Example:** 4/4 → 5/4
- Notes redistributed across new measure structure
- May create partial measures at boundaries

### 3. Metric Modulation Strategy

**Uses pivot rhythms for smooth transitions.**

```python
params = MeterConversionParams(
    strategy=ConversionStrategy.METRIC_MODULATION
)

result = converter.convert_meter(7, 8, params=params)

# Check pivot rhythm used
if 'pivot_rhythm' in result.stats:
    pivot = result.stats['pivot_rhythm']
    print(f"Pivot: {pivot['name']}")
    print(f"New tempo: {pivot['new_tempo']} BPM")
```

**Best for:**
- Professional-sounding transitions
- Maintaining musical flow
- When tempo can change

### 4. Phrase-Aware Strategy (Default)

**Preserves phrase structure and boundaries.**

```python
params = MeterConversionParams(
    strategy=ConversionStrategy.PHRASE_AWARE,
    preserve_phrase_structure=True
)

result = converter.convert_meter(3, 4, params=params)
```

**Best for:**
- Maintaining musical coherence
- Songs with clear phrase structure
- When phrase endings must align

**Example:**
- 8-measure phrase in 4/4 (32 beats)
- Convert to 3/4
- Result: ~11 measures (33 beats)
- Phrase boundaries preserved

---

## API Reference

### Classes

#### `MeterConverter`

Main conversion class.

```python
class MeterConverter:
    def __init__(self, midi_file: Union[str, MidiFile])

    def convert_meter(
        self,
        new_numerator: int,
        new_denominator: int,
        new_grouping: Optional[List[int]] = None,
        params: Optional[MeterConversionParams] = None
    ) -> ConversionResult
```

**Methods:**

| Method | Description |
|--------|-------------|
| `convert_meter()` | Main conversion method |
| `_detect_time_signature()` | Detect current time signature |
| `_detect_tempo()` | Detect current tempo |
| `_detect_phrase_boundaries()` | Find phrase boundaries |

#### `MetricModulator`

Handles metric modulation calculations.

```python
class MetricModulator:
    @staticmethod
    def calculate_tempo_relationship(
        old_time_sig: TimeSignatureInfo,
        new_time_sig: TimeSignatureInfo,
        pivot_note_value: Fraction = Fraction(1, 1)
    ) -> Dict

    @staticmethod
    def find_best_pivot(
        old_time_sig: TimeSignatureInfo,
        new_time_sig: TimeSignatureInfo
    ) -> Tuple[Fraction, float]
```

#### `PhrasePreserver`

Analyzes and preserves phrase structure.

```python
class PhrasePreserver:
    def __init__(self, midi_file: MidiFile, time_sig: TimeSignatureInfo)

    def analyze_phrases(self) -> List[Dict]

    def preserve_phrase_in_conversion(
        self,
        phrase: Dict,
        old_time_sig: TimeSignatureInfo,
        new_time_sig: TimeSignatureInfo
    ) -> Dict
```

#### `MeterUtilities`

Helper utilities.

```python
class MeterUtilities:
    @staticmethod
    def quantize_to_meter(
        tick: int,
        time_sig: TimeSignatureInfo,
        ppqn: int,
        strength: float = 1.0
    ) -> int

    @staticmethod
    def get_meter_accent_pattern(time_sig: TimeSignatureInfo) -> List[float]

    @staticmethod
    def validate_time_signature(numerator: int, denominator: int) -> bool
```

### Data Structures

#### `ConversionResult`

```python
@dataclass
class ConversionResult:
    success: bool
    new_midi: Optional[MidiFile]
    new_time_signature: Optional[TimeSignatureInfo]
    tempo_change_factor: float = 1.0
    warnings: List[str]
    stats: Dict[str, any]
```

#### `MeterConversionParams`

```python
@dataclass
class MeterConversionParams:
    strategy: ConversionStrategy = ConversionStrategy.PHRASE_AWARE
    preserve_phrase_structure: bool = True
    preserve_tempo_feel: bool = True
    adjust_tempo: bool = False
    target_tempo: Optional[int] = None
    maintain_durations: bool = False
    quantize_output: bool = True
    quantize_strength: float = 0.8
```

#### `TimeSignatureInfo`

```python
@dataclass
class TimeSignatureInfo:
    numerator: int
    denominator: int
    grouping: Optional[List[int]] = None
    family: Optional[MeterFamily] = None
    beats_per_measure: float
    ticks_per_measure: int
    ppqn: int = 480
```

---

## Examples

### Example 1: Jazz 4/4 to 5/4 (Take Five Style)

```python
from meter_converter import MeterConverter, ConversionStrategy, MeterConversionParams

# Load jazz standard in 4/4
converter = MeterConverter("autumn_leaves.mid")

# Convert to 5/4 with [3+2] grouping (Take Five)
result = converter.convert_meter(
    new_numerator=5,
    new_denominator=4,
    new_grouping=[3, 2],
    params=MeterConversionParams(
        strategy=ConversionStrategy.PHRASE_AWARE,
        preserve_phrase_structure=True
    )
)

result.new_midi.save("autumn_leaves_5_4.mid")
```

### Example 2: Pop 4/4 to 7/8 (Money Style)

```python
# Pink Floyd "Money" uses 7/4 with [4+3] grouping
converter = MeterConverter("pop_song.mid")

result = converter.convert_meter(
    new_numerator=7,
    new_denominator=8,
    new_grouping=[2, 2, 3],  # More common in 7/8
    params=MeterConversionParams(
        strategy=ConversionStrategy.METRIC_MODULATION
    )
)

print(f"Pivot rhythm: {result.stats.get('pivot_rhythm', {}).get('name')}")
result.new_midi.save("pop_song_7_8.mid")
```

### Example 3: Convert to Compound Meter (6/8)

```python
# Convert 4/4 march to 6/8 waltz feel
converter = MeterConverter("march.mid")

result = converter.convert_meter(
    new_numerator=6,
    new_denominator=8,
    params=MeterConversionParams(
        strategy=ConversionStrategy.STRETCH,
        adjust_tempo=True  # Maintain perceived speed
    )
)

result.new_midi.save("waltz.mid")
```

### Example 4: Progressive Metal Polymetric

```python
# Tool/Meshuggah style: 4/4 → 13/8
converter = MeterConverter("metal_riff.mid")

result = converter.convert_meter(
    new_numerator=13,
    new_denominator=8,
    new_grouping=[3, 3, 3, 2, 2],
    params=MeterConversionParams(
        strategy=ConversionStrategy.REDISTRIBUTE
    )
)

result.new_midi.save("metal_riff_13_8.mid")
```

### Example 5: Batch Conversion

```python
import os
from pathlib import Path

def batch_convert_to_waltz(input_dir, output_dir):
    """Convert all MIDI files to 3/4."""
    Path(output_dir).mkdir(exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.mid'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"waltz_{filename}")

            converter = MeterConverter(input_path)
            result = converter.convert_meter(3, 4)

            if result.success:
                result.new_midi.save(output_path)
                print(f"✓ Converted {filename}")
            else:
                print(f"✗ Failed {filename}")

# Use it
batch_convert_to_waltz("originals/", "waltzes/")
```

---

## Advanced Usage

### Custom Phrase Boundary Detection

```python
converter = MeterConverter("song.mid")

# Manually specify phrase boundaries
custom_boundaries = [
    PhraseBoundary(
        measure_number=4,
        tick_position=converter.current_time_sig.ticks_per_measure * 4,
        boundary_type='cadence',
        confidence=1.0
    ),
    PhraseBoundary(
        measure_number=8,
        tick_position=converter.current_time_sig.ticks_per_measure * 8,
        boundary_type='end',
        confidence=1.0
    )
]

converter.phrase_boundaries = custom_boundaries

result = converter.convert_meter(3, 4)
```

### Fine-Tune Quantization

```python
from meter_converter import MeterUtilities, TimeSignatureInfo

ts = TimeSignatureInfo(7, 8, grouping=[2, 2, 3])
ppqn = 480

# Quantize with different strengths
tick = 1250

full_quantize = MeterUtilities.quantize_to_meter(tick, ts, ppqn, strength=1.0)
half_quantize = MeterUtilities.quantize_to_meter(tick, ts, ppqn, strength=0.5)
no_quantize = MeterUtilities.quantize_to_meter(tick, ts, ppqn, strength=0.0)

print(f"Original: {tick}")
print(f"Full quantize: {full_quantize}")
print(f"Half quantize: {half_quantize}")
print(f"No quantize: {no_quantize}")
```

### Calculate Tempo Relationships

```python
from meter_converter import MetricModulator, TimeSignatureInfo
from fractions import Fraction

old_ts = TimeSignatureInfo(4, 4)
new_ts = TimeSignatureInfo(7, 8)

# Find best pivot rhythm
pivot, tempo_ratio = MetricModulator.find_best_pivot(old_ts, new_ts)

print(f"Best pivot: {pivot}")
print(f"Tempo ratio: {tempo_ratio}")
print(f"If old tempo = 120 BPM, new tempo = {120 * tempo_ratio} BPM")

# Try specific pivot
relationship = MetricModulator.calculate_tempo_relationship(
    old_ts, new_ts,
    pivot_note_value=Fraction(1, 2)  # Eighth note
)

print(f"With eighth note pivot:")
print(f"  Tempo ratio: {relationship['tempo_ratio']}")
print(f"  Pivots per old measure: {relationship['pivots_per_old_measure']}")
print(f"  Pivots per new measure: {relationship['pivots_per_new_measure']}")
```

### Analyze Phrase Structure

```python
from meter_converter import PhrasePreserver, TimeSignatureInfo

midi = MidiFile("song.mid")
ts = TimeSignatureInfo(4, 4, ppqn=midi.ticks_per_beat)

preserver = PhrasePreserver(midi, ts)
phrases = preserver.analyze_phrases()

print(f"Found {len(phrases)} phrases:")
for i, phrase in enumerate(phrases):
    print(f"Phrase {i+1}:")
    print(f"  Measures {phrase['start_measure']} - {phrase['end_measure']}")
    print(f"  Length: {phrase['length_measures']} measures")
```

---

## Technical Details

### Conversion Algorithms

#### Phrase-Aware Algorithm

1. **Detect phrases** using rest analysis and hypermeter
2. **Group notes** by phrase
3. **Convert each phrase** independently
4. **Maintain phrase lengths** (adjust measure count)
5. **Smooth boundaries** between phrases

#### Metric Modulation Algorithm

1. **Find pivot rhythm** (note value constant across meters)
2. **Calculate tempo ratio** using pivot
3. **Apply tempo change** to maintain rhythm duration
4. **Update time signature** markers

#### Stretch Algorithm

1. **Calculate stretch factor**: `new_beats / old_beats`
2. **Multiply all delta times** by stretch factor
3. **Update time signatures**
4. **Optionally adjust tempo** by inverse factor

#### Redistribution Algorithm

1. **Extract all notes** with absolute timing
2. **Group by measure** in source meter
3. **Map to target measures** proportionally
4. **Adjust positions** within measures
5. **Reconstruct MIDI** with new structure

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Load MIDI | O(n) | n = number of events |
| Detect phrases | O(n) | Single pass through notes |
| Basic conversion | O(n) | Update metadata only |
| Stretch conversion | O(n) | Scale delta times |
| Redistribute | O(n log n) | Sorting notes |
| Phrase-aware | O(n log n) | Phrase detection + sorting |

### Memory Usage

- **Input MIDI**: Original file size
- **Parsed structure**: ~3-5x file size (in-memory representation)
- **Output MIDI**: Similar to input (may vary ±20%)

### Limitations

1. **Preserves MIDI events only** - No audio processing
2. **Single meter per file** - No mixed meters
3. **Assumes Western meters** - May not handle non-Western rhythmic cycles optimally
4. **Quantization** - Very fine timing details may be lost
5. **Complex polyphony** - Dense polyphonic textures may need manual adjustment

---

## Research & References

### Academic Sources

1. **Elliott Carter - Metric Modulation**
   - Bernard, J.W. (1988). "The Evolution of Elliott Carter's Rhythmic Practice"
   - Smooth tempo changes via rhythmic relationships

2. **Phrase Structure**
   - Rothstein, W. (1989). "Phrase Rhythm in Tonal Music"
   - Classical phrase structure and hypermeter

3. **Meter Perception**
   - London, J. (2012). "Hearing in Time: Psychological Aspects of Musical Meter"
   - How listeners perceive meter and grouping

4. **Odd Meters**
   - Brubeck, D. (1959). "Time Out" album
   - Jazz in 5/4, 9/8, 6/4

5. **Polymetric Structures**
   - Tool, Meshuggah - Progressive metal techniques
   - Complex meter groupings (13/8, 15/8, etc.)

### Music Theory Resources

- **Kostka & Payne**: "Tonal Harmony" - Meter and phrase structure
- **Bartók**: "Mikrokosmos" - Asymmetric meters (Bulgarian rhythms)
- **Indian Tala System**: Carnatic and Hindustani rhythmic cycles
- **African Timeline Patterns**: Bell patterns and polyrhythm

### Implementation References

- **MusicXML Time Signature** specification
- **MIDI Specification** - MetaMessage time signature format
- **MuseScore** - Time signature conversion algorithms
- **Music21** - Meter detection and analysis

---

## Troubleshooting

### Common Issues

#### 1. Phrase Boundaries Not Detected

**Problem**: Conversion doesn't preserve phrases well.

**Solution**:
```python
# Manually specify phrase boundaries
converter.phrase_boundaries = [
    PhraseBoundary(measure_number=4, tick_position=..., boundary_type='end', confidence=1.0)
]
```

#### 2. Tempo Too Fast/Slow After Conversion

**Problem**: Music sounds rushed or dragged.

**Solution**:
```python
# Enable tempo adjustment
params = MeterConversionParams(
    strategy=ConversionStrategy.STRETCH,
    adjust_tempo=True  # Compensates for meter change
)
```

#### 3. Notes Sound Off-Grid

**Problem**: Rhythms don't align properly.

**Solution**:
```python
# Increase quantization strength
params = MeterConversionParams(
    quantize_output=True,
    quantize_strength=0.9  # 0-1, higher = more quantization
)
```

#### 4. Conversion Fails

**Problem**: `result.success == False`

**Solution**:
```python
result = converter.convert_meter(3, 4)

if not result.success:
    print("Warnings:")
    for warning in result.warnings:
        print(f"  - {warning}")

    # Try different strategy
    params = MeterConversionParams(strategy=ConversionStrategy.STRETCH)
    result = converter.convert_meter(3, 4, params=params)
```

---

## Performance Tips

1. **Use appropriate strategy**:
   - Simple conversions → `STRETCH`
   - Complex pieces → `PHRASE_AWARE`
   - Professional results → `METRIC_MODULATION`

2. **Disable phrase detection** if not needed:
   ```python
   params = MeterConversionParams(preserve_phrase_structure=False)
   ```

3. **Batch processing** - Reuse converter for same source meter:
   ```python
   converter = MeterConverter(midi)
   result_3_4 = converter.convert_meter(3, 4)
   result_5_4 = converter.convert_meter(5, 4)
   result_7_8 = converter.convert_meter(7, 8)
   ```

---

## Future Enhancements

Potential additions for future versions:

- [ ] Mixed meters within single file
- [ ] Gradual meter transitions (4/4 → 5/4 → 6/4)
- [ ] Non-Western rhythmic cycles (Tala, Usul, Clave)
- [ ] Machine learning phrase detection
- [ ] Visual editor integration
- [ ] Real-time conversion
- [ ] Swing feel preservation
- [ ] Microtonality support

---

## Contributing

This module is part of Agent 7's contribution to the 10-Agent Enhancement Project.

To extend functionality:

1. Inherit from `MeterConverter`
2. Add new `ConversionStrategy` enum values
3. Implement strategy method (e.g., `_convert_via_new_strategy`)
4. Add tests to `test_meter_converter.py`
5. Update documentation

---

## License

Part of HarmonyModule Library - Advanced Music/MIDI Python Framework

---

## Contact & Support

For issues, questions, or contributions:
- See main HarmonyModule documentation
- Check existing test cases for usage examples
- Review research references for theoretical background

---

**Created by Agent 7 - Modular Fusion Enhancement**
*Building the world's most advanced music generation system* 🎵
