# Big Band Generator - Quick Start

## Installation

```bash
# Clone the repository
git clone https://github.com/doseedo/Do.git
cd Do/midi_generator

# Install dependencies
pip install mido matplotlib
```

## Generate Your First Big Band Arrangement

```bash
# Default: C major, 140 BPM, jazz blues
python generate_big_band.py

# Custom tempo and key
python generate_big_band.py my_arrangement.mid 120 5  # F major, 120 BPM
```

## What You Get

A complete 17-track big band arrangement featuring:

### 🎷 Sax Section (6 tracks)
- Lead alto sax with bebop melody
- 5-part sax soli harmony (2 altos, 2 tenors, bari)

### 🎺 Brass Section (8 tracks)
- 4 trumpets + 4 trombones
- Punchy background figures and stabs

### 🎹 Rhythm Section (3 tracks)
- Piano comping (rootless voicings)
- Walking bass line
- Swing drums (ride cymbal + backbeat)

## Common Use Cases

### Jazz Club Swing (Medium tempo)
```bash
python generate_big_band.py club_swing.mid 140 3  # Eb major
```

### Ballad (Slow tempo)
```bash
python generate_big_band.py ballad.mid 80 2  # D major
```

### Up-tempo Bebop
```bash
python generate_big_band.py bebop.mid 200 10  # Bb major
```

### Classic Basie Sound
```bash
python generate_big_band.py basie.mid 140 7  # G major
```

## Key Reference

| Number | Key | Description |
|--------|-----|-------------|
| 0 | C | Beginner friendly |
| 2 | D | Bright sound |
| 3 | Eb | Classic jazz key |
| 5 | F | Blues key |
| 7 | G | Bright, happy |
| 10 | Bb | Jazz standard key |

## Output Format

- **Format**: Multi-track MIDI (Type 1)
- **Tracks**: 17 separate instrumental tracks
- **Compatibility**: All DAWs and notation software
- **General MIDI**: Works with any GM soundfont

## Next Steps

1. **Open in DAW**: Import into Logic, Ableton, FL Studio, etc.
2. **Adjust levels**: Mix individual instruments
3. **Add panning**: Create stereo width
4. **Humanize**: Add slight timing variations
5. **Effects**: Reverb, EQ, compression

## Full Documentation

See `README_BIG_BAND.md` for complete details, customization options, and troubleshooting.

---

**Happy arranging! 🎺🎷🎹**
