# Big Band Arrangement Generator

## Overview

The `generate_big_band.py` script creates complete big band arrangements in the style of Duke Ellington and Count Basie, featuring:

### Full Instrumentation (17 tracks)

**Saxophone Section (6 tracks):**
- Lead Alto Saxophone (melody)
- Alto Saxophone 1 & 2
- Tenor Saxophone 1 & 2
- Baritone Saxophone

**Brass Section (8 tracks):**
- Trumpet 1, 2, 3, 4
- Trombone 1, 2, 3, 4

**Rhythm Section (3 tracks):**
- Piano (syncopated comping)
- Acoustic Bass (walking bass)
- Drums (swing pattern with ride cymbal and hi-hat)

## Features

- **Authentic Swing Feel**: Medium swing (62% swing ratio)
- **Bebop Melodies**: Chromatic approach notes and bebop scales
- **Walking Bass Lines**: Quarter-note walking patterns with chord tones
- **Sax Soli**: 5-part close harmony (drop-2 voicings)
- **Brass Punches**: Powerful brass stabs on chord changes
- **Piano Comping**: Rootless voicings on upbeats (Basie style)
- **Swing Drums**: Ride cymbal pattern with backbeat

## Installation

### Required Dependencies

```bash
# Core requirement
pip install mido

# Optional for visualization
pip install matplotlib
```

## Usage

### Basic Usage

```bash
# Generate with defaults (C major, 140 BPM, 2 choruses of blues)
python generate_big_band.py

# Specify output file
python generate_big_band.py my_arrangement.mid

# Custom tempo and key
python generate_big_band.py output.mid 120 0    # C major, 120 BPM
python generate_big_band.py output.mid 140 -3   # Eb major, 140 BPM
python generate_big_band.py output.mid 180 2    # D major, 180 BPM
```

### Command-Line Arguments

```bash
python generate_big_band.py [output_file] [tempo] [key]
```

- `output_file`: Path to output MIDI file (default: `big_band_arrangement.mid`)
- `tempo`: Tempo in BPM (default: 140, typical range: 120-200)
- `key`: Key as pitch class (default: 0)
  - 0 = C, 1 = C#, 2 = D, 3 = Eb, 4 = E, 5 = F, 6 = F#, 7 = G, 8 = Ab, 9 = A, 10 = Bb, 11 = B

### Examples

```bash
# Slow ballad in F major
python generate_big_band.py ballad.mid 80 5

# Fast bebop in Bb major
python generate_big_band.py bebop.mid 200 10

# Medium swing in Eb (classic jazz key)
python generate_big_band.py swing.mid 140 3

# Up-tempo in G major
python generate_big_band.py uptempo.mid 180 7
```

## Generated Form

The script generates a **12-bar jazz blues** progression repeated for 2 choruses (24 bars total).

The jazz blues includes sophisticated changes:
- I7 → ivm7-VII7 → I7
- IV7 → I7
- ii7 → V7 → Imaj7 → vim7

## Output

The script creates a multi-track MIDI file with:

- **17 separate tracks** (one per instrument)
- **General MIDI instruments** (compatible with all DAWs and notation software)
- **Proper channel assignments** (drums on channel 9)
- **Tempo and time signature metadata**

### Track List

1. Lead Alto Sax (melody)
2. Alto Sax 1 (harmony)
3. Alto Sax 2 (harmony)
4. Tenor Sax 1 (harmony)
5. Tenor Sax 2 (harmony)
6. Baritone Sax (harmony)
7. Trumpet 1
8. Trumpet 2
9. Trumpet 3
10. Trumpet 4
11. Trombone 1
12. Trombone 2
13. Trombone 3
14. Trombone 4 (bass trombone)
15. Piano
16. Bass
17. Drums

## Working with the Output

### Import into DAWs

The generated MIDI file can be opened in:

- **Logic Pro**: Drag into arrangement window
- **Ableton Live**: Drag into session or arrangement view
- **FL Studio**: Import as separate channels
- **GarageBand**: Drag into timeline
- **Pro Tools**: Import MIDI file
- **Reaper**: Insert MIDI item

### Import into Notation Software

- **MuseScore**: File → Open
- **Sibelius**: File → Open → MIDI
- **Finale**: File → Import → MIDI
- **Dorico**: File → Import → MIDI

### Post-Processing Tips

1. **Adjust Velocities**: Add dynamic expression (crescendos, accents)
2. **Add Panning**:
   - Saxes: Center to slight left
   - Trumpets: Center to slight right
   - Trombones: Center
   - Rhythm section: Piano (center), Bass (center), Drums (wide stereo)
3. **Humanize**: Add subtle timing variations (5-15ms)
4. **Effects**:
   - Reverb: Medium room (jazz club ambience)
   - EQ: Boost high end on brass, warm low-mids on saxes
   - Compression: Light on individual tracks, gentle on master

## Customization

To modify the script for different arrangements:

### Change the Form

Edit line in `main()`:

```python
# Current (12-bar blues)
arrangement = generator.generate_arrangement(
    form=JazzForm.BLUES_12,
    num_choruses=2
)

# Change to rhythm changes
arrangement = generator.generate_arrangement(
    form=JazzForm.RHYTHM_CHANGES,
    num_choruses=2
)

# Change to AABA form
arrangement = generator.generate_arrangement(
    form=JazzForm.AABA_32,
    num_choruses=1
)
```

### Adjust Swing Feel

Modify the `BigBandGenerator.__init__` method:

```python
self.jazz_gen = JazzGenerator(
    style=JazzStyle.SWING,
    tempo=tempo,
    key=key,
    swing_feel=SwingFeel.LIGHT_SWING   # or HEAVY_SWING, TRIPLET
)
```

### Change Melody Density

In `_generate_melody` method:

```python
phrase = self.melody_gen.generate_phrase(
    chord,
    length_beats=4,
    density=0.9  # Higher = more notes (0.0-1.0)
)
```

## Technical Details

### Arranging Principles Used

1. **Sax Section**:
   - Drop-2 voicing (5-part close harmony)
   - Lead alto carries melody
   - Voices spaced within instrument comfortable ranges
   - Parallel motion maintains cohesion

2. **Brass Section**:
   - Punctuates on chord changes (not constant)
   - 4-part voicing (trumpets high, trombones low)
   - Short, accented figures (stabs)
   - Creates contrast with sustained sax section

3. **Rhythm Section**:
   - Piano: Rootless voicings on upbeats (2.5 and 4.5)
   - Bass: Walking quarter notes with chord tones and approach notes
   - Drums: Ride cymbal swing pattern, backbeat on 2 & 4

4. **Voice Leading**:
   - Smooth voice leading between chord changes
   - Instruments stay within comfortable ranges
   - Avoids large leaps where possible

### Swing Timing

The script applies authentic swing timing:
- Upbeat eighth notes delayed to 66.7% position
- Creates triplet-based swing feel
- Applied to drums, melody, and rhythm section

## Troubleshooting

### "ModuleNotFoundError: No module named 'mido'"

Install the mido library:
```bash
pip install mido
```

### Empty or Silent Tracks

- Check that your DAW/software supports General MIDI instrument numbers
- Verify channel assignments (drums must be on channel 9/10)
- Ensure MIDI import settings preserve all tracks

### Instruments Out of Range

The script automatically constrains pitches to instrument ranges. If notes sound wrong:
- Check transposition settings in your DAW
- Verify you're using the correct soundfont/instruments
- Some General MIDI implementations vary

### Swing Feel Not Working

- The swing is programmed into note timing
- Some DAWs may override with their own swing settings
- Disable DAW swing and let the MIDI timing control feel

## Related Files

- `genres/jazz.py` - Core jazz generation algorithms
- `examples/02_modal_jazz_composition.py` - Modal jazz example
- `transformation/arrangement_engine.py` - Arrangement utilities
- `examples/04_auto_arrangement.py` - Auto-arrangement from lead sheet

## License

MIT License - Part of the Dø (Doseedo) MIDI Generator Library

## Credits

- **Arranging Style**: Duke Ellington, Count Basie
- **Jazz Theory**: Mark Levine, Jerry Coker, George Russell
- **Implementation**: Phase 3 Consolidated Jazz Module
