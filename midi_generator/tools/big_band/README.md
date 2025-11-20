# Big Band Generator Tools

This directory contains big band arrangement generators with various features.

## Recommended Scripts

### 1. `generate_final.py` - Production Ready
The most stable, production-ready big band generator with all critical fixes:
- ✅ Proper swing timing with duration compensation
- ✅ Consistent chromatic grace notes in sax soli
- ✅ Professional swing drums with backbeat
- ✅ Varied piano comping patterns
- ✅ Walking bass lines

**Usage:**
```bash
python generate_final.py [name] [tempo] [key] [progression]

# Examples:
python generate_final.py swing 140 0 jazz_blues
python generate_final.py bebop 180 3 rhythm_changes
```

### 2. `generate_comprehensive.py` - Advanced Harmony
Uses the full harmony module ecosystem with 31+ progression types:
- ✅ All features from generate_final.py
- ✅ 31+ chord progression types across 5 categories
- ✅ Modal progressions (Dorian, Mixolydian, Lydian, etc.)
- ✅ Neo-Riemannian transformations (PLR, hexatonic cycles)
- ✅ Extended jazz progressions (Coltrane changes, Autumn Leaves, etc.)

**Usage:**
```bash
python generate_comprehensive.py [name] [tempo] [key] [progression_type]

# Examples:
python generate_comprehensive.py modal 140 0 dorian_vamp
python generate_comprehensive.py coltrane 180 0 coltrane_changes
python generate_comprehensive.py film 100 5 plr_film
```

**Available Progression Types:**
- Basic Jazz: jazz_blues, rhythm_changes, ii_V_I, minor_ii_V_i
- Extended Jazz: coltrane_changes, autumn_leaves, all_the_things, take_five, so_what, blue_bossa
- Modal: dorian_vamp, mixolydian_rock, lydian_dream, phrygian_spanish
- Neo-Riemannian: plr_film, hexatonic_northern, chromatic_mediant
- Advanced: modal_interchange, reharmonized_blues, quartal_harmony

### 3. `generate_proper.py` - Uses ArrangementEngine
Alternative implementation using the ArrangementEngine module.

## Archive

The `archive/` directory contains earlier versions for reference:
- generate_big_band.py - V1 (basic)
- generate_big_band_v2.py - V2 (experimental)
- generate_big_band_complete.py - Earlier complete version
- generate_big_band_complete_v3.py - V3
- Other experimental versions

These are kept for reference but not recommended for production use.

## Parameters

All generators accept these parameters:

1. **name** - Output filename (default: "swing" or "final")
2. **tempo** - BPM (default: 140)
3. **key** - Root note 0-11 (0=C, 1=Db, 2=D, etc.)
4. **progression** - Chord progression type (varies by generator)

## Output

Generates MIDI files with full big band instrumentation:
- Lead melody (alto sax)
- Sax section (2 altos, 2 tenors, bari)
- Brass section (4 trumpets, 4 trombones)
- Piano (rootless voicings)
- Walking bass
- Swing drums (ride, snare, hi-hat, kick)

## Tips

- Start with `generate_final.py` for reliable results
- Use `generate_comprehensive.py` for experimental harmony
- Adjust tempo: 100-120 (ballad), 140-160 (medium swing), 180+ (fast bebop)
- Keys: 0=C, 3=Eb (common for horns), 5=F, 7=G, 10=Bb
