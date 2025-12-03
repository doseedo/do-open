# Drum Track Handling Analysis

## Current Status: ⚠️ PARTIALLY INCORRECT

The system **detects** drum tracks but **does not handle them correctly** during transformation and discovery.

---

## What Works ✅

### 1. Drum Detection
**File:** `core/hierarchical_corpus.py:171-172`

```python
if channel == 9:
    return 'drums'
```

- ✅ Correctly identifies MIDI channel 9 as drums
- ✅ Stores `is_drum` flag in track metadata
- ✅ Labels drum tracks in corpus

### 2. Rhythm-Based Transforms
These transforms work correctly for drums:
- ✅ `time_shift` - Shifting drum pattern in time
- ✅ `time_scale` - Tempo changes (stretch/compress)
- ✅ `retrograde` - Reverse drum pattern
- ✅ `velocity_scale` - Accent changes
- ✅ Quantization - Rhythm alignment

---

## What's Broken ❌

### 1. Pitch Transforms Applied to Drums
**File:** `core/numpy_transforms.py:52-69`

```python
def transpose_semitone(batch: np.ndarray, semitones: int):
    """Transpose pitch by semitones."""
    # THIS GETS APPLIED TO DRUMS TOO!
    result[:, :, semitones:128] = pitch[:, :, :128-semitones]
```

**Problem:**
- ❌ System transposes drum MIDI notes
- ❌ Kick drum (MIDI 36) → Snare (MIDI 38) after T(+2)
- ❌ Hi-hat (MIDI 42) → Crash (MIDI 49) after T(+7)
- ❌ Destroys drum kit mapping completely

**Why This is Critical:**
In drums, MIDI note numbers are **categorical labels**, not pitches:
- MIDI 36 = Bass Drum (Kick)
- MIDI 38 = Snare
- MIDI 42 = Closed Hi-Hat
- MIDI 46 = Open Hi-Hat
- MIDI 49 = Crash Cymbal
- etc.

Transposing these is like renaming variables in code - it changes the meaning!

### 2. Inversion Applied to Drums
**File:** `core/numpy_transforms.py:72-94`

```python
def inversion(batch: np.ndarray, center: int = None):
    """Invert pitches around a center note."""
    # THIS ALSO GETS APPLIED TO DRUMS!
```

**Problem:**
- ❌ Inverts drum kit mapping
- ❌ No musical meaning for drums
- ❌ Kick ↔ High Tom swaps

### 3. Missing Drum-Specific Metadata in Discovery
**File:** `discovery/emergent_hierarchy.py:113-140`

```python
@dataclass
class MusicalObject:
    piece_id: str
    track_id: str
    start_time: int
    end_time: int
    tensor: np.ndarray
    # ❌ NO is_drum FIELD!
```

**Problem:**
- ❌ Discovery engine doesn't know which objects are drums
- ❌ Can't filter pitch transforms for drum objects
- ❌ All transforms applied blindly

---

## Impact on Current Discovery Run

### What's Happening Right Now:
1. ✅ System loads 1,731 MIDI files (including drums)
2. ✅ Detects drum tracks on channel 9
3. ❌ **Extracts drum segments as MusicalObject (loses is_drum flag)**
4. ❌ **Tries pitch transforms on drums:**
   - `transpose_semitone(7)` → Kick becomes high tom
   - `transpose_semitone(12)` → Entire kit shifts up octave
   - `inversion(60)` → Kit mapping inverted
5. ⚠️ **High error rates for drum transforms** (they don't match!)
6. ⚠️ **Drums likely marked as "sources"** (no valid derivations found)

### Expected Behavior:
- Drum tracks should only use rhythm transforms
- Pitch transforms should be skipped for drums
- Drum patterns should be discoverable (e.g., "swing hi-hat", "backbeat")

---

## What Drum Patterns Should Be Discovered

### Legitimate Drum Derivations:
1. **Rhythm Transforms:**
   - `time_shift(16)` - Offset by one bar
   - `time_scale(2.0)` - Half-time feel
   - `retrograde` - Pattern reversal

2. **Density Variations:**
   - `quantize_16th → quantize_8th` - Simplification
   - `velocity_scale(0.7)` - Ghost notes
   - Fill variations (high-level pattern)

3. **Style Transfers:**
   - Swing feel → Straight feel
   - Ride pattern → Hi-hat pattern (instrument substitution)
   - Backbeat → Syncopated (groove variation)

### What Should NOT Apply to Drums:
- ❌ `transpose_semitone` - Changes instruments
- ❌ `inversion` - Nonsensical for drums
- ❌ Any pitch-based harmony transforms

---

## Fix Required

### Phase 1: Immediate Fix (Block Pitch Transforms on Drums)

**1. Add `is_drum` to MusicalObject:**
```python
@dataclass
class MusicalObject:
    piece_id: str
    track_id: str
    start_time: int
    end_time: int
    tensor: np.ndarray
    is_drum: bool = False  # ADD THIS
```

**2. Pass drum metadata during object extraction:**
```python
def extract_objects(self, corpus: Dict, verbose: bool = True):
    for piece_id, piece_data in corpus.items():
        tracks = piece_data['tracks']
        metadata = piece_data.get('metadata', {})  # Get track metadata

        for track_id, track_tensor in tracks.items():
            is_drum = metadata.get(track_id, {}).get('is_drum', False)

            objects.append(MusicalObject(
                piece_id=piece_id,
                track_id=track_id,
                start_time=0,
                end_time=T,
                tensor=track_tensor,
                is_drum=is_drum  # PASS THIS
            ))
```

**3. Filter transforms in graph building:**
```python
for transform in transforms:
    # Skip pitch transforms for drums
    if source.is_drum and transform['name'] in ['transpose_semitone', 'inversion']:
        continue

    # Apply transform
    transformed = lib.apply_transform(...)
```

### Phase 2: Drum-Specific Patterns (Future Enhancement)

**1. Drum-specific transforms:**
- `drum_fill` - Add fill pattern
- `swing_feel` - Convert straight to swing
- `instrument_substitute` - Hi-hat → Ride, etc.
- `ghost_notes` - Add ghost notes at velocity ~30%

**2. Drum pattern templates:**
- Backbeat (kick on 1,3 / snare on 2,4)
- Jazz ride pattern
- Latin clave patterns
- Funk ghost note patterns

**3. Drum pattern similarity:**
- Compare rhythm only (ignore velocity)
- Allow instrument substitution (ride ≈ hi-hat)
- Groove-based matching

---

## Verification Tests Needed

Once fixed, verify:

```python
# Test 1: Drums skip pitch transforms
drum_obj = MusicalObject(..., is_drum=True)
melodic_obj = MusicalObject(..., is_drum=False)

# Should NOT find derivations like:
# drum_obj = transpose_semitone(7)(other_drum)

# Should find derivations like:
# drum_obj = time_shift(16)(other_drum)

# Test 2: Check discovered compositions
# For drum sources, compositions should only be rhythm-based
drum_compositions = [c for c in compositions if involves_drum(c)]
assert all('transpose_semitone' not in c for c in drum_compositions)
assert any('time_shift' in c for c in drum_compositions)
```

---

## Current Discovery Impact Assessment

### Likely Current Behavior:
1. **Drums treated as sources:** High error rates → no derivations found
2. **Drum compositions NOT discovered:** Rhythm patterns missed
3. **False negatives:** Valid drum variations rejected due to pitch mismatch
4. **Wasted computation:** Testing invalid pitch transforms on drums

### After Fix:
1. ✅ Drums correctly derive from other drums (rhythm only)
2. ✅ Drum pattern discovery enabled
3. ✅ Computational efficiency improved
4. ✅ More accurate derivation graph

---

## Recommendation: Fix After Current Run

**Why wait:**
- Current run is already ~45 minutes in (loading MIDI files)
- Fix requires code changes that would restart discovery
- Current run will still discover melodic patterns correctly
- Drums will be treated as independent sources (not incorrect, just incomplete)

**Action Plan:**
1. ✓ Let current run complete (~6 hours)
2. ✓ Document current results (melodic patterns only)
3. ✓ Implement drum-aware fixes (Phase 1)
4. ✓ Re-run discovery with drum support
5. ✓ Compare: melodic + drum patterns vs. melodic only

**Estimated Fix Time:** 2-3 hours for Phase 1

---

## Summary

| Aspect | Status | Impact |
|--------|--------|--------|
| Drum detection | ✅ Works | Identifies channel 9 |
| Rhythm transforms | ✅ Works | Time_shift, scale, retrograde OK |
| Pitch transforms | ❌ Broken | Applied incorrectly to drums |
| Drum metadata in discovery | ❌ Missing | Can't filter transforms |
| Drum pattern discovery | ❌ Disabled | Not finding drum derivations |
| Current run validity | ⚠️ Partial | Melodic patterns OK, drums ignored |

**Bottom Line:** The system is **not currently equipped to learn drum patterns correctly**. It will treat drums as irreducible sources and miss all drum-based derivations. Melodic discovery will work fine.

**Fix Priority:** Medium (after current run completes)

---

**Author:** Analysis - Drum Handling Assessment
**Date:** 2025-11-24
**Status:** Fix Required (Phase 1: 2-3 hours)
