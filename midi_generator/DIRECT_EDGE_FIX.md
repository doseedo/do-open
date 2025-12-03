# Direct Edge Extraction Fix

## Problem Identified

The previous composition discovery was extracting **round-trip cycles** instead of **direct arrangement relationships**.

### What Was Wrong

**Old behavior** (multi-hop path extraction):
```
For object brass_1[bar1]:
  Path: brass_1 → [T(12)] → brass_0 → [T(-12)] → brass_1

Result: TrackDerive(brass_1→brass_0) ∘ T(12) ∘ TrackDerive(brass_0→brass_1) ∘ T(-12)
Meaning: "Go from brass_1 to brass_0 and back" (IDENTITY/ROUND-TRIP)
```

**Problems**:
- All 100 compositions were meaningless round-trips
- Low frequencies (329 max vs 117,551 expected)
- No simple patterns like `time_shift(-16) ∘ T(7)`
- Missing section relationships entirely

### Root Cause

The `_get_path_to_root()` method followed multi-hop paths through the derivation graph, creating cycles when bidirectional edges existed (A→B and B→A).

## The Fix

**New behavior** (single-edge extraction):
```python
def find_compositions_from_paths(self, graph, verbose=True):
    """Extract DIRECT single-edge arrangement relationships."""

    for target, derivation in graph.items():
        if derivation.is_cross_track or derivation.is_cross_section:
            # This IS the arranging relationship
            pattern = format_direct_relationship(derivation)
            compositions[pattern] += 1
```

**Now extracts**:
- `TrackDerive(brass_0→brass_1) ∘ T(-12)` - "brass_1 is brass_0, octave down"
- `SectionDerive(sect_0→sect_1) ∘ time_shift(-16)` - "section 1 is 16 steps after section 0"
- `SectionTrackDerive(sect_0.trumpet→sect_1.trumpet) ∘ time_shift(-16)` - cross-section same-track

## Expected Results

### Before Fix
```
Top compositions:
1. TrackDerive(brass_1→brass_0) ∘ T(12) ∘ TrackDerive(brass_0→brass_1) ∘ T(-12)  freq=329
2. TrackDerive(brass_0→brass_1) ∘ T(-12) ∘ TrackDerive(brass_1→brass_0) ∘ T(12)  freq=329
   (All round-trips, low frequencies)
```

### After Fix
```
Top direct arrangement patterns:
1. SectionDerive(sect_0→sect_1) ∘ time_shift(-16.0)  freq=117,551+
2. TrackDerive(brass_0→brass_1) ∘ transpose_semitone(-12.0)  freq=65,000+
3. TrackDerive(reed_1→reed_2) ∘ transpose_semitone(-7.0)  freq=45,000+
4. SectionTrackDerive(sect_0.trumpet→sect_1.trumpet) ∘ time_shift(-16.0)  freq=35,000+
```

## Implementation Details

**File**: `1_approaches/transform_based/discovery/emergent_hierarchy.py`

**Function**: `find_compositions_from_paths()` (lines 814-895)

**Changes**:
1. Removed multi-hop path tracing via `_get_path_to_root()`
2. Removed path canonicalization (no longer needed for single edges)
3. Direct extraction: iterate through graph, count cross-track/cross-section edges
4. Updated docstring to reflect new behavior

**Compatibility**:
- Returns same signature: `(compositions, temporal_data)`
- `temporal_data` now empty dict (no longer tracks path variants)
- Composition format unchanged: `"Pattern ∘ transform(amount)"`

## Testing

Running discovery with fix:
```bash
python -u scripts/run_emergent_discovery.py > discovery_direct_edge_fix.log 2>&1
```

Expected improvements:
- Much higher frequencies for section relationships
- Simple, interpretable patterns
- No round-trip cycles
- Clear arranging relationships between instruments
