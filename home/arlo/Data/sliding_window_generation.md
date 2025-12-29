# Sliding Window Generation for Fast Musical Passages

## Problem

The ACE-Step model struggles with fast/complex MIDI inputs (e.g., 16th notes at 200 BPM). The existing workaround is the **tape speed hack**:

1. Slow down input audio/MIDI (e.g., 0.6x)
2. Generate at slower tempo (model handles it better)
3. Speed up output (e.g., 1.67x)

**Issue**: Time-stretching introduces artifacts (phase smearing, transient degradation).

## Solution: Sliding Window Generation

Instead of changing tempo, split the input into overlapping windows and generate each independently:

```
Fast 13-second passage
         |
         v
Split into 9 overlapping 3-second windows (50% overlap)
         |
         v
Generate each window independently
(model sees manageable note density per window)
         |
         v
Crossfade blend with raised-cosine windows
         |
         v
Seamless output at natural tempo (no artifacts)
```

## Key Advantages

| Approach | Artifacts | Speed | Quality |
|----------|-----------|-------|---------|
| Normal | None | Fast (1x) | Poor on fast passages |
| Tape Speed Hack | Time-stretch artifacts | Medium (2.5x) | Better pitch, smeared transients |
| Sliding Window | None (crossfade seams) | Slow (7-8x) | Best - natural tempo, no stretch |

## Usage

### Basic Usage

```python
from sliding_window_generation import generate_sliding_window
from genfrominterface import generate, load_model_any_ckpt

# Load model
model = load_model_any_ckpt(checkpoint, checkpoint_dir, manifest)

# Extract conditioning
piano_roll, amp, rframe, rbend, encodec_tokens, duration, window_slow = extract_conditioning(audio_path)

# Generate with sliding window
output_path = generate_sliding_window(
    model=model,
    generate_fn=generate,  # Pass the generate function
    piano_roll=piano_roll,
    amp=amp,
    rframe=rframe,
    rbend=rbend,
    group="winds",
    subgroup="sax",
    window_seconds=3.0,      # 3-second windows
    overlap_ratio=0.5,       # 50% overlap
    audio_file=audio_path,   # For GT latent extraction
    steps=40,
    seed=42,
    noise_level=0.9,
    # ... other generate() params
)
```

### Check if Sliding Window is Needed

```python
from sliding_window_generation import should_use_sliding_window

should_use, max_density, fast_pct = should_use_sliding_window(piano_roll)
print(f"Max density: {max_density:.1f} notes/sec")
print(f"Fast regions: {fast_pct:.1f}%")
print(f"Recommend sliding window: {should_use}")
```

### Integrated with genfrominterface.py

The sliding window is integrated into the main inference script with these parameters:

```python
run_generation(
    # ... other params ...
    sliding_window_mode=True,
    sliding_window_seconds=3.0,
    sliding_window_overlap=0.5,
)
```

## Parameters

### Window Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_seconds` | 3.0 | Duration of each window in seconds |
| `overlap_ratio` | 0.5 | Overlap between windows (0.5 = 50%) |
| `density_threshold` | 8.0 | Notes/sec threshold to trigger sliding window |

### Tuning Recommendations

**Window Size:**
- 2-3 seconds works well for most cases
- Shorter windows = more windows = better density handling but more seams
- Longer windows = fewer seams but may still struggle with dense passages

**Overlap:**
- 50% is a good default
- Higher overlap = smoother blending but slower generation
- 25% is faster with acceptable quality

## How It Works

### 1. Note Density Analysis

```python
def compute_note_density(piano_roll, window_frames=43):
    """
    Calculates notes-per-second at each frame by:
    1. Detecting note onsets (new notes appearing)
    2. Smoothing with 1-second moving average
    3. Converting to notes/second
    """
```

### 2. Window Splitting

```python
def split_conditioning_into_windows(piano_roll, amp, rframe, rbend, window_frames, overlap_ratio):
    """
    Splits all conditioning arrays into overlapping windows.
    Each window contains:
    - piano_roll slice
    - amp, rframe, rbend slices
    - start_frame, end_frame, actual_length metadata
    """
```

### 3. Per-Window Generation

For each window:
1. Slice source audio for GT latent extraction (if provided)
2. Call generate() with window conditioning
3. Trim output to actual window length

### 4. Crossfade Blending

```python
def crossfade_blend_audio(audio_segments, window_info):
    """
    Blends overlapping segments with raised-cosine crossfade:

    Window 1: [========------]
    Window 2:     [---========------]
    Window 3:          [---========------]

    Crossfade regions use: 0.5 * (1 - cos(pi * t))
    Ensures smooth transitions at overlap points.
    """
```

## Performance Comparison

Tested on 13.5-second passage with winds/sax at 40 steps:

| Method | Time | Notes |
|--------|------|-------|
| Normal | 37.5s | Single pass |
| Tape Speed (0.6x) | 96.1s | Slow + generate + speed up |
| Sliding Window | 292.4s | 9 windows + blending |

Sliding window is ~8x slower but avoids all time-stretch artifacts.

## When to Use

**Use Sliding Window when:**
- Input has fast passages (>8 notes/second)
- Time-stretch artifacts are unacceptable
- Quality is more important than speed

**Use Normal Generation when:**
- Input is moderate tempo
- Speed is critical
- Note density is manageable

**Use Tape Speed Hack when:**
- Need balance of quality and speed
- Some time-stretch artifacts are acceptable
- Input is very long (sliding window would be too slow)

## Files

- `sliding_window_generation.py` - Standalone module with all functions
- `sliding_window_generation.md` - This documentation
- `test_sliding_window.py` - Test script comparing all three approaches

## Integration Points

The sliding window functions are integrated into `genfrominterface.py`:

1. **Line 640-1044**: Function definitions (compute_note_density, split_conditioning_into_windows, crossfade_blend_audio, generate_sliding_window)

2. **Line 6737-6739**: Parameters added to run_generation():
   ```python
   sliding_window_mode: bool = False,
   sliding_window_seconds: float = 3.0,
   sliding_window_overlap: float = 0.5,
   ```

3. **Line 8199-8286**: Integration into monophonic mode generation with auto-detection based on note density.
