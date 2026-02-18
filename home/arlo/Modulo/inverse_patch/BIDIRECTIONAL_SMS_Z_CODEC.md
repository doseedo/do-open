# Bidirectional SMS-Z Codec

## The Problem

We have two representations of audio:
1. **SMS params** - sinusoidal model: 128 sine tracks (freq, amp, phase) + 8 noise bands per frame
2. **DCAE z** - neural latent: [8, 16, T] tensor, encodes full audio via ACE-Step

Previous attempts to map SMS → z failed because:
- SMS is **lossy** - it captures ~60-90% of audio energy (harmonics only, misses noise/transients)
- z = DCAE_encode(original_audio) captures **everything**
- The model was asked to predict z_audio from SMS, but SMS doesn't have enough info to fully determine z_audio
- Result: cos_sim ~0.93, blurry/averaged predictions for the parts SMS can't represent

## The Insight

**Train on SMS-rendered audio instead of original audio.**

```
SMS params → additive_synth → audio_sms → DCAE_encode → z_sms
```

In this setup:
- SMS params **fully determine** audio_sms (additive synthesis is deterministic)
- audio_sms **fully determines** z_sms (DCAE encode is deterministic)
- Therefore: SMS params **fully determine** z_sms
- There is **zero information gap** - the mapping is exact and invertible

This means a bidirectional model can achieve cos_sim ≈ 1.0 because there's nothing to guess.

## Architecture: Cycle-Consistent Bidirectional Mapping

Two models trained simultaneously:

```
G (forward):  compressed_SMS → z_sms
F (reverse):  z_sms → compressed_SMS
```

Four losses enforce perfect invertibility:

| Loss | Formula | What it enforces |
|------|---------|-----------------|
| Forward | \|G(sms) - z_sms\| | G accurately predicts z |
| Reverse | \|F(z) - sms\| | F accurately predicts SMS |
| Cycle Forward | \|G(F(z)) - z\| | z survives roundtrip through SMS |
| Cycle Reverse | \|F(G(sms)) - sms\| | SMS survives roundtrip through z |

The cycle losses are critical - they force G and F to be true inverses, not just good independent predictors.

## SMS Compression (Input Representation)

Raw SMS (128 sines) is sparse (~1.8 active per frame). We compress to a dense hierarchical program:

| Component | Dims | Description |
|-----------|------|-------------|
| group_f0s | 6 | Fundamental frequencies of harmonic groups |
| group_amps | 48 | Partial amplitudes (6 groups x 8 partials) |
| indep_freqs | 20 | Independent sine frequencies |
| indep_amps | 20 | Independent sine amplitudes |
| noise_amps | 8 | Noise band energies |
| **Total** | **102** | **Dense params per frame** |

Compression: 264 sparse dims → 102 dense dims (2.6x, but density goes from ~1% to ~100%).

## Phase 1: Perfect Codec (Current Implementation)

**Goal:** Achieve cos_sim ≈ 1.0 on SMS-rendered audio.

**Pipeline:**
```
1. Load SMS .pt files (freqs, amps, noise_amps)
2. Render each to audio via additive synthesis
3. Encode rendered audio with DCAE → z_sms
4. Compress SMS params → hierarchical program (102 dims)
5. Train bidirectional: G(sms) ↔ F(z_sms) with cycle losses
6. Test: verify roundtrip reconstruction in both directions
```

**Success criteria:**
- Forward cos_sim > 0.99
- Reverse MSE ≈ 0
- Cycle roundtrip error ≈ 0

## Phase 2: Extend the Operation Vocabulary

Once Phase 1 works, grow the parametric language beyond SMS:

```
Current operations:
  HarmonicSeries(f0, partials, weights)   ← tonal content
  IndependentSine(freq, amp)              ← isolated tones
  NoiseBand(band_idx, amp)                ← stochastic content

New operations to add:
  TransientBurst(time, spectrum, decay)   ← attacks, clicks
  SpectralTilt(slope, pivot_freq)         ← brightness/darkness
  FormantFilter(freqs, widths, gains)     ← vocal/resonance shapes
  AmplitudeModulation(carrier, mod_freq)  ← tremolo, AM synthesis
```

Each new operation type:
1. Render synthetic audio using that operation
2. Encode with DCAE → z
3. Train bidirectional mapping for that operation
4. Add to the vocabulary

The operation tree grows until it covers z-space completely.

## Phase 3: Neurosymbolic Bridge to Real Audio

The hard problem: find operation trees for real audio where z_real ≠ z_sms.

**Approach: Sparse Dictionary Learning in z-space**

```
z_real → encoder → sparse_code [K active from dictionary of N atoms]
                        ↓
              each atom = learned operation with parameters
                        ↓
sparse_code → decoder → z_reconstructed
```

**Training strategy:**
1. Initialize dictionary with Phase 1 SMS atoms (known operations)
2. Freeze SMS atoms, add N_new learnable atoms
3. Train on real audio: minimize |z_reconstructed - z_real| with sparsity penalty
4. New atoms emerge to capture what SMS can't: noise textures, transients, timbral qualities
5. Result: every audio has a sparse, interpretable operation tree

**Why this works:**
- Phase 1 anchors the dictionary with known, interpretable operations
- Sparsity constraint keeps representations interpretable (few active ops per sound)
- The dictionary atoms self-organize into audio-meaningful operations
- Bidirectional by construction (encoder finds the code, decoder reconstructs z)

## The End Goal

A bidirectional parametric codec for DCAE's z-space:

```
Any audio → DCAE_encode → z → sparse_operation_tree → editable params
                                                            ↓ (edit)
Edited audio ← DCAE_decode ← z' ← sparse_operation_tree' ← edited params
```

This gives you:
- **Analysis**: decompose any audio into interpretable operations
- **Editing**: modify individual operations (pitch, harmonics, noise, brightness)
- **Synthesis**: compose operations to create new sounds
- **Interpolation**: blend operation trees for morphing between sounds

All operating through DCAE's z-space, which guarantees high-quality audio output.
