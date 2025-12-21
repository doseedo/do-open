# Mute Translator Pipeline

Convert dry trumpet to muted (harmon) trumpet using latent space translation.

## Overview

```
                    TRAINING PIPELINE
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Step 1: Train Translator                                   │
│  ────────────────────────                                   │
│  dry_latent ──► MuteTranslator ──► muted_latent            │
│  (learns distribution mapping from 26 muted files)         │
│                                                             │
│  Step 2: Evaluate                                           │
│  ────────────────                                           │
│  Generate audio samples, listen, verify quality             │
│                                                             │
│  Step 3: Generate Synthetic Data                            │
│  ──────────────────────────────                             │
│  Apply translator to 229 hours of dry trumpet               │
│  Creates paired (dry, synthetic_muted) dataset              │
│                                                             │
│  Step 4: Train Student Model                                │
│  ──────────────────────────                                 │
│  Lightweight model for fast inference / VST export          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Files

```
mute_translator/
├── models.py           # Translator architectures
├── dataset.py          # Data loading for training
├── train_translator.py # Step 1: Train translator
├── evaluate_translator.py # Step 2: Evaluate & generate samples
├── generate_synthetic.py  # Step 3: Generate synthetic data
├── train_student.py    # Step 4: Train student model
├── inference.py        # Inference for both modes
└── run_pipeline.sh     # Main run script
```

## Quick Start

### With A100 attached:

```bash
# Step 1: Train the translator
./run_pipeline.sh 1

# Step 2: Evaluate (generates audio samples to listen to)
./run_pipeline.sh 2

# >>> LISTEN TO eval_output/*.wav <<<
# If the muted sound is good, continue:

# Step 3: Generate synthetic muted data from all dry files
./run_pipeline.sh 3

# Step 4: Train lightweight student model
./run_pipeline.sh 4

# Test on a new file
./run_pipeline.sh test /path/to/dry_trumpet.wav
```

## Manual Commands

### Step 1: Train Translator
```bash
python train_translator.py \
    --manifest /home/arlo/Data.backup/final_training_manifest_brass_only.json \
    --output_dir ./checkpoints \
    --batch_size 16 \
    --num_epochs 100
```

### Step 2: Evaluate
```bash
python evaluate_translator.py \
    --checkpoint ./checkpoints/best.pt \
    --output_dir ./eval_output \
    --num_samples 10
```

### Step 3: Generate Synthetic Data
```bash
python generate_synthetic.py \
    --checkpoint ./checkpoints/best.pt \
    --output_dir ./synthetic_data \
    --save_latents
```

### Step 4: Train Student
```bash
python train_student.py \
    --synthetic_manifest ./synthetic_data/synthetic_manifest.json \
    --output_dir ./student_checkpoints \
    --model_type mel \
    --num_epochs 100
```

### Inference
```bash
# Teacher mode (higher quality, uses DCAE)
python inference.py \
    --mode teacher \
    --translator_checkpoint ./checkpoints/best.pt \
    --input dry_trumpet.wav \
    --output muted_trumpet.wav

# Student mode (faster, VST-ready)
python inference.py \
    --mode student \
    --student_checkpoint ./student_checkpoints/best.pt \
    --input dry_trumpet.wav \
    --output muted_trumpet.wav

# Export to ONNX for VST
python inference.py \
    --mode student \
    --student_checkpoint ./student_checkpoints/best.pt \
    --export_onnx mute_converter.onnx \
    --input dummy.wav --output dummy_out.wav
```

## Data Stats

- **Dry trumpet**: 1,596 files, 229 hours
- **Muted trumpet**: 26 files, 37 minutes (verified GT)
- **Muted percentage**: 0.27%

## Architecture

### MuteTranslator (253K params)
- Residual network on DCAE latents [B, 8, H, T]
- 6 residual blocks with dilated convolutions
- Learns residual: `muted = dry + scale * f(dry)`

### MuteTranslatorLarge (154K params)
- Full 2D convolutions for H-T interaction
- 8 residual blocks
- Better for cross-frequency mute effects

### Student Models
- **MelStudentModel**: Operates on mel spectrograms (lighter)
- **WaveformStudentModel**: Operates on raw audio (heavier, higher quality)

## Expected Results

After training the translator:
- **Centroid shift**: +500-2000 Hz (muted sound is brighter)
- **High-frequency boost**: More energy >5kHz
- **Metallic timbre**: Characteristic harmon mute buzz
- **Preserved dynamics**: Same articulations and rhythm

## Notes

- Latent paths in manifest must be accessible (mount the disk with latents)
- DCAE checkpoint should be at `/home/arlo/Data/ACE-Step/checkpoints`
- A100 40GB is sufficient for all training steps
- Estimated training times:
  - Translator: ~1 hour
  - Synthetic generation: ~10-20 hours (for full corpus)
  - Student: ~2-4 hours
