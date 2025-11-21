# TensorBoard Quick Start Guide

## Overview

TensorBoard is now fully integrated into the domain encoder training pipeline. You can monitor training progress in real-time through an interactive web interface.

---

## Quick Start (3 Steps)

### 1. Start Training

```bash
# Train all domain encoders with TensorBoard logging
python examples/train_with_tensorboard.py

# Or train a specific domain
python examples/train_with_tensorboard.py --domain harmony
```

### 2. Launch TensorBoard

```bash
# Option 1: Use the launcher script (recommended)
./launch_tensorboard.sh

# Option 2: Direct command
tensorboard --logdir output/semantic_discovery/logs --port 6006
```

### 3. Open Browser

Navigate to: **http://localhost:6006**

---

## What You'll See

### Available Metrics

TensorBoard will display the following metrics in real-time:

| Metric | Description | Target |
|--------|-------------|--------|
| **Loss/train** | Training loss | < 10 |
| **Loss/val** | Validation loss | < 10 |
| **Loss/reconstruction_train** | Training reconstruction loss | < 5 |
| **Loss/reconstruction_val** | Validation reconstruction loss | < 5 |
| **Learning_Rate** | Current learning rate | Tracks scheduler |

### Key Views

1. **SCALARS Tab** - Line plots of all metrics over time
2. **HPARAMS Tab** - Hyperparameter comparison across runs
3. **GRAPHS Tab** - Neural network architecture visualization

---

## Interpreting Results

### Healthy Training (Example)

```
Epoch 1:  Loss = 1245.34
Epoch 5:  Loss = 234.56
Epoch 10: Loss = 45.67
Epoch 20: Loss = 8.92  ✅ CONVERGED
```

**What to look for:**
- Steady decrease in loss
- Training and validation losses tracking together
- Convergence to < 10 within 30-50 epochs

### Problem Signs

#### 1. Loss Not Decreasing

```
Epoch 1:  Loss = 5234.56
Epoch 10: Loss = 5198.23
Epoch 20: Loss = 5167.89  ❌ NOT CONVERGING
```

**Possible causes:**
- Features not normalized ❌
- Learning rate too low
- Model capacity insufficient

**Solution:**
- Verify using `NormalizedFeatureExtractor`
- Check features have mean ≈ 0, std ≈ 1
- Increase learning rate to 1e-2

#### 2. Loss Exploding

```
Epoch 1:  Loss = 123.45
Epoch 2:  Loss = 456.78
Epoch 3:  Loss = NaN  ❌ EXPLODED
```

**Possible causes:**
- Learning rate too high
- Gradient explosion
- Numerical instability

**Solution:**
- Reduce learning rate to 1e-3
- Add gradient clipping
- Check for NaN/Inf in features

#### 3. Overfitting

```
Epoch 20: Train Loss = 2.34  |  Val Loss = 45.67  ❌ OVERFITTING
```

**Possible causes:**
- Training too long
- Insufficient regularization
- Too few training samples

**Solution:**
- Increase dropout to 0.3
- Increase weight_decay to 1e-4
- Use early stopping

---

## Advanced Usage

### Comparing Multiple Runs

Train with different hyperparameters:

```bash
# Run 1: Default config
python examples/train_with_tensorboard.py --domain harmony

# Run 2: Higher learning rate
python examples/train_with_tensorboard.py --domain harmony --epochs 100

# Run 3: Larger batch size
python examples/train_with_tensorboard.py --domain harmony --batch-size 64
```

TensorBoard will show all runs together for comparison.

### Custom Log Directory

```bash
# Train with custom log directory
python examples/train_with_tensorboard.py --output-dir output/experiment_1

# Launch TensorBoard for that experiment
tensorboard --logdir output/experiment_1/logs
```

### Remote Access

If training on a remote server:

```bash
# On remote server
tensorboard --logdir output/semantic_discovery/logs --port 6006 --bind_all

# On local machine, set up SSH tunnel
ssh -L 6006:localhost:6006 user@remote-server

# Then open: http://localhost:6006
```

---

## TensorBoard Features

### 1. Smoothing

- Use the **Smoothing** slider to reduce noise in plots
- Recommended: 0.6-0.8 for training loss
- Helps identify trends vs. random fluctuations

### 2. Custom Scalars

Create custom layouts:

```python
from midi_generator.utils.tensorboard_logger import create_logger

logger = create_logger('harmony_encoder_v2')

# Log custom comparisons
logger.log_scalars({
    'Comparison/train_vs_val': train_loss - val_loss,
    'Comparison/improvement_rate': (prev_loss - curr_loss) / prev_loss
}, epoch)
```

### 3. Histograms

Monitor weight and gradient distributions:

```python
# Log model weights
logger.log_model_weights(encoder, step=epoch)

# Log gradients
logger.log_model_gradients(encoder, step=epoch)
```

### 4. Hyperparameter Tuning

Compare hyperparameters across runs:

```python
logger.log_hparams({
    'hidden_dim': 1024,
    'learning_rate': 0.01,
    'dropout': 0.2
}, {
    'final_val_loss': best_val_loss,
    'converged': val_loss < 10.0
})
```

---

## Troubleshooting

### Issue: Port already in use

**Error:**
```
TensorBoard port 6006 already in use
```

**Solutions:**

1. **Use different port:**
   ```bash
   ./launch_tensorboard.sh output/semantic_discovery/logs 6007
   ```

2. **Kill existing process:**
   ```bash
   kill $(lsof -t -i:6006)
   ./launch_tensorboard.sh
   ```

### Issue: No data shown in TensorBoard

**Possible causes:**
1. Training hasn't started yet
2. Wrong log directory
3. Logs not flushed to disk

**Solutions:**
1. Start training first
2. Verify log directory: `ls -la output/semantic_discovery/logs/`
3. Refresh browser (Ctrl+R)
4. Check training script has `logger.flush()` or `logger.close()`

### Issue: TensorBoard not installed

**Error:**
```
tensorboard: command not found
```

**Solution:**
```bash
pip install tensorboard
```

### Issue: Browser won't open

**Solutions:**
1. Manually navigate to: http://localhost:6006
2. Try different browser (Chrome/Firefox recommended)
3. Check firewall settings
4. Use `--bind_all` flag for external access

---

## Best Practices

### 1. Naming Conventions

Use descriptive experiment names:

```python
# Good ✅
logger = create_logger('harmony_encoder_v2_1024hidden_lr0.01')

# Bad ❌
logger = create_logger('test')
```

### 2. Regular Checkpointing

Save models at intervals:

```python
if epoch % 10 == 0:
    torch.save(encoder.state_dict(), f'checkpoint_epoch_{epoch}.pt')
```

### 3. Early Stopping

Stop when converged:

```python
if val_loss < 10.0:
    print(f"✅ Converged at epoch {epoch}!")
    break
```

### 4. Log Key Events

```python
logger.log_text('Events', f'Converged at epoch {epoch}', epoch)
logger.log_text('Config', str(config), step=0)
```

---

## Integration with Existing Code

### Minimal Integration

Add TensorBoard to existing training loop:

```python
from midi_generator.utils.tensorboard_logger import create_logger

# Initialize
logger = create_logger('my_experiment')

# In training loop
for epoch in range(num_epochs):
    train_loss = train_step()
    val_loss = validate()

    # Log metrics
    logger.log_scalars({
        'Loss/train': train_loss,
        'Loss/val': val_loss
    }, epoch)

# Close
logger.close()
```

### Full Integration

See: `examples/train_with_tensorboard.py`

---

## Useful Commands

```bash
# List all TensorBoard processes
ps aux | grep tensorboard

# Kill all TensorBoard processes
pkill -f tensorboard

# View specific experiment
tensorboard --logdir output/semantic_discovery/logs/harmony_encoder_v2

# Compare multiple experiments
tensorboard --logdir_spec harmony:output/logs/harmony,rhythm:output/logs/rhythm

# Save TensorBoard data to CSV
tensorboard --logdir output/logs --export_to_csv results.csv

# Delete old logs (careful!)
rm -rf output/semantic_discovery/logs/*
```

---

## Performance Tips

### 1. Reduce Logging Frequency

Log every N steps instead of every step:

```python
if step % 10 == 0:  # Log every 10 steps
    logger.log_scalar('Loss/train', loss, step)
```

### 2. Disable When Not Needed

```python
# For production runs without monitoring
logger = create_logger('experiment', enabled=False)  # No overhead
```

### 3. Clean Old Logs

```bash
# Keep only last 5 runs
cd output/semantic_discovery/logs
ls -t | tail -n +6 | xargs rm -rf
```

---

## Expected Training Curves

### Harmony Encoder (30 params)

```
Target: Loss < 10 within 20-30 epochs

Typical curve:
Epoch 1:  1234.56
Epoch 5:   234.56
Epoch 10:   45.67
Epoch 15:   12.34
Epoch 20:    6.78  ✅
```

### Rhythm Encoder (20 params)

```
Target: Loss < 10 within 15-25 epochs

Typical curve:
Epoch 1:   987.65
Epoch 5:   187.43
Epoch 10:   34.21
Epoch 15:    8.92  ✅
```

### All Encoders Summary

| Encoder | Params | Target Epochs | Target Loss |
|---------|--------|---------------|-------------|
| Harmony | 30 | 20-30 | < 10 |
| Rhythm | 20 | 15-25 | < 10 |
| Form | 15 | 15-20 | < 10 |
| Orchestration | 25 | 20-30 | < 10 |
| Texture | 20 | 15-25 | < 10 |
| Cross-Dimensional | 10 | 10-15 | < 1 |

---

## Resources

- **TensorBoard Documentation:** https://www.tensorflow.org/tensorboard
- **PyTorch TensorBoard Tutorial:** https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
- **Logger Utility:** `midi_generator/utils/tensorboard_logger.py`
- **Training Script:** `examples/train_with_tensorboard.py`
- **Migration Guide:** `CONVERGENCE_FIX_V2.0.md`

---

## Quick Reference Card

```bash
# START TRAINING
python examples/train_with_tensorboard.py

# LAUNCH TENSORBOARD
./launch_tensorboard.sh

# OPEN BROWSER
http://localhost:6006

# KEY METRICS TO WATCH
- Loss/train < 10 (convergence)
- Loss/val close to Loss/train (no overfitting)
- Learning_Rate decreasing over time

# STOP TENSORBOARD
Ctrl+C in terminal

# CLEAN UP
rm -rf output/semantic_discovery/logs/*
```

---

**Version:** 2.0
**Date:** 2025-11-21
**Status:** ✅ Ready to use
