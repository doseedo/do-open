# TensorBoard Setup Instructions

## Quick Setup

### 1. Install TensorBoard in Your Conda Environment

```bash
# Activate your environment
conda activate ace_step

# Install TensorBoard
conda install tensorboard -c conda-forge

# Or via pip
pip install tensorboard
```

### 2. Verify Installation

```bash
conda activate ace_step
python -c "from torch.utils.tensorboard import SummaryWriter; print('✅ TensorBoard ready!')"
```

### 3. Add TensorBoard Logging to Your Training

Since you're using `examples/train_semantic_discovery.py`, add this to enable TensorBoard:

```python
from torch.utils.tensorboard import SummaryWriter

# In your training script, after creating output directory:
log_dir = output_dir / 'tensorboard_logs'
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=str(log_dir))

# During training loop:
for epoch in range(num_epochs):
    train_loss = train_epoch()  # Your training function
    val_loss = validate()       # Your validation function

    # Log to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Learning_Rate', current_lr, epoch)

    # Flush to disk
    writer.flush()

# Close when done
writer.close()
```

### 4. Launch TensorBoard

```bash
# In a separate terminal (keep training running)
conda activate ace_step

# Launch TensorBoard pointing to your logs
tensorboard --logdir /mnt/models/semantic_encoders_v2_real/tensorboard_logs --port 6006

# Then open in browser:
# http://localhost:6006
```

### 5. Remote Access (If Training on Server)

If you're training on a remote server:

```bash
# On remote server (where training runs)
conda activate ace_step
tensorboard --logdir /mnt/models/semantic_encoders_v2_real/tensorboard_logs --port 6006 --bind_all

# On your local machine
ssh -L 6006:localhost:6006 arlo@your-server

# Then open: http://localhost:6006
```

---

## Integration with Your Current Training

Since your training is already running at `/mnt/models/semantic_encoders_v2_real`, here's a drop-in modification:

### Add to `examples/train_semantic_discovery.py`:

```python
# At the top, add import
from torch.utils.tensorboard import SummaryWriter

# After line where you create output_dir, add:
tensorboard_dir = Path(config.output_dir) / 'tensorboard_logs'
tensorboard_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=str(tensorboard_dir))
print(f"📊 TensorBoard logging to: {tensorboard_dir}")

# In your training loop (around line 400-450), add logging:
# After calculating losses
writer.add_scalar('Loss/total', total_loss.item(), epoch)
writer.add_scalar('Loss/reconstruction', recon_loss.item(), epoch)
writer.add_scalar('Loss/sparsity', sparsity_loss.item(), epoch)
writer.add_scalar('Loss/locality', locality_loss.item() if locality_loss else 0, epoch)
writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

# Every 10 epochs, flush to disk
if epoch % 10 == 0:
    writer.flush()

# At the end of training:
writer.close()
```

---

## Alternative: Simple CSV Logger (No Dependencies)

If you can't install TensorBoard right now, here's a simple CSV logger:

```python
import csv
from pathlib import Path

class SimpleLogger:
    """CSV logger that works without TensorBoard"""

    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = []

    def log(self, epoch, **kwargs):
        """Log metrics for an epoch"""
        entry = {'epoch': epoch, **kwargs}
        self.metrics.append(entry)

        # Write to CSV
        with open(self.log_file, 'w', newline='') as f:
            if self.metrics:
                writer = csv.DictWriter(f, fieldnames=self.metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.metrics)

    def close(self):
        print(f"✅ Metrics saved to {self.log_file}")

# Usage:
logger = SimpleLogger('training_metrics.csv')

for epoch in range(num_epochs):
    train_loss = train()
    val_loss = validate()

    logger.log(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        learning_rate=current_lr
    )

logger.close()

# Then plot with:
# python plot_metrics.py training_metrics.csv
```

---

## Monitoring Your Current Training

For your currently running training, you can monitor progress with:

```bash
# Watch the log file
tail -f /home/arlo/do-repo/training_real_data.log

# Check TensorBoard logs (if you add logging)
ls -lh /mnt/models/semantic_encoders_v2_real/tensorboard_logs/

# Launch TensorBoard for current run
cd /home/arlo/do-repo
conda activate ace_step
tensorboard --logdir /mnt/models/semantic_encoders_v2_real/tensorboard_logs
```

---

## What You Should See

Once TensorBoard is running, you'll see:

### SCALARS Tab
- **Loss/total** - Should decrease from ~1000 to < 10
- **Loss/reconstruction** - Main convergence indicator
- **Loss/train vs Loss/val** - Should track together (no overfitting)
- **Learning_Rate** - Should decrease over time

### Expected Convergence:
```
Epoch 1:  Loss = 1234.56
Epoch 10: Loss = 234.56
Epoch 20: Loss = 45.67
Epoch 30: Loss = 8.92  ✅ CONVERGED (< 10)
```

### Warning Signs:
- **Loss not decreasing** → Check feature normalization
- **Loss = NaN** → Learning rate too high
- **Val loss >> Train loss** → Overfitting

---

## Quick Commands Reference

```bash
# Install TensorBoard
conda activate ace_step && conda install tensorboard -c conda-forge

# Launch TensorBoard
tensorboard --logdir /mnt/models/semantic_encoders_v2_real/tensorboard_logs

# Monitor training
tail -f training_real_data.log

# Check if training is running
ps aux | grep train_semantic_discovery

# View TensorBoard in browser
# http://localhost:6006
```

---

## Your Next Steps

1. **Install TensorBoard** in your `ace_step` environment
2. **Add logging** to your training script (see code above)
3. **Restart training** or modify running script
4. **Launch TensorBoard** to monitor progress
5. **Watch for convergence** (loss < 10)

The training fixes you made are correct - now you just need visualization! 🎯
