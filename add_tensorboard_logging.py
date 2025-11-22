#!/usr/bin/env python3
"""
Quick script to add TensorBoard logging to train_semantic_discovery.py

This will patch your training script to add TensorBoard support.

Usage:
    python add_tensorboard_logging.py
"""

from pathlib import Path

def add_tensorboard_logging():
    """Add TensorBoard logging to training script"""

    script_path = Path('examples/train_semantic_discovery.py')

    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return

    print(f"📝 Reading {script_path}...")
    with open(script_path, 'r') as f:
        lines = f.readlines()

    # Find where to add imports
    import_added = False
    logger_init_added = False
    logger_close_added = False

    new_lines = []

    for i, line in enumerate(lines):
        new_lines.append(line)

        # Add import after other imports
        if not import_added and 'import torch' in line:
            new_lines.append('from torch.utils.tensorboard import SummaryWriter\n')
            import_added = True
            print("✅ Added TensorBoard import")

        # Add logger initialization after creating output directory
        if not logger_init_added and 'output_dir.mkdir' in line:
            new_lines.append('\n')
            new_lines.append('            # Initialize TensorBoard logger\n')
            new_lines.append('            tensorboard_dir = config.output_dir / "tensorboard_logs"\n')
            new_lines.append('            tensorboard_dir.mkdir(parents=True, exist_ok=True)\n')
            new_lines.append('            writer = SummaryWriter(log_dir=str(tensorboard_dir))\n')
            new_lines.append('            print(f"📊 TensorBoard logging to: {tensorboard_dir}")\n')
            new_lines.append('            print(f"   To view: tensorboard --logdir {tensorboard_dir}")\n')
            new_lines.append('\n')
            logger_init_added = True
            print("✅ Added TensorBoard logger initialization")

        # Add logging in training loop (after loss calculation)
        if 'total_loss.backward()' in line:
            # Add logging before backward pass
            indent = ' ' * 16
            new_lines.insert(-1, f'{indent}# Log to TensorBoard\n')
            new_lines.insert(-1, f'{indent}writer.add_scalar("Loss/total", total_loss.item(), epoch)\n')
            new_lines.insert(-1, f'{indent}writer.add_scalar("Loss/reconstruction", recon_loss.item(), epoch)\n')
            new_lines.insert(-1, f'{indent}writer.add_scalar("Loss/sparsity", sparsity_loss.item(), epoch)\n')
            new_lines.insert(-1, f'{indent}writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)\n')
            new_lines.insert(-1, f'{indent}if epoch % 10 == 0:\n')
            new_lines.insert(-1, f'{indent}    writer.flush()\n')
            new_lines.insert(-1, '\n')
            print("✅ Added TensorBoard logging in training loop")

        # Add writer.close() at the end
        if not logger_close_added and 'print("Training complete!")' in line:
            new_lines.append('\n')
            new_lines.append('            # Close TensorBoard writer\n')
            new_lines.append('            writer.close()\n')
            new_lines.append('            print(f"📊 TensorBoard logs saved to: {tensorboard_dir}")\n')
            logger_close_added = True
            print("✅ Added TensorBoard logger close")

    # Write modified script
    backup_path = script_path.with_suffix('.py.backup')
    print(f"\n💾 Creating backup: {backup_path}")
    with open(backup_path, 'w') as f:
        f.writelines(lines)

    print(f"📝 Writing modified script...")
    with open(script_path, 'w') as f:
        f.writelines(new_lines)

    print(f"\n✅ TensorBoard logging added to {script_path}!")
    print(f"\nNext steps:")
    print(f"1. conda activate ace_step")
    print(f"2. conda install tensorboard -c conda-forge")
    print(f"3. Run your training script")
    print(f"4. tensorboard --logdir /mnt/models/semantic_encoders_v2_real/tensorboard_logs")
    print(f"5. Open: http://localhost:6006")

if __name__ == '__main__':
    add_tensorboard_logging()
