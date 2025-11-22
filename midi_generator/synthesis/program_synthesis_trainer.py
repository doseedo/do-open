"""
Program Synthesis Training Infrastructure
==========================================

Train neural program synthesizer to learn MIDI transforms from examples.

Components:
1. SyntheticMIDIDataset - PyTorch Dataset for loading training data
2. ProgramSynthesisTrainer - Training loop with validation
3. Evaluation metrics - Exact match, functional equivalence

Author: Agent 8 - Transform Architecture
Phase: 4.4 (Training Infrastructure)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import pickle

from .neural_synthesizer import (
    NeuralProgramSynthesizer,
    midi_to_pianoroll
)
from .transform_dsl import DSLVocabulary, DSLProgram
from .synthetic_dataset import SyntheticExample


# ============================================================================
# PyTorch Dataset
# ============================================================================

class SyntheticMIDIDataset(Dataset):
    """
    PyTorch Dataset for neural program synthesis training.

    Loads synthetic examples and converts to tensors.
    """

    def __init__(self, examples: List[SyntheticExample],
                 max_time_steps: int = 500):
        """
        Args:
            examples: List from SyntheticDatasetGenerator.generate_dataset()
            max_time_steps: Maximum MIDI length (longer pieces truncated)
        """
        self.examples = examples
        self.max_time_steps = max_time_steps
        self.vocab = DSLVocabulary()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training example.

        Returns:
            {
                'input_midi': (time_steps, 128) pianoroll,
                'output_midi': (time_steps, 128) pianoroll,
                'dsl_tokens': (seq_len,) token indices,
                'amount': scalar,
                'lengths': {'input': int, 'output': int, 'program': int}
            }
        """
        example = self.examples[idx]

        # Convert MIDI to pianoroll
        input_pianoroll = midi_to_pianoroll(example.input_midi)
        output_pianoroll = midi_to_pianoroll(example.output_midi)

        # Truncate if too long
        if input_pianoroll.shape[0] > self.max_time_steps:
            input_pianoroll = input_pianoroll[:self.max_time_steps]
        if output_pianoroll.shape[0] > self.max_time_steps:
            output_pianoroll = output_pianoroll[:self.max_time_steps]

        # Convert DSL program to token sequence
        dsl_tokens = example.dsl_program.to_tokens()
        token_ids = [self.vocab.token_to_idx.get(t, self.vocab.token_to_idx['<UNK>'])
                    for t in dsl_tokens]

        return {
            'input_midi': torch.tensor(input_pianoroll, dtype=torch.float32),
            'output_midi': torch.tensor(output_pianoroll, dtype=torch.float32),
            'dsl_tokens': torch.tensor(token_ids, dtype=torch.long),
            'amount': torch.tensor(example.amount, dtype=torch.float32),
            'lengths': {
                'input': input_pianoroll.shape[0],
                'output': output_pianoroll.shape[0],
                'program': len(token_ids)
            }
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-length sequences.

    Pads MIDI and token sequences to batch maximum.
    """
    # Find max lengths in batch
    max_input_time = max(b['lengths']['input'] for b in batch)
    max_output_time = max(b['lengths']['output'] for b in batch)
    max_program_len = max(b['lengths']['program'] for b in batch)

    batch_size = len(batch)

    # Preallocate tensors
    input_midis = torch.zeros(batch_size, max_input_time, 128, dtype=torch.float32)
    output_midis = torch.zeros(batch_size, max_output_time, 128, dtype=torch.float32)
    dsl_tokens = torch.zeros(batch_size, max_program_len, dtype=torch.long)  # 0 = <PAD>
    amounts = torch.zeros(batch_size, dtype=torch.float32)

    # Fill tensors
    for i, b in enumerate(batch):
        input_len = b['lengths']['input']
        output_len = b['lengths']['output']
        program_len = b['lengths']['program']

        input_midis[i, :input_len] = b['input_midi']
        output_midis[i, :output_len] = b['output_midi']
        dsl_tokens[i, :program_len] = b['dsl_tokens']
        amounts[i] = b['amount']

    return {
        'input_midi': input_midis,
        'output_midi': output_midis,
        'dsl_tokens': dsl_tokens,
        'amounts': amounts
    }


def create_dataloaders(dataset: SyntheticMIDIDataset,
                      batch_size: int = 32,
                      train_split: float = 0.9,
                      num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        dataset: Complete dataset
        batch_size: Batch size
        train_split: Fraction for training (rest for validation)
        num_workers: Number of parallel workers

    Returns:
        train_loader, val_loader
    """
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# ============================================================================
# Trainer
# ============================================================================

class ProgramSynthesisTrainer:
    """
    Train neural program synthesizer.

    Handles:
    - Training loop with validation
    - Checkpointing
    - Metrics logging
    - Early stopping
    """

    def __init__(self,
                 model: NeuralProgramSynthesizer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 save_dir: Optional[Path] = None):
        """
        Initialize trainer.

        Args:
            model: NeuralProgramSynthesizer
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Save directory
        self.save_dir = Path(save_dir) if save_dir else Path('./checkpoints')
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 10

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc='Training')

        for batch in progress_bar:
            # Move to device
            input_midi = batch['input_midi'].to(self.device)
            output_midi = batch['output_midi'].to(self.device)
            target_program = batch['dsl_tokens'].to(self.device)

            # Forward pass
            logits = self.model(input_midi, output_midi, target_program)

            # Compute loss
            # Reshape for cross entropy: (batch * seq_len, vocab_size)
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            target_flat = target_program.reshape(-1)

            # Cross-entropy loss (ignore padding token 0)
            loss = F.cross_entropy(
                logits_flat,
                target_flat,
                ignore_index=0,  # Padding token
                reduction='mean'
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model.

        Returns:
            {
                'loss': validation loss,
                'token_accuracy': token-level accuracy,
                'exact_match': program exact match rate,
                'functional_match': functional equivalence rate (placeholder)
            }
        """
        self.model.eval()

        total_loss = 0
        total_correct_tokens = 0
        total_tokens = 0
        exact_matches = 0
        total_programs = 0

        num_batches = 0

        for batch in tqdm(self.val_loader, desc='Validation'):
            # Move to device
            input_midi = batch['input_midi'].to(self.device)
            output_midi = batch['output_midi'].to(self.device)
            target_program = batch['dsl_tokens'].to(self.device)

            # Forward pass
            logits = self.model(input_midi, output_midi, target_program)

            # Loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            target_flat = target_program.reshape(-1)

            loss = F.cross_entropy(
                logits_flat,
                target_flat,
                ignore_index=0,
                reduction='mean'
            )

            total_loss += loss.item()
            num_batches += 1

            # Token accuracy
            predictions = logits.argmax(dim=-1)  # (batch, seq_len)
            non_padding = (target_program != 0)
            correct = (predictions == target_program) & non_padding
            total_correct_tokens += correct.sum().item()
            total_tokens += non_padding.sum().item()

            # Exact match (entire program correct)
            for i in range(batch_size):
                pred_prog = predictions[i]
                target_prog = target_program[i]
                non_pad_mask = (target_prog != 0)

                if torch.equal(pred_prog[non_pad_mask], target_prog[non_pad_mask]):
                    exact_matches += 1

                total_programs += 1

        metrics = {
            'loss': total_loss / num_batches,
            'token_accuracy': total_correct_tokens / max(total_tokens, 1),
            'exact_match': exact_matches / max(total_programs, 1),
            'functional_match': 0.0  # Placeholder - would require execution
        }

        return metrics

    def train(self, num_epochs: int = 100, validate_every: int = 1) -> Dict[str, List]:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs
            validate_every: Validate every N epochs

        Returns:
            Training history
        """
        print(f"\n{'='*70}")
        print("Neural Program Synthesis Training")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*70}\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            print(f"Train loss: {train_loss:.4f}")

            # Validate
            if (epoch + 1) % validate_every == 0:
                metrics = self.validate()
                self.val_losses.append(metrics['loss'])

                print(f"Val loss: {metrics['loss']:.4f}")
                print(f"Token accuracy: {metrics['token_accuracy']:.2%}")
                print(f"Exact match: {metrics['exact_match']:.2%}")

                # Learning rate scheduling
                self.scheduler.step(metrics['loss'])

                # Save best model
                if metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = metrics['loss']
                    self.patience_counter = 0

                    checkpoint_path = self.save_dir / 'best_model.pt'
                    self.save_checkpoint(checkpoint_path, epoch, metrics)
                    print(f"✓ Saved best model (loss: {metrics['loss']:.4f})")

                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.max_patience:
                    print(f"\nEarly stopping triggered (patience: {self.max_patience})")
                    break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pt'
                self.save_checkpoint(checkpoint_path, epoch, metrics if epoch % validate_every == 0 else None)

        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")
        print(f"{'='*70}\n")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def save_checkpoint(self, path: Path, epoch: int, metrics: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }

        if metrics:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            print(f"Checkpoint metrics: {checkpoint['metrics']}")


# ============================================================================
# Training Entry Point
# ============================================================================

def train_neural_synthesizer(
    synthetic_examples: List[SyntheticExample],
    save_dir: Path,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: Optional[str] = None
) -> NeuralProgramSynthesizer:
    """
    Complete training pipeline.

    Args:
        synthetic_examples: Training data from SyntheticDatasetGenerator
        save_dir: Where to save checkpoints
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device ('cuda' or 'cpu', auto-detect if None)

    Returns:
        Trained model
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Create dataset
    print("Creating dataset...")
    dataset = SyntheticMIDIDataset(synthetic_examples)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=batch_size,
        train_split=0.9
    )

    # Create model
    print("Initializing model...")
    model = NeuralProgramSynthesizer(hidden_dim=512, num_layers=6)

    # Create trainer
    print("Initializing trainer...")
    trainer = ProgramSynthesisTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        save_dir=save_dir
    )

    # Train
    print("Starting training...")
    history = trainer.train(num_epochs=num_epochs)

    # Load best model
    best_model_path = save_dir / 'best_model.pt'
    if best_model_path.exists():
        trainer.load_checkpoint(best_model_path)

    return trainer.model
