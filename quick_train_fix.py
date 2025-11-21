#!/usr/bin/env python3
"""
Quick training fix - simplified training loop that actually works
"""
import sys
from pathlib import Path
sys.path.insert(0, '/home/arlo/do-repo')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Simple training that works
def train_encoder_simple(encoder, train_loader, val_loader, device, epochs=10):
    """Simple working training loop"""
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

    for epoch in range(epochs):
        encoder.train()
        total_loss = 0

        for batch_features in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features = batch_features.to(device)

            optimizer.zero_grad()

            # Forward pass
            semantic_features = encoder(batch_features)

            # Simple reconstruction loss
            reconstructed = encoder.decode(semantic_features) if hasattr(encoder, 'decode') else semantic_features
            loss = nn.MSELoss()(reconstructed, batch_features)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    return encoder

if __name__ == "__main__":
    print("Training fix loaded - use train_encoder_simple() function")
