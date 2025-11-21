"""
Semantic Feature Encoder Demo - Agent 3
========================================

Demonstration of the SemanticFeatureEncoder for discovering musical parameters.

This example shows:
1. Creating and configuring an encoder
2. Simulating training data
3. Computing losses
4. Extracting semantic features
5. Analyzing learned features
6. Saving and loading models

Author: Agent 3 - Neural Architecture
Date: November 21, 2025
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import numpy as np
    from midi_generator.learning.semantic_encoder import (
        EncoderConfig,
        SemanticFeatureEncoder,
        create_default_encoder,
        compute_reconstruction_quality,
        analyze_semantic_features
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("   Install: pip install torch numpy")
    DEPENDENCIES_AVAILABLE = False
    sys.exit(1)


def demo_basic_usage():
    """Demonstrate basic encoder usage"""
    print("\n" + "="*70)
    print("1. BASIC USAGE")
    print("="*70)

    # Create encoder with custom configuration
    print("\n📦 Creating encoder...")
    config = EncoderConfig(
        num_semantic_features=30,  # Discover 30 musical parameters
        num_locality_types=12,      # 12 musical transformations
        hidden_dim=512,
        reconstruction_weight=1.0,
        locality_weight=0.5,
        sparsity_weight=0.01
    )

    encoder = SemanticFeatureEncoder(config)
    print(f"   ✅ Created encoder with {config.num_semantic_features} semantic features")
    print(f"   📊 Architecture:")
    print(f"      Encoder: [200] → [512] → [30]")
    print(f"      Decoder: [30] → [512] → [200]")

    return encoder


def demo_forward_pass(encoder):
    """Demonstrate forward pass"""
    print("\n" + "="*70)
    print("2. FORWARD PASS")
    print("="*70)

    # Generate sample input (simulating 200D features from OptimizedFeatureExtractor)
    batch_size = 16
    print(f"\n📥 Input: Batch of {batch_size} feature vectors (200D)")
    features = torch.randn(batch_size, 200)

    # Forward pass
    print("🔄 Running forward pass...")
    output = encoder(features, compute_locality=False)

    print("✅ Output:")
    print(f"   Semantic features: {output['semantic_features'].shape}")
    print(f"   Reconstructed:     {output['reconstructed'].shape}")

    # Show sample semantic features
    z = output['semantic_features']
    print(f"\n📊 Sample semantic features (first 5):")
    print(f"   Range: [{z.min():.3f}, {z.max():.3f}]")
    print(f"   Mean:  {z.mean():.3f}")
    print(f"   Std:   {z.std():.3f}")

    return features, output


def demo_loss_computation(encoder, features):
    """Demonstrate loss computation"""
    print("\n" + "="*70)
    print("3. LOSS COMPUTATION")
    print("="*70)

    # Generate transformed features (simulating locality transformations)
    batch_size = features.shape[0]
    features_transformed = torch.randn(batch_size, 200)
    locality_labels = torch.randint(0, 12, (batch_size,))

    print(f"\n📥 Input:")
    print(f"   Original features: {features.shape}")
    print(f"   Transformed:       {features_transformed.shape}")
    print(f"   Locality labels:   {locality_labels.shape}")

    # Compute loss
    print("\n🔄 Computing losses...")
    loss_dict = encoder.compute_loss(
        features,
        features_transformed,
        locality_labels
    )

    print("✅ Loss components:")
    print(f"   Total loss:         {loss_dict['total_loss'].item():.4f}")
    print(f"   Reconstruction:     {loss_dict['reconstruction_loss'].item():.4f}")
    print(f"   Locality:           {loss_dict['locality_loss'].item():.4f}")
    print(f"   Sparsity:           {loss_dict['sparsity_loss'].item():.4f}")
    print(f"   Locality accuracy:  {loss_dict['locality_accuracy'].item():.2%}")

    return loss_dict


def demo_feature_extraction(encoder):
    """Demonstrate semantic feature extraction"""
    print("\n" + "="*70)
    print("4. SEMANTIC FEATURE EXTRACTION")
    print("="*70)

    # Single sample extraction
    print("\n📊 Single sample extraction:")
    single_features = torch.randn(200)
    encoder.eval()

    with torch.no_grad():
        semantic_features = encoder.extract_semantic_features(single_features)

    print(f"   Input shape:  {single_features.shape}")
    print(f"   Output shape: {semantic_features.shape}")
    print(f"   Features: {semantic_features[:5]}...")

    # Batch extraction
    print("\n📊 Batch extraction:")
    batch_features = torch.randn(100, 200)

    with torch.no_grad():
        semantic_batch = encoder.extract_semantic_features(batch_features, as_numpy=True)

    print(f"   Input shape:  {batch_features.shape}")
    print(f"   Output shape: {semantic_batch.shape}")
    print(f"   Type: {type(semantic_batch)}")

    return semantic_batch


def demo_feature_analysis(encoder):
    """Demonstrate feature analysis"""
    print("\n" + "="*70)
    print("5. FEATURE ANALYSIS")
    print("="*70)

    # Generate sample dataset
    print("\n📊 Generating sample dataset...")
    dataset = torch.randn(200, 200)

    # Analyze features
    print("🔍 Analyzing learned features...")
    analysis = analyze_semantic_features(encoder, dataset, top_k=10)

    print("\n✅ Analysis results:")

    # Feature importance
    importance = analysis['feature_importance']
    top_features = analysis['top_features']

    print(f"\n   Top 10 most important features:")
    for i, feat_idx in enumerate(top_features):
        print(f"      {i+1}. Feature {feat_idx}: importance = {importance[feat_idx]:.3f}")

    # Activation statistics
    stats = analysis['activation_statistics']
    print(f"\n   Activation statistics:")
    print(f"      Mean range:  [{stats['mean'].min():.3f}, {stats['mean'].max():.3f}]")
    print(f"      Std range:   [{stats['std'].min():.3f}, {stats['std'].max():.3f}]")

    # Sparsity
    sparsity = analysis['sparsity']
    print(f"\n   Sparsity: {sparsity:.1%} of activations are near-zero")

    return analysis


def demo_reconstruction_quality(encoder):
    """Demonstrate reconstruction quality metrics"""
    print("\n" + "="*70)
    print("6. RECONSTRUCTION QUALITY")
    print("="*70)

    # Generate test data
    test_features = torch.randn(100, 200)

    print("\n🔍 Computing reconstruction quality...")
    quality = compute_reconstruction_quality(encoder, test_features)

    print("✅ Quality metrics:")
    print(f"   MSE:         {quality['mse']:.4f}")
    print(f"   MAE:         {quality['mae']:.4f}")
    print(f"   R²:          {quality['r2']:.4f}")
    print(f"   Correlation: {quality['correlation']:.4f}")

    return quality


def demo_save_load(encoder):
    """Demonstrate saving and loading"""
    print("\n" + "="*70)
    print("7. SAVE AND LOAD")
    print("="*70)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "semantic_encoder.pt"

        # Save
        print(f"\n💾 Saving model to {model_path.name}...")
        encoder.save(model_path)
        print(f"   ✅ Saved ({model_path.stat().st_size / 1024:.1f} KB)")

        # Load
        print(f"\n📂 Loading model...")
        loaded_encoder = SemanticFeatureEncoder.load(model_path)
        print(f"   ✅ Loaded successfully")

        # Verify
        print(f"\n🔍 Verifying loaded model...")
        test_input = torch.randn(8, 200)
        encoder.eval()
        loaded_encoder.eval()

        with torch.no_grad():
            out1 = encoder.extract_semantic_features(test_input)
            out2 = loaded_encoder.extract_semantic_features(test_input)

        diff = torch.abs(out1 - out2).max().item()
        print(f"   Max difference: {diff:.6f}")

        if diff < 1e-5:
            print("   ✅ Models are identical!")
        else:
            print("   ⚠️  Models differ slightly")


def demo_training_simulation():
    """Simulate a simple training loop"""
    print("\n" + "="*70)
    print("8. TRAINING SIMULATION")
    print("="*70)

    print("\n📦 Creating fresh encoder for training...")
    config = EncoderConfig(
        num_semantic_features=20,
        learning_rate=1e-3
    )
    encoder = SemanticFeatureEncoder(config)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate)

    print(f"   ✅ Encoder created")
    print(f"   🎯 Optimizer: Adam (lr={config.learning_rate})")

    # Training loop
    num_epochs = 5
    print(f"\n🏋️  Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Generate fake training data
        features = torch.randn(32, 200)
        features_transformed = torch.randn(32, 200)
        locality_labels = torch.randint(0, 12, (32,))

        # Forward pass
        loss_dict = encoder.compute_loss(
            features,
            features_transformed,
            locality_labels
        )
        loss = loss_dict['total_loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        print(f"   Epoch {epoch+1}/{num_epochs}: loss = {loss.item():.4f}")

    print("\n✅ Training complete!")

    # Test inference
    print("\n🔍 Testing inference...")
    encoder.eval()
    with torch.no_grad():
        test_features = torch.randn(10, 200)
        semantic = encoder.extract_semantic_features(test_features)
    print(f"   ✅ Extracted {semantic.shape[1]} semantic features from {semantic.shape[0]} samples")


def main():
    """Main demonstration"""
    print("="*70)
    print("SEMANTIC FEATURE ENCODER DEMO - AGENT 3")
    print("="*70)
    print("\nThis demo shows how to use the SemanticFeatureEncoder")
    print("for discovering musical parameters from MIDI features.")

    if not DEPENDENCIES_AVAILABLE:
        return

    # Run demonstrations
    encoder = demo_basic_usage()
    features, output = demo_forward_pass(encoder)
    loss_dict = demo_loss_computation(encoder, features)
    semantic_features = demo_feature_extraction(encoder)
    analysis = demo_feature_analysis(encoder)
    quality = demo_reconstruction_quality(encoder)
    demo_save_load(encoder)
    demo_training_simulation()

    # Summary
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Integrate with Agent 4 (Gap Dataset) for real training data")
    print("2. Use Agent 1 (Musical Locality) for locality transformations")
    print("3. Train on real MIDI corpus")
    print("4. Interpret features with Agent 6")
    print("5. Register discovered parameters with UniversalParameterRegistry")
    print("\nFor more information, see:")
    print("  - midi_generator/learning/semantic_encoder.py")
    print("  - midi_generator/learning/test_semantic_encoder.py")
    print("  - docs/SEMANTIC_FEATURE_DISCOVERY.md (coming soon)")


if __name__ == "__main__":
    main()
