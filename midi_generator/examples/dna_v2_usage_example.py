"""
Musical DNA V2.0 - Usage Examples
=================================

Complete usage examples for the 300D hierarchical DNA system.

Author: Agent 3 - DNA Expansion & Hierarchical Architecture
Date: 2025-11-22
Version: 1.0.0
"""

from pathlib import Path
import numpy as np

# =============================================================================
# Example 1: Basic DNA Creation and Manipulation
# =============================================================================

def example_1_basic_dna():
    """Example 1: Create and manipulate Musical DNA v2.0"""
    print("="*70)
    print("EXAMPLE 1: Basic DNA Creation and Manipulation")
    print("="*70)

    from midi_generator.learning.musical_dna_v2 import MusicalDNA, create_random_dna

    # Create zero-initialized DNA
    print("\n1. Create zero DNA:")
    dna = MusicalDNA.from_zeros(source_file="example.mid")
    print(f"   Created DNA with {len(dna.to_vector())}D")
    print(f"   Version: {dna.version}")

    # Create random DNA for testing
    print("\n2. Create random DNA:")
    random_dna = create_random_dna(seed=42)
    print(f"   Created random DNA with {len(random_dna.to_vector())}D")

    # Access hierarchical levels
    print("\n3. Access hierarchical levels:")
    print(f"   Global params:    {len(random_dna.get_global_params())}D")
    print(f"   Sectional params: {len(random_dna.get_sectional_params())}D")
    print(f"   Local params:     {len(random_dna.get_local_params())}D")

    # Access individual components
    print("\n4. Access individual components:")
    print(f"   Key context:      {len(random_dna.key_context_params)}D")
    print(f"   Tempo feel:       {len(random_dna.tempo_feel_params)}D")
    print(f"   Genre style:      {len(random_dna.genre_style_params)}D")
    print(f"   Form structure:   {len(random_dna.form_structure_params)}D")
    print(f"   Harmony:          {len(random_dna.harmony_params)}D")
    print(f"   Melody:           {len(random_dna.melody_params)}D")
    print(f"   Rhythm:           {len(random_dna.rhythm_params)}D")
    print(f"   Voicing:          {len(random_dna.voicing_params)}D")
    print(f"   Texture:          {len(random_dna.texture_params)}D")
    print(f"   Orchestration:    {len(random_dna.orchestration_params)}D")

    # Save and load
    print("\n5. Save and load:")
    save_path = Path("/tmp/example_dna_v2.json")
    random_dna.save(save_path)
    print(f"   Saved to: {save_path}")

    loaded_dna = MusicalDNA.load(save_path)
    print(f"   Loaded successfully")

    # Print summary
    print("\n6. DNA Summary:")
    print(random_dna.summary())


# =============================================================================
# Example 2: Migrating from V1.0 to V2.0
# =============================================================================

def example_2_migration():
    """Example 2: Migrate v1.0 (120D) DNA to v2.0 (300D)"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Migrating from V1.0 to V2.0")
    print("="*70)

    from midi_generator.learning.dna_migration import migrate_120d_to_300d

    # Simulate v1.0 DNA
    print("\n1. Create v1.0 (120D) DNA:")
    old_dna_dict = {
        'version': '1.0',
        'harmony': np.random.randn(30).tolist(),
        'rhythm': np.random.randn(20).tolist(),
        'form': np.random.randn(15).tolist(),
        'orchestration': np.random.randn(25).tolist(),
        'texture': np.random.randn(20).tolist(),
        'cross_dimensional': np.random.randn(10).tolist(),
        'source_file': 'old_file.mid',
        'extraction_timestamp': '2024-01-01 12:00:00',
    }
    print("   Created v1.0 DNA (120D)")

    # Migrate to v2.0
    print("\n2. Migrate to v2.0 (300D):")
    new_dna = migrate_120d_to_300d(old_dna_dict)
    print(f"   Migrated: 120D → {len(new_dna.to_vector())}D")
    print(f"   Version: {new_dna.version}")

    # Show mapping
    print("\n3. Migration mapping:")
    print("   OLD → NEW:")
    print("   harmony (30D)          → harmony (60D) [first 30D preserved]")
    print("   rhythm (20D)           → rhythm (40D) [first 20D preserved]")
    print("   form (15D)             → form_structure (20D) [extended]")
    print("   orchestration (25D)    → orchestration (40D) [extended]")
    print("   texture (20D)          → texture (30D) [extended]")
    print("   cross_dimensional (10D) → genre_style (partial)")
    print("\n   NEW PARAMETERS (initialized to zeros):")
    print("   - key_context (12D)")
    print("   - tempo_feel (8D)")
    print("   - melody (40D)")
    print("   - voicing (30D)")

    # Save migrated DNA
    print("\n4. Save migrated DNA:")
    save_path = Path("/tmp/migrated_dna_v2.json")
    new_dna.save(save_path)
    print(f"   Saved to: {save_path}")


# =============================================================================
# Example 3: Using Encoders
# =============================================================================

def example_3_encoders():
    """Example 3: Create and use v2.0 encoders"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Using V2.0 Encoders")
    print("="*70)

    try:
        import torch
    except ImportError:
        print("\n⚠️  PyTorch not available - skipping encoder example")
        return

    from midi_generator.learning.global_encoder import GlobalEncoder
    from midi_generator.learning.melody_encoder import MelodyEncoder
    from midi_generator.learning.voicing_encoder import VoicingEncoder

    # Create encoders
    print("\n1. Create new encoders:")
    global_encoder = GlobalEncoder()
    melody_encoder = MelodyEncoder()
    voicing_encoder = VoicingEncoder()
    print("   ✓ GlobalEncoder (1150D → 60D)")
    print("   ✓ MelodyEncoder (200D → 40D)")
    print("   ✓ VoicingEncoder (400D → 30D)")

    # Test forward pass
    print("\n2. Test forward pass:")
    batch_size = 2

    # Global encoder
    x_global = torch.randn(batch_size, 1150)
    global_params = global_encoder(x_global)
    print(f"   GlobalEncoder: {x_global.shape} → {global_params.shape}")

    # Melody encoder
    x_melody = torch.randn(batch_size, 200)
    melody_params = melody_encoder(x_melody)
    print(f"   MelodyEncoder: {x_melody.shape} → {melody_params.shape}")

    # Voicing encoder
    x_voicing = torch.randn(batch_size, 400)
    voicing_params = voicing_encoder(x_voicing)
    print(f"   VoicingEncoder: {x_voicing.shape} → {voicing_params.shape}")

    # Extract components
    print("\n3. Extract components:")
    global_components = global_encoder.extract_components(x_global)
    for name, tensor in global_components.items():
        print(f"   {name}: {tensor.shape}")

    # Save encoder
    print("\n4. Save encoder:")
    save_path = Path("/tmp/melody_encoder_v2.pt")
    melody_encoder.save(save_path)
    print(f"   Saved to: {save_path}")

    # Load encoder
    print("\n5. Load encoder:")
    loaded_encoder = MelodyEncoder.load(save_path)
    print(f"   Loaded successfully")


# =============================================================================
# Example 4: Using ModularEncoderFactory V2
# =============================================================================

def example_4_factory():
    """Example 4: Use ModularEncoderFactory v2.0"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Using ModularEncoderFactory V2.0")
    print("="*70)

    try:
        import torch
    except ImportError:
        print("\n⚠️  PyTorch not available - skipping factory example")
        return

    try:
        from midi_generator.learning.modular_encoder_factory_v2 import (
            ModularEncoderFactoryV2,
            MusicalDimensionV2
        )
    except ImportError:
        print("\n⚠️  ModularEncoderFactoryV2 not available")
        return

    # Create factory
    print("\n1. Create factory:")
    factory = ModularEncoderFactoryV2()
    print("   ✓ Factory created")

    # Show architecture
    print("\n2. Architecture summary:")
    factory.print_architecture_summary()

    # Get parameter allocation
    print("\n3. Parameter allocation:")
    hierarchical = factory.get_hierarchical_allocation()
    for level, count in hierarchical.items():
        print(f"   {level:12s}: {count:3d}D")

    # Create all encoders
    print("\n4. Create all encoders:")
    try:
        encoders = factory.create_all_encoders(device='cpu')
        print(f"   ✓ Created {len(encoders)} encoders")
    except Exception as e:
        print(f"   ⚠️  Could not create encoders: {e}")
        return

    # Create hierarchical encoders
    print("\n5. Create hierarchical encoders:")
    hierarchical_encoders = factory.create_hierarchical_encoders(device='cpu')
    print(f"   Global encoders:    {len(hierarchical_encoders['global'])}")
    print(f"   Sectional encoders: {len(hierarchical_encoders['sectional'])}")
    print(f"   Local encoders:     {len(hierarchical_encoders['local'])}")

    # Save all encoders
    print("\n6. Save all encoders:")
    save_dir = Path("/tmp/encoders_v2")
    factory.save_all_encoders(save_dir)
    print(f"   ✓ Saved to: {save_dir}")


# =============================================================================
# Example 5: End-to-End Workflow
# =============================================================================

def example_5_end_to_end():
    """Example 5: Complete end-to-end workflow"""
    print("\n" + "="*70)
    print("EXAMPLE 5: End-to-End Workflow")
    print("="*70)

    from midi_generator.learning.musical_dna_v2 import MusicalDNA

    print("\nStep 1: Extract DNA from MIDI (simulation)")
    print("   In production:")
    print("   - Load MIDI file")
    print("   - Extract 1150D features using DeepFeatureExtractor")
    print("   - Pass features through encoders")
    print("   - Combine encoder outputs into MusicalDNA")

    print("\nStep 2: Create DNA (simulated)")
    dna = MusicalDNA(
        # Global (60D)
        key_context_params=np.random.randn(12),
        tempo_feel_params=np.random.randn(8),
        genre_style_params=np.random.randn(20),
        form_structure_params=np.random.randn(20),
        # Sectional (140D)
        harmony_params=np.random.randn(60),
        melody_params=np.random.randn(40),
        rhythm_params=np.random.randn(40),
        # Local (100D)
        voicing_params=np.random.randn(30),
        texture_params=np.random.randn(30),
        orchestration_params=np.random.randn(40),
        # Metadata
        source_file="example.mid"
    )
    print(f"   ✓ Created DNA: {len(dna.to_vector())}D")

    print("\nStep 3: Edit DNA parameters")
    print("   Original harmony mean: {:.3f}".format(np.mean(dna.harmony_params)))
    dna.harmony_params *= 1.2  # Increase harmonic complexity
    print("   Modified harmony mean: {:.3f}".format(np.mean(dna.harmony_params)))

    print("\nStep 4: Save DNA")
    save_path = Path("/tmp/edited_dna_v2.json")
    dna.save(save_path)
    print(f"   ✓ Saved to: {save_path}")

    print("\nStep 5: Load DNA")
    loaded_dna = MusicalDNA.load(save_path)
    print(f"   ✓ Loaded DNA: {len(loaded_dna.to_vector())}D")

    print("\nStep 6: Generate MIDI from DNA (future work)")
    print("   In production:")
    print("   - Pass DNA through decoder")
    print("   - Generate MIDI from decoded features")
    print("   - Save MIDI file")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MUSICAL DNA V2.0 - USAGE EXAMPLES")
    print("="*70)

    examples = [
        example_1_basic_dna,
        example_2_migration,
        example_3_encoders,
        example_4_factory,
        example_5_end_to_end,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n⚠️  {example_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
