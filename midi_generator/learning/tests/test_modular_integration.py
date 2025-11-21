"""
Integration Tests for Modular Semantic Discovery Pipeline
==========================================================

Comprehensive tests for Agent 8's integration components:
- ModularEncoderFactory
- ModularSemanticDiscoveryPipeline
- MusicalDNA
- Parallel training infrastructure

Author: Agent 8 - Integration Pipeline Builder
Date: November 21, 2025
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Check for dependencies
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

# Import modules to test
try:
    from midi_generator.learning.modular_encoder_factory import (
        ModularEncoderFactory,
        MusicalDimension,
        DimensionSpec,
        HARMONY_SPEC,
        RHYTHM_SPEC,
        create_default_factory
    )
    FACTORY_AVAILABLE = True
except ImportError:
    FACTORY_AVAILABLE = False

try:
    from midi_generator.learning.modular_discovery_pipeline import (
        ModularSemanticDiscoveryPipeline,
        ModularPipelineConfig,
        MusicalDNA
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


# =============================================================================
# Test Cases
# =============================================================================

@unittest.skipIf(not FACTORY_AVAILABLE, "ModularEncoderFactory not available")
class TestModularEncoderFactory(unittest.TestCase):
    """Test ModularEncoderFactory"""

    def setUp(self):
        """Set up test fixtures"""
        self.factory = create_default_factory()

    def test_factory_initialization(self):
        """Test factory initializes with all dimension specs"""
        self.assertEqual(len(self.factory.dimension_specs), 6)
        self.assertIn(MusicalDimension.HARMONY, self.factory.dimension_specs)
        self.assertIn(MusicalDimension.RHYTHM, self.factory.dimension_specs)
        self.assertIn(MusicalDimension.FORM, self.factory.dimension_specs)
        self.assertIn(MusicalDimension.ORCHESTRATION, self.factory.dimension_specs)
        self.assertIn(MusicalDimension.TEXTURE, self.factory.dimension_specs)
        self.assertIn(MusicalDimension.CROSS_DIMENSIONAL, self.factory.dimension_specs)

    def test_dimension_specs(self):
        """Test dimension specifications are correct"""
        # Harmony spec
        harmony_spec = self.factory.get_dimension_spec(MusicalDimension.HARMONY)
        self.assertEqual(harmony_spec.num_params, 30)
        self.assertEqual(harmony_spec.input_dim, 200)
        self.assertGreater(len(harmony_spec.locality_functions), 0)

        # Rhythm spec
        rhythm_spec = self.factory.get_dimension_spec(MusicalDimension.RHYTHM)
        self.assertEqual(rhythm_spec.num_params, 20)

        # Form spec
        form_spec = self.factory.get_dimension_spec(MusicalDimension.FORM)
        self.assertEqual(form_spec.num_params, 15)

        # Orchestration spec
        orch_spec = self.factory.get_dimension_spec(MusicalDimension.ORCHESTRATION)
        self.assertEqual(orch_spec.num_params, 25)

        # Texture spec
        texture_spec = self.factory.get_dimension_spec(MusicalDimension.TEXTURE)
        self.assertEqual(texture_spec.num_params, 20)

        # Cross-dimensional spec
        cross_spec = self.factory.get_dimension_spec(MusicalDimension.CROSS_DIMENSIONAL)
        self.assertEqual(cross_spec.num_params, 10)
        self.assertEqual(cross_spec.input_dim, 110)  # Concatenated domain params

    def test_total_parameters(self):
        """Test total parameter count is 120"""
        total = self.factory.get_total_parameters()
        self.assertEqual(total, 120)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_create_single_encoder(self):
        """Test creating a single encoder"""
        encoder = self.factory.create_encoder(MusicalDimension.HARMONY, device='cpu')
        self.assertIsNotNone(encoder)
        self.assertEqual(encoder.config.num_semantic_features, 30)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_create_all_encoders(self):
        """Test creating all encoders"""
        encoders = self.factory.create_all_encoders(device='cpu')
        self.assertEqual(len(encoders), 6)

        # Check each encoder exists
        self.assertIn(MusicalDimension.HARMONY, encoders)
        self.assertIn(MusicalDimension.RHYTHM, encoders)
        self.assertIn(MusicalDimension.FORM, encoders)
        self.assertIn(MusicalDimension.ORCHESTRATION, encoders)
        self.assertIn(MusicalDimension.TEXTURE, encoders)
        self.assertIn(MusicalDimension.CROSS_DIMENSIONAL, encoders)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_encoder_registry(self):
        """Test encoder registry tracking"""
        encoder = self.factory.create_encoder(MusicalDimension.HARMONY, device='cpu')

        # Check registry
        registered = self.factory.get_encoder(MusicalDimension.HARMONY)
        self.assertIsNotNone(registered)
        self.assertEqual(registered, encoder)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_save_load_encoders(self):
        """Test saving and loading encoders"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create and save encoders
            self.factory.create_all_encoders(device='cpu')
            self.factory.save_all_encoders(tmpdir)

            # Check files exist
            self.assertTrue((tmpdir / "harmony_encoder.pt").exists())
            self.assertTrue((tmpdir / "dimension_specs.json").exists())

            # Load encoders
            factory2 = create_default_factory()
            loaded = factory2.load_all_encoders(tmpdir, device='cpu')
            self.assertEqual(len(loaded), 6)


@unittest.skipIf(not PIPELINE_AVAILABLE, "ModularSemanticDiscoveryPipeline not available")
class TestMusicalDNA(unittest.TestCase):
    """Test MusicalDNA class"""

    def test_dna_initialization(self):
        """Test creating MusicalDNA"""
        dna = MusicalDNA(
            harmony_params=np.random.randn(30),
            rhythm_params=np.random.randn(20),
            form_params=np.random.randn(15),
            orchestration_params=np.random.randn(25),
            texture_params=np.random.randn(20),
            cross_params=np.random.randn(10),
            source_file="test.mid"
        )

        self.assertEqual(len(dna.harmony_params), 30)
        self.assertEqual(len(dna.rhythm_params), 20)
        self.assertEqual(len(dna.form_params), 15)
        self.assertEqual(len(dna.orchestration_params), 25)
        self.assertEqual(len(dna.texture_params), 20)
        self.assertEqual(len(dna.cross_params), 10)
        self.assertEqual(dna.source_file, "test.mid")

    def test_dna_to_vector(self):
        """Test converting DNA to 120D vector"""
        dna = MusicalDNA(
            harmony_params=np.ones(30),
            rhythm_params=np.ones(20) * 2,
            form_params=np.ones(15) * 3,
            orchestration_params=np.ones(25) * 4,
            texture_params=np.ones(20) * 5,
            cross_params=np.ones(10) * 6
        )

        vector = dna.to_vector()
        self.assertEqual(len(vector), 120)

        # Check concatenation order
        self.assertEqual(vector[0], 1.0)   # harmony
        self.assertEqual(vector[30], 2.0)  # rhythm
        self.assertEqual(vector[50], 3.0)  # form
        self.assertEqual(vector[65], 4.0)  # orchestration
        self.assertEqual(vector[90], 5.0)  # texture
        self.assertEqual(vector[110], 6.0) # cross

    def test_dna_from_vector(self):
        """Test creating DNA from 120D vector"""
        vector = np.arange(120, dtype=float)
        dna = MusicalDNA.from_vector(vector, source_file="test.mid")

        self.assertEqual(len(dna.harmony_params), 30)
        self.assertEqual(len(dna.rhythm_params), 20)
        self.assertEqual(len(dna.form_params), 15)
        self.assertEqual(len(dna.orchestration_params), 25)
        self.assertEqual(len(dna.texture_params), 20)
        self.assertEqual(len(dna.cross_params), 10)

        # Check values
        self.assertEqual(dna.harmony_params[0], 0.0)
        self.assertEqual(dna.rhythm_params[0], 30.0)
        self.assertEqual(dna.form_params[0], 50.0)
        self.assertEqual(dna.orchestration_params[0], 65.0)
        self.assertEqual(dna.texture_params[0], 90.0)
        self.assertEqual(dna.cross_params[0], 110.0)

    def test_dna_to_dict(self):
        """Test converting DNA to dictionary"""
        dna = MusicalDNA(
            harmony_params=np.ones(30),
            rhythm_params=np.ones(20),
            form_params=np.ones(15),
            orchestration_params=np.ones(25),
            texture_params=np.ones(20),
            cross_params=np.ones(10),
            source_file="test.mid"
        )

        dna_dict = dna.to_dict()

        self.assertIn('harmony', dna_dict)
        self.assertIn('rhythm', dna_dict)
        self.assertIn('form', dna_dict)
        self.assertIn('orchestration', dna_dict)
        self.assertIn('texture', dna_dict)
        self.assertIn('cross_dimensional', dna_dict)
        self.assertIn('source_file', dna_dict)

        self.assertEqual(len(dna_dict['harmony']), 30)
        self.assertEqual(dna_dict['source_file'], "test.mid")

    def test_dna_save_load(self):
        """Test saving and loading DNA"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dna_path = Path(tmpdir) / "test_dna.json"

            # Create and save DNA
            original = MusicalDNA(
                harmony_params=np.random.randn(30),
                rhythm_params=np.random.randn(20),
                form_params=np.random.randn(15),
                orchestration_params=np.random.randn(25),
                texture_params=np.random.randn(20),
                cross_params=np.random.randn(10),
                source_file="test.mid"
            )
            original.save(dna_path)

            # Load DNA
            loaded = MusicalDNA.load(dna_path)

            # Compare
            np.testing.assert_array_almost_equal(
                original.harmony_params,
                loaded.harmony_params
            )
            self.assertEqual(original.source_file, loaded.source_file)


@unittest.skipIf(not PIPELINE_AVAILABLE, "ModularSemanticDiscoveryPipeline not available")
class TestModularPipeline(unittest.TestCase):
    """Test ModularSemanticDiscoveryPipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.tmpdir = tempfile.mkdtemp()
        self.tmpdir = Path(self.tmpdir)

        self.config = ModularPipelineConfig(
            midi_corpus_dir=self.tmpdir / "corpus",
            output_dir=self.tmpdir / "output",
            max_files=5,
            max_epochs=2,
            verbose=False
        )

        # Create dummy corpus directory
        self.config.midi_corpus_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.tmpdir)

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = ModularSemanticDiscoveryPipeline(self.config)
        self.assertIsNotNone(pipeline.factory)
        self.assertEqual(len(pipeline.encoders), 0)  # Not created yet
        self.assertFalse(pipeline.is_trained)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_create_encoders(self):
        """Test creating encoders"""
        pipeline = ModularSemanticDiscoveryPipeline(self.config)
        pipeline.create_encoders()

        self.assertEqual(len(pipeline.encoders), 6)
        self.assertIn(MusicalDimension.HARMONY, pipeline.encoders)
        self.assertIn(MusicalDimension.CROSS_DIMENSIONAL, pipeline.encoders)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_save_load_pipeline(self):
        """Test saving and loading pipeline"""
        # Create pipeline
        pipeline = ModularSemanticDiscoveryPipeline(self.config)
        pipeline.create_encoders()
        pipeline.is_trained = True

        # Save
        save_dir = self.tmpdir / "pipeline_save"
        pipeline.save(save_dir)

        # Check files exist
        self.assertTrue((save_dir / "harmony_encoder.pt").exists())
        self.assertTrue((save_dir / "dimension_specs.json").exists())

        # Load
        pipeline2 = ModularSemanticDiscoveryPipeline(self.config)
        pipeline2.load(save_dir)

        self.assertEqual(len(pipeline2.encoders), 6)
        self.assertTrue(pipeline2.is_trained)


@unittest.skipIf(not PIPELINE_AVAILABLE, "ModularSemanticDiscoveryPipeline not available")
class TestParameterRegistry(unittest.TestCase):
    """Test unified parameter registry"""

    def test_registry_file_exists(self):
        """Test parameter registry file exists and is valid JSON"""
        registry_path = Path(__file__).parent.parent / "modular_parameter_registry.json"

        self.assertTrue(registry_path.exists(), "Parameter registry file not found")

        # Load and validate JSON
        with open(registry_path, 'r') as f:
            registry = json.load(f)

        self.assertIn('metadata', registry)
        self.assertIn('dimensions', registry)
        self.assertIn('locality_functions', registry)

    def test_registry_metadata(self):
        """Test registry metadata"""
        registry_path = Path(__file__).parent.parent / "modular_parameter_registry.json"

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        metadata = registry['metadata']
        self.assertEqual(metadata['total_parameters'], 120)
        self.assertIn('version', metadata)
        self.assertIn('author', metadata)

    def test_registry_dimensions(self):
        """Test registry dimensions are correct"""
        registry_path = Path(__file__).parent.parent / "modular_parameter_registry.json"

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        dimensions = registry['dimensions']

        # Check all dimensions present
        self.assertIn('harmony', dimensions)
        self.assertIn('rhythm', dimensions)
        self.assertIn('form', dimensions)
        self.assertIn('orchestration', dimensions)
        self.assertIn('texture', dimensions)
        self.assertIn('cross_dimensional', dimensions)

        # Check parameter counts
        self.assertEqual(dimensions['harmony']['count'], 30)
        self.assertEqual(dimensions['rhythm']['count'], 20)
        self.assertEqual(dimensions['form']['count'], 15)
        self.assertEqual(dimensions['orchestration']['count'], 25)
        self.assertEqual(dimensions['texture']['count'], 20)
        self.assertEqual(dimensions['cross_dimensional']['count'], 10)

        # Check total is 120
        total = sum(dim['count'] for dim in dimensions.values())
        self.assertEqual(total, 120)


# =============================================================================
# Test Suite
# =============================================================================

def suite():
    """Create test suite"""
    suite = unittest.TestSuite()

    # Factory tests
    if FACTORY_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestModularEncoderFactory))

    # Pipeline tests
    if PIPELINE_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestMusicalDNA))
        suite.addTest(unittest.makeSuite(TestModularPipeline))
        suite.addTest(unittest.makeSuite(TestParameterRegistry))

    return suite


if __name__ == '__main__':
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())

    # Print summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
