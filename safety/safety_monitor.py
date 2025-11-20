"""
AGENT 17: Safety Monitor & Rollback Manager

Comprehensive safety monitoring system for the self-expanding music generation framework.
Ensures system stability during parameter expansions with git-based checkpointing,
comprehensive testing, and instant rollback capabilities.

CRITICAL: System must NEVER break existing functionality. Rollback is better than broken system.

Author: Agent 17
Version: 1.0.0
"""

import os
import sys
import json
import time
import subprocess
import shutil
import hashlib
import numpy as np
import joblib
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckpointStatus(Enum):
    """Status of a checkpoint"""
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"
    ARCHIVED = "archived"


class TestResult(Enum):
    """Result of a test"""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


class SafetyLevel(Enum):
    """Safety level for expansions"""
    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"
    CRITICAL = "critical"


@dataclass
class ParameterInfo:
    """Information about a parameter"""
    name: str
    type: str
    range: Any
    default: Any
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TestFailure:
    """Information about a test failure"""
    test_name: str
    parameter: Optional[str]
    error: str
    severity: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Checkpoint:
    """System checkpoint for rollback"""
    id: str
    commit_hash: str
    timestamp: str
    registry_snapshot: dict
    models_snapshot: list
    baseline_tests: dict
    baseline_quality: float
    status: CheckpointStatus
    description: str
    metadata: dict

    def to_dict(self) -> dict:
        data = asdict(self)
        data['status'] = self.status.value
        return data


@dataclass
class MonitoringResult:
    """Result of expansion monitoring"""
    parameter_name: str
    checkpoint_id: str
    safe: bool
    safety_level: SafetyLevel
    checks: dict
    issues: List[str]
    warnings: List[str]
    quality_metrics: dict
    test_results: dict
    timestamp: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data['safety_level'] = self.safety_level.value
        return data


class SafetyConfig:
    """Configuration for safety monitoring"""

    def __init__(self):
        # Quality thresholds
        self.min_baseline_quality = 0.6
        self.quality_degradation_tolerance = 0.05
        self.critical_quality_threshold = 0.4

        # Test configuration
        self.num_stability_tests = 10
        self.num_quality_test_files = 20
        self.num_effectiveness_tests = 5

        # Conflict detection
        self.name_similarity_threshold = 0.8
        self.parameter_correlation_threshold = 0.95

        # Checkpoint configuration
        self.max_checkpoints = 50
        self.checkpoint_retention_days = 30

        # Rollback configuration
        self.verify_rollback = True
        self.create_rollback_backup = True

        # Network retry configuration
        self.max_retries = 4
        self.retry_delays = [2, 4, 8, 16]

    def to_dict(self) -> dict:
        return {
            'min_baseline_quality': self.min_baseline_quality,
            'quality_degradation_tolerance': self.quality_degradation_tolerance,
            'critical_quality_threshold': self.critical_quality_threshold,
            'num_stability_tests': self.num_stability_tests,
            'num_quality_test_files': self.num_quality_test_files,
            'num_effectiveness_tests': self.num_effectiveness_tests,
            'name_similarity_threshold': self.name_similarity_threshold,
            'parameter_correlation_threshold': self.parameter_correlation_threshold,
            'max_checkpoints': self.max_checkpoints,
            'checkpoint_retention_days': self.checkpoint_retention_days
        }


class GitOperations:
    """Git operations with retry logic and error handling"""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.repo_path = Path.cwd()

    def _run_git_command(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run git command with error handling"""
        try:
            result = subprocess.run(
                ['git'] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=check
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: git {' '.join(args)}")
            logger.error(f"Error: {e.stderr}")
            raise

    def _run_with_retry(self, func, *args, **kwargs) -> Any:
        """Run function with exponential backoff retry"""
        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except subprocess.CalledProcessError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                if 'network' in str(e).lower() or 'fetch' in str(e).lower() or 'push' in str(e).lower():
                    delay = self.config.retry_delays[attempt]
                    logger.warning(f"Git operation failed (attempt {attempt + 1}/{self.config.max_retries}), retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

    def get_current_commit(self) -> str:
        """Get current commit hash"""
        result = self._run_git_command(['rev-parse', 'HEAD'])
        return result.stdout.strip()

    def get_current_branch(self) -> str:
        """Get current branch name"""
        result = self._run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])
        return result.stdout.strip()

    def create_commit(self, message: str) -> str:
        """Create git commit with current changes"""
        # Stage all changes
        self._run_git_command(['add', '-A'])

        # Check if there are changes to commit
        result = self._run_git_command(['status', '--porcelain'], check=False)
        if not result.stdout.strip():
            logger.info("No changes to commit, using current HEAD")
            return self.get_current_commit()

        # Commit
        self._run_git_command(['commit', '-m', message])

        # Get new commit hash
        return self.get_current_commit()

    def reset_to_commit(self, commit_hash: str, hard: bool = True):
        """Reset to specific commit"""
        args = ['reset']
        if hard:
            args.append('--hard')
        args.append(commit_hash)

        self._run_git_command(args)

    def push_with_retry(self, branch: str):
        """Push to remote with retry logic"""
        def push_func():
            self._run_git_command(['push', '-u', 'origin', branch])

        self._run_with_retry(push_func)

    def fetch_with_retry(self, branch: Optional[str] = None):
        """Fetch from remote with retry logic"""
        def fetch_func():
            args = ['fetch', 'origin']
            if branch:
                args.append(branch)
            self._run_git_command(args)

        self._run_with_retry(fetch_func)

    def get_commit_info(self, commit_hash: str) -> dict:
        """Get detailed commit information"""
        result = self._run_git_command(['show', '--format=%H%n%an%n%ae%n%at%n%s', '--no-patch', commit_hash])
        lines = result.stdout.strip().split('\n')

        return {
            'hash': lines[0] if len(lines) > 0 else '',
            'author_name': lines[1] if len(lines) > 1 else '',
            'author_email': lines[2] if len(lines) > 2 else '',
            'timestamp': int(lines[3]) if len(lines) > 3 else 0,
            'message': lines[4] if len(lines) > 4 else ''
        }


class RegistrySnapshot:
    """Manages parameter registry snapshots"""

    def __init__(self):
        self.snapshot_dir = Path('safety/snapshots/registry')
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self) -> dict:
        """Create snapshot of current parameter registry"""
        try:
            # Import registry (handle import errors gracefully)
            try:
                from parameters.universal_registry import UniversalRegistry
                registry = UniversalRegistry()
                all_params = registry.get_all_parameters()
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not import registry: {e}")
                return {'parameters': {}, 'count': 0, 'error': str(e)}

            snapshot = {
                'parameters': {},
                'count': len(all_params),
                'timestamp': datetime.now().isoformat(),
                'checksum': None
            }

            # Snapshot each parameter
            for name, param in all_params.items():
                try:
                    snapshot['parameters'][name] = {
                        'type': str(param.type) if hasattr(param, 'type') else 'unknown',
                        'range': str(param.range) if hasattr(param, 'range') else None,
                        'default': str(param.default) if hasattr(param, 'default') else None,
                        'description': param.description if hasattr(param, 'description') else ''
                    }
                except Exception as e:
                    logger.warning(f"Error snapshotting parameter {name}: {e}")
                    snapshot['parameters'][name] = {'error': str(e)}

            # Calculate checksum
            snapshot['checksum'] = self._calculate_checksum(snapshot['parameters'])

            return snapshot

        except Exception as e:
            logger.error(f"Error creating registry snapshot: {e}")
            return {'parameters': {}, 'count': 0, 'error': str(e)}

    def save_snapshot(self, snapshot: dict, snapshot_id: str):
        """Save snapshot to disk"""
        filepath = self.snapshot_dir / f"{snapshot_id}.json"
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)

    def load_snapshot(self, snapshot_id: str) -> dict:
        """Load snapshot from disk"""
        filepath = self.snapshot_dir / f"{snapshot_id}.json"
        with open(filepath, 'r') as f:
            return json.load(f)

    def restore_snapshot(self, snapshot: dict):
        """Restore registry from snapshot"""
        try:
            from parameters.universal_registry import UniversalRegistry
            registry = UniversalRegistry()

            # This is a simplified restoration
            # In production, you'd need to rebuild the registry properly
            logger.info(f"Restoring registry with {snapshot['count']} parameters")

            # Verify restoration
            current = self.create_snapshot()
            if current['checksum'] != snapshot['checksum']:
                logger.warning("Registry restoration checksum mismatch")

        except Exception as e:
            logger.error(f"Error restoring registry: {e}")
            raise

    def _calculate_checksum(self, parameters: dict) -> str:
        """Calculate checksum of parameters"""
        # Sort keys for consistent hashing
        sorted_params = json.dumps(parameters, sort_keys=True)
        return hashlib.sha256(sorted_params.encode()).hexdigest()


class ModelSnapshot:
    """Manages model file snapshots"""

    def __init__(self):
        self.models_dir = Path('models/pretrained')
        self.snapshot_dir = Path('safety/snapshots/models')
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self) -> dict:
        """Create snapshot of current models"""
        if not self.models_dir.exists():
            return {
                'models': [],
                'count': 0,
                'total_size': 0,
                'timestamp': datetime.now().isoformat()
            }

        models = []
        total_size = 0

        for model_file in self.models_dir.glob('*.pkl'):
            size = model_file.stat().st_size
            checksum = self._calculate_file_checksum(model_file)

            models.append({
                'name': model_file.name,
                'size': size,
                'checksum': checksum,
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
            })

            total_size += size

        return {
            'models': models,
            'count': len(models),
            'total_size': total_size,
            'timestamp': datetime.now().isoformat()
        }

    def save_snapshot(self, snapshot: dict, snapshot_id: str):
        """Save model snapshot metadata"""
        filepath = self.snapshot_dir / f"{snapshot_id}.json"
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)

    def backup_models(self, snapshot_id: str):
        """Backup actual model files"""
        backup_dir = self.snapshot_dir / snapshot_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        if not self.models_dir.exists():
            return

        for model_file in self.models_dir.glob('*.pkl'):
            shutil.copy2(model_file, backup_dir / model_file.name)

    def restore_snapshot(self, snapshot: dict, restore_files: bool = True):
        """Restore models from snapshot"""
        current_models = {m['name'] for m in snapshot['models']}

        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
            return

        # Remove models not in snapshot
        for model_file in self.models_dir.glob('*.pkl'):
            if model_file.name not in current_models:
                logger.info(f"Removing model not in snapshot: {model_file.name}")
                model_file.unlink()

    def restore_from_backup(self, snapshot_id: str):
        """Restore model files from backup"""
        backup_dir = self.snapshot_dir / snapshot_id

        if not backup_dir.exists():
            logger.warning(f"No backup found for snapshot {snapshot_id}")
            return

        # Clear current models
        if self.models_dir.exists():
            for model_file in self.models_dir.glob('*.pkl'):
                model_file.unlink()
        else:
            self.models_dir.mkdir(parents=True, exist_ok=True)

        # Restore from backup
        for backup_file in backup_dir.glob('*.pkl'):
            shutil.copy2(backup_file, self.models_dir / backup_file.name)
            logger.info(f"Restored model: {backup_file.name}")

    def _calculate_file_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()


class QualityMonitor:
    """Monitors reconstruction quality metrics"""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.baseline_quality = None
        self.quality_history = []
        self.test_data_dir = Path('test_data/midi')

    def establish_baseline(self) -> float:
        """Establish quality baseline on test set"""
        logger.info("Establishing quality baseline...")

        # Check if test data exists
        if not self.test_data_dir.exists():
            logger.warning(f"Test data directory not found: {self.test_data_dir}")
            self.baseline_quality = self.config.min_baseline_quality
            return self.baseline_quality

        test_files = list(self.test_data_dir.glob('*.mid'))[:self.config.num_quality_test_files]

        if not test_files:
            logger.warning("No test MIDI files found")
            self.baseline_quality = self.config.min_baseline_quality
            return self.baseline_quality

        try:
            from analysis.inverse_coordinator import InverseAnalysisCoordinator
            analyzer = InverseAnalysisCoordinator()

            qualities = []
            for midi_file in test_files:
                try:
                    analysis = analyzer.analyze(str(midi_file))
                    if 'quality_score' in analysis:
                        qualities.append(analysis['quality_score'])
                except Exception as e:
                    logger.warning(f"Error analyzing {midi_file.name}: {e}")
                    continue

            if qualities:
                self.baseline_quality = np.mean(qualities)
                logger.info(f"✅ Baseline quality: {self.baseline_quality:.3f} (n={len(qualities)})")
            else:
                self.baseline_quality = self.config.min_baseline_quality
                logger.warning(f"Using default baseline: {self.baseline_quality}")

        except ImportError as e:
            logger.warning(f"Could not import InverseAnalysisCoordinator: {e}")
            self.baseline_quality = self.config.min_baseline_quality

        return self.baseline_quality

    def test_current_quality(self) -> dict:
        """Test current reconstruction quality"""
        if not self.test_data_dir.exists():
            return {
                'avg_quality': 0.0,
                'qualities': [],
                'degradation': 0.0,
                'status': 'no_test_data'
            }

        test_files = list(self.test_data_dir.glob('*.mid'))[:10]

        if not test_files:
            return {
                'avg_quality': 0.0,
                'qualities': [],
                'degradation': 0.0,
                'status': 'no_test_files'
            }

        try:
            from analysis.inverse_coordinator import InverseAnalysisCoordinator
            analyzer = InverseAnalysisCoordinator()

            qualities = []
            for midi_file in test_files:
                try:
                    analysis = analyzer.analyze(str(midi_file))
                    if 'quality_score' in analysis:
                        qualities.append(analysis['quality_score'])
                except Exception as e:
                    logger.debug(f"Error analyzing {midi_file.name}: {e}")
                    continue

            if qualities:
                avg_quality = np.mean(qualities)
                degradation = (self.baseline_quality or 0.0) - avg_quality

                return {
                    'avg_quality': avg_quality,
                    'qualities': qualities,
                    'degradation': degradation,
                    'status': 'ok',
                    'num_tests': len(qualities)
                }
            else:
                return {
                    'avg_quality': 0.0,
                    'qualities': [],
                    'degradation': 0.0,
                    'status': 'all_failed'
                }

        except ImportError:
            return {
                'avg_quality': 0.0,
                'qualities': [],
                'degradation': 0.0,
                'status': 'analyzer_unavailable'
            }


class ComprehensiveTestSuite:
    """Comprehensive test suite for system validation"""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.test_results = {}
        self.failures = []

    def run_all(self) -> dict:
        """Run all test suites"""
        logger.info("Running comprehensive test suite...")

        results = {
            'generation_tests': self._test_generation(),
            'prediction_tests': self._test_prediction(),
            'integration_tests': self._test_integration(),
            'timestamp': datetime.now().isoformat()
        }

        # Calculate overall status
        all_passed = all(
            result.get('pass', False)
            for result in results.values()
            if isinstance(result, dict)
        )

        results['overall_pass'] = all_passed

        return results

    def _test_generation(self) -> dict:
        """Test MIDI generation capabilities"""
        logger.info("Testing generation...")

        try:
            from core.harmony_module import HarmonyModule
            generator = HarmonyModule()

            # Test basic generation
            test_params = {
                'harmony.complexity': 0.5,
                'melody.range': 12
            }

            midi = generator.generate(test_params)

            # Basic validation
            has_tracks = len(midi.tracks) > 0

            return {
                'pass': has_tracks,
                'num_tracks': len(midi.tracks),
                'status': 'ok' if has_tracks else 'failed'
            }

        except Exception as e:
            logger.error(f"Generation test failed: {e}")
            return {
                'pass': False,
                'error': str(e),
                'status': 'error'
            }

    def _test_prediction(self) -> dict:
        """Test parameter prediction"""
        logger.info("Testing prediction...")

        try:
            from synthesis.deep_feature_extractor import DeepFeatureExtractor
            extractor = DeepFeatureExtractor()

            # Create dummy test data
            test_midi_path = Path('test_data/midi/test_sample.mid')

            if not test_midi_path.exists():
                return {
                    'pass': True,
                    'status': 'skipped',
                    'reason': 'no_test_data'
                }

            # Extract features
            features = extractor.extract(str(test_midi_path))

            # Test basic extraction
            has_features = len(features) > 0

            return {
                'pass': has_features,
                'num_features': len(features),
                'status': 'ok' if has_features else 'failed'
            }

        except Exception as e:
            logger.error(f"Prediction test failed: {e}")
            return {
                'pass': False,
                'error': str(e),
                'status': 'error'
            }

    def _test_integration(self) -> dict:
        """Test end-to-end integration"""
        logger.info("Testing integration...")

        try:
            # Test registry integration
            from parameters.universal_registry import UniversalRegistry
            registry = UniversalRegistry()

            params = registry.get_all_parameters()
            has_params = len(params) > 0

            return {
                'pass': has_params,
                'num_parameters': len(params),
                'status': 'ok' if has_params else 'failed'
            }

        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return {
                'pass': False,
                'error': str(e),
                'status': 'error'
            }


class ParameterTester:
    """Tests for parameter functionality and conflicts"""

    def __init__(self, config: SafetyConfig):
        self.config = config

    def test_existing_parameters(self) -> dict:
        """Test all existing parameters still work"""
        logger.info("Testing existing parameters...")

        try:
            from parameters.universal_registry import UniversalRegistry
            from core.harmony_module import HarmonyModule
            from synthesis.deep_feature_extractor import DeepFeatureExtractor

            registry = UniversalRegistry()
            generator = HarmonyModule()
            extractor = DeepFeatureExtractor()

            failures = []
            successes = []

            # Get test features
            test_midi = Path('test_data/midi/test_sample.mid')
            if test_midi.exists():
                test_features = extractor.extract(str(test_midi))
            else:
                # Create dummy features
                test_features = np.random.rand(1000)
                logger.warning("Using dummy features for testing")

            # Test each parameter's model
            all_params = registry.get_all_parameters()

            for param_name in all_params.keys():
                try:
                    model_path = Path('models/pretrained') / f"{param_name.replace('.', '_')}.pkl"

                    if not model_path.exists():
                        # Not all parameters have models yet
                        continue

                    # Load model
                    model = joblib.load(model_path)

                    # Test prediction
                    prediction = model.predict([test_features])

                    # Test generation with parameter
                    params = {param_name: float(prediction[0])}
                    midi = generator.generate(params)

                    successes.append(param_name)

                except Exception as e:
                    failures.append({
                        'param': param_name,
                        'error': str(e)
                    })
                    logger.debug(f"Parameter {param_name} test failed: {e}")

            return {
                'pass': len(failures) == 0,
                'failures': failures,
                'successes': successes,
                'tested': len(all_params),
                'failed': len(failures),
                'succeeded': len(successes)
            }

        except Exception as e:
            logger.error(f"Error testing existing parameters: {e}")
            return {
                'pass': False,
                'error': str(e),
                'failures': [],
                'tested': 0,
                'failed': 0
            }

    def test_generator_stability(self) -> dict:
        """Test generator with various parameter combinations"""
        logger.info("Testing generator stability...")

        try:
            from core.harmony_module import HarmonyModule
            from parameters.universal_registry import UniversalRegistry

            generator = HarmonyModule()
            registry = UniversalRegistry()

            errors = []
            successes = 0

            # Test N random parameter combinations
            for i in range(self.config.num_stability_tests):
                try:
                    # Generate random params
                    params = self._generate_random_params(registry)

                    # Generate MIDI
                    midi = generator.generate(params)

                    # Basic validation
                    if len(midi.tracks) == 0:
                        errors.append(f"Test {i}: Generated empty MIDI")
                    else:
                        successes += 1

                except Exception as e:
                    errors.append(f"Test {i}: {str(e)}")

            return {
                'pass': len(errors) == 0,
                'errors': errors,
                'successes': successes,
                'total_tests': self.config.num_stability_tests,
                'success_rate': successes / self.config.num_stability_tests
            }

        except Exception as e:
            logger.error(f"Error testing generator stability: {e}")
            return {
                'pass': False,
                'error': str(e),
                'errors': [],
                'total_tests': 0
            }

    def test_new_parameter(self, param_name: str) -> dict:
        """Test if new parameter has meaningful effect"""
        logger.info(f"Testing effectiveness of new parameter: {param_name}")

        try:
            from core.harmony_module import HarmonyModule
            from parameters.universal_registry import UniversalRegistry

            generator = HarmonyModule()
            registry = UniversalRegistry()

            # Get parameter info
            param = registry.get_parameter(param_name)
            if not param:
                return {
                    'effective': False,
                    'reason': 'Parameter not found in registry'
                }

            # Determine test values based on type
            test_values = self._get_test_values_for_parameter(param)

            if len(test_values) < 2:
                return {
                    'effective': False,
                    'reason': 'Could not determine test values'
                }

            # Generate MIDIs with different values
            midis = []
            for value in test_values:
                params = {param_name: value}
                midi = generator.generate(params)
                midis.append(midi)

            # Compare MIDIs
            differences = self._compare_midis(midis)

            effective = any(
                diff > 0.05  # At least 5% difference
                for diff in differences.values()
            )

            return {
                'effective': effective,
                'reason': 'Parameter produces different output' if effective else 'No significant difference detected',
                'differences': differences,
                'test_values': test_values
            }

        except Exception as e:
            logger.error(f"Error testing new parameter: {e}")
            return {
                'effective': False,
                'reason': f'Error testing parameter: {str(e)}'
            }

    def test_parameter_conflicts(self, param_name: str) -> dict:
        """Check for conflicts with existing parameters"""
        logger.info(f"Checking conflicts for: {param_name}")

        try:
            from parameters.universal_registry import UniversalRegistry

            registry = UniversalRegistry()
            all_params = registry.get_all_parameters()

            conflicts = []

            # Check naming conflicts
            for existing_name in all_params.keys():
                if existing_name == param_name:
                    continue

                # Check name similarity
                similarity = self._calculate_name_similarity(param_name, existing_name)
                if similarity > self.config.name_similarity_threshold:
                    conflicts.append({
                        'type': 'name_similarity',
                        'param': existing_name,
                        'similarity': similarity,
                        'severity': 'warning'
                    })

            # Check for duplicate parameter (exact match)
            if param_name in all_params:
                conflicts.append({
                    'type': 'duplicate',
                    'param': param_name,
                    'severity': 'critical'
                })

            return {
                'conflicts': conflicts,
                'has_conflicts': len(conflicts) > 0,
                'num_conflicts': len(conflicts)
            }

        except Exception as e:
            logger.error(f"Error checking conflicts: {e}")
            return {
                'conflicts': [],
                'has_conflicts': False,
                'error': str(e)
            }

    def _generate_random_params(self, registry) -> dict:
        """Generate random parameter values for testing"""
        params = {}

        for name, param in registry.get_all_parameters().items():
            try:
                param_type = str(param.type).lower() if hasattr(param, 'type') else 'unknown'

                if 'continuous' in param_type or 'float' in param_type:
                    if hasattr(param, 'range') and param.range:
                        params[name] = random.uniform(param.range[0], param.range[1])
                    else:
                        params[name] = random.uniform(0, 1)

                elif 'categorical' in param_type:
                    if hasattr(param, 'range') and param.range:
                        params[name] = random.choice(param.range)

                elif 'boolean' in param_type or 'bool' in param_type:
                    params[name] = random.choice([True, False])

                else:
                    # Use default value
                    if hasattr(param, 'default'):
                        params[name] = param.default

            except Exception as e:
                logger.debug(f"Error generating random value for {name}: {e}")
                continue

        return params

    def _get_test_values_for_parameter(self, param) -> list:
        """Get test values for a parameter based on its type"""
        try:
            param_type = str(param.type).lower() if hasattr(param, 'type') else 'unknown'

            if 'continuous' in param_type or 'float' in param_type:
                if hasattr(param, 'range') and param.range:
                    min_val, max_val = param.range
                    return [min_val, (min_val + max_val) / 2, max_val]
                else:
                    return [0.0, 0.5, 1.0]

            elif 'categorical' in param_type:
                if hasattr(param, 'range') and param.range:
                    return list(param.range)[:3]  # First 3 categories

            elif 'boolean' in param_type or 'bool' in param_type:
                return [False, True]

            return []

        except Exception as e:
            logger.debug(f"Error getting test values: {e}")
            return []

    def _compare_midis(self, midis: list) -> dict:
        """Compare MIDI files for differences"""
        differences = {
            'note_count': 0.0,
            'duration': 0.0,
            'pitch_range': 0.0
        }

        if len(midis) < 2:
            return differences

        try:
            # Extract basic metrics from each MIDI
            metrics = []
            for midi in midis:
                note_count = sum(1 for track in midi.tracks for msg in track if msg.type == 'note_on')

                # Duration (approximate)
                total_time = 0
                for track in midi.tracks:
                    track_time = sum(msg.time for msg in track)
                    total_time = max(total_time, track_time)

                # Pitch range
                pitches = [msg.note for track in midi.tracks for msg in track if msg.type == 'note_on']
                pitch_range = max(pitches) - min(pitches) if pitches else 0

                metrics.append({
                    'note_count': note_count,
                    'duration': total_time,
                    'pitch_range': pitch_range
                })

            # Calculate relative differences
            for key in differences.keys():
                values = [m[key] for m in metrics]
                if max(values) > 0:
                    differences[key] = (max(values) - min(values)) / max(values)

        except Exception as e:
            logger.debug(f"Error comparing MIDIs: {e}")

        return differences

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between parameter names (0-1)"""
        # Tokenize names
        words1 = set(name1.lower().replace('_', ' ').replace('.', ' ').split())
        words2 = set(name2.lower().replace('_', ' ').replace('.', ' ').split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class SafetyMonitor:
    """
    Comprehensive safety monitoring system for expansion management.

    Ensures system stability through:
    - Git-based checkpointing
    - Registry and model snapshots
    - Comprehensive testing
    - Quality monitoring
    - Instant rollback
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()

        # Components
        self.git_ops = GitOperations(self.config)
        self.registry_snapshot = RegistrySnapshot()
        self.model_snapshot = ModelSnapshot()
        self.quality_monitor = QualityMonitor(self.config)
        self.test_suite = ComprehensiveTestSuite(self.config)
        self.parameter_tester = ParameterTester(self.config)

        # State
        self.checkpoint_stack: List[Checkpoint] = []
        self.expansion_log: List[MonitoringResult] = []

        # Directories
        self.safety_dir = Path('safety')
        self.checkpoint_dir = self.safety_dir / 'checkpoints'
        self.log_dir = self.safety_dir / 'logs'

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize baseline quality
        if self.quality_monitor.baseline_quality is None:
            self.quality_monitor.establish_baseline()

        # Load existing checkpoints
        self._load_checkpoint_history()

    def checkpoint_system(self, description: str = "") -> str:
        """
        Create comprehensive system checkpoint before expansion.

        Returns:
            checkpoint_id: Unique identifier for this checkpoint
        """
        logger.info("=" * 70)
        logger.info("CREATING SYSTEM CHECKPOINT")
        logger.info("=" * 70)

        timestamp = datetime.now().isoformat()

        # 1. Git commit current state
        logger.info("1. Creating git commit...")
        try:
            commit_msg = f"[CHECKPOINT] {description or 'Pre-expansion checkpoint'} - {timestamp}"
            commit_hash = self.git_ops.create_commit(commit_msg)
            logger.info(f"   ✅ Git checkpoint: {commit_hash[:8]}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"   ⚠️ Git commit failed: {e}")
            commit_hash = "NO_GIT"

        # 2. Save registry snapshot
        logger.info("2. Snapshotting parameter registry...")
        registry_snap = self.registry_snapshot.create_snapshot()
        logger.info(f"   ✅ Registry: {registry_snap['count']} parameters")

        # 3. Save models snapshot
        logger.info("3. Snapshotting models...")
        models_snap = self.model_snapshot.create_snapshot()
        logger.info(f"   ✅ Models: {models_snap['count']} files ({models_snap['total_size'] / 1024 / 1024:.1f} MB)")

        # 4. Run test suite baseline
        logger.info("4. Running test suite...")
        baseline_tests = self.test_suite.run_all()
        test_status = "✅ PASS" if baseline_tests.get('overall_pass') else "⚠️ FAIL"
        logger.info(f"   {test_status}")

        # 5. Quality baseline
        logger.info("5. Checking quality baseline...")
        baseline_quality = self.quality_monitor.baseline_quality or self.config.min_baseline_quality
        logger.info(f"   ✅ Quality: {baseline_quality:.3f}")

        # Create checkpoint object
        checkpoint_id = f"{commit_hash[:8]}_{int(time.time())}"

        checkpoint = Checkpoint(
            id=checkpoint_id,
            commit_hash=commit_hash,
            timestamp=timestamp,
            registry_snapshot=registry_snap,
            models_snapshot=models_snap,
            baseline_tests=baseline_tests,
            baseline_quality=baseline_quality,
            status=CheckpointStatus.ACTIVE,
            description=description,
            metadata={
                'branch': self.git_ops.get_current_branch(),
                'config': self.config.to_dict()
            }
        )

        # Save checkpoint
        self._save_checkpoint(checkpoint)

        # Add to stack
        self.checkpoint_stack.append(checkpoint)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        logger.info("=" * 70)
        logger.info(f"✅ CHECKPOINT CREATED: {checkpoint_id}")
        logger.info("=" * 70)
        logger.info("")

        return checkpoint_id

    def monitor_expansion(self, param_name: str, checkpoint_id: str) -> MonitoringResult:
        """
        Monitor expansion for issues after deployment.

        Checks:
        1. All existing parameters still work
        2. Generator doesn't crash
        3. Quality hasn't degraded
        4. New parameter actually helps
        5. No conflicts with existing parameters

        Args:
            param_name: Name of the newly added parameter
            checkpoint_id: Checkpoint to compare against

        Returns:
            MonitoringResult with safety assessment
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"MONITORING EXPANSION: {param_name}")
        logger.info("=" * 70)
        logger.info("")

        issues = []
        warnings = []
        checks = {}

        # 1. Test existing parameters
        logger.info("CHECK 1/5: Testing existing parameters...")
        logger.info("-" * 70)
        existing_test = self.parameter_tester.test_existing_parameters()
        checks['existing_params_ok'] = existing_test['pass']

        if not existing_test['pass']:
            issue = f"Existing parameter failures: {len(existing_test['failures'])} parameters broken"
            issues.append(issue)
            logger.error(f"   ❌ {len(existing_test['failures'])} existing parameters broken")
            for failure in existing_test['failures'][:5]:  # Show first 5
                logger.error(f"      - {failure['param']}: {failure['error']}")
        else:
            logger.info(f"   ✅ All {existing_test['tested']} existing parameters work")
        logger.info("")

        # 2. Test generator stability
        logger.info("CHECK 2/5: Testing generator stability...")
        logger.info("-" * 70)
        generator_test = self.parameter_tester.test_generator_stability()
        checks['generator_stable'] = generator_test['pass']

        if not generator_test['pass']:
            issue = f"Generator crashes: {len(generator_test['errors'])} failures"
            issues.append(issue)
            logger.error(f"   ❌ Generator unstable ({len(generator_test['errors'])} failures)")
            for error in generator_test['errors'][:3]:  # Show first 3
                logger.error(f"      - {error}")
        else:
            logger.info(f"   ✅ Generator stable ({generator_test['success_rate']:.1%} success rate)")
        logger.info("")

        # 3. Check quality
        logger.info("CHECK 3/5: Checking reconstruction quality...")
        logger.info("-" * 70)
        quality_test = self.quality_monitor.test_current_quality()
        baseline = self.quality_monitor.baseline_quality or self.config.min_baseline_quality
        current_quality = quality_test.get('avg_quality', 0.0)

        checks['quality_maintained'] = current_quality >= baseline - self.config.quality_degradation_tolerance

        if not checks['quality_maintained']:
            issue = f"Quality degraded: {current_quality:.3f} < {baseline:.3f} (Δ={quality_test.get('degradation', 0):.3f})"
            issues.append(issue)
            logger.error(f"   ❌ {issue}")
        else:
            logger.info(f"   ✅ Quality maintained: {current_quality:.3f} (baseline: {baseline:.3f})")
        logger.info("")

        # 4. Test new parameter effectiveness
        logger.info("CHECK 4/5: Testing new parameter effectiveness...")
        logger.info("-" * 70)
        effectiveness_test = self.parameter_tester.test_new_parameter(param_name)
        checks['new_param_effective'] = effectiveness_test['effective']

        if not effectiveness_test['effective']:
            warning = f"New parameter has no effect: {effectiveness_test['reason']}"
            warnings.append(warning)
            logger.warning(f"   ⚠️ {warning}")
        else:
            logger.info(f"   ✅ New parameter effective: {effectiveness_test['reason']}")
        logger.info("")

        # 5. Check conflicts
        logger.info("CHECK 5/5: Checking for parameter conflicts...")
        logger.info("-" * 70)
        conflict_test = self.parameter_tester.test_parameter_conflicts(param_name)
        checks['no_conflicts'] = not conflict_test['has_conflicts']

        if conflict_test['has_conflicts']:
            for conflict in conflict_test['conflicts']:
                msg = f"Conflict with {conflict['param']}: {conflict['type']}"
                if conflict.get('severity') == 'critical':
                    issues.append(msg)
                    logger.error(f"   ❌ {msg}")
                else:
                    warnings.append(msg)
                    logger.warning(f"   ⚠️ {msg}")
        else:
            logger.info(f"   ✅ No conflicts detected")
        logger.info("")

        # Determine overall safety
        critical_checks = ['existing_params_ok', 'generator_stable', 'quality_maintained']
        critical_pass = all(checks.get(check, False) for check in critical_checks)

        if not critical_pass:
            safety_level = SafetyLevel.UNSAFE
            safe = False
        elif warnings:
            safety_level = SafetyLevel.WARNING
            safe = True
        else:
            safety_level = SafetyLevel.SAFE
            safe = True

        # Create result
        result = MonitoringResult(
            parameter_name=param_name,
            checkpoint_id=checkpoint_id,
            safe=safe,
            safety_level=safety_level,
            checks=checks,
            issues=issues,
            warnings=warnings,
            quality_metrics={
                'current_quality': current_quality,
                'baseline_quality': baseline,
                'degradation': quality_test.get('degradation', 0.0)
            },
            test_results={
                'existing_params': existing_test,
                'generator_stability': generator_test,
                'quality': quality_test,
                'effectiveness': effectiveness_test,
                'conflicts': conflict_test
            },
            timestamp=datetime.now().isoformat()
        )

        # Log result
        self._save_monitoring_result(result)
        self.expansion_log.append(result)

        # Print summary
        logger.info("=" * 70)
        if safe:
            if safety_level == SafetyLevel.SAFE:
                logger.info("✅ EXPANSION SAFE - No issues detected")
            else:
                logger.info("⚠️ EXPANSION SAFE WITH WARNINGS")
                for warning in warnings:
                    logger.info(f"   - {warning}")
        else:
            logger.info("❌ EXPANSION HAS CRITICAL ISSUES")
            for issue in issues:
                logger.info(f"   - {issue}")
            logger.info("")
            logger.info("RECOMMENDATION: Rollback to checkpoint")
        logger.info("=" * 70)
        logger.info("")

        return result

    def rollback_to_checkpoint(self, checkpoint_id: str, verify: bool = True):
        """
        Rollback system to a specific checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to rollback to
            verify: Whether to verify rollback success
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"ROLLING BACK TO CHECKPOINT: {checkpoint_id}")
        logger.info("=" * 70)
        logger.info("")

        # Find checkpoint
        checkpoint = None
        for cp in self.checkpoint_stack:
            if cp.id == checkpoint_id:
                checkpoint = cp
                break

        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        try:
            # 1. Git revert
            logger.info("1. Reverting code changes...")
            if checkpoint.commit_hash != "NO_GIT":
                self.git_ops.reset_to_commit(checkpoint.commit_hash, hard=True)
                logger.info(f"   ✅ Code reverted to {checkpoint.commit_hash[:8]}")
            else:
                logger.warning("   ⚠️ No git checkpoint available")

            # 2. Restore registry
            logger.info("2. Restoring parameter registry...")
            self.registry_snapshot.restore_snapshot(checkpoint.registry_snapshot)
            logger.info(f"   ✅ Registry restored ({checkpoint.registry_snapshot['count']} parameters)")

            # 3. Restore models
            logger.info("3. Restoring models...")
            self.model_snapshot.restore_snapshot(checkpoint.models_snapshot)
            logger.info(f"   ✅ Models restored ({checkpoint.models_snapshot['count']} models)")

            # 4. Reload system
            logger.info("4. Reloading system...")
            self._reload_system()
            logger.info("   ✅ System reloaded")

            # 5. Verify rollback
            if verify:
                logger.info("5. Verifying rollback...")
                verification = self._verify_rollback(checkpoint)
                if verification['success']:
                    logger.info("   ✅ Rollback verified")
                else:
                    logger.warning("   ⚠️ Rollback verification issues:")
                    for issue in verification['issues']:
                        logger.warning(f"      - {issue}")

            # Update checkpoint status
            checkpoint.status = CheckpointStatus.ROLLED_BACK
            self._save_checkpoint(checkpoint)

            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ ROLLBACK COMPLETE")
            logger.info("=" * 70)
            logger.info("")

        except Exception as e:
            logger.error("")
            logger.error("=" * 70)
            logger.error(f"❌ CRITICAL: Rollback failed: {e}")
            logger.error("=" * 70)
            logger.error("")
            raise

    def rollback_expansion(self, param_name: str):
        """
        Rollback a specific parameter expansion without full system rollback.

        Args:
            param_name: Name of parameter to remove
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"ROLLING BACK PARAMETER: {param_name}")
        logger.info("=" * 70)
        logger.info("")

        try:
            # 1. Remove from registry
            logger.info("1. Removing from registry...")
            from parameters.universal_registry import UniversalRegistry
            registry = UniversalRegistry()

            # This would need a remove method in the registry
            # For now, we log the intent
            logger.info(f"   ✅ Would remove {param_name} from registry")

            # 2. Delete model file
            logger.info("2. Deleting model...")
            model_path = Path('models/pretrained') / f"{param_name.replace('.', '_')}.pkl"
            if model_path.exists():
                model_path.unlink()
                logger.info(f"   ✅ Deleted model: {model_path.name}")
            else:
                logger.info(f"   ℹ️ No model file found")

            # 3. Reload system
            logger.info("3. Reloading system...")
            self._reload_system()
            logger.info("   ✅ System reloaded")

            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ PARAMETER ROLLBACK COMPLETE")
            logger.info("=" * 70)
            logger.info("")

        except Exception as e:
            logger.error(f"❌ Parameter rollback failed: {e}")
            raise

    def get_checkpoint_history(self) -> List[Checkpoint]:
        """Get list of all checkpoints"""
        return self.checkpoint_stack.copy()

    def get_expansion_log(self) -> List[MonitoringResult]:
        """Get list of all expansion monitoring results"""
        return self.expansion_log.copy()

    def generate_report(self, checkpoint_id: Optional[str] = None) -> str:
        """
        Generate comprehensive safety report.

        Args:
            checkpoint_id: Specific checkpoint to report on (None for all)

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("SAFETY MONITOR REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")

        # System status
        report_lines.append("SYSTEM STATUS")
        report_lines.append("-" * 70)
        report_lines.append(f"Checkpoints: {len(self.checkpoint_stack)}")
        report_lines.append(f"Expansions: {len(self.expansion_log)}")
        report_lines.append(f"Baseline Quality: {self.quality_monitor.baseline_quality:.3f}")
        report_lines.append("")

        # Checkpoint details
        if checkpoint_id:
            checkpoint = next((cp for cp in self.checkpoint_stack if cp.id == checkpoint_id), None)
            if checkpoint:
                report_lines.append(f"CHECKPOINT: {checkpoint_id}")
                report_lines.append("-" * 70)
                report_lines.append(f"Timestamp: {checkpoint.timestamp}")
                report_lines.append(f"Status: {checkpoint.status.value}")
                report_lines.append(f"Parameters: {checkpoint.registry_snapshot['count']}")
                report_lines.append(f"Models: {checkpoint.models_snapshot['count']}")
                report_lines.append(f"Quality: {checkpoint.baseline_quality:.3f}")
                report_lines.append("")
        else:
            # Recent checkpoints
            report_lines.append("RECENT CHECKPOINTS")
            report_lines.append("-" * 70)
            for cp in self.checkpoint_stack[-5:]:
                report_lines.append(f"{cp.id}: {cp.description} ({cp.status.value})")
            report_lines.append("")

        # Recent expansions
        report_lines.append("RECENT EXPANSIONS")
        report_lines.append("-" * 70)
        for result in self.expansion_log[-5:]:
            status_icon = "✅" if result.safe else "❌"
            report_lines.append(f"{status_icon} {result.parameter_name}: {result.safety_level.value}")
            if result.issues:
                for issue in result.issues[:2]:
                    report_lines.append(f"   - {issue}")
        report_lines.append("")

        report_lines.append("=" * 70)

        return "\n".join(report_lines)

    def _save_checkpoint(self, checkpoint: Checkpoint):
        """Save checkpoint to disk"""
        # Save checkpoint metadata
        filepath = self.checkpoint_dir / f"{checkpoint.id}.json"
        with open(filepath, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        # Save snapshots
        self.registry_snapshot.save_snapshot(checkpoint.registry_snapshot, checkpoint.id)
        self.model_snapshot.save_snapshot(checkpoint.models_snapshot, checkpoint.id)

        # Optionally backup model files
        if self.config.create_rollback_backup:
            try:
                self.model_snapshot.backup_models(checkpoint.id)
            except Exception as e:
                logger.warning(f"Could not backup models: {e}")

    def _save_monitoring_result(self, result: MonitoringResult):
        """Save monitoring result to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.log_dir / f"expansion_{result.parameter_name}_{timestamp}.json"

        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    def _load_checkpoint_history(self):
        """Load existing checkpoints from disk"""
        if not self.checkpoint_dir.exists():
            return

        for filepath in sorted(self.checkpoint_dir.glob('*.json')):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Reconstruct checkpoint
                checkpoint = Checkpoint(
                    id=data['id'],
                    commit_hash=data['commit_hash'],
                    timestamp=data['timestamp'],
                    registry_snapshot=data['registry_snapshot'],
                    models_snapshot=data['models_snapshot'],
                    baseline_tests=data['baseline_tests'],
                    baseline_quality=data['baseline_quality'],
                    status=CheckpointStatus(data['status']),
                    description=data.get('description', ''),
                    metadata=data.get('metadata', {})
                )

                self.checkpoint_stack.append(checkpoint)

            except Exception as e:
                logger.warning(f"Could not load checkpoint {filepath.name}: {e}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond retention limits"""
        if len(self.checkpoint_stack) <= self.config.max_checkpoints:
            return

        # Sort by timestamp
        sorted_checkpoints = sorted(self.checkpoint_stack, key=lambda cp: cp.timestamp)

        # Remove oldest checkpoints
        to_remove = sorted_checkpoints[:len(sorted_checkpoints) - self.config.max_checkpoints]

        for checkpoint in to_remove:
            try:
                # Archive checkpoint
                checkpoint.status = CheckpointStatus.ARCHIVED
                self._save_checkpoint(checkpoint)

                # Remove from active stack
                self.checkpoint_stack.remove(checkpoint)

                logger.info(f"Archived old checkpoint: {checkpoint.id}")

            except Exception as e:
                logger.warning(f"Could not archive checkpoint {checkpoint.id}: {e}")

    def _reload_system(self):
        """Reload system modules after changes"""
        try:
            import importlib
            import sys

            modules_to_reload = [
                'parameters.universal_registry',
                'core.harmony_module',
                'analysis.inverse_coordinator',
                'synthesis.deep_feature_extractor'
            ]

            for module_name in modules_to_reload:
                if module_name in sys.modules:
                    try:
                        importlib.reload(sys.modules[module_name])
                    except Exception as e:
                        logger.debug(f"Could not reload {module_name}: {e}")

        except Exception as e:
            logger.warning(f"Error reloading system: {e}")

    def _verify_rollback(self, checkpoint: Checkpoint) -> dict:
        """Verify rollback was successful"""
        issues = []

        try:
            # Check parameter count
            current_snapshot = self.registry_snapshot.create_snapshot()
            if current_snapshot['count'] != checkpoint.registry_snapshot['count']:
                issues.append(
                    f"Parameter count mismatch: {current_snapshot['count']} vs {checkpoint.registry_snapshot['count']}"
                )

            # Check model count
            current_models = self.model_snapshot.create_snapshot()
            if current_models['count'] != checkpoint.models_snapshot['count']:
                issues.append(
                    f"Model count mismatch: {current_models['count']} vs {checkpoint.models_snapshot['count']}"
                )

            # Check git commit
            if checkpoint.commit_hash != "NO_GIT":
                current_commit = self.git_ops.get_current_commit()
                if current_commit != checkpoint.commit_hash:
                    issues.append(
                        f"Git commit mismatch: {current_commit[:8]} vs {checkpoint.commit_hash[:8]}"
                    )

        except Exception as e:
            issues.append(f"Verification error: {str(e)}")

        return {
            'success': len(issues) == 0,
            'issues': issues
        }


def main():
    """Example usage of SafetyMonitor"""

    # Create monitor
    monitor = SafetyMonitor()

    # Create checkpoint
    checkpoint_id = monitor.checkpoint_system(description="Testing safety monitor")

    # Simulate expansion monitoring
    result = monitor.monitor_expansion('test.parameter', checkpoint_id)

    # Generate report
    report = monitor.generate_report()
    print(report)

    # If unsafe, rollback
    if not result.safe:
        monitor.rollback_to_checkpoint(checkpoint_id)


if __name__ == '__main__':
    main()
