"""
AGENT 16: Expansion Orchestrator
=================================

Master controller for automated system expansion workflow.

This orchestrator coordinates the complete expansion process:
1. MIDI input → inverse analysis
2. Gap detection
3. Parameter proposal (LLM)
4. Validation (musical correctness)
5. Code generation (LLM)
6. Code validation
7. Training data generation
8. Model training
9. Quality verification
10. Deployment or rollback

This is the central nervous system of the self-expanding music generation system.

Author: Agent 16 - Expansion Orchestrator
License: MIT
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import os

# Import our agents
try:
    from midi_generator.llm.code_generator import LLMCodeGenerationAgent, GeneratedImplementation
    from midi_generator.validation.musical_validator import MusicalValidator, ParameterValidationResult
    from midi_generator.training.synthetic_data_generator import SyntheticTrainingDataGenerator, TrainingDataset
    from midi_generator.training.model_trainer import ModelTrainingSpecialist, ModelTrainingResult
except ImportError:
    print("WARNING: Could not import all agent modules. Some functionality may be limited.")


class ExpansionStage(Enum):
    """Stages of expansion workflow"""
    INITIALIZATION = "initialization"
    INVERSE_ANALYSIS = "inverse_analysis"
    GAP_DETECTION = "gap_detection"
    PARAMETER_PROPOSAL = "parameter_proposal"
    PARAMETER_VALIDATION = "parameter_validation"
    CODE_GENERATION = "code_generation"
    CODE_VALIDATION = "code_validation"
    TRAINING_DATA_GENERATION = "training_data_generation"
    MODEL_TRAINING = "model_training"
    QUALITY_VERIFICATION = "quality_verification"
    DEPLOYMENT = "deployment"
    ROLLBACK = "rollback"
    COMPLETED = "completed"
    FAILED = "failed"


class ExpansionStatus(Enum):
    """Status of expansion"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ExpansionCheckpoint:
    """System checkpoint for rollback"""
    checkpoint_id: str
    timestamp: str
    backup_paths: Dict[str, Path]
    parameter_count: int
    model_count: int
    description: str


@dataclass
class ParameterExpansion:
    """Single parameter expansion record"""
    parameter_name: str
    proposal: dict
    validation_result: Optional[ParameterValidationResult] = None
    generated_code: Optional[GeneratedImplementation] = None
    training_dataset: Optional[TrainingDataset] = None
    training_result: Optional[ModelTrainingResult] = None
    status: ExpansionStatus = ExpansionStatus.PENDING
    error: Optional[str] = None
    stage: ExpansionStage = ExpansionStage.INITIALIZATION


@dataclass
class ExpansionWorkflowResult:
    """Complete expansion workflow result"""
    success: bool
    expansions_deployed: List[str]
    quality_improvement: float
    initial_quality: float
    final_quality: float
    expansion_details: List[ParameterExpansion]
    failure_reason: Optional[str] = None
    checkpoint_id: Optional[str] = None
    total_time: float = 0.0


class SafetyMonitor:
    """Safety monitor for system checkpoints and rollbacks"""

    def __init__(self, checkpoints_dir: Path = Path('.checkpoints')):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: List[ExpansionCheckpoint] = []

    def create_checkpoint(self, description: str = "System checkpoint") -> ExpansionCheckpoint:
        """
        Create system checkpoint for potential rollback

        Args:
            description: Checkpoint description

        Returns:
            ExpansionCheckpoint
        """
        checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_dir = self.checkpoints_dir / checkpoint_id

        print(f"Creating checkpoint: {checkpoint_id}")

        # Backup critical files
        backup_paths = {}

        files_to_backup = [
            'midi_generator/parameters/universal_registry.py',
            'midi_generator/parameters/registry.json',
        ]

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files_to_backup:
            src = Path(file_path)
            if src.exists():
                dst = checkpoint_dir / src.name
                shutil.copy2(src, dst)
                backup_paths[file_path] = dst

        # Count current parameters and models
        param_count = self._count_parameters()
        model_count = self._count_models()

        checkpoint = ExpansionCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            backup_paths=backup_paths,
            parameter_count=param_count,
            model_count=model_count,
            description=description
        )

        self.checkpoints.append(checkpoint)

        print(f"  ✅ Checkpoint created: {checkpoint_id}")
        print(f"     Parameters: {param_count}, Models: {model_count}")

        return checkpoint

    def rollback_to_checkpoint(self, checkpoint: ExpansionCheckpoint):
        """
        Rollback system to checkpoint

        Args:
            checkpoint: Checkpoint to rollback to
        """
        print(f"\n{'='*80}")
        print(f"ROLLING BACK TO CHECKPOINT: {checkpoint.checkpoint_id}")
        print(f"{'='*80}\n")

        # Restore backed up files
        for original_path, backup_path in checkpoint.backup_paths.items():
            if backup_path.exists():
                src = Path(original_path)
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_path, src)
                print(f"  ✅ Restored: {original_path}")

        print(f"\n✅ Rollback complete to checkpoint: {checkpoint.checkpoint_id}")

    def _count_parameters(self) -> int:
        """Count registered parameters"""
        try:
            registry_file = Path('midi_generator/parameters/registry.json')
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return len(data)
                    elif isinstance(data, list):
                        return len(data)
            return 0
        except:
            return 0

    def _count_models(self) -> int:
        """Count trained models"""
        models_dir = Path('models/pretrained')
        if models_dir.exists():
            return len(list(models_dir.glob('*.pkl')))
        return 0


class InverseAnalysisCoordinator:
    """Coordinator for inverse MIDI analysis (placeholder)"""

    def analyze(self, midi_file: Path) -> dict:
        """
        Analyze MIDI file and compute reconstruction quality

        Args:
            midi_file: Path to MIDI file

        Returns:
            Analysis dict with quality_score, feature_errors, etc.
        """
        # Mock implementation
        print(f"  Analyzing MIDI: {midi_file}")

        # In real implementation, would:
        # 1. Extract features from MIDI
        # 2. Predict parameters using existing models
        # 3. Generate new MIDI from predicted parameters
        # 4. Compare features and compute error

        # Mock result
        return {
            'quality_score': 0.65,  # Simulated quality score
            'feature_errors': {
                'harmony_complexity': 0.25,
                'voicing_density': 0.30,
                'rhythmic_syncopation': 0.20
            },
            'high_error_features': [
                'harmony_complexity',
                'voicing_density',
                'rhythmic_syncopation'
            ]
        }

    def _reload_models(self):
        """Reload models after new parameters added"""
        print("  Reloading prediction models...")
        # In real implementation, reload all XGBoost models
        pass


class IntelligentGapDetector:
    """Intelligent gap detector (placeholder)"""

    def detect_gaps(self, feature_errors: dict, existing_params: Set[str]) -> List[dict]:
        """
        Detect parameter gaps from feature errors

        Args:
            feature_errors: Dictionary of feature -> error
            existing_params: Set of existing parameter names

        Returns:
            List of gap proposals
        """
        print("  Detecting parameter gaps...")

        # Mock implementation
        gaps = [
            {
                'suggested_parameter': 'harmony.voicing.quartal_probability',
                'impact_score': 0.85,
                'priority': 'HIGH',
                'affected_features': ['harmony_complexity', 'voicing_density'],
                'rationale': 'High error in voicing-related features suggests missing quartal voicing control'
            },
            {
                'suggested_parameter': 'rhythm.syncopation.intensity',
                'impact_score': 0.75,
                'priority': 'MEDIUM',
                'affected_features': ['rhythmic_syncopation'],
                'rationale': 'Rhythmic syncopation errors indicate need for explicit syncopation control'
            }
        ]

        return gaps


class LLMParameterProposalAgent:
    """LLM-powered parameter proposal agent (placeholder)"""

    def propose_parameter(self, gap: dict) -> dict:
        """
        Generate parameter proposal from gap

        Args:
            gap: Gap detection result

        Returns:
            Parameter proposal dict
        """
        print(f"  Proposing parameter for: {gap['suggested_parameter']}")

        # Mock proposal
        return {
            'name': gap['suggested_parameter'],
            'type': 'CONTINUOUS',
            'range': (0.0, 1.0),
            'default': 0.3,
            'description': 'Probability of using quartal (fourth-based) voicings instead of tertian',
            'musical_context': 'Quartal harmony is common in modal jazz. Higher values create more open, modern sounds.',
            'implementation_strategy': 'In voicing generation, check probability before creating chord voicings. If triggered, build voicings in fourths instead of thirds.',
            'affected_features': gap['affected_features'],
            'generator_integration_points': [
                'generators/advanced_harmony_generator.py::HarmonyGenerator.generate_voicing()'
            ],
            'test_cases': [
                {'value': 0.0, 'expected': 'No quartal voicings'},
                {'value': 0.5, 'expected': 'Mix of quartal and tertian'},
                {'value': 1.0, 'expected': 'All quartal voicings'}
            ],
            'example_values': {
                'modal_jazz': 0.7,
                'bebop': 0.1,
                'fusion': 0.6
            }
        }


class ExpansionOrchestrator:
    """
    Master orchestrator for automated system expansion

    Coordinates all agents to execute complete expansion workflow:
    - Inverse analysis
    - Gap detection
    - Parameter proposal
    - Validation
    - Code generation
    - Training data generation
    - Model training
    - Deployment/rollback
    """

    def __init__(self, api_key: Optional[str] = None, mock_mode: bool = False):
        """
        Initialize expansion orchestrator

        Args:
            api_key: Anthropic API key for LLM agents
            mock_mode: If True, use mock implementations
        """
        self.mock_mode = mock_mode

        # Initialize agents
        print("Initializing expansion agents...")

        try:
            self.code_generator = LLMCodeGenerationAgent(api_key=api_key, mock_mode=mock_mode)
            self.validator = MusicalValidator(api_key=api_key, mock_mode=mock_mode)
            self.data_generator = SyntheticTrainingDataGenerator()
            self.model_trainer = ModelTrainingSpecialist()
        except:
            print("WARNING: Some agents could not be initialized, using mock mode")
            self.mock_mode = True

        # Coordinators (mostly placeholders for now)
        self.inverse_analyzer = InverseAnalysisCoordinator()
        self.gap_detector = IntelligentGapDetector()
        self.param_proposer = LLMParameterProposalAgent()

        # Safety
        self.safety_monitor = SafetyMonitor()

        # History
        self.expansion_history: List[ExpansionWorkflowResult] = []

    def expand_from_midi(self,
                        input_midi: Path,
                        auto_approve: bool = False,
                        max_expansions: int = 3,
                        min_improvement: float = 0.05) -> ExpansionWorkflowResult:
        """
        Complete expansion workflow from MIDI input

        Args:
            input_midi: Path to MIDI file that system cannot reconstruct well
            auto_approve: If False, require human approval before deployment
            max_expansions: Maximum parameters to add in one run
            min_improvement: Minimum quality improvement to accept (0.0-1.0)

        Returns:
            ExpansionWorkflowResult
        """
        print(f"\n{'='*80}")
        print("STARTING AUTOMATED EXPANSION WORKFLOW")
        print(f"{'='*80}")
        print(f"Input MIDI: {input_midi}")
        print(f"Auto-approve: {auto_approve}")
        print(f"Max expansions: {max_expansions}")
        print(f"Min improvement: {min_improvement}")
        print()

        start_time = time.time()

        # Create checkpoint
        checkpoint = self.safety_monitor.create_checkpoint("Pre-expansion checkpoint")

        try:
            # STAGE 1: INVERSE ANALYSIS
            print(f"\n{'='*80}")
            print("STAGE 1: INVERSE ANALYSIS")
            print(f"{'='*80}")

            analysis = self.inverse_analyzer.analyze(input_midi)

            print(f"Quality score: {analysis['quality_score']:.3f}")
            print(f"High-error features: {len(analysis['high_error_features'])}")

            if analysis['quality_score'] > 0.8:
                print("\n✅ Quality already sufficient (>0.8), no expansion needed")
                return ExpansionWorkflowResult(
                    success=True,
                    expansions_deployed=[],
                    quality_improvement=0.0,
                    initial_quality=analysis['quality_score'],
                    final_quality=analysis['quality_score'],
                    expansion_details=[],
                    total_time=time.time() - start_time
                )

            initial_quality = analysis['quality_score']

            # STAGE 2: GAP DETECTION
            print(f"\n{'='*80}")
            print("STAGE 2: GAP DETECTION")
            print(f"{'='*80}")

            gaps = self.gap_detector.detect_gaps(
                analysis['feature_errors'],
                set()  # Would load from registry
            )

            if not gaps:
                print("⚠️ No actionable gaps found")
                return ExpansionWorkflowResult(
                    success=False,
                    expansions_deployed=[],
                    quality_improvement=0.0,
                    initial_quality=initial_quality,
                    final_quality=initial_quality,
                    expansion_details=[],
                    failure_reason='No actionable gaps detected',
                    total_time=time.time() - start_time
                )

            print(f"Detected {len(gaps)} potential parameter gaps:")
            for i, gap in enumerate(gaps[:max_expansions], 1):
                print(f"\n  {i}. {gap['suggested_parameter']}")
                print(f"     Impact: {gap['impact_score']:.2f} | Priority: {gap['priority']}")
                print(f"     Affects: {', '.join(gap['affected_features'][:3])}")

            # STAGE 3-10: PROPOSE AND DEPLOY PARAMETERS
            expansions = []
            deployed = []

            for i, gap in enumerate(gaps[:max_expansions], 1):
                print(f"\n{'='*80}")
                print(f"PROCESSING PARAMETER {i}/{min(len(gaps), max_expansions)}")
                print(f"{'='*80}")

                expansion = self._process_single_parameter(gap, auto_approve, checkpoint)
                expansions.append(expansion)

                if expansion.status == ExpansionStatus.SUCCESS:
                    deployed.append(expansion.parameter_name)

            # STAGE 11: VERIFY IMPROVEMENT
            if deployed:
                print(f"\n{'='*80}")
                print("STAGE 11: QUALITY VERIFICATION")
                print(f"{'='*80}")

                print("Re-analyzing MIDI with new parameters...")
                new_analysis = self.inverse_analyzer.analyze(input_midi)
                final_quality = new_analysis['quality_score']
                improvement = final_quality - initial_quality

                print(f"Initial quality: {initial_quality:.3f}")
                print(f"Final quality:   {final_quality:.3f}")
                print(f"Improvement:     {improvement:+.3f} ({improvement*100:+.1f}%)")

                if improvement >= min_improvement:
                    print(f"\n✅ EXPANSION SUCCESSFUL!")
                    print(f"Deployed {len(deployed)} new parameters:")
                    for param in deployed:
                        print(f"  - {param}")

                    result = ExpansionWorkflowResult(
                        success=True,
                        expansions_deployed=deployed,
                        quality_improvement=improvement,
                        initial_quality=initial_quality,
                        final_quality=final_quality,
                        expansion_details=expansions,
                        checkpoint_id=checkpoint.checkpoint_id,
                        total_time=time.time() - start_time
                    )

                    self._record_expansion(result)
                    return result

                else:
                    print(f"\n⚠️ Insufficient improvement ({improvement:.3f}), rolling back...")
                    self.safety_monitor.rollback_to_checkpoint(checkpoint)

                    return ExpansionWorkflowResult(
                        success=False,
                        expansions_deployed=[],
                        quality_improvement=improvement,
                        initial_quality=initial_quality,
                        final_quality=final_quality,
                        expansion_details=expansions,
                        failure_reason=f'Insufficient quality improvement ({improvement:.3f} < {min_improvement})',
                        checkpoint_id=checkpoint.checkpoint_id,
                        total_time=time.time() - start_time
                    )
            else:
                print("\n❌ No parameters successfully deployed")
                return ExpansionWorkflowResult(
                    success=False,
                    expansions_deployed=[],
                    quality_improvement=0.0,
                    initial_quality=initial_quality,
                    final_quality=initial_quality,
                    expansion_details=expansions,
                    failure_reason='All parameter deployments failed',
                    total_time=time.time() - start_time
                )

        except Exception as e:
            print(f"\n💥 CRITICAL ERROR: {e}")
            print("Rolling back all changes...")
            self.safety_monitor.rollback_to_checkpoint(checkpoint)

            return ExpansionWorkflowResult(
                success=False,
                expansions_deployed=[],
                quality_improvement=0.0,
                initial_quality=0.0,
                final_quality=0.0,
                expansion_details=[],
                failure_reason=f'Critical error: {e}',
                checkpoint_id=checkpoint.checkpoint_id,
                total_time=time.time() - start_time
            )

    def _process_single_parameter(self,
                                  gap: dict,
                                  auto_approve: bool,
                                  checkpoint: ExpansionCheckpoint) -> ParameterExpansion:
        """
        Process single parameter expansion

        Args:
            gap: Gap detection result
            auto_approve: Auto-approve deployment
            checkpoint: System checkpoint

        Returns:
            ParameterExpansion
        """
        expansion = ParameterExpansion(
            parameter_name=gap['suggested_parameter'],
            proposal={}
        )

        try:
            # STAGE 3: PARAMETER PROPOSAL
            expansion.stage = ExpansionStage.PARAMETER_PROPOSAL
            print(f"\nStage 3: Proposing parameter...")

            proposal = self.param_proposer.propose_parameter(gap)
            expansion.proposal = proposal

            # STAGE 4: PARAMETER VALIDATION
            expansion.stage = ExpansionStage.PARAMETER_VALIDATION
            print(f"\nStage 4: Validating parameter...")

            validation = self.validator.validate_parameter(proposal)
            expansion.validation_result = validation

            if not validation.valid:
                expansion.status = ExpansionStatus.FAILED
                expansion.error = f"Validation failed: {validation.errors}"
                print(f"  ❌ Validation failed: {validation.errors}")
                return expansion

            # STAGE 5: CODE GENERATION
            expansion.stage = ExpansionStage.CODE_GENERATION
            print(f"\nStage 5: Generating code...")

            generated_code = self.code_generator.generate_implementation(proposal)
            expansion.generated_code = generated_code

            # STAGE 6: CODE VALIDATION
            expansion.stage = ExpansionStage.CODE_VALIDATION
            print(f"\nStage 6: Validating code...")

            code_validation = self.validator.validate_code(
                {
                    'registry_update': generated_code.registry_update,
                    'generator_modifications': generated_code.generator_modifications,
                    'new_methods': generated_code.new_methods,
                    'test_code': generated_code.test_code
                },
                proposal
            )

            if not code_validation.valid:
                expansion.status = ExpansionStatus.FAILED
                expansion.error = f"Code validation failed"
                print(f"  ❌ Code validation failed")
                return expansion

            # Request approval if needed
            if not auto_approve:
                if not self._request_approval(expansion):
                    expansion.status = ExpansionStatus.FAILED
                    expansion.error = "Deployment rejected by user"
                    print("  ❌ Deployment rejected by user")
                    return expansion

            # STAGE 7: TRAINING DATA GENERATION
            expansion.stage = ExpansionStage.TRAINING_DATA_GENERATION
            print(f"\nStage 7: Generating training data...")

            try:
                training_dataset = self.data_generator.generate_training_data(
                    param_name=proposal['name'],
                    param_def=proposal,
                    n_examples=100,  # Reduced for speed
                    output_dir=Path('training_data')
                )
                expansion.training_dataset = training_dataset
                print(f"  ✅ Generated {training_dataset.n_examples} examples")
            except Exception as e:
                expansion.status = ExpansionStatus.FAILED
                expansion.error = f"Training data generation failed: {e}"
                print(f"  ❌ Training data generation failed: {e}")
                return expansion

            # STAGE 8: MODEL TRAINING
            expansion.stage = ExpansionStage.MODEL_TRAINING
            print(f"\nStage 8: Training model...")

            try:
                # Prepare training data
                training_data = [
                    {
                        'features': ex.features,
                        'parameter_value': ex.parameter_value
                    }
                    for ex in training_dataset.examples
                ]

                training_result = self.model_trainer.train_parameter_model(
                    param_name=proposal['name'],
                    param_def=proposal,
                    training_data=training_data
                )
                expansion.training_result = training_result

                if not training_result.success:
                    expansion.status = ExpansionStatus.FAILED
                    expansion.error = f"Model training failed: {training_result.error}"
                    print(f"  ❌ Model training failed: {training_result.error}")
                    return expansion

                print(f"  ✅ Model trained successfully")

            except Exception as e:
                expansion.status = ExpansionStatus.FAILED
                expansion.error = f"Model training failed: {e}"
                print(f"  ❌ Model training failed: {e}")
                return expansion

            # STAGE 9: DEPLOYMENT
            expansion.stage = ExpansionStage.DEPLOYMENT
            print(f"\nStage 9: Deploying parameter...")

            try:
                self._deploy_parameter(expansion)
                expansion.status = ExpansionStatus.SUCCESS
                expansion.stage = ExpansionStage.COMPLETED
                print(f"  ✅ Parameter deployed successfully")

            except Exception as e:
                expansion.status = ExpansionStatus.FAILED
                expansion.error = f"Deployment failed: {e}"
                print(f"  ❌ Deployment failed: {e}")
                return expansion

            return expansion

        except Exception as e:
            expansion.status = ExpansionStatus.FAILED
            expansion.error = str(e)
            print(f"  ❌ Error: {e}")
            return expansion

    def _request_approval(self, expansion: ParameterExpansion) -> bool:
        """Request human approval for deployment"""

        proposal = expansion.proposal
        validation = expansion.validation_result

        print(f"\n{'='*80}")
        print("APPROVAL REQUIRED")
        print(f"{'='*80}")
        print(f"\nParameter: {proposal['name']}")
        print(f"Type: {proposal.get('type', 'UNKNOWN')}")
        print(f"Range: {proposal.get('range', 'UNKNOWN')}")
        print(f"Default: {proposal.get('default', 'UNKNOWN')}")
        print(f"\nDescription: {proposal.get('description', 'No description')}")
        print(f"\nMusical Context: {proposal.get('musical_context', 'No context')}")

        if validation:
            print(f"\nValidation Score: {validation.overall_score:.2f}")

            if validation.warnings:
                print(f"\n⚠️ Warnings:")
                for warn in validation.warnings[:3]:
                    print(f"  - {warn}")

        response = input("\nApprove deployment? (yes/no): ").strip().lower()
        return response in ['yes', 'y']

    def _deploy_parameter(self, expansion: ParameterExpansion):
        """
        Deploy parameter to system

        Args:
            expansion: Parameter expansion
        """
        # In real implementation, would:
        # 1. Apply code changes to generator files
        # 2. Update parameter registry
        # 3. Register model with inverse analyzer
        # 4. Update documentation

        print("  Deploying code changes...")
        print("  Updating parameter registry...")
        print("  Registering model...")

        # Mock deployment
        pass

    def _record_expansion(self, result: ExpansionWorkflowResult):
        """Record expansion in history"""

        self.expansion_history.append(result)

        # Save to JSON
        history_file = Path('expansion_history.json')

        history_data = {
            'timestamp': datetime.now().isoformat(),
            'success': result.success,
            'expansions_deployed': result.expansions_deployed,
            'quality_improvement': result.quality_improvement,
            'initial_quality': result.initial_quality,
            'final_quality': result.final_quality,
            'total_time': result.total_time
        }

        # Load existing history
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        else:
            history = []

        history.append(history_data)

        # Save
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def batch_expand_from_dataset(self,
                                  midi_files: List[Path],
                                  max_expansions_per_file: int = 3) -> Dict[str, ExpansionWorkflowResult]:
        """
        Expand system by analyzing multiple MIDI files

        Args:
            midi_files: List of MIDI files to analyze
            max_expansions_per_file: Max expansions per file

        Returns:
            Dictionary mapping filename to expansion result
        """
        print(f"\n{'='*80}")
        print(f"BATCH EXPANSION FROM {len(midi_files)} MIDI FILES")
        print(f"{'='*80}\n")

        results = {}

        for i, midi_file in enumerate(midi_files, 1):
            print(f"\n[{i}/{len(midi_files)}] Processing: {midi_file}")
            print("-" * 80)

            try:
                result = self.expand_from_midi(
                    input_midi=midi_file,
                    auto_approve=True,  # Auto-approve for batch
                    max_expansions=max_expansions_per_file
                )
                results[str(midi_file)] = result

                if result.success:
                    print(f"  ✅ Success: {len(result.expansions_deployed)} parameters deployed")
                else:
                    print(f"  ❌ Failed: {result.failure_reason}")

            except Exception as e:
                print(f"  ❌ Error: {e}")

        # Print batch summary
        self._print_batch_summary(results)

        return results

    def _print_batch_summary(self, results: Dict[str, ExpansionWorkflowResult]):
        """Print batch expansion summary"""

        print(f"\n{'='*80}")
        print("BATCH EXPANSION SUMMARY")
        print(f"{'='*80}")

        total = len(results)
        successful = sum(1 for r in results.values() if r.success)
        failed = total - successful

        total_params = sum(len(r.expansions_deployed) for r in results.values())
        avg_improvement = np.mean([r.quality_improvement for r in results.values() if r.success]) if successful > 0 else 0.0

        print(f"\nTotal files: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total parameters added: {total_params}")
        print(f"Avg quality improvement: {avg_improvement:+.3f}")

        if successful > 0:
            print(f"\nSuccessful Expansions:")
            for filename, result in results.items():
                if result.success:
                    print(f"  ✅ {Path(filename).name}: {len(result.expansions_deployed)} params, +{result.quality_improvement:.3f} quality")

        print(f"\n{'='*80}\n")

    def get_expansion_statistics(self) -> dict:
        """Get statistics about expansion history"""

        if not self.expansion_history:
            return {
                'total_expansions': 0,
                'successful_expansions': 0,
                'total_parameters_added': 0,
                'avg_quality_improvement': 0.0
            }

        successful = [h for h in self.expansion_history if h.success]

        return {
            'total_expansions': len(self.expansion_history),
            'successful_expansions': len(successful),
            'failed_expansions': len(self.expansion_history) - len(successful),
            'total_parameters_added': sum(len(h.expansions_deployed) for h in successful),
            'avg_quality_improvement': np.mean([h.quality_improvement for h in successful]) if successful else 0.0,
            'total_time': sum(h.total_time for h in self.expansion_history)
        }


# Example usage
if __name__ == '__main__':
    import numpy as np  # For batch summary

    # Create orchestrator
    orchestrator = ExpansionOrchestrator(mock_mode=True)

    # Example MIDI file
    example_midi = Path('examples/problematic_reconstruction.mid')

    # Run expansion workflow
    result = orchestrator.expand_from_midi(
        input_midi=example_midi,
        auto_approve=True,  # Auto-approve for demo
        max_expansions=2
    )

    # Print result
    print(f"\n{'='*80}")
    print("EXPANSION RESULT")
    print(f"{'='*80}")
    print(f"Success: {result.success}")
    print(f"Parameters deployed: {len(result.expansions_deployed)}")
    print(f"Quality improvement: {result.quality_improvement:+.3f}")
    print(f"Total time: {result.total_time:.2f}s")

    if result.success:
        print(f"\nDeployed parameters:")
        for param in result.expansions_deployed:
            print(f"  - {param}")
    else:
        print(f"\nFailure reason: {result.failure_reason}")

    # Get statistics
    stats = orchestrator.get_expansion_statistics()
    print(f"\n{'='*80}")
    print("EXPANSION STATISTICS")
    print(f"{'='*80}")
    for key, value in stats.items():
        print(f"{key}: {value}")
