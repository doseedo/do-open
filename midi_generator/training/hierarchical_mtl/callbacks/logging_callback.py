"""
Logging Callback for Hierarchical MTL Training.

Author: Agent 06
Date: November 20, 2025
"""

from typing import Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime


class LoggingCallback:
    """
    Callback for logging training metrics.

    Supports console logging and optional integration with Wandb/MLflow.
    """

    def __init__(
        self,
        log_dir: Path,
        use_wandb: bool = False,
        use_mlflow: bool = False,
        experiment_name: str = "hierarchical_mtl",
        run_name: Optional[str] = None,
        log_every_n_steps: int = 10,
        verbose: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_every_n_steps = log_every_n_steps
        self.verbose = verbose

        # Initialize experiment tracking
        self.wandb_run = None
        self.mlflow_run = None

        if use_wandb:
            self._init_wandb()

        if use_mlflow:
            self._init_mlflow()

        # Training history
        self.history = {
            'train': [],
            'val': [],
            'test': []
        }

    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.experiment_name,
                name=self.run_name,
                reinit=True
            )
            if self.verbose:
                print(f"✓ Initialized Wandb tracking: {self.experiment_name}/{self.run_name}")
        except ImportError:
            print("Warning: wandb not installed, skipping wandb tracking")
            self.use_wandb = False

    def _init_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_run = mlflow.start_run(run_name=self.run_name)
            if self.verbose:
                print(f"✓ Initialized MLflow tracking: {self.experiment_name}/{self.run_name}")
        except ImportError:
            print("Warning: mlflow not installed, skipping mlflow tracking")
            self.use_mlflow = False

    def on_train_begin(self, config: Dict[str, Any]):
        """Called at the beginning of training."""
        # Log config
        if self.use_wandb:
            import wandb
            wandb.config.update(config)

        if self.use_mlflow:
            import mlflow
            mlflow.log_params(self._flatten_dict(config))

        # Save config to file
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Called at the end of each epoch."""
        # Store in history
        self.history['train'].append({
            'epoch': epoch,
            **train_metrics
        })
        self.history['val'].append({
            'epoch': epoch,
            **val_metrics
        })

        # Console logging
        if self.verbose:
            print(f"\nEpoch {epoch}:")
            print(f"  Train - {self._format_metrics(train_metrics)}")
            print(f"  Val   - {self._format_metrics(val_metrics)}")

        # Log to wandb
        if self.use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()}
            })

        # Log to mlflow
        if self.use_mlflow:
            import mlflow
            for k, v in train_metrics.items():
                mlflow.log_metric(f'train_{k}', v, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f'val_{k}', v, step=epoch)

    def on_train_end(self, test_metrics: Optional[Dict[str, float]] = None):
        """Called at the end of training."""
        # Log test metrics if provided
        if test_metrics is not None:
            self.history['test'].append(test_metrics)

            if self.verbose:
                print(f"\nFinal Test Results:")
                print(f"  {self._format_metrics(test_metrics)}")

            if self.use_wandb:
                import wandb
                wandb.log({f'test/{k}': v for k, v in test_metrics.items()})

            if self.use_mlflow:
                import mlflow
                for k, v in test_metrics.items():
                    mlflow.log_metric(f'test_{k}', v)

        # Save history
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Close wandb run
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()

        # Close mlflow run
        if self.use_mlflow and self.mlflow_run is not None:
            import mlflow
            mlflow.end_run()

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for console display."""
        return ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def __repr__(self) -> str:
        return f"LoggingCallback(wandb={self.use_wandb}, mlflow={self.use_mlflow})"
