#!/usr/bin/env python3
"""
Dataset Utilities for MIDI Corpus Training
Agent 03: Metadata & Labeling Manager

Comprehensive utilities for managing labeled MIDI datasets including:
    - Dataset format specification and validation
    - PyTorch dataset classes
    - Train/validation/test splitting
    - Data loading and batching
    - Export to multiple formats (JSON, CSV, HDF5, pickle)
    - Dataset statistics and analysis

Author: Agent 03 - Metadata & Labeling Manager
License: MIT
"""

import json
import pickle
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import warnings

# Optional imports
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Install with: pip install torch")

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("WARNING: h5py not available. Install with: pip install h5py")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("WARNING: pandas not available. Install with: pip install pandas")


# ==============================================================================
# DATASET FORMAT SPECIFICATION
# ==============================================================================

@dataclass
class LabeledDatasetEntry:
    """
    Single entry in the labeled dataset.

    Combines auto-extracted labels, manual labels, and metadata.
    """
    file_id: str
    file_path: str

    # Labels
    level1_labels: Dict[str, Any]
    level2_labels: Dict[str, Any]
    level3_labels: Dict[str, Any]

    # Labeling metadata
    auto_labeled: bool = True
    manually_labeled: bool = False
    labeler_id: Optional[str] = None
    labeling_timestamp: Optional[str] = None

    # Quality scores
    quality_score: Optional[float] = None
    confidence_score: Optional[float] = None

    # Notes and flags
    notes: str = ""
    flagged: bool = False
    flag_reason: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'file_id': self.file_id,
            'file_path': self.file_path,
            'labels': {
                'level1': self.level1_labels,
                'level2': self.level2_labels,
                'level3': self.level3_labels
            },
            'metadata': {
                'auto_labeled': self.auto_labeled,
                'manually_labeled': self.manually_labeled,
                'labeler_id': self.labeler_id,
                'labeling_timestamp': self.labeling_timestamp,
                'quality_score': self.quality_score,
                'confidence_score': self.confidence_score,
                'notes': self.notes,
                'flagged': self.flagged,
                'flag_reason': self.flag_reason
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LabeledDatasetEntry':
        """Create from dictionary."""
        return cls(
            file_id=data['file_id'],
            file_path=data['file_path'],
            level1_labels=data['labels']['level1'],
            level2_labels=data['labels']['level2'],
            level3_labels=data['labels']['level3'],
            auto_labeled=data['metadata'].get('auto_labeled', True),
            manually_labeled=data['metadata'].get('manually_labeled', False),
            labeler_id=data['metadata'].get('labeler_id'),
            labeling_timestamp=data['metadata'].get('labeling_timestamp'),
            quality_score=data['metadata'].get('quality_score'),
            confidence_score=data['metadata'].get('confidence_score'),
            notes=data['metadata'].get('notes', ''),
            flagged=data['metadata'].get('flagged', False),
            flag_reason=data['metadata'].get('flag_reason', '')
        )

    def get_all_labels_flat(self) -> Dict[str, Any]:
        """Get all labels as a flat dictionary."""
        flat_labels = {}
        for level_labels in [self.level1_labels, self.level2_labels, self.level3_labels]:
            flat_labels.update(level_labels)
        return flat_labels


# ==============================================================================
# PYTORCH DATASET CLASSES
# ==============================================================================

if TORCH_AVAILABLE:
    class MIDILabeledDataset(Dataset):
        """
        PyTorch Dataset for labeled MIDI files.

        Returns (features, labels) pairs where:
            - features: extracted musical features (to be computed by feature extractor)
            - labels: hierarchical parameter labels
        """

        def __init__(self,
                     dataset_entries: List[LabeledDatasetEntry],
                     feature_extractor: Optional[Any] = None,
                     transform: Optional[Any] = None,
                     target_transform: Optional[Any] = None):
            """
            Initialize dataset.

            Args:
                dataset_entries: List of LabeledDatasetEntry objects
                feature_extractor: Feature extraction function/class
                transform: Optional transform to apply to features
                target_transform: Optional transform to apply to labels
            """
            self.entries = dataset_entries
            self.feature_extractor = feature_extractor
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self) -> int:
            return len(self.entries)

        def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
            """
            Get item at index.

            Returns:
                (features, labels) tuple
            """
            entry = self.entries[idx]

            # Extract features (if feature_extractor provided)
            if self.feature_extractor:
                features = self.feature_extractor.extract(entry.file_path)
            else:
                # Placeholder: return file path for lazy loading
                features = entry.file_path

            # Get labels
            labels = entry.get_all_labels_flat()

            # Apply transforms
            if self.transform:
                features = self.transform(features)
            if self.target_transform:
                labels = self.target_transform(labels)

            return features, labels

        def get_label_statistics(self) -> Dict[str, Dict[str, float]]:
            """Compute statistics for all labels."""
            stats = defaultdict(lambda: {'values': []})

            for entry in self.entries:
                for param, value in entry.get_all_labels_flat().items():
                    if isinstance(value, (int, float)) and value is not None:
                        stats[param]['values'].append(value)

            # Compute statistics
            for param, data in stats.items():
                values = data['values']
                if values:
                    stats[param] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }

            return dict(stats)


    class HierarchicalMTLDataset(Dataset):
        """
        PyTorch Dataset for hierarchical multi-task learning.

        Returns (features, level1_labels, level2_labels, level3_labels)
        to support hierarchical prediction.
        """

        def __init__(self,
                     dataset_entries: List[LabeledDatasetEntry],
                     feature_extractor: Optional[Any] = None):
            """Initialize dataset."""
            self.entries = dataset_entries
            self.feature_extractor = feature_extractor

        def __len__(self) -> int:
            return len(self.entries)

        def __getitem__(self, idx: int):
            """Get item with hierarchical labels."""
            entry = self.entries[idx]

            # Extract features
            if self.feature_extractor:
                features = self.feature_extractor.extract(entry.file_path)
            else:
                features = entry.file_path

            return (
                features,
                entry.level1_labels,
                entry.level2_labels,
                entry.level3_labels
            )


# ==============================================================================
# TRAIN/VAL/TEST SPLITTING
# ==============================================================================

class DatasetSplitter:
    """
    Utilities for splitting dataset into train/validation/test sets.

    Supports stratified splitting by genre.
    """

    @staticmethod
    def stratified_split(
            entries: List[LabeledDatasetEntry],
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            test_ratio: float = 0.15,
            stratify_by: str = 'genre.primary',
            random_seed: int = 42
    ) -> Tuple[List, List, List]:
        """
        Split dataset with stratification by specified parameter.

        Args:
            entries: List of dataset entries
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            stratify_by: Parameter to stratify by (e.g., 'genre.primary')
            random_seed: Random seed for reproducibility

        Returns:
            (train_entries, val_entries, test_entries)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        np.random.seed(random_seed)

        # Group entries by stratification key
        stratified_groups = defaultdict(list)
        for entry in entries:
            # Get stratification value (from level1 labels)
            strat_value = entry.level1_labels.get(stratify_by, 'unknown')
            stratified_groups[strat_value].append(entry)

        # Split each group
        train_set = []
        val_set = []
        test_set = []

        for group_name, group_entries in stratified_groups.items():
            # Shuffle
            indices = np.random.permutation(len(group_entries))
            shuffled_entries = [group_entries[i] for i in indices]

            # Split
            n = len(shuffled_entries)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_set.extend(shuffled_entries[:train_end])
            val_set.extend(shuffled_entries[train_end:val_end])
            test_set.extend(shuffled_entries[val_end:])

            print(f"  {group_name}: {len(shuffled_entries)} total → "
                  f"train={train_end}, val={val_end - train_end}, test={n - val_end}")

        print(f"\nTotal: {len(entries)} → "
              f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

        return train_set, val_set, test_set

    @staticmethod
    def save_split(
            train: List[LabeledDatasetEntry],
            val: List[LabeledDatasetEntry],
            test: List[LabeledDatasetEntry],
            output_dir: Path
    ):
        """Save train/val/test split to separate files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            output_file = output_dir / f'{split_name}.json'

            data = {
                'split': split_name,
                'count': len(split_data),
                'entries': [entry.to_dict() for entry in split_data]
            }

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"Saved {split_name} set ({len(split_data)} entries) to {output_file}")

    @staticmethod
    def load_split(split_file: Path) -> List[LabeledDatasetEntry]:
        """Load a split from file."""
        with open(split_file, 'r') as f:
            data = json.load(f)

        entries = [LabeledDatasetEntry.from_dict(entry_dict) for entry_dict in data['entries']]

        print(f"Loaded {data['split']} set with {len(entries)} entries")

        return entries


# ==============================================================================
# DATASET EXPORT UTILITIES
# ==============================================================================

class DatasetExporter:
    """Export dataset to various formats."""

    @staticmethod
    def to_json(entries: List[LabeledDatasetEntry], output_file: Path):
        """Export to JSON format."""
        data = {
            'version': '2.0',
            'count': len(entries),
            'entries': [entry.to_dict() for entry in entries]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(entries)} entries to JSON: {output_file}")

    @staticmethod
    def to_csv(entries: List[LabeledDatasetEntry], output_file: Path):
        """Export to CSV format (flattened)."""
        if not entries:
            print("No entries to export")
            return

        # Get all possible parameter names
        all_params = set()
        for entry in entries:
            all_params.update(entry.get_all_labels_flat().keys())

        all_params = sorted(all_params)

        # Write CSV
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['file_id', 'file_path'] + all_params + \
                        ['auto_labeled', 'manually_labeled', 'quality_score']

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in entries:
                row = {
                    'file_id': entry.file_id,
                    'file_path': entry.file_path,
                    'auto_labeled': entry.auto_labeled,
                    'manually_labeled': entry.manually_labeled,
                    'quality_score': entry.quality_score or ''
                }

                # Add labels
                labels = entry.get_all_labels_flat()
                for param in all_params:
                    row[param] = labels.get(param, '')

                writer.writerow(row)

        print(f"Exported {len(entries)} entries to CSV: {output_file}")

    @staticmethod
    def to_hdf5(entries: List[LabeledDatasetEntry], output_file: Path):
        """Export to HDF5 format."""
        if not HDF5_AVAILABLE:
            print("ERROR: h5py not available. Cannot export to HDF5.")
            return

        # Organize data
        file_ids = []
        file_paths = []
        labels_dict = defaultdict(list)

        for entry in entries:
            file_ids.append(entry.file_id)
            file_paths.append(entry.file_path)

            # Collect all labels
            for param, value in entry.get_all_labels_flat().items():
                # Handle None values
                if value is None:
                    value = -999.0  # Sentinel value for missing
                # Convert categorical to string/int
                if isinstance(value, str):
                    value = value.encode('utf-8')

                labels_dict[param].append(value)

        # Write HDF5
        with h5py.File(output_file, 'w') as f:
            # Metadata
            f.attrs['version'] = '2.0'
            f.attrs['count'] = len(entries)

            # File info
            f.create_dataset('file_ids', data=np.array(file_ids, dtype='S'))
            f.create_dataset('file_paths', data=np.array(file_paths, dtype='S'))

            # Labels (each parameter as separate dataset)
            labels_group = f.create_group('labels')
            for param, values in labels_dict.items():
                # Determine dtype
                if all(isinstance(v, bytes) for v in values):
                    dtype = 'S'
                else:
                    dtype = 'f'

                labels_group.create_dataset(param, data=np.array(values, dtype=dtype))

        print(f"Exported {len(entries)} entries to HDF5: {output_file}")

    @staticmethod
    def to_pickle(entries: List[LabeledDatasetEntry], output_file: Path):
        """Export to pickle format."""
        with open(output_file, 'wb') as f:
            pickle.dump(entries, f)

        print(f"Exported {len(entries)} entries to pickle: {output_file}")

    @staticmethod
    def to_pandas(entries: List[LabeledDatasetEntry]) -> 'pd.DataFrame':
        """Convert to pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            print("ERROR: pandas not available")
            return None

        # Flatten to rows
        rows = []
        for entry in entries:
            row = {
                'file_id': entry.file_id,
                'file_path': entry.file_path,
                'auto_labeled': entry.auto_labeled,
                'manually_labeled': entry.manually_labeled,
                'quality_score': entry.quality_score
            }
            row.update(entry.get_all_labels_flat())
            rows.append(row)

        return pd.DataFrame(rows)


# ==============================================================================
# DATASET LOADER
# ==============================================================================

class LabeledDatasetLoader:
    """
    Load and manage labeled datasets.
    """

    @staticmethod
    def load_from_json(json_file: Path) -> List[LabeledDatasetEntry]:
        """Load dataset from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)

        entries = [LabeledDatasetEntry.from_dict(entry_dict) for entry_dict in data['entries']]

        print(f"Loaded {len(entries)} entries from {json_file}")

        return entries

    @staticmethod
    def merge_auto_and_manual_labels(
            auto_labels_file: Path,
            manual_labels_dir: Path
    ) -> List[LabeledDatasetEntry]:
        """
        Merge auto-extracted labels with manual labels.

        Args:
            auto_labels_file: JSON file with auto-extracted labels
            manual_labels_dir: Directory containing manual label JSON files

        Returns:
            List of merged LabeledDatasetEntry objects
        """
        # Load auto-extracted labels
        with open(auto_labels_file, 'r') as f:
            auto_data = json.load(f)

        # Create entries from auto-labels
        entries_dict = {}
        for auto_entry in auto_data.get('labels', []):
            file_id = auto_entry['file_id']

            entry = LabeledDatasetEntry(
                file_id=file_id,
                file_path=auto_entry['file_path'],
                level1_labels=auto_entry['labels']['level1'],
                level2_labels=auto_entry['labels']['level2'],
                level3_labels=auto_entry['labels']['level3'],
                auto_labeled=True,
                manually_labeled=False
            )

            entries_dict[file_id] = entry

        # Merge manual labels
        manual_labels_dir = Path(manual_labels_dir)
        manual_files = list(manual_labels_dir.glob('*.json'))

        print(f"Found {len(manual_files)} manual label files")

        for manual_file in manual_files:
            with open(manual_file, 'r') as f:
                manual_data = json.load(f)

            file_id = manual_data['file_id']

            if file_id in entries_dict:
                entry = entries_dict[file_id]

                # Merge manual labels into appropriate levels
                manual_labels = manual_data.get('manual_labels', {})

                for param, value in manual_labels.items():
                    if value is not None:
                        # Determine level
                        if param in ['energy.level', 'complexity.overall']:
                            entry.level1_labels[param] = value
                        elif param.startswith('harmony.') or param.startswith('melody.'):
                            entry.level2_labels[param] = value
                        else:
                            entry.level3_labels[param] = value

                # Update metadata
                entry.manually_labeled = True
                entry.labeler_id = manual_data.get('labeler_id')
                entry.labeling_timestamp = manual_data.get('timestamp')
                entry.notes = manual_data.get('notes', '')

                print(f"  ✓ Merged manual labels for {file_id}")
            else:
                print(f"  ⚠ Manual labels for {file_id} not found in auto-labels")

        return list(entries_dict.values())

    @staticmethod
    def create_pytorch_dataloader(
            entries: List[LabeledDatasetEntry],
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 4,
            **kwargs
    ) -> 'DataLoader':
        """
        Create PyTorch DataLoader.

        Args:
            entries: Dataset entries
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker processes
            **kwargs: Additional arguments for DataLoader

        Returns:
            DataLoader
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        dataset = MIDILabeledDataset(entries)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def validate_dataset(entries: List[LabeledDatasetEntry]) -> Dict[str, Any]:
    """
    Validate dataset and return validation report.

    Checks:
        - Missing values
        - Out-of-range values
        - Consistency checks
        - Genre distribution
    """
    report = {
        'total_entries': len(entries),
        'issues': [],
        'statistics': {}
    }

    # Check for missing values
    missing_counts = defaultdict(int)
    for entry in entries:
        for param, value in entry.get_all_labels_flat().items():
            if value is None:
                missing_counts[param] += 1

    if missing_counts:
        report['issues'].append({
            'type': 'missing_values',
            'details': dict(missing_counts)
        })

    # Check value ranges (for continuous params)
    range_violations = defaultdict(list)
    for entry in entries:
        for param, value in entry.get_all_labels_flat().items():
            if isinstance(value, (int, float)) and value is not None:
                # Most continuous params should be in [0, 1]
                if param not in ['tempo.bpm', 'melody.range_semitones', 'dynamics.overall_level']:
                    if not (0.0 <= value <= 1.0):
                        range_violations[param].append((entry.file_id, value))

    if range_violations:
        report['issues'].append({
            'type': 'range_violations',
            'details': {k: len(v) for k, v in range_violations.items()}
        })

    # Genre distribution
    genre_counts = defaultdict(int)
    for entry in entries:
        genre = entry.level1_labels.get('genre.primary', 'unknown')
        genre_counts[genre] += 1

    report['statistics']['genre_distribution'] = dict(genre_counts)

    # Manual vs auto labeling
    manual_count = sum(1 for e in entries if e.manually_labeled)
    report['statistics']['manual_labeled_count'] = manual_count
    report['statistics']['auto_labeled_count'] = len(entries) - manual_count

    return report


def print_dataset_summary(entries: List[LabeledDatasetEntry]):
    """Print a summary of the dataset."""
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(f"\nTotal entries: {len(entries)}")

    # Genre distribution
    print("\nGenre Distribution:")
    genre_counts = defaultdict(int)
    for entry in entries:
        genre = entry.level1_labels.get('genre.primary', 'unknown')
        genre_counts[genre] += 1

    for genre, count in sorted(genre_counts.items()):
        percentage = (count / len(entries)) * 100
        print(f"  {genre}: {count} ({percentage:.1f}%)")

    # Manual vs auto
    manual_count = sum(1 for e in entries if e.manually_labeled)
    auto_count = len(entries) - manual_count
    print(f"\nLabeling Method:")
    print(f"  Auto-labeled: {auto_count}")
    print(f"  Manually-labeled: {manual_count}")

    # Missing values
    print("\nMissing Values:")
    missing_counts = defaultdict(int)
    for entry in entries:
        for param, value in entry.get_all_labels_flat().items():
            if value is None:
                missing_counts[param] += 1

    if missing_counts:
        for param, count in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / len(entries)) * 100
            print(f"  {param}: {count} ({percentage:.1f}%)")
    else:
        print("  No missing values!")

    print("\n" + "=" * 80)


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python dataset_utils.py <command> <args>")
        print("\nCommands:")
        print("  split <input_json> <output_dir> - Split dataset into train/val/test")
        print("  export <input_json> <format> <output> - Export to format (json/csv/hdf5/pickle)")
        print("  validate <input_json> - Validate dataset")
        print("  summary <input_json> - Print dataset summary")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'split':
        input_file = Path(sys.argv[2])
        output_dir = Path(sys.argv[3])

        entries = LabeledDatasetLoader.load_from_json(input_file)
        train, val, test = DatasetSplitter.stratified_split(entries)
        DatasetSplitter.save_split(train, val, test, output_dir)

    elif command == 'export':
        input_file = Path(sys.argv[2])
        format_type = sys.argv[3]
        output_file = Path(sys.argv[4])

        entries = LabeledDatasetLoader.load_from_json(input_file)

        if format_type == 'json':
            DatasetExporter.to_json(entries, output_file)
        elif format_type == 'csv':
            DatasetExporter.to_csv(entries, output_file)
        elif format_type == 'hdf5':
            DatasetExporter.to_hdf5(entries, output_file)
        elif format_type == 'pickle':
            DatasetExporter.to_pickle(entries, output_file)
        else:
            print(f"Unknown format: {format_type}")

    elif command == 'validate':
        input_file = Path(sys.argv[2])
        entries = LabeledDatasetLoader.load_from_json(input_file)
        report = validate_dataset(entries)
        print(json.dumps(report, indent=2))

    elif command == 'summary':
        input_file = Path(sys.argv[2])
        entries = LabeledDatasetLoader.load_from_json(input_file)
        print_dataset_summary(entries)

    else:
        print(f"Unknown command: {command}")
