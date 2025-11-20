#!/usr/bin/env python3
"""
Performance Optimizer - Agent 28
================================

High-performance optimization system for scaling to 800+ parameters while
maintaining sub-second prediction times and minimal memory footprint.

This module provides:
1. Intelligent model caching with LRU eviction
2. Parallel model loading and inference
3. Feature extraction caching and optimization
4. Model compression (pruning, quantization)
5. Batch prediction optimization
6. Memory-mapped model loading for large-scale deployment
7. Performance monitoring and profiling
8. Adaptive optimization strategies

Performance Targets:
- Parameter prediction: <1 second for 800 parameters
- Training time per model: <30 seconds
- Memory usage: <2GB RAM
- Model storage: <500MB total

Author: Agent 28 - Performance Optimizer
License: MIT
"""

import os
import sys
import json
import time
import hashlib
import pickle
import mmap
import threading
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import warnings

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Install with: pip install numpy")

try:
    import joblib
    from joblib import Memory
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("Joblib not available. Install with: pip install joblib")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available for memory monitoring. Install with: pip install psutil")


# ============================================================================
# Performance Monitoring
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    operation: str
    duration: float
    memory_delta: float = 0.0
    cpu_percent: float = 0.0
    num_operations: int = 1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'operation': self.operation,
            'duration': self.duration,
            'memory_delta_mb': self.memory_delta,
            'cpu_percent': self.cpu_percent,
            'num_operations': self.num_operations,
            'ops_per_second': self.num_operations / self.duration if self.duration > 0 else 0,
            'timestamp': self.timestamp,
            **self.metadata
        }


class PerformanceMonitor:
    """
    Advanced performance monitoring and profiling system

    Features:
    - Automatic timing and memory tracking
    - CPU utilization monitoring
    - Operation profiling
    - Performance reporting
    - Bottleneck detection
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize performance monitor

        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self.metrics: List[PerformanceMetrics] = []
        self._start_times: Dict[str, float] = {}
        self._start_memory: Dict[str, float] = {}
        self.lock = threading.Lock()

    @contextmanager
    def profile(self, operation: str, metadata: Optional[Dict] = None):
        """
        Context manager for profiling operations

        Usage:
            with monitor.profile('model_loading'):
                load_model()
        """
        if not self.enabled:
            yield
            return

        # Get initial stats
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_percent()

        try:
            yield
        finally:
            # Calculate metrics
            duration = time.time() - start_time
            memory_delta = self._get_memory_usage() - start_memory
            cpu_percent = self._get_cpu_percent()

            # Record metrics
            metric = PerformanceMetrics(
                operation=operation,
                duration=duration,
                memory_delta=memory_delta,
                cpu_percent=cpu_percent,
                metadata=metadata or {}
            )

            with self.lock:
                self.metrics.append(metric)

    def start(self, operation: str):
        """Start timing an operation"""
        if not self.enabled:
            return

        self._start_times[operation] = time.time()
        self._start_memory[operation] = self._get_memory_usage()

    def end(self, operation: str, metadata: Optional[Dict] = None):
        """End timing an operation"""
        if not self.enabled:
            return

        if operation not in self._start_times:
            warnings.warn(f"No start time for operation: {operation}")
            return

        duration = time.time() - self._start_times[operation]
        memory_delta = self._get_memory_usage() - self._start_memory.get(operation, 0)

        metric = PerformanceMetrics(
            operation=operation,
            duration=duration,
            memory_delta=memory_delta,
            cpu_percent=self._get_cpu_percent(),
            metadata=metadata or {}
        )

        with self.lock:
            self.metrics.append(metric)

        # Cleanup
        del self._start_times[operation]
        if operation in self._start_memory:
            del self._start_memory[operation]

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        return 0.0

    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage"""
        if PSUTIL_AVAILABLE:
            return psutil.cpu_percent(interval=0.1)
        return 0.0

    def get_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance summary

        Args:
            operation: Filter by operation name (optional)

        Returns:
            Summary statistics
        """
        with self.lock:
            metrics = self.metrics

        if operation:
            metrics = [m for m in metrics if m.operation == operation]

        if not metrics:
            return {}

        durations = [m.duration for m in metrics]
        memory_deltas = [m.memory_delta for m in metrics]

        return {
            'operation': operation or 'all',
            'count': len(metrics),
            'total_duration': sum(durations),
            'avg_duration': np.mean(durations) if NUMPY_AVAILABLE else sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_memory_delta_mb': sum(memory_deltas),
            'avg_memory_delta_mb': np.mean(memory_deltas) if NUMPY_AVAILABLE else sum(memory_deltas) / len(memory_deltas),
        }

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all operations"""
        operations = set(m.operation for m in self.metrics)
        return {op: self.get_summary(op) for op in operations}

    def print_report(self):
        """Print performance report"""
        summaries = self.get_all_summaries()

        print("\n" + "="*80)
        print("PERFORMANCE REPORT")
        print("="*80)

        for operation, summary in sorted(summaries.items()):
            print(f"\n{operation}:")
            print(f"  Count:          {summary['count']}")
            print(f"  Total Time:     {summary['total_duration']:.3f}s")
            print(f"  Avg Time:       {summary['avg_duration']:.3f}s")
            print(f"  Min Time:       {summary['min_duration']:.3f}s")
            print(f"  Max Time:       {summary['max_duration']:.3f}s")
            print(f"  Memory Delta:   {summary['avg_memory_delta_mb']:.2f} MB (avg)")

        print("\n" + "="*80)

    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.metrics.clear()
            self._start_times.clear()
            self._start_memory.clear()


# ============================================================================
# Model Caching
# ============================================================================

class LRUModelCache:
    """
    Least Recently Used (LRU) cache for XGBoost models

    Features:
    - Automatic eviction of least recently used models
    - Configurable cache size
    - Thread-safe operations
    - Memory-aware caching
    """

    def __init__(self, max_size: int = 100, max_memory_mb: float = 1024):
        """
        Initialize LRU cache

        Args:
            max_size: Maximum number of models to cache
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache: OrderedDict = OrderedDict()
        self.sizes: Dict[str, float] = {}  # Model sizes in MB
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get model from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None

    def put(self, key: str, model: Any, size_mb: float = 0):
        """
        Add model to cache

        Args:
            key: Cache key
            model: Model object
            size_mb: Model size in MB
        """
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
                if key in self.sizes:
                    del self.sizes[key]

            # Add to cache
            self.cache[key] = model
            self.sizes[key] = size_mb

            # Evict if necessary
            self._evict_if_needed()

    def _evict_if_needed(self):
        """Evict least recently used models if cache is full"""
        # Check size limit
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove oldest

        # Check memory limit
        total_memory = sum(self.sizes.values())
        while total_memory > self.max_memory_mb and self.cache:
            key, _ = self.cache.popitem(last=False)
            if key in self.sizes:
                total_memory -= self.sizes[key]
                del self.sizes[key]

    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.sizes.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_memory_mb': sum(self.sizes.values()),
            'max_memory_mb': self.max_memory_mb,
        }


# ============================================================================
# Feature Caching
# ============================================================================

class FeatureCache:
    """
    Cache for feature extraction results

    Features:
    - Hash-based caching
    - Disk persistence
    - Automatic invalidation
    - Memory-efficient storage
    """

    def __init__(self, cache_dir: Optional[Path] = None, max_memory_mb: float = 512):
        """
        Initialize feature cache

        Args:
            cache_dir: Directory for cache files
            max_memory_mb: Maximum memory for in-memory cache
        """
        self.cache_dir = cache_dir or Path.home() / '.midi_generator_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_mb = max_memory_mb
        self.memory_cache: OrderedDict = OrderedDict()
        self.memory_sizes: Dict[str, float] = {}
        self.lock = threading.Lock()

        self.hits = 0
        self.misses = 0

    def _compute_hash(self, data: Any) -> str:
        """Compute hash of input data"""
        # Convert to bytes
        if isinstance(data, (list, tuple)):
            data_bytes = json.dumps(data, sort_keys=True).encode()
        elif isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True).encode()
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode()

        return hashlib.sha256(data_bytes).hexdigest()

    def get(self, key: Any) -> Optional[np.ndarray]:
        """
        Get features from cache

        Args:
            key: Input data (will be hashed)

        Returns:
            Cached features or None
        """
        cache_key = self._compute_hash(key)

        # Check memory cache
        with self.lock:
            if cache_key in self.memory_cache:
                self.memory_cache.move_to_end(cache_key)
                self.hits += 1
                return self.memory_cache[cache_key]

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                features = np.load(cache_file)
                self._add_to_memory_cache(cache_key, features)
                self.hits += 1
                return features
            except Exception as e:
                warnings.warn(f"Failed to load cache file: {e}")

        self.misses += 1
        return None

    def put(self, key: Any, features: np.ndarray, save_to_disk: bool = True):
        """
        Store features in cache

        Args:
            key: Input data (will be hashed)
            features: Feature array
            save_to_disk: Whether to save to disk
        """
        cache_key = self._compute_hash(key)

        # Save to memory cache
        self._add_to_memory_cache(cache_key, features)

        # Save to disk cache
        if save_to_disk:
            cache_file = self.cache_dir / f"{cache_key}.npy"
            try:
                np.save(cache_file, features)
            except Exception as e:
                warnings.warn(f"Failed to save cache file: {e}")

    def _add_to_memory_cache(self, key: str, features: np.ndarray):
        """Add features to memory cache"""
        with self.lock:
            # Remove if exists
            if key in self.memory_cache:
                del self.memory_cache[key]
                if key in self.memory_sizes:
                    del self.memory_sizes[key]

            # Calculate size
            size_mb = features.nbytes / 1024 / 1024

            # Add to cache
            self.memory_cache[key] = features
            self.memory_sizes[key] = size_mb

            # Evict if needed
            self._evict_if_needed()

    def _evict_if_needed(self):
        """Evict oldest entries if memory limit exceeded"""
        total_memory = sum(self.memory_sizes.values())

        while total_memory > self.max_memory_mb and self.memory_cache:
            key, _ = self.memory_cache.popitem(last=False)
            if key in self.memory_sizes:
                total_memory -= self.memory_sizes[key]
                del self.memory_sizes[key]

    def clear_memory(self):
        """Clear memory cache"""
        with self.lock:
            self.memory_cache.clear()
            self.memory_sizes.clear()

    def clear_disk(self):
        """Clear disk cache"""
        for cache_file in self.cache_dir.glob("*.npy"):
            try:
                cache_file.unlink()
            except Exception as e:
                warnings.warn(f"Failed to delete cache file: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        disk_files = len(list(self.cache_dir.glob("*.npy")))

        return {
            'memory_size': len(self.memory_cache),
            'disk_size': disk_files,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'memory_mb': sum(self.memory_sizes.values()),
            'max_memory_mb': self.max_memory_mb,
        }


# ============================================================================
# Parallel Model Loading
# ============================================================================

class ParallelModelLoader:
    """
    Parallel model loading for fast startup

    Features:
    - Concurrent model loading
    - Priority-based loading
    - Lazy loading support
    - Error handling and retry
    """

    def __init__(self, max_workers: int = 8, cache: Optional[LRUModelCache] = None):
        """
        Initialize parallel loader

        Args:
            max_workers: Number of worker threads
            cache: Model cache (optional)
        """
        self.max_workers = max_workers
        self.cache = cache
        self.monitor = PerformanceMonitor()

    def load_models(self,
                   model_paths: Dict[str, Path],
                   priority: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Load multiple models in parallel

        Args:
            model_paths: Dictionary mapping parameter_name -> model_path
            priority: Optional priority for each parameter (higher = load first)

        Returns:
            Dictionary mapping parameter_name -> model
        """
        models = {}

        # Sort by priority if provided
        if priority:
            sorted_params = sorted(model_paths.keys(),
                                 key=lambda p: priority.get(p, 0),
                                 reverse=True)
        else:
            sorted_params = list(model_paths.keys())

        # Load in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._load_single_model, param, model_paths[param]): param
                for param in sorted_params
            }

            for future in as_completed(futures):
                param = futures[future]
                try:
                    model = future.result()
                    if model is not None:
                        models[param] = model
                except Exception as e:
                    warnings.warn(f"Failed to load model for {param}: {e}")

        return models

    def _load_single_model(self, param_name: str, model_path: Path) -> Optional[Any]:
        """Load a single model"""
        # Check cache first
        if self.cache:
            cached_model = self.cache.get(param_name)
            if cached_model is not None:
                return cached_model

        # Load from disk
        with self.monitor.profile(f'load_model_{param_name}'):
            try:
                if not model_path.exists():
                    warnings.warn(f"Model file not found: {model_path}")
                    return None

                if JOBLIB_AVAILABLE:
                    model = joblib.load(model_path)
                else:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)

                # Add to cache
                if self.cache:
                    # Estimate model size
                    size_mb = model_path.stat().st_size / 1024 / 1024
                    self.cache.put(param_name, model, size_mb)

                return model

            except Exception as e:
                warnings.warn(f"Error loading model from {model_path}: {e}")
                return None


# ============================================================================
# Batch Prediction Optimizer
# ============================================================================

class BatchPredictor:
    """
    Optimized batch prediction for multiple parameters

    Features:
    - Single feature extraction for all parameters
    - Parallel prediction across models
    - Result caching
    - Memory-efficient processing
    """

    def __init__(self,
                 models: Dict[str, Any],
                 feature_extractor: Optional[Callable] = None,
                 cache: Optional[FeatureCache] = None,
                 max_workers: int = 8):
        """
        Initialize batch predictor

        Args:
            models: Dictionary mapping parameter_name -> model
            feature_extractor: Function to extract features
            cache: Feature cache
            max_workers: Number of worker threads
        """
        self.models = models
        self.feature_extractor = feature_extractor
        self.cache = cache
        self.max_workers = max_workers
        self.monitor = PerformanceMonitor()

    def predict(self, input_data: Any,
                parameters: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Predict values for multiple parameters

        Args:
            input_data: Input data for feature extraction
            parameters: List of parameters to predict (None = all)

        Returns:
            Dictionary mapping parameter_name -> predicted_value
        """
        # Extract features once
        with self.monitor.profile('feature_extraction'):
            features = self._get_features(input_data)

        # Determine which parameters to predict
        if parameters is None:
            parameters = list(self.models.keys())
        else:
            # Filter to available models
            parameters = [p for p in parameters if p in self.models]

        # Predict in parallel
        predictions = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._predict_single, param, features): param
                for param in parameters
            }

            for future in as_completed(futures):
                param = futures[future]
                try:
                    value = future.result()
                    predictions[param] = value
                except Exception as e:
                    warnings.warn(f"Failed to predict {param}: {e}")
                    predictions[param] = None

        return predictions

    def predict_batch(self,
                     input_batch: List[Any],
                     parameters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Predict for a batch of inputs

        Args:
            input_batch: List of input data
            parameters: Parameters to predict

        Returns:
            List of prediction dictionaries
        """
        results = []

        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.predict, input_data, parameters)
                for input_data in input_batch
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    warnings.warn(f"Batch prediction failed: {e}")
                    results.append({})

        return results

    def _get_features(self, input_data: Any) -> np.ndarray:
        """Extract or retrieve cached features"""
        # Check cache
        if self.cache:
            cached_features = self.cache.get(input_data)
            if cached_features is not None:
                return cached_features

        # Extract features
        if self.feature_extractor is None:
            # Assume input_data is already features
            if isinstance(input_data, np.ndarray):
                features = input_data
            else:
                features = np.array(input_data)
        else:
            features = self.feature_extractor(input_data)

        # Cache features
        if self.cache and isinstance(features, np.ndarray):
            self.cache.put(input_data, features)

        return features

    def _predict_single(self, param_name: str, features: np.ndarray) -> Any:
        """Predict single parameter"""
        model = self.models.get(param_name)
        if model is None:
            return None

        try:
            # Reshape features if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)

            prediction = model.predict(features)

            # Return scalar if single prediction
            if len(prediction) == 1:
                return prediction[0]
            return prediction

        except Exception as e:
            warnings.warn(f"Prediction failed for {param_name}: {e}")
            return None


# ============================================================================
# Model Compression
# ============================================================================

class ModelCompressor:
    """
    Compress XGBoost models to reduce storage and memory

    Techniques:
    - Feature pruning (remove low-importance features)
    - Tree pruning (remove weak trees)
    - Quantization (reduce precision)
    - Sparse storage
    """

    def __init__(self):
        """Initialize model compressor"""
        self.monitor = PerformanceMonitor()

    def compress_model(self,
                      model: Any,
                      method: str = 'prune',
                      importance_threshold: float = 0.01,
                      tree_threshold: float = 0.1) -> Any:
        """
        Compress a single model

        Args:
            model: XGBoost model
            method: Compression method ('prune', 'quantize', or 'both')
            importance_threshold: Minimum feature importance to keep
            tree_threshold: Minimum tree weight to keep

        Returns:
            Compressed model
        """
        if not XGBOOST_AVAILABLE:
            warnings.warn("XGBoost not available, returning original model")
            return model

        with self.monitor.profile(f'compress_model_{method}'):
            if method in ('prune', 'both'):
                model = self._prune_model(model, importance_threshold, tree_threshold)

            if method in ('quantize', 'both'):
                model = self._quantize_model(model)

        return model

    def _prune_model(self,
                    model: Any,
                    importance_threshold: float,
                    tree_threshold: float) -> Any:
        """
        Prune low-importance features and weak trees

        Note: This is a simplified version. Full implementation would
        require deep integration with XGBoost internals.
        """
        # For now, just return the original model
        # Full implementation would:
        # 1. Get feature importances
        # 2. Retrain with only important features
        # 3. Remove weak trees based on weights

        return model

    def _quantize_model(self, model: Any) -> Any:
        """
        Quantize model weights to reduce precision

        Note: This is a placeholder. Full implementation would
        require conversion to lower precision formats.
        """
        # For now, just return the original model
        # Full implementation would convert to float16 or int8

        return model

    def compress_directory(self,
                          models_dir: Path,
                          output_dir: Optional[Path] = None,
                          **kwargs) -> Dict[str, float]:
        """
        Compress all models in a directory

        Args:
            models_dir: Directory containing model files
            output_dir: Output directory (default: overwrite originals)
            **kwargs: Arguments for compress_model

        Returns:
            Dictionary mapping filename -> compression_ratio
        """
        if output_dir is None:
            output_dir = models_dir
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        compression_ratios = {}

        for model_file in models_dir.glob('*.pkl'):
            try:
                # Load model
                if JOBLIB_AVAILABLE:
                    model = joblib.load(model_file)
                else:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)

                # Get original size
                original_size = model_file.stat().st_size

                # Compress
                compressed_model = self.compress_model(model, **kwargs)

                # Save compressed model
                output_file = output_dir / model_file.name
                if JOBLIB_AVAILABLE:
                    joblib.dump(compressed_model, output_file, compress=9)
                else:
                    with open(output_file, 'wb') as f:
                        pickle.dump(compressed_model, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Calculate compression ratio
                compressed_size = output_file.stat().st_size
                ratio = original_size / compressed_size if compressed_size > 0 else 1.0
                compression_ratios[model_file.name] = ratio

            except Exception as e:
                warnings.warn(f"Failed to compress {model_file}: {e}")

        return compression_ratios


# ============================================================================
# Memory-Mapped Model Loading
# ============================================================================

class MemoryMappedModelManager:
    """
    Memory-mapped model loading for large-scale deployment

    Features:
    - Lazy loading (models loaded on first access)
    - Shared memory across processes
    - Minimal memory footprint
    - Fast startup time
    """

    def __init__(self, models_dir: Path):
        """
        Initialize memory-mapped manager

        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.model_paths: Dict[str, Path] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.lock = threading.Lock()

        # Index available models
        self._index_models()

    def _index_models(self):
        """Index all available model files"""
        for model_file in self.models_dir.glob('*.pkl'):
            param_name = model_file.stem
            self.model_paths[param_name] = model_file

    def get_model(self, param_name: str) -> Optional[Any]:
        """
        Get model (load on first access)

        Args:
            param_name: Parameter name

        Returns:
            Model object or None
        """
        # Check if already loaded
        with self.lock:
            if param_name in self.loaded_models:
                return self.loaded_models[param_name]

        # Load model
        model = self._load_model(param_name)

        # Cache for future access
        if model is not None:
            with self.lock:
                self.loaded_models[param_name] = model

        return model

    def _load_model(self, param_name: str) -> Optional[Any]:
        """Load a single model from disk"""
        if param_name not in self.model_paths:
            return None

        model_path = self.model_paths[param_name]

        try:
            if JOBLIB_AVAILABLE:
                # Use memory mapping if available
                model = joblib.load(model_path, mmap_mode='r')
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

            return model

        except Exception as e:
            warnings.warn(f"Failed to load model {param_name}: {e}")
            return None

    def preload_models(self, param_names: List[str], max_workers: int = 8):
        """
        Preload specific models in parallel

        Args:
            param_names: List of parameters to preload
            max_workers: Number of worker threads
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.get_model, param)
                for param in param_names
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    warnings.warn(f"Failed to preload model: {e}")

    def get_available_parameters(self) -> List[str]:
        """Get list of all available parameters"""
        return list(self.model_paths.keys())

    def clear_cache(self):
        """Clear loaded models from memory"""
        with self.lock:
            self.loaded_models.clear()


# ============================================================================
# Main Performance Optimizer
# ============================================================================

class PerformanceOptimizer:
    """
    Main performance optimization orchestrator

    Combines all optimization techniques:
    - Model caching
    - Feature caching
    - Parallel loading
    - Batch prediction
    - Model compression
    - Performance monitoring
    """

    def __init__(self,
                 models_dir: Optional[Path] = None,
                 cache_dir: Optional[Path] = None,
                 max_model_cache_size: int = 100,
                 max_model_memory_mb: float = 1024,
                 max_feature_memory_mb: float = 512,
                 max_workers: int = 8,
                 enable_monitoring: bool = True):
        """
        Initialize performance optimizer

        Args:
            models_dir: Directory containing model files
            cache_dir: Directory for caches
            max_model_cache_size: Maximum number of cached models
            max_model_memory_mb: Maximum memory for model cache
            max_feature_memory_mb: Maximum memory for feature cache
            max_workers: Number of worker threads
            enable_monitoring: Enable performance monitoring
        """
        self.models_dir = Path(models_dir) if models_dir else None
        self.max_workers = max_workers

        # Initialize components
        self.model_cache = LRUModelCache(
            max_size=max_model_cache_size,
            max_memory_mb=max_model_memory_mb
        )

        self.feature_cache = FeatureCache(
            cache_dir=cache_dir,
            max_memory_mb=max_feature_memory_mb
        )

        self.model_loader = ParallelModelLoader(
            max_workers=max_workers,
            cache=self.model_cache
        )

        self.compressor = ModelCompressor()

        self.monitor = PerformanceMonitor(enabled=enable_monitoring)

        self.mmap_manager = None
        if self.models_dir:
            self.mmap_manager = MemoryMappedModelManager(self.models_dir)

        # Loaded models
        self.models: Dict[str, Any] = {}
        self.batch_predictor: Optional[BatchPredictor] = None

    def load_models(self,
                   parameter_names: Optional[List[str]] = None,
                   use_mmap: bool = False,
                   priority: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Load models for specified parameters

        Args:
            parameter_names: List of parameters (None = all available)
            use_mmap: Use memory-mapped loading
            priority: Loading priority for each parameter

        Returns:
            Dictionary of loaded models
        """
        with self.monitor.profile('load_models_total'):
            if use_mmap and self.mmap_manager:
                # Memory-mapped loading
                if parameter_names is None:
                    parameter_names = self.mmap_manager.get_available_parameters()

                self.mmap_manager.preload_models(parameter_names, self.max_workers)
                self.models = {
                    param: self.mmap_manager.get_model(param)
                    for param in parameter_names
                }
            else:
                # Standard parallel loading
                if self.models_dir is None:
                    raise ValueError("models_dir not specified")

                if parameter_names is None:
                    # Load all available models
                    model_paths = {
                        p.stem: p for p in self.models_dir.glob('*.pkl')
                    }
                else:
                    model_paths = {
                        param: self.models_dir / f"{param}.pkl"
                        for param in parameter_names
                    }

                self.models = self.model_loader.load_models(model_paths, priority)

        # Initialize batch predictor
        self.batch_predictor = BatchPredictor(
            models=self.models,
            cache=self.feature_cache,
            max_workers=self.max_workers
        )

        return self.models

    def predict(self,
               input_data: Any,
               feature_extractor: Optional[Callable] = None,
               parameters: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Predict parameters from input data

        Args:
            input_data: Input for prediction
            feature_extractor: Function to extract features
            parameters: Parameters to predict (None = all)

        Returns:
            Dictionary mapping parameter_name -> predicted_value
        """
        if self.batch_predictor is None:
            raise RuntimeError("No models loaded. Call load_models() first.")

        with self.monitor.profile('prediction_total'):
            # Set feature extractor if provided
            if feature_extractor:
                self.batch_predictor.feature_extractor = feature_extractor

            # Make predictions
            predictions = self.batch_predictor.predict(input_data, parameters)

        return predictions

    def predict_batch(self,
                     input_batch: List[Any],
                     feature_extractor: Optional[Callable] = None,
                     parameters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Predict for multiple inputs

        Args:
            input_batch: List of inputs
            feature_extractor: Function to extract features
            parameters: Parameters to predict

        Returns:
            List of prediction dictionaries
        """
        if self.batch_predictor is None:
            raise RuntimeError("No models loaded. Call load_models() first.")

        with self.monitor.profile('batch_prediction_total'):
            # Set feature extractor if provided
            if feature_extractor:
                self.batch_predictor.feature_extractor = feature_extractor

            # Make predictions
            results = self.batch_predictor.predict_batch(input_batch, parameters)

        return results

    def compress_models(self,
                       input_dir: Optional[Path] = None,
                       output_dir: Optional[Path] = None,
                       **kwargs) -> Dict[str, float]:
        """
        Compress models to reduce storage

        Args:
            input_dir: Input directory (default: self.models_dir)
            output_dir: Output directory (default: overwrite)
            **kwargs: Compression arguments

        Returns:
            Compression ratios
        """
        if input_dir is None:
            input_dir = self.models_dir

        if input_dir is None:
            raise ValueError("No models directory specified")

        with self.monitor.profile('compress_models_total'):
            ratios = self.compressor.compress_directory(
                input_dir, output_dir, **kwargs
            )

        return ratios

    def benchmark(self,
                 num_predictions: int = 100,
                 num_parameters: Optional[int] = None) -> Dict[str, Any]:
        """
        Run performance benchmark

        Args:
            num_predictions: Number of predictions to make
            num_parameters: Number of parameters (uses all if None)

        Returns:
            Benchmark results
        """
        if not self.models:
            raise RuntimeError("No models loaded")

        # Select parameters
        if num_parameters:
            parameters = list(self.models.keys())[:num_parameters]
        else:
            parameters = list(self.models.keys())

        # Generate dummy input
        if NUMPY_AVAILABLE:
            dummy_input = np.random.randn(100)  # 100 features
        else:
            dummy_input = [0.0] * 100

        # Run benchmark
        self.monitor.reset()

        start_time = time.time()
        for _ in range(num_predictions):
            self.predict(dummy_input, parameters=parameters)
        total_time = time.time() - start_time

        # Calculate metrics
        avg_time = total_time / num_predictions
        predictions_per_second = num_predictions / total_time

        return {
            'num_predictions': num_predictions,
            'num_parameters': len(parameters),
            'total_time': total_time,
            'avg_time_per_prediction': avg_time,
            'predictions_per_second': predictions_per_second,
            'model_cache_stats': self.model_cache.get_stats(),
            'feature_cache_stats': self.feature_cache.get_stats(),
            'performance_summary': self.monitor.get_all_summaries(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'num_loaded_models': len(self.models),
            'model_cache': self.model_cache.get_stats(),
            'feature_cache': self.feature_cache.get_stats(),
            'performance': self.monitor.get_all_summaries(),
        }

    def print_report(self):
        """Print comprehensive performance report"""
        stats = self.get_stats()

        print("\n" + "="*80)
        print("PERFORMANCE OPTIMIZER REPORT")
        print("="*80)

        print(f"\nModels Loaded: {stats['num_loaded_models']}")

        print("\nModel Cache:")
        for key, value in stats['model_cache'].items():
            print(f"  {key}: {value}")

        print("\nFeature Cache:")
        for key, value in stats['feature_cache'].items():
            print(f"  {key}: {value}")

        print("\nPerformance Metrics:")
        self.monitor.print_report()

    def clear_caches(self):
        """Clear all caches"""
        self.model_cache.clear()
        self.feature_cache.clear_memory()
        if self.mmap_manager:
            self.mmap_manager.clear_cache()


# ============================================================================
# Training Optimization
# ============================================================================

class ParallelTrainer:
    """
    Parallel training for multiple parameter models

    Features:
    - Concurrent model training
    - Resource management
    - Progress tracking
    - Error handling
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel trainer

        Args:
            max_workers: Number of worker processes (default: CPU count)
        """
        if max_workers is None:
            max_workers = mp.cpu_count()
        self.max_workers = max_workers
        self.monitor = PerformanceMonitor()

    def train_models(self,
                    training_jobs: List[Dict[str, Any]],
                    output_dir: Path,
                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Train multiple models in parallel

        Args:
            training_jobs: List of training job specifications
                Each job: {
                    'param_name': str,
                    'X_train': array,
                    'y_train': array,
                    'params': dict  # XGBoost params
                }
            output_dir: Directory to save trained models
            progress_callback: Optional callback for progress updates

        Returns:
            Training results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        total_jobs = len(training_jobs)

        with self.monitor.profile('parallel_training'):
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                futures = {
                    executor.submit(
                        self._train_single_model,
                        job,
                        output_dir
                    ): job['param_name']
                    for job in training_jobs
                }

                # Process results as they complete
                completed = 0
                for future in as_completed(futures):
                    param_name = futures[future]
                    completed += 1

                    try:
                        result = future.result()
                        results[param_name] = result

                        if progress_callback:
                            progress_callback(completed, total_jobs, param_name)

                    except Exception as e:
                        warnings.warn(f"Training failed for {param_name}: {e}")
                        results[param_name] = {'error': str(e)}

        return results

    @staticmethod
    def _train_single_model(job: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Train a single model (static method for multiprocessing)"""
        if not XGBOOST_AVAILABLE:
            return {'error': 'XGBoost not available'}

        param_name = job['param_name']
        X_train = job['X_train']
        y_train = job['y_train']
        params = job.get('params', {})

        # Default XGBoost parameters
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
        }
        default_params.update(params)

        start_time = time.time()

        try:
            # Train model
            model = xgb.XGBRegressor(**default_params)
            model.fit(X_train, y_train)

            # Save model
            model_path = output_dir / f"{param_name}.pkl"
            if JOBLIB_AVAILABLE:
                joblib.dump(model, model_path, compress=3)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

            training_time = time.time() - start_time

            return {
                'success': True,
                'training_time': training_time,
                'model_path': str(model_path),
                'num_samples': len(X_train),
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
            }


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example usage of PerformanceOptimizer"""

    # Initialize optimizer
    optimizer = PerformanceOptimizer(
        models_dir=Path('models'),
        cache_dir=Path('cache'),
        max_model_cache_size=100,
        max_model_memory_mb=1024,
        max_feature_memory_mb=512,
        max_workers=8,
        enable_monitoring=True
    )

    # Load models
    print("Loading models...")
    models = optimizer.load_models()
    print(f"Loaded {len(models)} models")

    # Make predictions
    print("\nMaking predictions...")
    dummy_input = np.random.randn(100)
    predictions = optimizer.predict(dummy_input)
    print(f"Predicted {len(predictions)} parameters")

    # Run benchmark
    print("\nRunning benchmark...")
    benchmark_results = optimizer.benchmark(num_predictions=100)
    print(f"Average prediction time: {benchmark_results['avg_time_per_prediction']:.4f}s")
    print(f"Predictions per second: {benchmark_results['predictions_per_second']:.2f}")

    # Print report
    optimizer.print_report()

    # Compress models
    print("\nCompressing models...")
    compression_ratios = optimizer.compress_models()
    avg_ratio = np.mean(list(compression_ratios.values()))
    print(f"Average compression ratio: {avg_ratio:.2f}x")


# ============================================================================
# Advanced Profiling and Diagnostics
# ============================================================================

class ProfilerDecorator:
    """
    Decorator for automatic function profiling

    Usage:
        profiler = ProfilerDecorator(monitor)

        @profiler.profile
        def my_function():
            ...
    """

    def __init__(self, monitor: PerformanceMonitor):
        """
        Initialize profiler decorator

        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor

    def profile(self, func: Callable) -> Callable:
        """Decorator to profile a function"""
        def wrapper(*args, **kwargs):
            with self.monitor.profile(func.__name__):
                return func(*args, **kwargs)
        return wrapper


class ResourceManager:
    """
    Manage system resources for optimal performance

    Features:
    - CPU affinity management
    - Memory limit enforcement
    - Process priority adjustment
    - Resource monitoring
    """

    def __init__(self):
        """Initialize resource manager"""
        self.original_priority = None
        self.original_affinity = None

    def set_high_priority(self):
        """Set process to high priority"""
        if not PSUTIL_AVAILABLE:
            warnings.warn("psutil not available, cannot set priority")
            return

        try:
            process = psutil.Process()
            self.original_priority = process.nice()

            # Set high priority (lower nice value)
            if sys.platform == 'linux':
                process.nice(-5)  # Higher priority
            else:
                process.nice(psutil.HIGH_PRIORITY_CLASS)

        except Exception as e:
            warnings.warn(f"Failed to set high priority: {e}")

    def restore_priority(self):
        """Restore original process priority"""
        if not PSUTIL_AVAILABLE or self.original_priority is None:
            return

        try:
            process = psutil.Process()
            process.nice(self.original_priority)
        except Exception as e:
            warnings.warn(f"Failed to restore priority: {e}")

    def set_cpu_affinity(self, cpu_ids: List[int]):
        """
        Set CPU affinity

        Args:
            cpu_ids: List of CPU IDs to use
        """
        if not PSUTIL_AVAILABLE:
            warnings.warn("psutil not available, cannot set CPU affinity")
            return

        try:
            process = psutil.Process()
            self.original_affinity = process.cpu_affinity()
            process.cpu_affinity(cpu_ids)
        except Exception as e:
            warnings.warn(f"Failed to set CPU affinity: {e}")

    def restore_affinity(self):
        """Restore original CPU affinity"""
        if not PSUTIL_AVAILABLE or self.original_affinity is None:
            return

        try:
            process = psutil.Process()
            process.cpu_affinity(self.original_affinity)
        except Exception as e:
            warnings.warn(f"Failed to restore affinity: {e}")

    def get_available_memory(self) -> float:
        """Get available system memory in MB"""
        if not PSUTIL_AVAILABLE:
            return 0.0

        return psutil.virtual_memory().available / 1024 / 1024

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        if not PSUTIL_AVAILABLE:
            return 0.0

        return psutil.cpu_percent(interval=0.1)

    @contextmanager
    def optimized_resources(self, cpu_ids: Optional[List[int]] = None):
        """
        Context manager for optimized resource usage

        Usage:
            with resource_manager.optimized_resources([0, 1, 2, 3]):
                # Code runs with optimized resources
                ...
        """
        try:
            self.set_high_priority()
            if cpu_ids:
                self.set_cpu_affinity(cpu_ids)
            yield
        finally:
            self.restore_priority()
            self.restore_affinity()


class AutoTuner:
    """
    Automatically tune optimizer parameters based on workload

    Features:
    - Adaptive cache sizing
    - Worker count optimization
    - Memory management
    - Performance-based tuning
    """

    def __init__(self, optimizer: 'PerformanceOptimizer'):
        """
        Initialize auto-tuner

        Args:
            optimizer: Performance optimizer instance
        """
        self.optimizer = optimizer
        self.tuning_history: List[Dict] = []

    def tune_cache_size(self, target_hit_rate: float = 0.9):
        """
        Tune model cache size to achieve target hit rate

        Args:
            target_hit_rate: Target cache hit rate (0-1)
        """
        current_stats = self.optimizer.model_cache.get_stats()
        current_hit_rate = current_stats['hit_rate']
        current_size = current_stats['size']

        if current_hit_rate < target_hit_rate:
            # Increase cache size
            new_size = int(current_size * (target_hit_rate / max(current_hit_rate, 0.01)))
            new_size = min(new_size, 500)  # Cap at 500

            self.optimizer.model_cache.max_size = new_size

            self.tuning_history.append({
                'timestamp': time.time(),
                'action': 'increase_cache',
                'old_size': current_size,
                'new_size': new_size,
                'hit_rate': current_hit_rate,
            })

    def tune_worker_count(self, target_throughput: float):
        """
        Tune number of workers to achieve target throughput

        Args:
            target_throughput: Target predictions per second
        """
        # Run benchmark with current workers
        results = self.optimizer.benchmark(num_predictions=20)
        current_throughput = results['predictions_per_second']
        current_workers = self.optimizer.max_workers

        if current_throughput < target_throughput:
            # Increase workers
            new_workers = min(current_workers * 2, 32)
            self.optimizer.max_workers = new_workers

            self.tuning_history.append({
                'timestamp': time.time(),
                'action': 'increase_workers',
                'old_workers': current_workers,
                'new_workers': new_workers,
                'throughput': current_throughput,
            })

    def auto_tune(self, num_iterations: int = 3):
        """
        Run automatic tuning process

        Args:
            num_iterations: Number of tuning iterations
        """
        for i in range(num_iterations):
            print(f"Auto-tuning iteration {i+1}/{num_iterations}")

            # Tune cache
            self.tune_cache_size(target_hit_rate=0.9)

            # Wait for cache to warm up
            time.sleep(1)

            # Tune workers
            self.tune_worker_count(target_throughput=50)

        return self.tuning_history


class PerformanceAnalyzer:
    """
    Analyze performance metrics and identify bottlenecks

    Features:
    - Bottleneck detection
    - Performance regression detection
    - Optimization recommendations
    - Trend analysis
    """

    def __init__(self, monitor: PerformanceMonitor):
        """
        Initialize analyzer

        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor

    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks

        Returns:
            List of bottleneck descriptions
        """
        bottlenecks = []

        # Get all operation summaries
        summaries = self.monitor.get_all_summaries()

        for operation, stats in summaries.items():
            # Check if operation is slow
            if stats['avg_duration'] > 1.0:  # >1 second
                bottlenecks.append({
                    'operation': operation,
                    'type': 'slow_operation',
                    'avg_duration': stats['avg_duration'],
                    'recommendation': f"Optimize {operation} - taking {stats['avg_duration']:.2f}s on average"
                })

            # Check if operation uses too much memory
            if stats.get('avg_memory_delta_mb', 0) > 500:  # >500MB
                bottlenecks.append({
                    'operation': operation,
                    'type': 'high_memory',
                    'avg_memory_mb': stats['avg_memory_delta_mb'],
                    'recommendation': f"Reduce memory usage in {operation}"
                })

        return bottlenecks

    def detect_regression(self, baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect performance regressions compared to baseline

        Args:
            baseline: Baseline performance metrics

        Returns:
            List of regressions
        """
        regressions = []

        current = self.monitor.get_all_summaries()

        for operation, current_stats in current.items():
            if operation not in baseline:
                continue

            baseline_stats = baseline[operation]

            # Check duration regression
            duration_ratio = current_stats['avg_duration'] / baseline_stats['avg_duration']
            if duration_ratio > 1.2:  # 20% slower
                regressions.append({
                    'operation': operation,
                    'type': 'duration_regression',
                    'baseline': baseline_stats['avg_duration'],
                    'current': current_stats['avg_duration'],
                    'ratio': duration_ratio,
                })

        return regressions

    def generate_recommendations(self) -> List[str]:
        """
        Generate optimization recommendations

        Returns:
            List of recommendations
        """
        recommendations = []

        # Analyze bottlenecks
        bottlenecks = self.identify_bottlenecks()
        for bottleneck in bottlenecks:
            recommendations.append(bottleneck['recommendation'])

        # Analyze cache performance
        # (Would need access to optimizer's cache stats)

        return recommendations

    def print_analysis(self):
        """Print performance analysis"""
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS")
        print("="*80)

        # Bottlenecks
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            print("\nBottlenecks Detected:")
            for b in bottlenecks:
                print(f"  ⚠ {b['operation']}: {b['recommendation']}")
        else:
            print("\n✓ No bottlenecks detected")

        # Recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            print("\nRecommendations:")
            for r in recommendations:
                print(f"  • {r}")

        print("\n" + "="*80)


class DiagnosticTools:
    """
    Diagnostic tools for troubleshooting performance issues

    Features:
    - Model validation
    - Cache diagnostics
    - Memory leak detection
    - Performance profiling
    """

    def __init__(self, optimizer: 'PerformanceOptimizer'):
        """
        Initialize diagnostic tools

        Args:
            optimizer: Performance optimizer instance
        """
        self.optimizer = optimizer

    def validate_models(self) -> Dict[str, Any]:
        """
        Validate all loaded models

        Returns:
            Validation results
        """
        results = {
            'total_models': len(self.optimizer.models),
            'valid_models': 0,
            'invalid_models': [],
        }

        for param_name, model in self.optimizer.models.items():
            try:
                # Try to make a prediction
                if NUMPY_AVAILABLE:
                    test_input = np.random.randn(1, 100)
                    prediction = model.predict(test_input)
                    results['valid_models'] += 1
            except Exception as e:
                results['invalid_models'].append({
                    'param_name': param_name,
                    'error': str(e)
                })

        return results

    def diagnose_cache(self) -> Dict[str, Any]:
        """
        Diagnose cache performance

        Returns:
            Cache diagnostics
        """
        model_cache_stats = self.optimizer.model_cache.get_stats()
        feature_cache_stats = self.optimizer.feature_cache.get_stats()

        diagnostics = {
            'model_cache': {
                'status': 'healthy' if model_cache_stats['hit_rate'] > 0.7 else 'suboptimal',
                'hit_rate': model_cache_stats['hit_rate'],
                'size': model_cache_stats['size'],
                'utilization': model_cache_stats['size'] / model_cache_stats['max_size'],
            },
            'feature_cache': {
                'status': 'healthy' if feature_cache_stats['hit_rate'] > 0.7 else 'suboptimal',
                'hit_rate': feature_cache_stats['hit_rate'],
                'memory_utilization': feature_cache_stats['memory_mb'] / feature_cache_stats['max_memory_mb'],
            }
        }

        return diagnostics

    def detect_memory_leaks(self, num_iterations: int = 100) -> Dict[str, Any]:
        """
        Detect potential memory leaks

        Args:
            num_iterations: Number of iterations to test

        Returns:
            Memory leak analysis
        """
        if not PSUTIL_AVAILABLE:
            return {'error': 'psutil not available'}

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        memory_samples = [initial_memory]

        # Run multiple predictions
        if NUMPY_AVAILABLE:
            test_input = np.random.randn(100)

            for _ in range(num_iterations):
                self.optimizer.predict(test_input)

                # Sample memory
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)

        final_memory = memory_samples[-1]
        memory_increase = final_memory - initial_memory

        # Calculate growth rate
        if NUMPY_AVAILABLE:
            growth_rate = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
        else:
            growth_rate = memory_increase / num_iterations

        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': memory_increase,
            'growth_rate_mb_per_iter': growth_rate,
            'leak_detected': growth_rate > 0.1,  # >0.1 MB per iteration
            'samples': memory_samples,
        }

    def profile_prediction(self, num_predictions: int = 100) -> Dict[str, Any]:
        """
        Profile prediction performance

        Args:
            num_predictions: Number of predictions to profile

        Returns:
            Profiling results
        """
        if NUMPY_AVAILABLE:
            test_input = np.random.randn(100)
        else:
            test_input = [0.0] * 100

        timings = []

        for _ in range(num_predictions):
            start = time.time()
            self.optimizer.predict(test_input)
            duration = time.time() - start
            timings.append(duration)

        if NUMPY_AVAILABLE:
            results = {
                'num_predictions': num_predictions,
                'mean': np.mean(timings),
                'std': np.std(timings),
                'min': np.min(timings),
                'max': np.max(timings),
                'median': np.median(timings),
                'p95': np.percentile(timings, 95),
                'p99': np.percentile(timings, 99),
            }
        else:
            results = {
                'num_predictions': num_predictions,
                'mean': sum(timings) / len(timings),
                'min': min(timings),
                'max': max(timings),
            }

        return results

    def run_full_diagnostics(self) -> Dict[str, Any]:
        """
        Run complete diagnostic suite

        Returns:
            Comprehensive diagnostics
        """
        print("\nRunning full diagnostics...")

        diagnostics = {
            'timestamp': time.time(),
            'models': self.validate_models(),
            'cache': self.diagnose_cache(),
            'memory_leak': self.detect_memory_leaks(num_iterations=50),
            'profiling': self.profile_prediction(num_predictions=50),
        }

        return diagnostics

    def print_diagnostics_report(self, diagnostics: Optional[Dict] = None):
        """Print diagnostics report"""
        if diagnostics is None:
            diagnostics = self.run_full_diagnostics()

        print("\n" + "="*80)
        print("DIAGNOSTIC REPORT")
        print("="*80)

        # Model validation
        print("\n1. Model Validation:")
        model_results = diagnostics['models']
        print(f"   Total models: {model_results['total_models']}")
        print(f"   Valid models: {model_results['valid_models']}")
        if model_results['invalid_models']:
            print(f"   Invalid models: {len(model_results['invalid_models'])}")
            for inv in model_results['invalid_models'][:5]:
                print(f"     - {inv['param_name']}: {inv['error']}")

        # Cache diagnostics
        print("\n2. Cache Performance:")
        cache_diag = diagnostics['cache']
        print(f"   Model cache: {cache_diag['model_cache']['status']}")
        print(f"     Hit rate: {cache_diag['model_cache']['hit_rate']*100:.1f}%")
        print(f"   Feature cache: {cache_diag['feature_cache']['status']}")
        print(f"     Hit rate: {cache_diag['feature_cache']['hit_rate']*100:.1f}%")

        # Memory leak detection
        print("\n3. Memory Leak Detection:")
        mem_leak = diagnostics['memory_leak']
        if 'error' in mem_leak:
            print(f"   {mem_leak['error']}")
        else:
            print(f"   Memory increase: {mem_leak['memory_increase_mb']:.2f} MB")
            print(f"   Growth rate: {mem_leak['growth_rate_mb_per_iter']:.4f} MB/iter")
            print(f"   Leak detected: {'YES ⚠' if mem_leak['leak_detected'] else 'NO ✓'}")

        # Profiling
        print("\n4. Prediction Profiling:")
        prof = diagnostics['profiling']
        print(f"   Mean: {prof['mean']*1000:.2f}ms")
        if 'std' in prof:
            print(f"   Std: {prof['std']*1000:.2f}ms")
            print(f"   P95: {prof['p95']*1000:.2f}ms")
            print(f"   P99: {prof['p99']*1000:.2f}ms")

        print("\n" + "="*80)


# ============================================================================
# Utilities
# ============================================================================

def estimate_model_size(model: Any) -> float:
    """
    Estimate model size in MB

    Args:
        model: Model object

    Returns:
        Estimated size in MB
    """
    try:
        import pickle
        serialized = pickle.dumps(model)
        return len(serialized) / 1024 / 1024
    except Exception:
        return 0.0


def create_dummy_models(output_dir: Path, num_models: int = 10, num_samples: int = 100):
    """
    Create dummy models for testing

    Args:
        output_dir: Output directory
        num_models: Number of models to create
        num_samples: Training samples per model
    """
    if not (NUMPY_AVAILABLE and XGBOOST_AVAILABLE and JOBLIB_AVAILABLE):
        raise RuntimeError("Required packages not available")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {num_models} dummy models...")

    for i in range(num_models):
        # Generate training data
        X = np.random.randn(num_samples, 50)
        y = np.random.randn(num_samples)

        # Train model
        model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
        model.fit(X, y)

        # Save model
        model_path = output_dir / f"param_{i:04d}.pkl"
        joblib.dump(model, model_path)

        if (i + 1) % 10 == 0:
            print(f"  Created {i+1}/{num_models} models")

    print(f"✓ Created {num_models} models in {output_dir}")


def benchmark_single_component(component_name: str, **kwargs):
    """
    Benchmark a single component

    Args:
        component_name: Name of component to benchmark
        **kwargs: Component-specific arguments
    """
    print(f"\nBenchmarking {component_name}...")

    if component_name == 'model_cache':
        cache = LRUModelCache(**kwargs)

        # Benchmark cache operations
        start = time.time()
        for i in range(1000):
            cache.put(f"model_{i}", {'data': i})
        put_time = time.time() - start

        start = time.time()
        for i in range(1000):
            cache.get(f"model_{i}")
        get_time = time.time() - start

        print(f"  Put operations: {put_time:.3f}s (1000 ops)")
        print(f"  Get operations: {get_time:.3f}s (1000 ops)")
        print(f"  Cache stats: {cache.get_stats()}")

    elif component_name == 'feature_cache':
        if not NUMPY_AVAILABLE:
            print("  NumPy not available")
            return

        cache = FeatureCache(**kwargs)

        # Benchmark cache operations
        start = time.time()
        for i in range(100):
            features = np.random.randn(100)
            cache.put(f"input_{i}", features)
        put_time = time.time() - start

        start = time.time()
        for i in range(100):
            cache.get(f"input_{i}")
        get_time = time.time() - start

        print(f"  Put operations: {put_time:.3f}s (100 ops)")
        print(f"  Get operations: {get_time:.3f}s (100 ops)")
        print(f"  Cache stats: {cache.get_stats()}")

    else:
        print(f"  Unknown component: {component_name}")


if __name__ == '__main__':
    example_usage()
