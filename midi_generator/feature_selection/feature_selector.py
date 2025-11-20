"""
Feature Selection Pipeline - Agent 04
======================================

Comprehensive feature selection system to reduce from 1000+ features to 200 optimal features
for efficient multi-task learning.

Implements 7 feature selection methods:
1. Filter-based (Correlation)
2. Univariate Statistical Tests (F-test, Chi-squared, Mutual Information)
3. Tree-based Feature Importance (XGBoost, Random Forest)
4. L1 Regularization (Lasso)
5. Recursive Feature Elimination (RFE)
6. Principal Component Analysis (PCA)
7. Domain Knowledge Curation

Plus ensemble feature selection combining multiple methods.

Author: Agent 04 - Feature Selection Optimizer
License: MIT
"""

import json
import pickle
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    from sklearn.feature_selection import (
        SelectKBest,
        f_regression,
        f_classif,
        chi2,
        mutual_info_regression,
        mutual_info_classif,
        RFE,
    )
    from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not installed. Install with: pip install scikit-learn")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not installed. Install with: pip install xgboost")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class FeatureSelectionResult:
    """Result from a feature selection method"""
    method_name: str
    selected_features: List[str]
    feature_scores: Dict[str, float]
    n_features_selected: int
    selection_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleSelectionResult:
    """Result from ensemble feature selection"""
    selected_features: List[str]
    feature_vote_counts: Dict[str, int]
    feature_avg_scores: Dict[str, float]
    methods_used: List[str]
    n_features_selected: int
    selection_time: float


@dataclass
class FeatureCategoryInfo:
    """Information about a feature category"""
    category_name: str
    feature_indices: List[int]
    feature_names: List[str]
    n_features: int
    target_n_features: int  # How many to select from this category


# ============================================================================
# Main Feature Selector
# ============================================================================

class ComprehensiveFeatureSelector:
    """
    Comprehensive feature selection system with 7+ methods.

    This class implements all feature selection methods required by Agent 04:
    1. Filter-based (correlation)
    2. Univariate statistical tests
    3. Tree-based importance
    4. L1 regularization (Lasso)
    5. Recursive Feature Elimination
    6. PCA
    7. Domain knowledge curation
    8. Ensemble selection (combines multiple methods)

    Usage:
        selector = ComprehensiveFeatureSelector(
            feature_matrix=X,
            feature_names=names,
            target_n_features=200
        )

        # Run all methods
        results = selector.run_all_methods(target_values=y, parameter_name='tempo')

        # Get ensemble selection
        ensemble_result = selector.ensemble_selection(results, min_votes=3)

        # Save selected features
        selector.save_selected_features(ensemble_result, 'selected_features_200.json')
    """

    def __init__(
        self,
        feature_matrix: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        target_n_features: int = 200,
        cache_dir: Path = Path('midi_generator/feature_selection/cache'),
        random_state: int = 42
    ):
        """
        Initialize Feature Selector.

        Args:
            feature_matrix: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            target_n_features: Target number of features to select
            cache_dir: Directory for caching results
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn")

        self.feature_matrix = feature_matrix
        self.feature_names = feature_names or []
        self.target_n_features = target_n_features
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Internal state
        self.n_features = len(feature_names) if feature_names else 0
        self.n_samples = feature_matrix.shape[0] if feature_matrix is not None else 0

        # Results storage
        self.selection_results: Dict[str, FeatureSelectionResult] = {}
        self.ensemble_result: Optional[EnsembleSelectionResult] = None

        # Feature categories (will be populated)
        self.feature_categories: Dict[str, FeatureCategoryInfo] = {}

    def set_data(
        self,
        feature_matrix: np.ndarray,
        feature_names: List[str]
    ):
        """Set or update feature data"""
        self.feature_matrix = feature_matrix
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.n_samples = feature_matrix.shape[0]

    # ========================================================================
    # Method 1: Filter-based Selection (Correlation)
    # ========================================================================

    def filter_based_selection(
        self,
        target_values: np.ndarray,
        parameter_name: str,
        n_features: Optional[int] = None
    ) -> FeatureSelectionResult:
        """
        Select features based on correlation with target parameter.

        Fast method, good baseline. Uses Pearson correlation for continuous
        targets and point-biserial correlation for binary targets.

        Args:
            target_values: Target parameter values
            parameter_name: Name of parameter
            n_features: Number of features to select (default: self.target_n_features)

        Returns:
            FeatureSelectionResult
        """
        import time
        start_time = time.time()

        n_features = n_features or self.target_n_features

        print(f"\n[Method 1] Filter-based selection for {parameter_name}")
        print(f"Computing correlations with target...")

        # Compute correlations
        correlations = {}
        for i, fname in enumerate(self.feature_names):
            try:
                corr = np.corrcoef(self.feature_matrix[:, i], target_values)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                correlations[fname] = abs(corr)
            except:
                correlations[fname] = 0.0

        # Select top N by correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:n_features]]

        result = FeatureSelectionResult(
            method_name='filter_correlation',
            selected_features=selected_features,
            feature_scores=correlations,
            n_features_selected=len(selected_features),
            selection_time=time.time() - start_time
        )

        self.selection_results['filter_correlation'] = result
        print(f"✅ Selected {len(selected_features)} features in {result.selection_time:.2f}s")

        return result

    # ========================================================================
    # Method 2: Univariate Statistical Tests
    # ========================================================================

    def univariate_statistical_selection(
        self,
        target_values: np.ndarray,
        parameter_name: str,
        n_features: Optional[int] = None,
        test_method: str = 'auto'
    ) -> FeatureSelectionResult:
        """
        Select features using univariate statistical tests.

        Uses F-test for regression, chi-squared for classification,
        or mutual information for both.

        Args:
            target_values: Target parameter values
            parameter_name: Name of parameter
            n_features: Number of features to select
            test_method: 'auto', 'f_test', 'mutual_info'

        Returns:
            FeatureSelectionResult
        """
        import time
        start_time = time.time()

        n_features = n_features or self.target_n_features

        print(f"\n[Method 2] Univariate statistical selection for {parameter_name}")

        # Determine if classification or regression
        is_classification = len(np.unique(target_values)) < 20

        # Choose test method
        if test_method == 'auto':
            test_method = 'mutual_info'  # Works for both

        if test_method == 'f_test':
            if is_classification:
                score_func = f_classif
            else:
                score_func = f_regression
        elif test_method == 'mutual_info':
            if is_classification:
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
        else:
            raise ValueError(f"Unknown test method: {test_method}")

        print(f"Using {test_method} ({'classification' if is_classification else 'regression'})")

        # Select features
        selector = SelectKBest(score_func=score_func, k=n_features)
        selector.fit(self.feature_matrix, target_values)

        # Get scores
        scores = selector.scores_
        selected_indices = selector.get_support(indices=True)

        feature_scores = {
            self.feature_names[i]: float(scores[i]) if not np.isnan(scores[i]) else 0.0
            for i in range(len(self.feature_names))
        }

        selected_features = [self.feature_names[i] for i in selected_indices]

        result = FeatureSelectionResult(
            method_name=f'univariate_{test_method}',
            selected_features=selected_features,
            feature_scores=feature_scores,
            n_features_selected=len(selected_features),
            selection_time=time.time() - start_time,
            metadata={'test_type': 'classification' if is_classification else 'regression'}
        )

        self.selection_results[f'univariate_{test_method}'] = result
        print(f"✅ Selected {len(selected_features)} features in {result.selection_time:.2f}s")

        return result

    # ========================================================================
    # Method 3: Tree-based Feature Importance
    # ========================================================================

    def tree_based_selection(
        self,
        target_values: np.ndarray,
        parameter_name: str,
        n_features: Optional[int] = None,
        model_type: str = 'xgboost'
    ) -> FeatureSelectionResult:
        """
        Select features using tree-based model feature importance.

        Uses XGBoost or Random Forest to compute feature importances.

        Args:
            target_values: Target parameter values
            parameter_name: Name of parameter
            n_features: Number of features to select
            model_type: 'xgboost' or 'random_forest'

        Returns:
            FeatureSelectionResult
        """
        import time
        start_time = time.time()

        n_features = n_features or self.target_n_features

        print(f"\n[Method 3] Tree-based selection ({model_type}) for {parameter_name}")

        # Determine if classification or regression
        is_classification = len(np.unique(target_values)) < 20

        # Train model
        if model_type == 'xgboost' and XGBOOST_AVAILABLE:
            if is_classification:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1
                )
        else:
            # Fall back to Random Forest
            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            model_type = 'random_forest'

        print(f"Training {model_type} model...")
        model.fit(self.feature_matrix, target_values)

        # Get feature importances
        importances = model.feature_importances_

        feature_scores = {
            self.feature_names[i]: float(importances[i])
            for i in range(len(self.feature_names))
        }

        # Select top N
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:n_features]]

        result = FeatureSelectionResult(
            method_name=f'tree_{model_type}',
            selected_features=selected_features,
            feature_scores=feature_scores,
            n_features_selected=len(selected_features),
            selection_time=time.time() - start_time,
            metadata={'model_type': model_type}
        )

        self.selection_results[f'tree_{model_type}'] = result
        print(f"✅ Selected {len(selected_features)} features in {result.selection_time:.2f}s")

        return result

    # ========================================================================
    # Method 4: L1 Regularization (Lasso)
    # ========================================================================

    def lasso_selection(
        self,
        target_values: np.ndarray,
        parameter_name: str,
        n_features: Optional[int] = None,
        alpha: Optional[float] = None
    ) -> FeatureSelectionResult:
        """
        Select features using L1 regularization (Lasso).

        Lasso drives coefficients of irrelevant features to zero.

        Args:
            target_values: Target parameter values
            parameter_name: Name of parameter
            n_features: Number of features to select
            alpha: Regularization strength (auto-selected if None)

        Returns:
            FeatureSelectionResult
        """
        import time
        start_time = time.time()

        n_features = n_features or self.target_n_features

        print(f"\n[Method 4] Lasso selection for {parameter_name}")

        # Determine if classification or regression
        is_classification = len(np.unique(target_values)) < 20

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.feature_matrix)

        if is_classification:
            # Use Logistic Regression with L1
            print("Using Logistic Regression with L1 penalty...")
            model = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=1.0 if alpha is None else 1.0/alpha,
                random_state=self.random_state,
                max_iter=1000
            )
        else:
            # Use Lasso regression
            if alpha is None:
                print("Auto-selecting alpha with LassoCV...")
                model = LassoCV(
                    cv=5,
                    random_state=self.random_state,
                    max_iter=10000,
                    n_jobs=-1
                )
            else:
                model = Lasso(
                    alpha=alpha,
                    random_state=self.random_state,
                    max_iter=10000
                )

        model.fit(X_scaled, target_values)

        # Get coefficients
        if is_classification and len(model.coef_.shape) > 1:
            # Multi-class: average across classes
            coefficients = np.mean(np.abs(model.coef_), axis=0)
        else:
            coefficients = np.abs(model.coef_).ravel()

        feature_scores = {
            self.feature_names[i]: float(coefficients[i])
            for i in range(len(self.feature_names))
        }

        # Select top N by absolute coefficient value
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:n_features] if f[1] > 0]

        # If not enough features with non-zero coefficients, add more
        if len(selected_features) < n_features:
            remaining = [f[0] for f in sorted_features if f[0] not in selected_features]
            selected_features.extend(remaining[:n_features - len(selected_features)])

        result = FeatureSelectionResult(
            method_name='lasso_l1',
            selected_features=selected_features,
            feature_scores=feature_scores,
            n_features_selected=len(selected_features),
            selection_time=time.time() - start_time,
            metadata={
                'alpha': model.alpha_ if hasattr(model, 'alpha_') else alpha,
                'n_nonzero': sum(1 for v in coefficients if v > 1e-10)
            }
        )

        self.selection_results['lasso_l1'] = result
        print(f"✅ Selected {len(selected_features)} features in {result.selection_time:.2f}s")
        print(f"   Non-zero coefficients: {result.metadata['n_nonzero']}")

        return result

    # ========================================================================
    # Method 5: Recursive Feature Elimination (RFE)
    # ========================================================================

    def rfe_selection(
        self,
        target_values: np.ndarray,
        parameter_name: str,
        n_features: Optional[int] = None,
        step: float = 0.1
    ) -> FeatureSelectionResult:
        """
        Select features using Recursive Feature Elimination.

        Iteratively removes least important features.
        More expensive but often more effective.

        Args:
            target_values: Target parameter values
            parameter_name: Name of parameter
            n_features: Number of features to select
            step: Fraction of features to remove at each iteration

        Returns:
            FeatureSelectionResult
        """
        import time
        start_time = time.time()

        n_features = n_features or self.target_n_features

        print(f"\n[Method 5] Recursive Feature Elimination for {parameter_name}")

        # Determine if classification or regression
        is_classification = len(np.unique(target_values)) < 20

        # Choose base estimator
        if is_classification:
            estimator = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )

        print(f"Running RFE (target: {n_features} features, step: {step})...")

        # Run RFE
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=step,
            verbose=0
        )

        selector.fit(self.feature_matrix, target_values)

        # Get selected features
        selected_indices = selector.get_support(indices=True)
        rankings = selector.ranking_

        feature_scores = {
            self.feature_names[i]: float(1.0 / rankings[i])  # Convert rank to score
            for i in range(len(self.feature_names))
        }

        selected_features = [self.feature_names[i] for i in selected_indices]

        result = FeatureSelectionResult(
            method_name='rfe',
            selected_features=selected_features,
            feature_scores=feature_scores,
            n_features_selected=len(selected_features),
            selection_time=time.time() - start_time,
            metadata={'step': step}
        )

        self.selection_results['rfe'] = result
        print(f"✅ Selected {len(selected_features)} features in {result.selection_time:.2f}s")

        return result

    # ========================================================================
    # Method 6: PCA
    # ========================================================================

    def pca_selection(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95
    ) -> FeatureSelectionResult:
        """
        Select principal components explaining desired variance.

        Note: This transforms features rather than selecting them,
        so it's less interpretable but can be more effective.

        Args:
            n_components: Number of components (default: target_n_features)
            variance_threshold: Cumulative variance to explain

        Returns:
            FeatureSelectionResult
        """
        import time
        start_time = time.time()

        n_components = n_components or self.target_n_features

        print(f"\n[Method 6] PCA selection")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.feature_matrix)

        # Fit PCA
        pca = PCA(n_components=min(n_components, self.n_features, self.n_samples))
        pca.fit(X_scaled)

        # Find components needed for variance threshold
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_needed = np.argmax(cumulative_variance >= variance_threshold) + 1

        # Get feature contributions to top components
        components = pca.components_[:n_components_needed]
        feature_importance = np.sum(np.abs(components), axis=0)

        feature_scores = {
            self.feature_names[i]: float(feature_importance[i])
            for i in range(len(self.feature_names))
        }

        # Select top N features by PCA loadings
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:n_components]]

        result = FeatureSelectionResult(
            method_name='pca',
            selected_features=selected_features,
            feature_scores=feature_scores,
            n_features_selected=len(selected_features),
            selection_time=time.time() - start_time,
            metadata={
                'n_components': n_components_needed,
                'explained_variance': float(cumulative_variance[n_components_needed-1]),
                'total_variance': float(np.sum(pca.explained_variance_ratio_))
            }
        )

        self.selection_results['pca'] = result
        print(f"✅ Selected {len(selected_features)} features in {result.selection_time:.2f}s")
        print(f"   {n_components_needed} components explain {result.metadata['explained_variance']:.1%} variance")

        return result

    # ========================================================================
    # Method 7: Domain Knowledge Curation
    # ========================================================================

    def domain_knowledge_selection(
        self,
        must_have_features: Optional[List[str]] = None,
        prefer_categories: Optional[Dict[str, int]] = None,
        n_features: Optional[int] = None
    ) -> FeatureSelectionResult:
        """
        Select features based on domain knowledge and musical importance.

        This method ensures critical musical dimensions are covered.

        Args:
            must_have_features: Features that must be included
            prefer_categories: {category: n_features} - preference per category
            n_features: Total number of features to select

        Returns:
            FeatureSelectionResult
        """
        import time
        start_time = time.time()

        n_features = n_features or self.target_n_features
        must_have_features = must_have_features or []

        print(f"\n[Method 7] Domain knowledge curation")

        selected_features = list(must_have_features)
        feature_scores = {}

        # Score features by category importance
        category_scores = {
            'harmony': 1.0,
            'melody': 0.9,
            'rhythm': 0.95,
            'dynamics': 0.7,
            'texture': 0.6,
            'structure': 0.5
        }

        # Assign scores based on category
        for fname in self.feature_names:
            if fname in must_have_features:
                feature_scores[fname] = 10.0  # Highest priority
            else:
                # Infer category from name
                category = self._infer_feature_category(fname)
                base_score = category_scores.get(category, 0.5)

                # Boost certain features
                if any(keyword in fname.lower() for keyword in [
                    'chord', 'progression', 'voice_leading', 'tension',
                    'contour', 'interval', 'density', 'complexity',
                    'syncopation', 'groove', 'swing'
                ]):
                    base_score *= 1.5

                feature_scores[fname] = base_score

        # Select remaining features by score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        for fname, score in sorted_features:
            if fname not in selected_features and len(selected_features) < n_features:
                selected_features.append(fname)

        result = FeatureSelectionResult(
            method_name='domain_knowledge',
            selected_features=selected_features[:n_features],
            feature_scores=feature_scores,
            n_features_selected=len(selected_features[:n_features]),
            selection_time=time.time() - start_time,
            metadata={
                'n_must_have': len(must_have_features),
                'categories_used': list(category_scores.keys())
            }
        )

        self.selection_results['domain_knowledge'] = result
        print(f"✅ Selected {len(selected_features[:n_features])} features in {result.selection_time:.2f}s")

        return result

    def _infer_feature_category(self, feature_name: str) -> str:
        """Infer feature category from name"""
        fname_lower = feature_name.lower()

        if any(kw in fname_lower for kw in ['chord', 'harmony', 'voicing', 'progression', 'voice']):
            return 'harmony'
        elif any(kw in fname_lower for kw in ['melody', 'melodic', 'interval', 'contour', 'pitch']):
            return 'melody'
        elif any(kw in fname_lower for kw in ['rhythm', 'beat', 'tempo', 'syncopation', 'groove', 'swing']):
            return 'rhythm'
        elif any(kw in fname_lower for kw in ['dynamic', 'velocity', 'accent', 'articulation']):
            return 'dynamics'
        elif any(kw in fname_lower for kw in ['texture', 'density', 'polyphony', 'layer']):
            return 'texture'
        elif any(kw in fname_lower for kw in ['structure', 'form', 'section', 'repetition']):
            return 'structure'
        else:
            return 'unknown'

    # ========================================================================
    # Method 8: Ensemble Selection
    # ========================================================================

    def ensemble_selection(
        self,
        results: Optional[Dict[str, FeatureSelectionResult]] = None,
        min_votes: int = 3,
        n_features: Optional[int] = None
    ) -> EnsembleSelectionResult:
        """
        Combine multiple feature selection methods using voting.

        Features selected by >= min_votes methods are included.
        Increases robustness by aggregating multiple perspectives.

        Args:
            results: Dictionary of selection results (uses self.selection_results if None)
            min_votes: Minimum number of methods that must select a feature
            n_features: Target number of features (default: self.target_n_features)

        Returns:
            EnsembleSelectionResult
        """
        import time
        start_time = time.time()

        results = results or self.selection_results
        n_features = n_features or self.target_n_features

        print(f"\n[Ensemble] Combining {len(results)} feature selection methods")
        print(f"Minimum votes required: {min_votes}")

        # Count votes for each feature
        vote_counts = Counter()
        score_sums = defaultdict(float)
        score_counts = defaultdict(int)

        for method_name, result in results.items():
            for feature in result.selected_features:
                vote_counts[feature] += 1
                if feature in result.feature_scores:
                    # Normalize scores to [0, 1] for each method
                    max_score = max(result.feature_scores.values())
                    if max_score > 0:
                        normalized_score = result.feature_scores[feature] / max_score
                        score_sums[feature] += normalized_score
                        score_counts[feature] += 1

        # Calculate average scores
        avg_scores = {
            feature: score_sums[feature] / score_counts[feature] if score_counts[feature] > 0 else 0.0
            for feature in vote_counts.keys()
        }

        # Select features with >= min_votes
        selected_by_votes = [
            feature for feature, count in vote_counts.items()
            if count >= min_votes
        ]

        print(f"Features with >= {min_votes} votes: {len(selected_by_votes)}")

        # If we have too many, select by average score
        if len(selected_by_votes) > n_features:
            selected_by_votes.sort(key=lambda f: (vote_counts[f], avg_scores[f]), reverse=True)
            selected_features = selected_by_votes[:n_features]
        # If we don't have enough, add more by vote count
        elif len(selected_by_votes) < n_features:
            remaining = [
                feature for feature in vote_counts.keys()
                if feature not in selected_by_votes
            ]
            remaining.sort(key=lambda f: (vote_counts[f], avg_scores[f]), reverse=True)
            selected_features = selected_by_votes + remaining[:n_features - len(selected_by_votes)]
        else:
            selected_features = selected_by_votes

        result = EnsembleSelectionResult(
            selected_features=selected_features,
            feature_vote_counts=dict(vote_counts),
            feature_avg_scores=avg_scores,
            methods_used=list(results.keys()),
            n_features_selected=len(selected_features),
            selection_time=time.time() - start_time
        )

        self.ensemble_result = result

        print(f"✅ Ensemble selected {len(selected_features)} features in {result.selection_time:.2f}s")
        print(f"   Vote distribution: {dict(Counter(vote_counts.values()))}")

        return result

    # ========================================================================
    # Convenience Methods
    # ========================================================================

    def run_all_methods(
        self,
        target_values: np.ndarray,
        parameter_name: str,
        methods: Optional[List[str]] = None
    ) -> Dict[str, FeatureSelectionResult]:
        """
        Run all feature selection methods.

        Args:
            target_values: Target parameter values
            parameter_name: Name of parameter
            methods: List of methods to run (all if None)

        Returns:
            Dictionary of results
        """
        if methods is None:
            methods = [
                'filter',
                'univariate',
                'tree_xgboost',
                'tree_rf',
                'lasso',
                'rfe',
                'pca',
                'domain'
            ]

        print(f"\n{'='*70}")
        print(f"Running {len(methods)} feature selection methods for: {parameter_name}")
        print(f"Target features: {self.target_n_features}")
        print(f"{'='*70}")

        results = {}

        if 'filter' in methods:
            results['filter'] = self.filter_based_selection(target_values, parameter_name)

        if 'univariate' in methods:
            results['univariate'] = self.univariate_statistical_selection(target_values, parameter_name)

        if 'tree_xgboost' in methods and XGBOOST_AVAILABLE:
            results['tree_xgboost'] = self.tree_based_selection(
                target_values, parameter_name, model_type='xgboost'
            )

        if 'tree_rf' in methods:
            results['tree_rf'] = self.tree_based_selection(
                target_values, parameter_name, model_type='random_forest'
            )

        if 'lasso' in methods:
            results['lasso'] = self.lasso_selection(target_values, parameter_name)

        if 'rfe' in methods:
            results['rfe'] = self.rfe_selection(target_values, parameter_name)

        if 'pca' in methods:
            results['pca'] = self.pca_selection()

        if 'domain' in methods:
            results['domain'] = self.domain_knowledge_selection()

        print(f"\n{'='*70}")
        print(f"All methods complete!")
        print(f"{'='*70}\n")

        return results

    def save_selected_features(
        self,
        result: Union[FeatureSelectionResult, EnsembleSelectionResult],
        output_path: Path,
        include_scores: bool = True
    ):
        """Save selected features to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'n_features_selected': result.n_features_selected,
            'selected_features': result.selected_features,
            'selection_timestamp': datetime.now().isoformat(),
        }

        if isinstance(result, EnsembleSelectionResult):
            data['method'] = 'ensemble'
            data['methods_used'] = result.methods_used
            if include_scores:
                data['feature_vote_counts'] = result.feature_vote_counts
                data['feature_avg_scores'] = result.feature_avg_scores
        else:
            data['method'] = result.method_name
            if include_scores:
                data['feature_scores'] = result.feature_scores
            if result.metadata:
                data['metadata'] = result.metadata

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Saved selected features to {output_path}")

    def load_selected_features(self, input_path: Path) -> List[str]:
        """Load selected features from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)

        return data['selected_features']

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of feature selection results"""
        summary = {
            'n_original_features': self.n_features,
            'n_samples': self.n_samples,
            'target_n_features': self.target_n_features,
            'methods_run': list(self.selection_results.keys()),
            'n_methods': len(self.selection_results),
        }

        if self.ensemble_result:
            summary['ensemble'] = {
                'n_selected': self.ensemble_result.n_features_selected,
                'methods_used': self.ensemble_result.methods_used
            }

        return summary

    def print_summary(self):
        """Print summary of feature selection"""
        summary = self.get_summary()

        print("\n" + "="*70)
        print("FEATURE SELECTION SUMMARY")
        print("="*70)
        print(f"Original features: {summary['n_original_features']}")
        print(f"Target features: {summary['target_n_features']}")
        print(f"Samples: {summary['n_samples']}")
        print(f"\nMethods run: {summary['n_methods']}")
        for method in summary['methods_run']:
            result = self.selection_results[method]
            print(f"  - {method}: {result.n_features_selected} features, {result.selection_time:.2f}s")

        if 'ensemble' in summary:
            ens = summary['ensemble']
            print(f"\nEnsemble selection:")
            print(f"  Features selected: {ens['n_selected']}")
            print(f"  Methods combined: {len(ens['methods_used'])}")

        print("="*70 + "\n")


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_feature_selection(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    target_values: np.ndarray,
    parameter_name: str,
    target_n_features: int = 200,
    output_dir: Optional[Path] = None
) -> EnsembleSelectionResult:
    """
    Quick feature selection with all methods + ensemble.

    Args:
        feature_matrix: Feature matrix
        feature_names: Feature names
        target_values: Target values
        parameter_name: Parameter name
        target_n_features: Target number of features
        output_dir: Output directory (saves if provided)

    Returns:
        EnsembleSelectionResult
    """
    selector = ComprehensiveFeatureSelector(
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        target_n_features=target_n_features
    )

    # Run all methods
    results = selector.run_all_methods(target_values, parameter_name)

    # Ensemble selection
    ensemble_result = selector.ensemble_selection(results, min_votes=3)

    # Save if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save ensemble result
        selector.save_selected_features(
            ensemble_result,
            output_dir / f'{parameter_name}_selected_features.json'
        )

        # Save all individual results
        for method_name, result in results.items():
            selector.save_selected_features(
                result,
                output_dir / f'{parameter_name}_{method_name}_features.json'
            )

    return ensemble_result


if __name__ == "__main__":
    print("="*70)
    print("COMPREHENSIVE FEATURE SELECTOR - AGENT 04")
    print("="*70)

    # Demo with synthetic data
    print("\n📊 Creating synthetic data for demonstration...")
    np.random.seed(42)

    n_samples = 500
    n_features = 1000
    target_n_features = 200

    # Create feature matrix
    X = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i:04d}" for i in range(n_features)]

    # Create target (continuous)
    # Make some features actually predictive
    y = (X[:, 0] * 2.0 +
         X[:, 10] * 1.5 +
         X[:, 50] * 1.0 +
         X[:, 100] * 0.8 +
         np.random.randn(n_samples) * 0.5)

    print(f"Data: {n_samples} samples, {n_features} features")
    print(f"Target: {target_n_features} features")

    # Create selector
    selector = ComprehensiveFeatureSelector(
        feature_matrix=X,
        feature_names=feature_names,
        target_n_features=target_n_features
    )

    # Run quick selection (subset of methods for demo speed)
    print("\n🚀 Running feature selection methods...")
    results = selector.run_all_methods(
        target_values=y,
        parameter_name='demo_parameter',
        methods=['filter', 'univariate', 'lasso', 'domain']
    )

    # Ensemble
    print("\n🎯 Running ensemble selection...")
    ensemble = selector.ensemble_selection(results, min_votes=2)

    # Summary
    selector.print_summary()

    print("\n✅ Demo complete!")
    print("\nFor production use:")
    print("  from midi_generator.feature_selection import ComprehensiveFeatureSelector")
    print("  selector = ComprehensiveFeatureSelector(X, feature_names, target_n=200)")
    print("  results = selector.run_all_methods(y, 'tempo')")
    print("  ensemble = selector.ensemble_selection(results, min_votes=3)")
