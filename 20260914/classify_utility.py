import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
from joblib import Parallel, delayed, parallel_config
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import inspect
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import inspect
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve




MODELS_NEED_WEIGHTS = ['SVM_Linear', 'LR_L1', 'LR_L2']

class ClassifierManager:
    """Pure classifier configuration descriptor and instance factory"""
    
    _REGISTRY = {
        'LR_L1': LogisticRegression,
        'SVM_Linear': LinearSVC, 
        'LR_L2': LogisticRegression
    }
    
    def __init__(
        self, 
        name, 
        param_grid, 
        fixed_kwargs=None, 
        need_scale=False, 
        has_weights=False):
        if name not in self._REGISTRY:
            raise ValueError(f"Unregistered classifier: {name}")
        
        self.name = name
        self._clazz = self._REGISTRY[name]
        self.param_grid = param_grid
        self.fixed_kwargs = fixed_kwargs if fixed_kwargs is not None else {}
        self.need_scale = need_scale
        self.has_weights = has_weights
        
        # Parse signature once at initialization and cache result (only computed once in entire lifecycle)
        sig = inspect.signature(self._clazz.__init__).parameters
        self._accepts_random_state = 'random_state' in sig
        
        # Seal classifier internal parallelism to prevent thread explosion with GridSearchCV(n_jobs=-1)
        if name not in ['LR_L1', 'LR_L2']:
            if 'n_jobs' in sig:
                self.fixed_kwargs['n_jobs'] = 1
    
    def retrieve_classifier(self, random_state=None):
        """Factory method: returns an instantiated object with fixed parameters already injected"""
        params = self.fixed_kwargs.copy()
        if random_state is not None and self._accepts_random_state:
            params['random_state'] = random_state
        return self._clazz(**params)


class RemoveZeroColumns(BaseEstimator, TransformerMixin):
    """Pure Pandas version, uses vectorized operations to extremely quickly remove all-zero columns, absolutely preserves row/column Index"""
    
    def fit(self, X, y=None):
        # If numpy is received unexpectedly, first convert to DataFrame as a safeguard (prevents accidental chain breakage)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Core: .any(axis=0) is a pure C-level Pandas vectorized operation, much faster than np.any
        # It returns a boolean Series with column names
        self.non_zero_mask_ = (X != 0).any(axis=0)
        
        # (Optional) Print info for monitoring
        n_removed = (~self.non_zero_mask_).sum()
        if n_removed > 0:
            print(f"🧹 [Remove zero columns] Removed {n_removed} all-zero features, retained {self.non_zero_mask_.sum()}.")
            
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Core: must use .loc for column name / boolean array slicing!
        # .loc slicing is not only extremely fast, but also 100% ensures Index (row and column names) is passed intact to the next step's ComBat
        return X.loc[:, self.non_zero_mask_]



def build_pipeline_and_param_grid(classifier_manager, reducer_manager, random_state=None):
    
    pipeline_steps = []
    param_grid = {}
    
    # Step 1: Fixed first step, remove all-zero columns
    pipeline_steps.append(('remove_zero', RemoveZeroColumns()))
    
    # Step 2: Whether to standardize before dimensionality reduction
    if reducer_manager:
        if reducer_manager.need_scale_input:
            pipeline_steps.append(('scaler_dr', StandardScaler()))
            
        # Step 3: Reducer
        pipeline_steps.append(('reducer', reducer_manager.get_reducer(random_state=random_state)))
        
    # Step 4: Whether to standardize before classification
    if classifier_manager.need_scale:
        pipeline_steps.append(('scaler_clf', StandardScaler()))
        
    # Step 5: Classifier
    pipeline_steps.append(('clf', classifier_manager.retrieve_classifier(random_state=random_state)))
    
    # Aggregate parameter grid
    if reducer_manager:
        param_grid.update({f'reducer__{k}': v for k, v in reducer_manager.param_grid.items()})
        
    param_grid.update({f'clf__{k}': v for k, v in classifier_manager.param_grid.items()})
        
    pipeline = Pipeline(pipeline_steps)
    return pipeline, param_grid



def extract_indices_for_significant_features(
    feature_matrix, 
    label_matrix, 
    p_threshold: float = 0.05, 
    min_frequency: float = 0.5,
    need_correction: bool = False):  
    
    positive_indices = np.where(label_matrix == 1)[0]
    negative_indices = np.where(label_matrix == 0)[0]
    positive_features = feature_matrix[positive_indices]
    negative_features = feature_matrix[negative_indices]
    
    num_features = feature_matrix.shape[1]
    
    topo_p_values = np.full(num_features, np.nan)  
    strength_p_values = np.full(num_features, np.nan)
    
    for i in range(num_features):
        pos_feature = positive_features[:, i]
        neg_feature = negative_features[:, i]
        
        pos_nonzero_mask = pos_feature > 0
        neg_nonzero_mask = neg_feature > 0
        pos_ratio = np.mean(pos_nonzero_mask)
        neg_ratio = np.mean(neg_nonzero_mask)
        
        # --- Compute strength P-value ---
        is_high_freq_feature = (pos_ratio >= min_frequency) and (neg_ratio >= min_frequency)
        if is_high_freq_feature:
            pos_nonzero_values = pos_feature[pos_nonzero_mask]
            neg_nonzero_values = neg_feature[neg_nonzero_mask]
            if len(pos_nonzero_values) > 3 and len(neg_nonzero_values) > 3:
                try:
                    # Optimization: if both groups have identical non-zero values, variance is 0, mannwhitneyu will error, which is a reasonable skip
                    if np.std(pos_nonzero_values) > 1e-8 or np.std(neg_nonzero_values) > 1e-8:
                        _, p_strength = mannwhitneyu(pos_nonzero_values, neg_nonzero_values, alternative='two-sided')
                        strength_p_values[i] = p_strength
                except ValueError as e:
                    # Catch specific errors to avoid swallowing unknown bugs
                    # print(f"Feature {i} strength test failed (possibly zero variance): {e}")
                    pass

    # =========================================================================
    # Phase 2: Filter by rules (logic is perfect, no modification needed)
    # =========================================================================
    def _filter_by_rules(p_values, threshold, apply_correction):
        valid_mask = ~np.isnan(p_values)
        valid_indices = np.where(valid_mask)[0]
        valid_p_values = p_values[valid_mask]
        
        if len(valid_p_values) == 0:
            return np.array([], dtype=int)
            
        if apply_correction:
            if threshold >= 1:
                raise ValueError(f"When FDR correction is enabled, threshold ({threshold}) must be < 1, otherwise Q-values cannot be computed.")
            
            reject, q_values, _, _ = multipletests(valid_p_values, alpha=threshold, method='fdr_bh')
            return valid_indices[reject]
            
        else:
            if threshold >= 1:
                k = min(int(threshold), len(valid_p_values))
                smallest_p_indices = np.argsort(valid_p_values)[:k]
                return valid_indices[smallest_p_indices]
            else:
                pass_threshold_mask = valid_p_values < threshold
                return valid_indices[pass_threshold_mask]

    selected_feature_indices = _filter_by_rules(strength_p_values, p_threshold, need_correction)
    
    return selected_feature_indices


def _evaluate_single_cv_pass(
    round_num, 
    all_sample_features, 
    all_sample_labels, 
    current_classifier, 
    stratify_labels=None, 
    test_params=None
) -> tuple[list, list, list, np.ndarray | None, np.ndarray | None, np.ndarray | None, list]:
    
    if not isinstance(all_sample_features, pd.DataFrame):
        all_sample_features = pd.DataFrame(all_sample_features)
    
    if stratify_labels is None:
        stratify_labels = all_sample_labels
        
    pipeline, param_grid = build_pipeline_and_param_grid(current_classifier, None, random_state=round_num)
    
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=round_num)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=round_num+10000)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=1
    )
    
    num_features = all_sample_features.shape[-1]
    num_outer_folds = 5
    
    accuracy_list, auc_list, f1_list, all_fold_predictions = [], [], [], []
    whole_brain_importance_matrix = np.zeros((num_outer_folds, num_features))
    whole_brain_valid_mask_matrix = np.zeros((num_outer_folds, num_features), dtype=bool)
    
    # Determine whether current model produces true zero weights
    produces_sparse_weights = current_classifier.name in ['LR_L1', 'RandomForest'] 
    
    for outer_fold_idx, (train_indices, test_indices) in enumerate(outer_cv.split(all_sample_features, stratify_labels)):
        
        train_samples = all_sample_features.iloc[train_indices]
        test_samples = all_sample_features.iloc[test_indices]
        
        train_matrix = train_samples.to_numpy()
        test_matrix = test_samples.to_numpy()
            
            
        train_labels = all_sample_labels[train_indices]
        test_labels = all_sample_labels[test_indices]
        
        if test_params is not None:
            retained_feature_indices = extract_indices_for_significant_features(train_matrix, train_labels, **test_params)
        else:
            retained_feature_indices = np.arange(train_matrix.shape[1])
        
        if len(retained_feature_indices) < 10:
            print(f"Warning: Too few differential features (only {len(retained_feature_indices)} left), this fold may fail")
            
        train_matrix = train_matrix[:, retained_feature_indices]
        test_matrix = test_matrix[:, retained_feature_indices]
        
        grid_search.fit(train_matrix, train_labels)
        best_model = grid_search.best_estimator_
        
        if current_classifier.has_weights:
            clf_instance = best_model.named_steps['clf']
            current_fold_importance = np.zeros(num_features)
            current_fold_valid_mask = np.zeros(num_features, dtype=bool)
            
            subspace_importance = None
            if current_classifier.name in ['SVM_Linear', 'LR_L1', 'LR_L2']:
                # Core: Extract raw weights and perform L2 normalization to eliminate cross-fold model magnitude differences
                raw_weights = clf_instance.coef_[0]
                l2_norm = np.linalg.norm(raw_weights)
                normalized_weights = raw_weights / l2_norm if l2_norm > 1e-8 else raw_weights
                subspace_importance = np.abs(normalized_weights)
                
            if subspace_importance is not None:
                # Core: Strict dimension validation, reject dangerous guess alignment
                if len(subspace_importance) != len(retained_feature_indices):
                    print(f"Outer fold {outer_fold_idx+1} critical error: weight dimension ({len(subspace_importance)}) does not match filtered feature count ({len(retained_feature_indices)})! Weights for this fold discarded!")
                else:
                    current_fold_importance[retained_feature_indices] = subspace_importance
                    if produces_sparse_weights:
                        current_fold_valid_mask[retained_feature_indices] = subspace_importance > 1e-6
                    else:
                        current_fold_valid_mask[retained_feature_indices] = True
            
            whole_brain_importance_matrix[outer_fold_idx] = current_fold_importance
            whole_brain_valid_mask_matrix[outer_fold_idx] = current_fold_valid_mask
        
        try:
            predicted_proba = best_model.predict_proba(test_matrix)[:, 1]
        except AttributeError:
            test_raw_scores = best_model.decision_function(test_matrix)
            predicted_proba = (test_raw_scores - test_raw_scores.min()) / (test_raw_scores.max() - test_raw_scores.min())
            
        current_best_threshold = 0.5
        predicted_labels = (predicted_proba >= current_best_threshold).astype(int)   
        
        accuracy_list.append(accuracy_score(test_labels, predicted_labels))
        auc_list.append(roc_auc_score(test_labels, predicted_proba))
        f1_list.append(f1_score(test_labels, predicted_labels, average='binary'))
        
        all_fold_predictions.append({
            'true': test_labels,
            'proba': predicted_proba
        })
        
    # Align with inner selection version, return 7 elements, 6th (frequency matrix) is None
    return accuracy_list, auc_list, f1_list, whole_brain_importance_matrix, whole_brain_valid_mask_matrix, None, all_fold_predictions


def _evaluate_stability_selection_pass(
    round_num, 
    all_sample_features, 
    all_sample_labels, 
    current_classifier, 
    stratify_labels=None, 
    test_params=None,
    stability_freq_threshold=1.0
) -> tuple[list, list, list, np.ndarray, list, list]:
    
    if not isinstance(all_sample_features, pd.DataFrame):
        all_sample_features = pd.DataFrame(all_sample_features)
    
    if stratify_labels is None:
        stratify_labels = all_sample_labels
        
    pipeline, param_grid = build_pipeline_and_param_grid(current_classifier, None, random_state=round_num)
    
    num_features = all_sample_features.shape[-1]
    num_splits = 5
    outer_cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=round_num)
    phase1_cv = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=10, random_state=round_num+10000)
    phase2_cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=round_num+20000)
    
    accuracy_list, auc_list, f1_list = [], [], []
    all_test_labels = []
    all_predicted_probas = []
    whole_brain_feature_freq_matrix = np.zeros((num_splits, num_features))
    
    for outer_fold_idx, (train_indices, test_indices) in enumerate(outer_cv.split(all_sample_features, stratify_labels)):
        
        train_samples = all_sample_features.iloc[train_indices]
        test_samples = all_sample_features.iloc[test_indices]
        
        train_matrix = train_samples.to_numpy()
        test_matrix = test_samples.to_numpy()
            
        train_labels = all_sample_labels[train_indices]
        test_labels = all_sample_labels[test_indices]
        
        if test_params is not None:
            retained_feature_indices = extract_indices_for_significant_features(train_matrix, train_labels, **test_params)
        else:
            retained_feature_indices = np.arange(train_matrix.shape[1])
        
        if len(retained_feature_indices) < 10:
            print(f"Warning: Too few differential features (only {len(retained_feature_indices)} left), this fold may fail")
            
        train_matrix_filtered = train_matrix[:, retained_feature_indices]
        test_matrix_filtered = test_matrix[:, retained_feature_indices]
        
        current_fold_feature_freq = np.zeros(num_features)
        
        feature_occurrence_count = np.zeros(len(retained_feature_indices))
        feature_weight_sign_cumsum = np.zeros(len(retained_feature_indices))
        
        grid_search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=param_grid,
            cv=phase1_cv,
            scoring='roc_auc',
            n_jobs=1
        )
        
        grid_search.fit(train_matrix_filtered, train_labels)
        best_params = grid_search.best_params_
        
        for inner_train_idx, _ in phase1_cv.split(train_matrix_filtered, train_labels):
            inner_train_samples = train_matrix_filtered[inner_train_idx]
            inner_train_labels = train_labels[inner_train_idx]
            
            current_fold_model = clone(pipeline).set_params(**best_params)
            current_fold_model.fit(inner_train_samples, inner_train_labels)
            clf_instance = current_fold_model.named_steps['clf']
            
            fold_raw_weights = clf_instance.coef_[0]
            
            if len(fold_raw_weights) != len(retained_feature_indices):
                continue
            
            feature_weight_sign_cumsum += np.sign(fold_raw_weights)
            feature_occurrence_count += (fold_raw_weights != 0).astype(int)
            
        total_inner_folds = phase1_cv.get_n_splits()  # 50
        freq_array = feature_occurrence_count / total_inner_folds
        
        # Direction consistency filtering (eliminate flip-floppers)
        direction_consistency_array = np.abs(feature_weight_sign_cumsum / total_inner_folds)
        direction_consistency_threshold = 0.8
        inconsistent_mask = direction_consistency_array < direction_consistency_threshold
        freq_array[inconsistent_mask] = 0.0
        
        num_killed = np.sum(inconsistent_mask)
        if num_killed > 0:
            print(f"Outer fold {outer_fold_idx+1}: Cleared {num_killed} flip-flopper features with inconsistent direction (<{direction_consistency_threshold})")
        
        current_fold_feature_freq[retained_feature_indices] = freq_array
        whole_brain_feature_freq_matrix[outer_fold_idx] = current_fold_feature_freq
        
        stable_feature_local_indices = np.where(freq_array >= stability_freq_threshold)[0]
        
        if len(stable_feature_local_indices) == 0:
            print(f"Outer fold {outer_fold_idx+1}: No features with frequency>={stability_freq_threshold}! Degrading to features with frequency>0.5")
            stable_feature_local_indices = np.where(freq_array >= 0.5)[0]
            if len(stable_feature_local_indices) == 0:
                print(f"Outer fold {outer_fold_idx+1}: Still no high-frequency features, using all differential features")
                stable_feature_local_indices = np.arange(len(retained_feature_indices))
        
        stable_feature_local_mask = np.zeros(len(retained_feature_indices), dtype=bool)
        stable_feature_local_mask[stable_feature_local_indices] = True
        
        print(f"Outer fold {outer_fold_idx+1}: {len(retained_feature_indices)} differential features -> {len(stable_feature_local_indices)} stable features")
        
        # Reduce to stable features
        train_matrix_final = train_matrix_filtered[:, stable_feature_local_mask]
        test_matrix_final = test_matrix_filtered[:, stable_feature_local_mask]
        
        # ================================================================
        # Phase 2: Fine hyperparameter tuning (on stable features only)
        # ================================================================
        fine_grid = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=param_grid,
            cv=phase2_cv,
            scoring='roc_auc',
            n_jobs=1,
            return_train_score=True
        )
        
        fine_grid.fit(train_matrix_final, train_labels)
        best_params = fine_grid.best_params_
        final_model = clone(pipeline).set_params(**best_params)
        final_model.fit(train_matrix_final, train_labels)
        
        # predicted_proba = final_model.predict_proba(test_matrix_final)[:, 1]
        
        try:
            predicted_proba = final_model.predict_proba(test_matrix_final)[:, 1]
        except AttributeError:
            test_raw_scores = final_model.decision_function(test_matrix_final)
            predicted_proba = (test_raw_scores - test_raw_scores.min()) / (test_raw_scores.max() - test_raw_scores.min())
            predicted_proba = np.clip(predicted_proba, 0, 1)
        
        optimal_threshold = 0.5
        predicted_labels = (predicted_proba >= optimal_threshold).astype(int)
        
        accuracy_list.append(accuracy_score(test_labels, predicted_labels))
        auc_list.append(roc_auc_score(test_labels, predicted_proba))
        f1_list.append(f1_score(test_labels, predicted_labels, average='binary'))
        
        all_test_labels.append(test_labels)
        all_predicted_probas.append(predicted_proba)
        
    return accuracy_list, auc_list, f1_list, whole_brain_feature_freq_matrix, all_test_labels, all_predicted_probas

def process_single_feature_set(
    all_sample_features, 
    all_sample_labels, 
    current_classifier, 
    all_stratify_labels=None, 
    test_params=None,
    stability_freq_threshold=0.90,
    n_jobs=5):
    
    if not isinstance(all_sample_features, pd.DataFrame):
        all_sample_features = pd.DataFrame(all_sample_features)
        
    if all_stratify_labels is None: 
        all_stratify_labels = all_sample_labels
        
    def _safe_single_eval(round_num):
        try:
            return _evaluate_stability_selection_pass(
                round_num, 
                all_sample_features, 
                all_sample_labels, 
                current_classifier, 
                all_stratify_labels, 
                test_params,
                stability_freq_threshold=stability_freq_threshold
            )
        except Exception as e:
            print(f'Round {round_num} fatal error: {e}')
            import traceback
            traceback.print_exc() 
            return None
    
    with parallel_config(n_jobs=n_jobs, backend='loky', inner_max_num_threads=1):    
        results_list = Parallel(verbose=10)(
            delayed(_safe_single_eval)(j) for j in range(10)
        )
    
        final_results_list = []
    for result in results_list:
        if result is None:
            continue
        
        # Result structure: (acc_list, auc_list, f1_list, freq_matrix, test_labels_list, pred_probas_list)
        # Check whether the first three metric lists are valid
        try:
            # Concatenate the first three lists into an array and check if all values are finite
            metrics_arr = np.array(result[:3])
            if not np.all(np.isfinite(metrics_arr)):
                continue
        except Exception:
            continue
            
        final_results_list.append(result)
        
    if not final_results_list:
        return {
            'acc_mean': np.nan, 'acc_std': np.nan,
            'auc_mean': np.nan, 'auc_std': np.nan,
            'f1_mean': np.nan, 'f1_std': np.nan,
            'roc_data': None,
            'avg_weights': None,
            'feat_freq': None
        }

    # ================= Metric aggregation =================
    all_accuracies, all_aucs, all_f1s = [], [], []
    all_y_true = []       # Store true labels from all outer folds
    all_y_proba = []      # Store predicted probabilities from all outer folds
    
    valid_freq_matrices = [] # Store the feature frequency matrix from each round
    
    for result in final_results_list:
        # 1. Collect scalar metrics
        all_accuracies.extend(result[0])
        all_aucs.extend(result[1])
        all_f1s.extend(result[2])
        
        # 2. Collect feature frequency matrix (index 3)
        if result[3] is not None:
            valid_freq_matrices.append(result[3])
            
        # 3. Collect true labels and predicted probabilities (indices 4 and 5)
        # result[4] and result[5] are lists of length 5 (one per outer fold)
        if result[4] is not None and result[5] is not None:
            all_y_true.extend(result[4])
            all_y_proba.extend(result[5])
            
    # ================= Feature frequency aggregation =================
    final_avg_freq = None
    final_avg_weights = None
    if valid_freq_matrices:
        # Vertically stack all matrices: shape = (num_valid_rounds * 5, num_features)
        global_freq_matrix = np.vstack(valid_freq_matrices)
        # Compute the average occurrence frequency of each feature across all valid folds
        final_avg_freq = global_freq_matrix.mean(axis=0)
        # Use the average frequency directly as the final weights (modify here if other weight logic is needed)
        final_avg_weights = final_avg_freq 

    # ================= ROC curve computation =================
    mean_fpr = np.linspace(0, 1, 100)
    tpr_list = []
    
    # Iterate over all successfully completed outer test folds (already flattened into a 1D list)
    for y_true, y_proba in zip(all_y_true, all_y_proba):
        # ROC can only be computed if both classes are present
        if len(np.unique(y_true)) < 2:
            continue
            
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        # Interpolate to the common FPR grid
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_list.append(interp_tpr)
        
    roc_data = None
    if tpr_list:
        tpr_array = np.array(tpr_list)
        mean_tpr = np.mean(tpr_array, axis=0)
        std_tpr = np.std(tpr_array, axis=0)
        mean_tpr[-1] = 1.0
        
        roc_data = {
            'fpr': mean_fpr,
            'tpr_mean': mean_tpr,
            'tpr_upper': np.minimum(mean_tpr + std_tpr, 1),
            'tpr_lower': np.maximum(mean_tpr - std_tpr, 0),
            # Reuse the already collected AUC list directly
            'auc_mean': np.mean(all_aucs), 
            'auc_std': np.std(all_aucs),
            'round_tprs': tpr_list
        }
    # =========================================================

    return {
        'acc_mean': np.mean(all_accuracies),
        'acc_std': np.std(all_accuracies),
        'auc_mean': np.mean(all_aucs),
        'auc_std': np.std(all_aucs),
        'f1_mean': np.mean(all_f1s),
        'f1_std': np.std(all_f1s),
        'roc_data': roc_data,
        'avg_weights': final_avg_weights,
        'feat_freq': final_avg_freq
    }