import os
from pathlib import Path
import time
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from GraphUtilities import gsp_compute_graph_learning_theta_batch, calculate_signal_difference, sparsify_fc
from MultiGraphADMMLearner import MultiGraphADMMLearner
import NumbaGraph
import Proximities


import numpy as np

from StateRecorder import StateRecorder
from learn_graph_fdpg_log_degrees import learn_graph_FDPG
from learn_graph_log_degree import learn_graph_log_degrees
from learn_graph_mm import learn_graph_mm

class PearsonCorrelationLearner:
    """
    Functional connectivity matrix calculator based on Pearson correlation coefficient.
    Input shape: (batch_size, time_series_length, num_nodes)
    Output shape: (batch_size, num_nodes, num_nodes)
    """
    def __init__(self, 
                 num_nodes: int, 
                 log_prefix: str="Computing FC matrix", 
                 verbose: bool=False):
        
        self.log_prefix = log_prefix
        self.verbose = verbose
        self.num_nodes = num_nodes

    def __call__(self, 
                 input_data: np.ndarray) -> np.ndarray:
        """
        Compute Pearson correlation coefficients.
        """
        if self.verbose:
            print(f"{self.log_prefix}: Start computing...")

        # Check input dimensions (Batch, Time, Nodes)
        if input_data.ndim != 3:
            raise ValueError("Input data dimensions must be (batch_size, time_series_length, num_nodes)")
        
        # Get dimension info
        timepoints = input_data.shape[1]

        # Standardize data (subtract mean, divide by standard deviation)
        # axis=1 is the time axis
        # keepdims=True preserves dimensions for broadcasting when computing mean and std
        mean = np.mean(input_data, axis=1, keepdims=True)
        std = np.std(input_data, axis=1, keepdims=True, ddof=1) # ddof=1 for sample standard deviation
        
        # Avoid division by zero (if a node's time series is constant, std is 0)
        std[std == 0] = 1.0
        
        z_scores = (input_data - mean) / std

        # Compute correlation coefficient matrix
        # Corr = Z @ Z.T / (T-1)
        # Use np.matmul for batch matrix multiplication
        # Standardized data shape: (B, T, N) -> transpose to (B, N, T)
        # Result shape: (B, N, N)
        # i.e., (Z^T @ Z) / (T-1), but since Z is already standardized, the dot product directly gives the correlation
        # Note: Dividing by (T-1) is the definition of covariance, but with Z-scores already divided by std,
        # the dot product divided by (T-1) yields Pearson r.
        # In fact, dot product of Z-score vectors / (T-1) = Pearson correlation coefficient
        
        fc_matrix = np.matmul(z_scores.transpose(0, 2, 1), z_scores) / (timepoints - 1)

        if self.verbose:
            print(f"{self.log_prefix}: Computation completed.")
            
        return fc_matrix


def PreparePearsonFCLearner(
    num_nodes: int, 
    verbose: bool=False) -> 'PearsonCorrelationLearner':
    """
    Prepare an instance of the Pearson correlation-based functional connectivity learner.
    
    Parameters:
        num_nodes: Number of brain regions / features
        verbose: Whether to output computation progress messages
    """
    
    # Create learner instance
    # Removed all ADMM-related parameters (iterations, tolerance, lambda values, prox operators, etc.)
    learner = PearsonCorrelationLearner(num_nodes=num_nodes, log_prefix="Computing Pearson FC", verbose=verbose)
                     
    return learner


def PrepareLogDegreeGraphLearner(
    num_nodes: int,
    max_iterations: int=2000,
    max_tolerance: float=1e-6) -> 'MultiGraphADMMLearner':
    
    """Prepare graph learner instance"""
    # Create graph operator
    graph_operator = NumbaGraph.GraphOperator_Unsafe(num_nodes)
    relax_alpha = 1.8
    degree_coeff = 1.0
    lam4 = 0.0
    lam5 = 0.0
    lam6 = 0.5

    # Create proximity operators
    prox_degree = Proximities.LogProx(degree_coeff)
    prox_nn = Proximities.NNSoft()
    
    # Create graph learner
    learner = MultiGraphADMMLearner(graph_operator=graph_operator, 
                lam4=lam4, 
                lam5=lam5, 
                lam6=lam6, 
                relax_alpha=relax_alpha, 
                nn_prox=prox_nn, 
                adjmat_prox=None, 
                degree_prox=prox_degree, 
                lapmat_prox=None, 
                max_iterations=max_iterations, 
                max_tolerance=max_tolerance,
                log_prefix="Running",
                verbose=False) # Disable verbose output for individual learners during parallel execution to avoid log clutter
                     
    return learner

def PrepareLogDegreeEntropyGraphLearner(
    entropy_strength: float=0.01,
    max_iteration=50000, 
    max_tolerance=1e-2) -> Callable[[int], 'MultiGraphADMMLearner']:
    
    def _prepare(num_nodes: int) -> 'MultiGraphADMMLearner':
    
        """Prepare graph learner instance"""
        # Create graph operator
        graph_operator = NumbaGraph.GraphOperator_Unsafe(num_nodes)
        relax_alpha = 1.8
        degree_coeff = 1.0
        lam4 = 0.0
        lam5 = 0.0
        lam6 = 0.5

        # Create proximity operators
        prox_log = Proximities.LogProx(degree_coeff)
        prox_entropy = Proximities.EntropyProx(entropy_strength)
        prox_degree = (prox_log, prox_entropy)
        
        prox_nn = Proximities.NNSoft()
        
        # Create graph learner
        learner = MultiGraphADMMLearner(graph_operator=graph_operator, 
                    lam4=lam4, 
                    lam5=lam5, 
                    lam6=lam6, 
                    relax_alpha=relax_alpha, 
                    nn_prox=prox_nn, 
                    adjmat_prox=None, 
                    degree_prox=prox_degree, 
                    lapmat_prox=None, 
                    max_iterations=max_iteration, 
                    max_tolerance=max_tolerance,
                    log_prefix="Running",
                    verbose=False) # Disable verbose output for individual learners during parallel execution to avoid log clutter
                        
        return learner
    return _prepare


def PrepareLogDegreeVarianceGraphLearner(
    L2_strength: float=0.01,
    max_iteration=50000, 
    max_tolerance=1e-2) -> Callable[[int], 'MultiGraphADMMLearner']:
    
    def _prepare(num_nodes: int) -> 'MultiGraphADMMLearner':
    
        """Prepare graph learner instance"""
        # Create graph operator
        graph_operator = NumbaGraph.GraphOperator_Unsafe(num_nodes)
        relax_alpha = 1.8
        degree_coeff = 1.0
        lam4 = 0.0
        lam5 = 0.0
        lam6 = 0.5

        # Create proximity operators
        prox_log = Proximities.LogProx(degree_coeff)
        prox_proj = Proximities.ProjProx(np.ones(shape=(1, num_nodes)), L2_strength)
        prox_degree = (prox_log, prox_proj)
        
        prox_nn = Proximities.NNSoft()
        
        # Create graph learner
        learner = MultiGraphADMMLearner(graph_operator=graph_operator, 
                    lam4=lam4, 
                    lam5=lam5, 
                    lam6=lam6, 
                    relax_alpha=relax_alpha, 
                    nn_prox=prox_nn, 
                    adjmat_prox=None, 
                    degree_prox=prox_degree, 
                    lapmat_prox=None, 
                    max_iterations=max_iteration, 
                    max_tolerance=max_tolerance,
                    log_prefix="Running",
                    verbose=False) # Disable verbose output for individual learners during parallel execution to avoid log clutter
        
        return learner
    return _prepare





def SingleGraphLearnerWrapper(
    input_path: str|Path, 
    output_prefix: str|Path, 
    edge_ratios: list[float],
    learner_maker: Callable[[int], 'MultiGraphADMMLearner']|Callable[[int], 'PearsonCorrelationLearner']|None=None,
    remove_mean: bool=False,
    post_processor: Callable[[np.ndarray], np.ndarray]|None=None):
    """
    Internal function to process a single npy file, intended for concurrent calls via joblib.
    """
    
    if not isinstance(input_path, Path):
        input_path = Path(input_path)
        
    if not isinstance(output_prefix, Path):
            output_prefix = Path(output_prefix)
            
    check_filename = f'{output_prefix}-{edge_ratios[1]:.02f}.npy'
    if os.path.exists(check_filename):
        return True
    
    eps_tolerance = 1e-10
    try:
        # --- 1. Load Time Series ---
        if input_path.suffix == '.npy':
            time_series = np.load(input_path)
        elif input_path.suffix == '.xlsx':
            df = pd.read_excel(input_path)
            roi_cols = [c for c in df.columns if c.startswith('ROI_')]
            time_series = df[roi_cols].to_numpy()
        elif input_path.suffix == '.joblib':
            time_series = joblib.load(input_path)
        else:
            # Default to CSV
            df = pd.read_csv(input_path)
            roi_cols = [c for c in df.columns if c.startswith('ROI_')]
            time_series = df[roi_cols].to_numpy()
        
        if remove_mean:
            time_series -= np.mean(time_series, axis=0, keepdims=True)
        
        norms = np.linalg.norm(time_series, axis=0)
        
        active_indices = np.where(norms>eps_tolerance)[0]
        num_nodes = len(active_indices)
        
        graph_operator = NumbaGraph.GraphOperator_Unsafe(num_nodes)
        
        time_series = time_series[:, active_indices]
        time_series /= norms[active_indices]
        
        # 11. Run graph learner optimization
        print("\nStarting graph learner optimization...")
        
        if learner_maker is None:
            learner = PrepareLogDegreeGraphLearner(num_nodes)
        else:
            learner = learner_maker(num_nodes)
             
        start = time.perf_counter()
        if isinstance(learner, MultiGraphADMMLearner):
            
            signal_difference = calculate_signal_difference(time_series)
            dist_matrix = graph_operator.Q(signal_difference)
            
            edge_ratio_arr = np.array(edge_ratios)
                    
            print(f"Signal difference edge vector shape: {dist_matrix.shape}")
            
            # 4. Compute theta bounds
            expected_edges = np.array([max(int((num_nodes-1) * ratio), 1) for ratio in edge_ratios])
            theta_values = gsp_compute_graph_learning_theta_batch(
                dist_matrix.squeeze(), 
                expected_edges, 
                geom_mean=False, 
                is_sorted=False)
            
            signal_diff_weighted = theta_values[..., np.newaxis] * signal_difference
            init_weights = np.exp(-signal_diff_weighted/2)
            init_weights = sparsify_fc(init_weights, edge_ratios)
            
            print(f"Initial edge weights shape: {init_weights.shape}")
            
            weights_optimized, _, _ = learner(
                edges=init_weights,
                signal_diff=signal_diff_weighted,
                expected_edge_ratio=edge_ratio_arr)
                                
        else: # isinstance(learner, PearsonCorrelationLearner)
            weights_optimized = learner(time_series[np.newaxis, ...])
            graph_operator = NumbaGraph.GraphOperator_Unsafe(num_nodes)
            weights_vector = graph_operator.Qstar(weights_optimized)/2
            weights_optimized = sparsify_fc(
                edges=weights_vector, 
                expected_edge_ratios=edge_ratios)
        
        end = time.perf_counter()
        print ('Execution time: %f' % (end-start))
        
        if post_processor is not None:
            weights_optimized = post_processor(weights_optimized)
        
        # 5. Save results
        for i, ratio in enumerate(edge_ratios):
            check_filename = f'{output_prefix}-{ratio:.02f}.npy'
            np.save(check_filename, weights_optimized[i])
        return True
        
    except Exception as e:
        # Printing in multiprocessing can cause garbled output; returning error message for unified printing in main process is safer
        return f"Error in {input_path}: {str(e)}"
    
    
    
def compute_dice_coefficient(matrix_a, matrix_b, threshold=1e-8):
    """Compute the Dice coefficient between two binary vectors."""
    a_bin = (matrix_a > threshold).astype(int)
    b_bin = (matrix_b > threshold).astype(int)

    intersection = np.sum(a_bin & b_bin)
    size_a = np.sum(a_bin)
    size_b = np.sum(b_bin)

    if size_a + size_b == 0:
        return 1.0

    return 2 * intersection / (size_a + size_b)



def extract_w_sequence(recorder):
    iters = sorted(recorder.state_list.keys())
    ws = [recorder.state_list[i][0] for i in iters]
    return iters, ws


def compute_dice_sequence(recorder):
    _, ws = extract_w_sequence(recorder)
    return np.array([compute_dice_coefficient(ws[i], ws[i+1]) for i in range(len(ws)-1)])


def compute_nonzero_ratio_sequence(ws, threshold=1e-8):
    """
    calculate edge density of w in each iteration
    """
    return np.array([
        np.count_nonzero(w > threshold) / w.size
        for w in ws
    ])




def compute_log10_rel_errors(ws, w_final):
    norm_final = np.linalg.norm(w_final)
    if norm_final < 1e-30:
        return np.full(len(ws), -300.0)
    return np.array([
        np.log10(max(np.linalg.norm(w - w_final) / norm_final, 1e-300))
        for w in ws
    ])


# ---------- Run a Model with lamda list ----------
def run_model_multi_lam(model_type, n_nodes, init_edges, signal_diff, expected_ratio,
              lambda_list, max_iter=1000, tol=1e-12, record_points=None):
    """
    model_type: 'Entropy', 'Variance'
    """
    
    if record_points is None:
        record_points = list(range(1, max_iter + 1))
    results = {}
    for lam in lambda_list:
        recorder = StateRecorder(record_points)
        if model_type == 'Entropy':
            learner_factory = PrepareLogDegreeEntropyGraphLearner(
                entropy_strength=lam, max_tolerance=tol, max_iteration=max_iter
            )
        else:  # Variance
            learner_factory = PrepareLogDegreeVarianceGraphLearner(
                L2_strength=lam, max_tolerance=tol, max_iteration=max_iter
            )
        learner = learner_factory(n_nodes)
        learner(
            edges=init_edges[np.newaxis, :].copy(),
            signal_diff=signal_diff[np.newaxis, :],
            expected_edge_ratio=np.array([expected_ratio]),
            iteration_record=recorder
        )
        iters, ws = extract_w_sequence(recorder)
        w_final = ws[-1]
        log10_rel = compute_log10_rel_errors(ws, w_final)
        dices = compute_dice_sequence(recorder)
        results[lam] = {
            'iters': iters,
            'ws': ws,
            'w_final': w_final,
            'log10_rel': log10_rel,
            'dices': dices,
        }
    return results

# ---------- Run an algorithm without lamda list ----------
def run_algorithm(alg_type, n_nodes, init_edges, signal_diff, expected_ratio,
              maxiter=1000, tol=1e-12, record_points=None):
    """
    model_type: 'baseline', 'fdgp', 'mm', 'admm'
    """
    a = 1.0
    b = 1.0
    graph_op = NumbaGraph.GraphOperator_Unsafe(n_nodes)
    
    if record_points is None:
        record_points = list(range(1, maxiter + 1))
        
    recorder = StateRecorder(record_points)
    if alg_type.lower() == 'admm':
        admm_learner = PrepareLogDegreeGraphLearner(
            n_nodes, max_tolerance=0.0, max_iterations=maxiter
        )
        admm_learner(
            edges=init_edges[np.newaxis, :].copy(),
            signal_diff=signal_diff[np.newaxis, :],
            expected_edge_ratio=np.array([expected_ratio]),
            iteration_record=recorder
        )
    elif alg_type.lower() == 'fdpg':
        learn_graph_FDPG(
            z=signal_diff, n=n_nodes,
            graph=graph_op,
            a=a, b=b,
            max_iterations=maxiter, maxerror=tol,
            record=recorder,
            verbose=False
        )
    elif alg_type.lower() == 'mm':
        learn_graph_mm(
            z=signal_diff, n=n_nodes,
            graph=graph_op,
            a=a, b=b,
            max_iterations=maxiter, maxerror=tol,
            record=recorder,
            verbose=False
        )
    else:
        learn_graph_log_degrees(
            z=signal_diff, n=n_nodes,
            graph=graph_op,
            a=a, b=b, gamma=0.999999,
            max_iterations=maxiter, maxerror=tol,
            record=recorder,
            w_init=init_edges.copy(),
            verbose=False
        )
    
    iters, ws = extract_w_sequence(recorder)
    w_final = ws[-1]
    log10_rel = compute_log10_rel_errors(ws, w_final)
    dices = compute_dice_sequence(recorder)
    result = {
        'iters': iters,
        'ws': ws,
        'w_final': w_final,
        'log10_rel': log10_rel,
        'dices': dices
    }
    return result

def get_record_points(max_iter: int, num_log_points: int = 100) -> np.ndarray:
    """
    Generate log-uniform sampling points balanced for curve fidelity and memory.
    Storing full intermediate solutions at every iteration is prohibitive
    (each snapshot is O(|V|) across multiple methods × ROI configs),
    so we subsample to ~1% of iterations. Log-spacing ensures each order
    of magnitude is equally represented, capturing both early transients
    and late-stage convergence without over-sampling the tail.
    np.unique removes duplicates from rounding when max_iter < num_log_points.
    """
    early_points = np.arange(1, min(10, max_iter + 1))
    log_points = np.unique(
        np.logspace(0, np.log10(max_iter + 1), num=num_log_points).astype(int)
    )
    return np.unique(np.concatenate([early_points, log_points]))


