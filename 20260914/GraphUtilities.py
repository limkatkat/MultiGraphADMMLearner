from typing import cast
import numpy as np
from numpy.typing import NDArray


# Calculate the difference between signals
def calculate_signal_difference(
    signals: NDArray, 
    output_buffer: NDArray|None=None) -> NDArray:
    
    batch_size: tuple[int, ...] = signals.shape[:-2]
    num_nodes: int = signals.shape[-1]
    num_codes: int = num_nodes * (num_nodes-1) // 2
    
    if output_buffer is None:
        output_buffer = np.empty(shape=(batch_size+(num_codes,)), dtype=signals.dtype)
    output_buffer = cast(NDArray, output_buffer)
    
    li = 0
    for i in range(num_nodes):
        ui = li + num_nodes - (i+1)
        output_buffer[..., li:ui] = np.sum(np.square(signals[..., i, np.newaxis] - signals[..., i+1:]), axis=-2)
        li = ui
    
    return output_buffer


def gsp_compute_theta_bounds(Z, geom_mean=False, is_sorted=False):
    """
    Kalofolias 2019
    Compute the values of parameter theta (controlling sparsity) that should
    be expected to give each sparsity level. Return upper and lower bounds 
    for each sparsity level k = [1, ..., n-1] neighbors/node.

    Parameters
    ----------
    Z : ndarray (n, n)
        Zero-diagonal pairwise distance matrix between nodes.
    geom_mean : bool, optional
        Use geometric mean instead of arithmetic mean? (default: False)
    is_sorted : bool, optional
        Is Z already sorted? (default: False)

    Returns
    -------
    theta_l : ndarray (n-1,)
        Lower bound of theta for each sparsity level.
    theta_u : ndarray (n-1,)
        Upper bound of theta for each sparsity level.
    Z_sorted : ndarray (n, n-1)
        Sorted version of Z (each row without its diagonal element).
    """
    # Ensure Z is a 2D array
    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D matrix.")

    if is_sorted:
        Z_sorted = Z.copy()
    else:
        n = Z.shape[0]
        # For each node, take all distances except the self-distance (diagonal), then sort ascending
        Z_sorted = np.zeros((n, n - 1))
        for i in range(n):
            row_without_diag = np.delete(Z[i], i)
            Z_sorted[i, :] = np.sort(row_without_diag)
    
    m, n_sorted = Z_sorted.shape   # n_sorted = n-1
    k_vals = np.arange(1, n_sorted + 1)   # [1, 2, ..., n-1]
    K_mat = np.tile(k_vals, (m, 1))       # shape (n, n-1)

    B_k = np.cumsum(Z_sorted, axis=1)     # cumulative sum along each row

    # Compute the argument of the mean: 1 / sqrt(k * z^2 - B_k * z)
    # Avoiding division by zero: the terms are theoretically positive for distinct distances.
    eps_tolerance = 1e-10
    arg = 1.0 / np.sqrt(np.maximum(K_mat * Z_sorted**2 - B_k * Z_sorted, eps_tolerance))

    if geom_mean:
        # Geometric mean across rows
        theta_u = np.exp(np.mean(np.log(arg), axis=0))
    else:
        # Arithmetic mean across rows
        theta_u = np.mean(arg, axis=0)

    # Lower bounds are just the upper bounds shifted by one (theta_l[k] = theta_u[k+1])
    theta_l = np.concatenate([theta_u[1:], [0.0]])   # last lower bound set to 0

    return theta_l, theta_u, Z_sorted


def gsp_compute_graph_learning_theta(Z, k, geom_mean=False, is_sorted=False):
    """
    Kalofolias 2019
    Compute a single theta value for graph learning, given desired sparsity level k.

    Parameters
    ----------
    Z : ndarray (n, n)
        Zero-diagonal pairwise distance matrix.
    k : int
        Desired sparsity level (number of neighbors per node), between 1 and n-1.
    geom_mean : bool, optional
        Use geometric mean instead of arithmetic mean? (default: False)
    is_sorted : bool, optional
        Is Z already sorted? (default: False)

    Returns
    -------
    theta : float
        Recommended theta (geometric mean of bounds, or 1.1 * theta_min for k=1).
    theta_min : float
        Lower bound for theta at sparsity k.
    theta_max : float
        Upper bound for theta at sparsity k.
    """
    theta_l, theta_u, _ = gsp_compute_theta_bounds(Z, geom_mean, is_sorted)

    # Convert 1-based k to 0-based index (k=1..n-1)
    if k < 1 or k > len(theta_l):
        raise ValueError(f"k must be between 1 and {len(theta_l)} (number of nodes - 1).")

    idx = k - 1
    theta_min = theta_l[idx]
    theta_max = theta_u[idx]

    if k > 1:
        theta = np.sqrt(theta_min * theta_max)
    else:
        theta = theta_min * 1.1

    return theta, theta_min, theta_max

def gsp_compute_graph_learning_theta_batch(Z, k_list, geom_mean=False, is_sorted=False):
    """
    Kalofolias 2019
    Compute theta values for a single distance matrix, given a LIST of desired 
    sparsity levels.

    Parameters
    ----------
    Z : ndarray (n, n)
        Zero-diagonal pairwise distance matrix.
    k_list : list[int] | NDArray[int]
        A list or 1D array of desired sparsity levels (number of neighbors per node), 
        each between 1 and n-1.
    geom_mean : bool, optional
        Use geometric mean instead of arithmetic mean? (default: False)
    is_sorted : bool, optional
        Is Z already sorted? (default: False)

    Returns
    -------
    thetas : NDArray[float]
        Recommended theta values corresponding to each k in k_list.
    """
    # Get bounds for all k
    theta_l, theta_u, _ = gsp_compute_theta_bounds(Z, geom_mean, is_sorted)
    
    # Convert input to numpy array for vectorized operations
    k_arr = np.asarray(k_list)
    max_k = len(theta_l)
    
    # Boundary check
    if np.any(k_arr < 1) or np.any(k_arr > max_k):
        raise ValueError(f"All k must be between 1 and {max_k} (number of nodes - 1).")
        
    # Convert 1-based k to 0-based index
    idx = k_arr - 1
    
    # Extract bounds corresponding to k
    theta_min_arr = theta_l[idx]
    theta_max_arr = theta_u[idx]
    
    # Default: compute using geometric mean (for k > 1)
    thetas = np.sqrt(theta_min_arr * theta_max_arr)
    
    # Handle k == 1 case
    mask_k1 = (k_arr == 1)
    thetas[mask_k1] = theta_min_arr[mask_k1] * 1.1
        
    return thetas

import numpy as np

def sparsify_fc(
    edges: np.ndarray, 
    expected_edge_ratios: list[float]
) -> np.ndarray:
    """
    Sparsifies the functional connectivity matrix based on edge ratios.

    1. If edges is 1D (or 1xN):
       - Sorts all elements by absolute value descending.
       - For each ratio in expected_edge_ratios, keeps the largest values (setting others to 0).
       - Returns shape (len(ratios), num_edges).

    2. If edges is k x N (k > 1):
       - Checks if k matches the length of expected_edge_ratios.
       - For each row i, sparsifies it using the corresponding ratio i.
       - Returns shape (k, num_edges).

    Parameters
    ----------
    edges : np.ndarray
        Input vector (1D) or matrix (2D).
    expected_edge_ratios : list[float]
        List of edge ratios to keep (e.g., [0.1, 0.2, 0.5]).
        
    Returns
    -------
    NDArray
        Sparsified matrix/vector.
    """
    edges = np.asarray(edges)
    
    # Case 1: 1D Input (or 1xN matrix treated as 1D)
    if edges.ndim == 1 or (edges.ndim == 2 and edges.shape[0] == 1):
        # Flatten to ensure 1D
        flat_edges = edges.flatten()
        num_edges = flat_edges.size
        num_ratios = len(expected_edge_ratios)
        
        # 1. Sort by absolute value descending
        # argsort returns indices for ascending order, so reverse with [::-1]
        sorted_indices = np.argsort(np.abs(flat_edges))[::-1]
        
        # 2. Initialize result array
        # Shape: (num_ratios, num_edges)
        out = np.zeros((num_ratios, num_edges), dtype=flat_edges.dtype)
        
        # 3. Keep top-k elements for each ratio
        for i, ratio in enumerate(expected_edge_ratios):
            num_keep = int(ratio * num_edges)
            
            if num_keep > 0:
                # Get the indices of the top values
                top_k_indices = sorted_indices[:num_keep]
                # Assign values to their original positions (sparse format)
                out[i, top_k_indices] = flat_edges[top_k_indices]
                
        return out

    # Case 2: k x N Input
    elif edges.ndim == 2:
        k, num_edges = edges.shape
        
        # Check if number of rows matches the number of ratios
        if k != len(expected_edge_ratios):
            raise ValueError(
                f"Number of rows in edges ({k}) must match the length of expected_edge_ratios ({len(expected_edge_ratios)})."
            )
        
        # Initialize result array
        out = np.zeros_like(edges)
        
        # Process each row independently
        for i in range(k):
            row = edges[i]
            ratio = expected_edge_ratios[i]
            
            # Sort this row by absolute value descending
            sorted_indices = np.argsort(np.abs(row))[::-1]
            
            num_keep = int(ratio * num_edges)
            
            if num_keep > 0:
                top_k_indices = sorted_indices[:num_keep]
                # Assign values to their original positions in the row
                out[i, top_k_indices] = row[top_k_indices]
        
        return out

    else:
        raise ValueError("Input edges must be a 1D or 2D numpy array.")

