import numpy as np
import time



def learn_graph_FDPG(z, n, graph, 
                     a=1.0, b=1.0, 
                     max_iterations=2000, maxerror=1e-6, reset=100, eps_tolerance=1e-5,
                     verbose=False, record=None):
    """
    FDPG (Fast Dual Proximal Gradient) for graph learning with log-degree prior.
    Python equivalent of the MATLAB reference implementation by Saboksayr & Mateos.
    
    Parameters
    ----------
    Z : ndarray
        Vector of length E (upper triangular of pairwise distances),
        or an NxN symmetric matrix.
    n : int
        Number of nodes.
    graph : object
        Graph operator instance. Must provide the following methods:
        - P(w):     Edge weight vector (E,) -> Degree vector (N,)
        - Pstar(d): Degree vector (N,)   -> Edge weight vector (E,)
        - Q(w):     Edge weight vector (E,) -> N x N symmetric adjacency matrix (diagonal is 0)
        - Qstar(W): NxN symmetric matrix -> Strict upper triangular vector (E,)
    a : float
        Log prior constant (bigger a -> bigger weights).
    b : float
        ||W||_F^2 prior constant (bigger b -> more dense W).
    maxiter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance (maximum error).
    reset : int
        Nesterov momentum restart interval.
    thr : float
        Float threshold. Edges with weights below this value will be set to 0. 
        Also used as a judgment threshold for support stability.
    verbose : bool
        Whether to print iteration information.
    record : object, optional
        External iteration recorder. Must provide:
        - should_record: Attribute or method to determine if current iteration needs recording.
        - record(k, data_tuple): Record data for k-th iteration.
        data_tuple format: (w, rel_norm_w, rel_norm_lam, obj_val, dual_obj_val)
    
    Returns
    -------
    W_hat : ndarray, shape (N, N)
        Learned weighted adjacency matrix (symmetric, zero diagonal).
    stat : dict
        Internal statistics, containing 'obj_val', 'dual_obj_val', 'time', 'Lambda'.
    """
    np.random.seed(12345)
    
    # --- Parse Z ---
    if z.ndim == 1 or (z.ndim == 2 and min(z.shape) == 1):
        z = np.asarray(z, dtype=np.float64).ravel()
    else:
        z = np.asarray(z, dtype=np.float64)
        z = graph.Qstar(z)
    
    z = z.ravel()
    E = n * (n - 1) // 2
    if len(z) != E:
        raise ValueError(f"Input data length {len(z)} does not match number of edges {E} corresponding to node count {n}")
    
    # --- Operators via graph operator instance ---
    P_op = graph.P
    Pstar_op = graph.Pstar
    
    L = (n - 1) / b
    prox_g = lambda x: (x + np.sqrt(x**2 + 4 * a * L)) / 2.0
    
    f_eval = lambda w: b * np.linalg.norm(w)**2 + 2.0 * w.T @ z
    g_eval = lambda w: -a * np.sum(np.log(P_op(w)))
    
    # Fenchel conjugates for dual objective
    F_eval = lambda x: (Pstar_op(x)).T @ np.maximum(0.0, (Pstar_op(x) - 2 * z) / (2 * b)) \
                      - 2.0 * np.maximum(0.0, (Pstar_op(x) - 2 * z) / (2 * b)).T @ z \
                      - b * np.linalg.norm(np.maximum(0.0, (Pstar_op(x) - 2 * z) / (2 * b)))**2
    G_eval = lambda x: a * np.sum(np.log(a / x)) - a * n
    
    # --- FDPG algorithm ---
    stat = {
        'obj_val': np.full(max_iterations, np.nan),
        'dual_obj_val': np.full(max_iterations, np.nan),
        'time': float
    }
    
    omega_k = np.random.rand(n)
    lambda_k = omega_k.copy()
    w_hat_k = np.maximum(np.finfo(float).eps, (Pstar_op(lambda_k) - 2 * z) / (2 * b))
    tk = 1.0
    
    # ---- Support convergence auxiliary variables ----
    w_support_prev = None
    
    tic = time.time()
    
    for k in range(max_iterations):
        # 1. Primal variable update from extrapolation point
        w_bar = np.maximum(np.finfo(float).eps, (Pstar_op(omega_k) - 2 * z) / (2 * b))
        
        # 2. Proximal step
        u = prox_g(P_op(w_bar) - L * omega_k)
        
        # 3. Dual update
        lambda_K = omega_k - (1.0 / L) * (P_op(w_bar) - u)
        
        # 4. Nesterov acceleration
        tK = (1 + np.sqrt(1 + 4 * tk**2)) / 2
        omega_K = lambda_K + ((tk - 1) / tK) * (lambda_K - lambda_k)
        
        # 5. Recover primal from new dual
        w_hat_K = np.maximum(np.finfo(float).eps, (Pstar_op(lambda_K) - 2 * z) / (2 * b))
        
        # 6. Convergence metrics (relative norms)
        rel_norm_w = np.linalg.norm(w_hat_K - w_hat_k) / (np.linalg.norm(w_hat_k) + 1e-30)
        rel_norm_lam = np.linalg.norm(lambda_K - lambda_k) / (np.linalg.norm(lambda_k) + 1e-30)
        
        # ---- Compute current support and stability ----
        # Use float threshold to determine if an edge belongs to the support, consistent with final post-processing truncation standard
        w_support_curr = w_hat_K > eps_tolerance
        support_is_stable = (w_support_prev is not None and np.array_equal(w_support_curr, w_support_prev))
        
        # 7. Compute objectives
        obj_val = g_eval(w_hat_K) + f_eval(w_hat_K)
        dual_obj_val = F_eval(lambda_K) + G_eval(lambda_K)
        
        # 8. Record to internal stat
        stat['obj_val'][k] = obj_val
        stat['dual_obj_val'][k] = dual_obj_val
        
        # 9. External record (adapt iteration recorder)
        if record and record.should_record(k):
            record_data = (w_hat_K.copy(), rel_norm_w, rel_norm_lam, obj_val, dual_obj_val)
            record.record(k, record_data)
        
        # 10. Verbose output
        if verbose:
            nnz = np.sum(w_support_curr)
            print(f"iteration {k+1:4d}: {rel_norm_w:6.4e} *** {rel_norm_lam:6.4e} *** {obj_val:6.3e} *** nnz={nnz}")
        
        # 11. Restart (O'Donoghue & Candès, 2015) - Fix restart logic
        if (k + 1) % reset == 0:
            tk = 1.0
            lambda_k = lambda_K.copy()
            omega_k = lambda_K.copy()  # Extrapolation point rolls back to current point upon restart
            w_hat_k = w_hat_K.copy()
        else:
            # 12. Normal update state
            tk = tK
            lambda_k = lambda_K.copy()
            omega_k = omega_K.copy()
            w_hat_k = w_hat_K.copy()
        
        # 13. Convergence check
        # Condition: dual and primal relative errors satisfied AND support stable
        if rel_norm_w < maxerror and rel_norm_lam < maxerror and support_is_stable:
            break
        
        # Update previous step support record
        w_support_prev = w_support_curr
    
    stat['time'] = time.time() - tic
    stat['Lambda'] = lambda_K.copy()
    
    if verbose:
        print("Done!")
    
    # Remove unused NaN entries
    stat['obj_val'] = stat['obj_val'][:k+1]
    stat['dual_obj_val'] = stat['dual_obj_val'][:k+1]
    
    # --- Thresholding the weighted learned graph ---
    w_hat_K = w_hat_K.copy()
    w_hat_K[w_hat_K < eps_tolerance] = 0.0
    W_hat = graph.Q(w_hat_K)
    
    return W_hat, stat
