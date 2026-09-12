import numpy as np

import NumbaGraph
from StateRecorder import StateRecorder

def learn_graph_mm(
    z: np.ndarray, 
    n: int, 
    graph: NumbaGraph.GraphOperator_Unsafe,
    a: float, 
    b: float, 
    max_iterations: int=1000, 
    maxerror: float=1e-8, 
    eps_tolerance: float=1e-12, 
    verbose: bool=False,
    record: StateRecorder|None=None
):
    
    total_edges = n * (n - 1) // 2
    w0 = np.ones(total_edges, dtype=np.float64)

    f0 = None
    prev_nonzero_mask = None
    support_is_stable = False

    it = 0
    while it < max_iterations:
        it += 1

        degree = graph.P(w0)
        if np.any(degree <= 0):
            degree = np.maximum(degree, eps_tolerance)

        inv_degree = 1.0 / degree
        c = a * w0 * graph.Pstar(inv_degree)

        sqrt_term = np.sqrt(4 * z * z + 8 * b * c)
        w_new = (-2 * z + sqrt_term) / (4 * b)

        w_new = np.maximum(w_new, 0.0)

        # ---- Calculate objective function value ----
        deg_new = graph.P(w_new)
        if np.any(deg_new <= 0):
            deg_new = np.maximum(deg_new, eps_tolerance)
        term1 = 2.0 * np.dot(w_new, z)
        term2_new = -a * np.sum(np.log(deg_new))
        term3 = b * np.dot(w_new, w_new)
        f_new = term1 + term2_new + term3

        # ---- Calculate relative change ----
        f_rel_change = abs(f0 - f_new) / (abs(f0) + 1e-30) if f0 is not None else np.inf
        w_rel_change = np.linalg.norm(w_new - w0) / (np.linalg.norm(w0) + 1e-30)

        # ---- Support stability check ----
        current_nonzero_mask = w_new > eps_tolerance
        if prev_nonzero_mask is not None:
            support_is_stable = np.array_equal(current_nonzero_mask, prev_nonzero_mask)

        # ---- Iteration recording ----
        if record and record.should_record(it):
            record.record(it, (w_new.copy(), w_rel_change, f_rel_change, f_new))
            
        # ---- Convergence check: Support stable + Function value close + Vector close ----
        if (f0 is not None
                and support_is_stable
                and w_rel_change < maxerror
                and f_rel_change < maxerror):
            if verbose:
                print(f"Iter {it}: Support stable and function/vector errors satisfied, exiting "
                      f"(w_rel={w_rel_change:.2e}, f_rel={f_rel_change:.2e})")
            break

        # ---- Update previous step record ----
        prev_nonzero_mask = current_nonzero_mask
        f0 = f_new
        w0 = w_new

        if verbose and (it % 100 == 0):
            nnz = np.sum(current_nonzero_mask)
            print(f"Iter {it}: f={f_new:.6e}, w_rel={w_rel_change:.2e}, "
                  f"f_rel={f_rel_change:.2e}, nnz={nnz}/{total_edges}")

    return w0
