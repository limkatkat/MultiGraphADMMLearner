import numpy as np


def _prox_sum_log(x, gamma_a):
    """prox_{gamma * a * sum(log)}(x) = (x + sqrt(x^2 + 4*gamma_a)) / 2"""
    return 0.5 * (x + np.sqrt(x * x + 4.0 * gamma_a))

def learn_graph_log_degrees(z, n, graph, a=1.0, b=1.0, gamma=0.5,
                            max_iterations=1000, maxerror=1e-5, verbose=True,
                            record=None, w_init=None):
    """
    FBF primal-dual graph learning (supports warm start and iteration recording).

    Parameters
    ----------
    maxerror : float
        Convergence threshold. If set to 0, ignores primal-dual error and forces running for max_iterations steps.
    record : StateRecorder | None
        If provided, records keyframes using the same interface as ADMM (should_record/record).
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    E = n * (n - 1) // 2

    S_op, St_op = graph.P, graph.Pstar

    # ---- Parameters ----
    norm_K = np.sqrt(2.0 * (n - 1))
    mu     = 2.0 * b + norm_K
    step   = gamma / mu
    s_diff = 2.0 * step * z
    tau1   = 1.0 - step * 2.0 * b
    tau2   = a * step

    # ---- Initialization ----
    if w_init is not None:
        w = np.asarray(w_init, dtype=np.float64).ravel()
        if len(w) != E:
            raise ValueError(f"w_init length {len(w)} does not match expected edge count {E}")
    else:
        w = np.zeros(E, dtype=np.float64)

    v = S_op(w)
    sw = v.copy()

    # ---- Convergence Mode ----
    use_error_convergence = (maxerror > 0)
    
    # ---- Support Convergence Auxiliary Variables ----
    w_support_prev = None
    support_eps = 1e-12  # Numerical threshold to determine if an edge belongs to the support, preventing floating-point noise interference

    # ---- Main Loop ----
    it = 0
    rp = np.inf
    rd = np.inf

    while it < max_iterations:
        it += 1

        # Forward primal
        Y_n = tau1 * w - step * St_op(v)
        P_n = np.maximum(Y_n - s_diff, 0.0)

        # Forward dual
        y_n = v + step * sw
        p_n = y_n - tau2 * _prox_sum_log(y_n / tau2, 1.0 / tau2)

        # Backward primal
        Q_n = tau1 * P_n - step * St_op(p_n)

        # Backward dual
        q_n = p_n + step * S_op(P_n)

        # Errors (always calculated for recording and verbose)
        rp = np.linalg.norm(-Y_n + Q_n) / (np.linalg.norm(w) + 1e-30)
        rd = np.linalg.norm(-y_n + q_n) / (np.linalg.norm(v) + 1e-30)

        # Update
        w  = w  - Y_n + Q_n
        v  = v  - y_n + q_n
        sw = S_op(w)

        # ---- Compute Current Support ----
        # Note: w may have tiny negative values during FBF iterations, which will be truncated to 0 in post-processing
        # Using > support_eps instead of > 0 avoids support jitter caused by tiny floating-point numbers
        w_support_curr = w > support_eps

        # ---- Iteration Recording ----
        if record and record.should_record(it):
                d_rec = S_op(w)
                fval_rec = (2.0 * w @ z
                            + b * w @ w
                            - a * np.sum(np.log(np.clip(d_rec, 1e-300, None))))
                record.record(it, (w.copy(), rp, rd, fval_rec))

        # ---- Convergence Check ----
        if use_error_convergence and rp < maxerror and rd < maxerror:
            # New condition: The support of the solution is the same for two consecutive iterations
            if w_support_prev is not None and np.array_equal(w_support_curr, w_support_prev):
                break
        
        # Update previous step support record
        w_support_prev = w_support_curr

    # ---- Post-processing ----
    w[w < 0] = 0.0

    if verbose:
        d = S_op(w)
        fval = 2.0 * w @ z + b * w @ w - a * np.sum(np.log(np.clip(d, 1e-300, None)))
        print(f"iter={it}, fval={fval:.4e}, rp={rp:.2e}, rd={rd:.2e}")

    return w

