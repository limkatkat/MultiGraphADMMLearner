import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from GraphUtilities import (
    gsp_compute_graph_learning_theta_batch,
    calculate_signal_difference,
    sparsify_fc)

import NumbaGraph
from learn_graph_utility import PrepareLogDegreeGraphLearner


os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def run_admm_only():
    ts_dir = Path('Schaefer')
    roi_counts = [100, 500, 1000]
    edge_ratios = [0.04, 0.08, 0.16, 0.32, 0.64]
    float_tol = 1e-2

    maxiter = 10000

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------
    bar = "=" * 78
    print(bar)
    print(" ADMM-only benchmark".center(78))
    print(bar)
    print(f"  ROIs        : {roi_counts}")
    print(f"  Edge ratios : {edge_ratios}")
    print(f"  Max iters   : {maxiter}")
    print(f"  Tolerance   : {float_tol}")
    print(bar)

    # ---------------------------------------------------------------
    # Collect timings for a final summary table
    # ---------------------------------------------------------------
    timings = {}   # timings[(n_rois, edge_ratio)] = elapsed seconds
    overall_start = time.perf_counter()

    for n_rois in roi_counts:
        input_filename = ts_dir / f'{n_rois}.csv'
        time_series = pd.read_csv(input_filename, delimiter=',')
        time_series = time_series[list(time_series.columns)].to_numpy()

        ts_norms = np.linalg.norm(time_series, axis=0)
        non_zero_cols = ts_norms > float_tol
        valid_indices = np.where(non_zero_cols)[0]
        num_nodes = len(valid_indices)

        graph_op = NumbaGraph.GraphOperator_Unsafe(num_nodes)
        time_series = time_series[:, valid_indices]
        time_series /= ts_norms[valid_indices]
        signal_diff_vec = calculate_signal_difference(time_series)
        dist_matrix = graph_op.Q(signal_diff_vec)

        print(f"\n[ROI = {n_rois}]  nodes = {num_nodes}")
        print("-" * 78)
        print(f"  {'edge_ratio':>10} | {'edges':>8} | {'time (s)':>12}")
        print("-" * 78)

        for edge_ratio in edge_ratios:
            expected_edges = max(int((num_nodes - 1) * edge_ratio), 1)
            theta_vals = gsp_compute_graph_learning_theta_batch(
                dist_matrix.squeeze(), [expected_edges],
                geom_mean=False, is_sorted=False
            )

            weighted_signal_diff = theta_vals[0] * signal_diff_vec

            t_start = time.perf_counter()
            init_edge_weights = np.exp(-weighted_signal_diff / 2)
            init_edge_weights = sparsify_fc(init_edge_weights, [edge_ratio])

            admm_learner = PrepareLogDegreeGraphLearner(
                num_nodes, max_tolerance=float_tol, max_iterations=maxiter
            )
            admm_learner(
                edges=init_edge_weights[np.newaxis, :].copy(),
                signal_diff=weighted_signal_diff[np.newaxis, :],
                expected_edge_ratio=np.array([edge_ratio]),
                show_error_interval=1
            )
            t_end = time.perf_counter()

            elapsed = t_end - t_start
            timings[(n_rois, edge_ratio)] = elapsed
            print(f"  {edge_ratio:>10.2f} | {expected_edges:>8d} | {elapsed:>12.3f}")

        print("-" * 78)

    overall_end = time.perf_counter()

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    print("\n" + bar)
    print(" Summary".center(78))
    print(bar)

    # Header row
    header = f"  {'ROI':>6} | " + " | ".join(f"{r:>8.2f}" for r in edge_ratios) + f" | {'row total':>10}"
    print(header)
    print("-" * len(header))

    for n_rois in roi_counts:
        row_total = 0.0
        cells = []
        for edge_ratio in edge_ratios:
            t = timings.get((n_rois, edge_ratio), float('nan'))
            row_total += t if not np.isnan(t) else 0.0
            cells.append(f"{t:>8.3f}")
        print(f"  {n_rois:>6} | " + " | ".join(cells) + f" | {row_total:>10.3f}")

    print("-" * len(header))

    # Column totals
    col_totals = []
    for edge_ratio in edge_ratios:
        s = sum(timings.get((n, edge_ratio), 0.0) for n in roi_counts)
        col_totals.append(f"{s:>8.3f}")
    grand_total = sum(timings.values())
    print(f"  {'total':>6} | " + " | ".join(col_totals) + f" | {grand_total:>10.3f}")

    print(bar)
    print(f"  Grand total elapsed : {overall_end - overall_start:.3f} s")
    print(bar)


if __name__ == '__main__':
    run_admm_only()
    
    