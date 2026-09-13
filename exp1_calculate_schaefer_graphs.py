import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from GraphUtilities import gsp_compute_graph_learning_theta_batch, calculate_signal_difference, sparsify_fc
import NumbaGraph

from learn_graph_utility import (
    compute_log10_rel_errors,
    compute_nonzero_ratio_sequence,
    run_algorithm
)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def load_or_compute_results(ts_dir, roi_counts, edge_ratios, float_tol, cache_dir):
    """
    Load cached results if available; otherwise run full computation and cache.
    Cache is stored per (ROI, ratio) to allow incremental recomputation.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    all_results = {}

    for n_rois in roi_counts:
        all_results[n_rois] = []

        # -- Load Data --
        input_filename = ts_dir / f'{n_rois}.csv'
        time_series = pd.read_csv(input_filename, delimiter=',')
        roi_col_names = [col for col in time_series.columns]
        time_series = time_series[roi_col_names].to_numpy()

        ts_norms = np.linalg.norm(time_series, axis=0)
        non_zero_cols = ts_norms > float_tol
        valid_indices = np.where(non_zero_cols)[0]
        num_nodes = len(valid_indices)

        graph_op = NumbaGraph.GraphOperator_Unsafe(num_nodes)
        time_series = time_series[:, valid_indices]
        time_series /= ts_norms[valid_indices]
        signal_diff_vec = calculate_signal_difference(time_series)
        dist_matrix = graph_op.Q(signal_diff_vec)

        for edge_ratio in edge_ratios:
            cache_file = cache_dir / f'ROI{n_rois}_ratio{edge_ratio}.pkl'

            # Try loading from cache first
            if cache_file.exists():
                print(f"[CACHE HIT] ROI={n_rois}, ratio={edge_ratio} -> loading from {cache_file.name}")
                with open(cache_file, 'rb') as f:
                    config_results = pickle.load(f)
                all_results[n_rois].append(config_results)
                continue

            # No cache: full computation
            print(f"\n[COMPUTE] ROI={n_rois}, nodes={num_nodes}, ratio={edge_ratio}")

            expected_edges = max(int((num_nodes - 1) * edge_ratio), 1)
            theta_vals = gsp_compute_graph_learning_theta_batch(
                dist_matrix.squeeze(), [expected_edges],
                geom_mean=False, is_sorted=False
            )
            weighted_signal_diff = theta_vals[0] * signal_diff_vec

            init_edge_weights = np.exp(-weighted_signal_diff / 2)
            init_edge_weights = sparsify_fc(init_edge_weights, [edge_ratio])

            # 1. Baseline
            print("  Running Baseline ...")
            res_baseline = run_algorithm('baseline', num_nodes, init_edge_weights, weighted_signal_diff,
                                         edge_ratio, 10000, float_tol, None)
            baseline_max_iter = res_baseline['iters'][-1]
            print(f"    Baseline converged at {baseline_max_iter} iterations.")

            # Log-uniform sampling
            record_points = np.unique(
                np.logspace(0, np.log10(baseline_max_iter + 1), num=100).astype(int)
            )

            # 2. FDPG
            print(f"  Running FDPG (max {baseline_max_iter} iters) ...")
            res_fdpg = run_algorithm('fdpg', num_nodes, init_edge_weights, weighted_signal_diff,
                                     edge_ratio, baseline_max_iter, 0.0, record_points)

            # 3. MM
            print(f"  Running MM (max {baseline_max_iter} iters) ...")
            res_mm = run_algorithm('mm', num_nodes, init_edge_weights, weighted_signal_diff,
                                   edge_ratio, baseline_max_iter, 0.0, record_points)

            # 4. ADMM
            print(f"  Running ADMM (max {baseline_max_iter} iters) ...")
            res_admm = run_algorithm('admm', num_nodes, init_edge_weights, weighted_signal_diff,
                                     edge_ratio, baseline_max_iter, 0.0, record_points)

            # ---------- Flatten results and compute additional metrics ----------
            config_results = {'ratio': edge_ratio}
            w_baseline_final = res_baseline['w_final']

            method_results = {
                'baseline': res_baseline,
                'fdpg': res_fdpg,
                'mm': res_mm,
                'admm': res_admm
            }

            for method_name, method_res in method_results.items():
                prefix = method_name.lower()
                config_results[f'{prefix}_iters'] = method_res['iters']
                config_results[f'{prefix}_dice'] = method_res['dices']
                config_results[f'{prefix}_actual_iters'] = method_res['iters'][-1]
                config_results[f'{prefix}_rel_err_arr'] = compute_log10_rel_errors(method_res['ws'], w_baseline_final)
                config_results[f'{prefix}_nn_ratio'] = compute_nonzero_ratio_sequence(method_res['ws'])
                config_results[f'{prefix}_corr_gt'] = np.zeros(len(method_res['ws']))

            config_results['f_baseline_final'] = 0.0

            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(config_results, f)
            print(f"  [CACHED] Saved to {cache_file.name}")

            all_results[n_rois].append(config_results)

    return all_results


def create_convergence_summary_tables(all_results, roi_counts, edge_ratios, output_dir):
    """
    Generates one LaTeX table per edge ratio.
    Layout:
      - 10 rows   : thresholds 10^{-k}, k = 1..10
      - 3 groups  : N = 100 / 500 / 1000
      - 4 subcols : methods, denoted by circled numbers (see caption)
    Cell: first recorded iteration at which log10 relative error reaches -k.
    '--' = level not attained within the recorded budget.
    """
    print(f"\nGenerating LaTeX convergence tables (one per edge ratio)...")
    ks = range(1, 11)
    methods = ['Baseline', 'MM', 'FDPG', 'ADMM']
    n_m = len(methods)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for ratio in edge_ratios:
        
        per_roi = {}
        for n_rois in roi_counts:
            if n_rois not in all_results:
                continue
            res_map = {r['ratio']: r for r in all_results[n_rois]}
            if ratio not in res_map:
                continue
            res = res_map[ratio]
            cell = {}
            for k in ks:
                for method in methods:
                    iters = np.array(res[f'{method.lower()}_iters'])
                    rel_errs = np.array(res[f'{method.lower()}_rel_err_arr'])
                    min_len = min(len(iters), len(rel_errs))
                    iters, rel_errs = iters[:min_len], rel_errs[:min_len]
                    indices = np.where(rel_errs <= -k)[0]
                    cell[(k, method)] = float(iters[indices[0]]) if len(indices) > 0 else np.nan
            per_roi[n_rois] = cell

        if not per_roi:
            continue

        # ---------- 生成 LaTeX ----------
        col_spec = 'c' + 'r' * (n_m * len(per_roi))

        lines = []
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        caption = (
            r'\caption{Number of iterations required to first reach a relative error of '
            r'$10^{-k}$ (w.r.t.\ the converged reference solution) for target density $r=%g$.}'
            % (ratio)
        )
        lines.append(caption)
        lines.append(r'\label{tab:conv_r%d}' % int(round(ratio * 100)))
        lines.append(r'\footnotesize')
        lines.append(r'\setlength{\tabcolsep}{4pt}')
        lines.append(r'\begin{tabular}{' + col_spec + '}')
        lines.append(r'\toprule')

        header1 = ['']
        for n_rois in per_roi.keys():
            header1.append(r'\multicolumn{%d}{c}{$N=%d$}' % (n_m, n_rois))
        lines.append(' & '.join(header1) + r' \\')

        cm, start = [], 2
        for _ in per_roi.keys():
            cm.append(r'\cmidrule(lr){%d-%d}' % (start, start + n_m - 1))
            start += n_m
        lines.append(' '.join(cm))

        header2 = ['']
        for _ in per_roi.keys():
            for method in methods:
                header2.append(method)
        lines.append(' & '.join(header2) + r' \\')
        lines.append(r'\midrule')

        for k in ks:
            row = ['$10^{-%d}$' % k]
            for n_rois, cell in per_roi.items():
                for method in methods:
                    v = cell[(k, method)]
                    row.append('--' if np.isnan(v) else f'{v:.0f}')
            lines.append(' & '.join(row) + r' \\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')

        latex_code = '\n'.join(lines)
        out_file = output_dir / f'convergence_table_ratio{ratio}.tex'
        with open(out_file, 'w') as f:
            f.write(latex_code + '\n')
        print(f"[OK] LaTeX table (r={ratio}) saved to: {out_file}")
        print("\n" + latex_code + "\n")


def main():
    ts_dir = Path('Schaefer')
    roi_counts = [100, 500, 1000]
    edge_ratios = [0.04, 0.08, 0.16, 0.32, 0.64]
    float_tol = 1e-10

    conv_data_dir = Path('Convergence_Data')
    conv_data_dir.mkdir(exist_ok=True)
    cache_dir = Path('Cache_exp1')

    # ===============================================================
    # Data Computation Section (with caching)
    # ===============================================================
    all_results = load_or_compute_results(ts_dir, roi_counts, edge_ratios, float_tol, cache_dir)

    # ===============================================================
    # Global Plot Settings
    # ===============================================================
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'legend.title_fontsize': 20,
        'figure.titlesize': 26,
        'lines.linewidth': 3.0,
    })

    method_styles = {
        'Baseline': {'color': "#0c0c0c", 'linestyle': '-', 'lw': 2.5, 'label': 'Log-degree'},
        'MM':       {'color': '#2ca02c', 'linestyle': '-', 'lw': 2.5, 'label': 'MM'},
        'FDPG':     {'color': "#1d0be2", 'linestyle': '--', 'lw': 2.5, 'label': 'FDPG'},
        'ADMM':     {'color': '#d62728', 'linestyle': '--', 'lw': 2.5, 'label': 'ADMM'},
    }
    plot_order = ['Baseline', 'MM', 'FDPG', 'ADMM']
    method_map = {'Baseline': 'baseline', 'MM': 'mm', 'FDPG': 'fdpg', 'ADMM': 'admm'}

    legend_elements = [
        plt.Line2D([0], [0], **method_styles['ADMM']),
        plt.Line2D([0], [0], **method_styles['FDPG']),
        plt.Line2D([0], [0], **method_styles['Baseline']),
        plt.Line2D([0], [0], **method_styles['MM'])
    ]

    # ===============================================================
    # Helper: Draw a single supplementary-style figure (3 rows x 5 cols)
    # ===============================================================
    def _draw_supp_figure_proper(metric_suffix, ylabel, filename_stem):
        fig, axes = plt.subplots(3, 5, figsize=(24, 15))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95),
                   fontsize=18, title='Algorithms', title_fontsize=18)

        ylim_map = {
            'dice':        (0.9, 1.01),
            'nn_ratio':    (-0.1, 1.01),
            'rel_err_arr': (-10, 1),
        }

        for row_idx, n_rois in enumerate(roi_counts):
            for col_idx, res in enumerate(all_results[n_rois]):
                ax = axes[row_idx, col_idx]
                
                for method in plot_order:
                    prefix = method_map[method]
                    x = res[f'{prefix}_iters']
                    y = res[f'{prefix}_{metric_suffix}']
                    
                    if metric_suffix == 'dice':
                        x = x[1:]
                    
                    ax.plot(x, y, **method_styles[method])

                # ---------- Dice stable point ----------
                if metric_suffix == 'dice':
                    for method in plot_order:
                        prefix = method_map[method]
                        raw_x = np.array(res[f'{prefix}_iters'])
                        raw_y = np.array(res[f'{prefix}_dice'])
                        
                        plot_x = raw_x[1:]
                        plot_y = raw_y[1:]
                        
                        is_stable_mask = plot_y >= (1.0 - 1e-5)
                        unstable_indices = np.where(~is_stable_mask)[0]
                        
                        if len(unstable_indices) > 0:
                            last_unstable_idx = unstable_indices[-1]
                            first_stable_idx = last_unstable_idx + 1
                        else:
                            first_stable_idx = 0

                        if first_stable_idx < len(plot_y) and np.all(is_stable_mask[first_stable_idx:]):
                            stable_iter = plot_x[first_stable_idx]
                            color = method_styles[method]['color']
                            
                            ax.plot(stable_iter, 0, marker='s', color=color, 
                                    transform=ax.get_xaxis_transform(), markersize=8, 
                                    markeredgecolor='white', markeredgewidth=0.5, clip_on=False, zorder=10)
                # ------------------------------------------

                ax.set_xlabel('Iteration', fontsize=18)
                ax.set_ylabel(ylabel, fontsize=18)

                # lim of y
                if metric_suffix in ylim_map:
                    if metric_suffix == 'rel_err_arr':
                        
                        min_admm_err = np.min(res['admm_rel_err_arr'])
                        y_low = -10
                        if min_admm_err < -10:
                            y_low = int(np.ceil(min_admm_err)) - 1
                        ax.set_ylim(y_low, 1)
                    else:
                        ax.set_ylim(*ylim_map[metric_suffix])

                if metric_suffix == 'rel_err_arr':
                    admm_x = np.asarray(res['admm_rel_err_arr'])
                    admm_iters = res['admm_iters']
                    
                    # minimal error of ADMM
                    admm_min_err = np.min(admm_x)
                    ax.axhline(y=admm_min_err, color='gray', linestyle='--', 
                               lw=1.5, alpha=0.7, zorder=0)
                    
                    ax.text(-0.08, admm_min_err, f'{admm_min_err:.2f}',
                            transform=ax.get_yaxis_transform(),
                            va='center', ha='right', fontsize=12, color='gray',
                            fontweight='bold', zorder=5)
                    
                    for k in range(1, 11):
                        threshold = -k
                        reach_indices = np.where(admm_x <= threshold)[0]
                        if len(reach_indices) == 0:
                            continue
                        idx = reach_indices[0]
                        ax.annotate(
                            str(admm_iters[idx]),
                            xy=(admm_iters[idx], admm_x[idx]),
                            xytext=(6, 6), textcoords='offset points',
                            fontsize=14, color=method_styles['ADMM']['color'],
                            ha='left', va='bottom',
                            arrowprops=dict(arrowstyle='->', color=method_styles['ADMM']['color'],
                                            lw=0.8, shrinkA=0, shrinkB=2),
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                      edgecolor=method_styles['ADMM']['color'],
                                      alpha=0.85, linewidth=0.6),
                        )

                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        fig.text(0.5, 0.02, 'Rows: ROI number; Columns: Edge ratios',
                 ha='center', va='center', fontsize=22, fontweight='bold')
        plt.tight_layout(rect=(0, 0.04, 0.95, 0.95))
        save_path = conv_data_dir / f'{filename_stem}_supplement_large_font.pdf'
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] {ylabel} (Supplement) saved: {save_path}")

    _draw_supp_figure_proper('dice', 'Dice Coefficient', 'dice_coefficient_allROI')
    _draw_supp_figure_proper('rel_err_arr', 'Relative Error', 'relative_error_allROI')
    _draw_supp_figure_proper('nn_ratio', 'Edge Densities', 'nonzero_edge_ratio_allROI')

    # ===============================================================
    # Figure 4 (Main Text): ROI=500
    # ===============================================================
    target_roi = 500
    if target_roi in all_results:
        print(f"\nGenerating main text figure (ROI={target_roi}, 3x5)...")
        target_data = all_results[target_roi]

        fig_paper, axes_paper = plt.subplots(3, 5, figsize=(24, 15))
        plt.subplots_adjust(hspace=0.35, wspace=0.3)
        fig_paper.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95),
                         fontsize=18, title='Algorithms', title_fontsize=18)

        row_configs = [
            ('dice',        'Dice Coefficient',              True),
            ('nn_ratio',    'Edge Densities', False),
            ('rel_err_arr', 'Log₁₀ Relative Error',          False),
        ]

        for col_idx, res in enumerate(target_data):
            for row_idx, (metric_suffix, ylabel, skip_first) in enumerate(row_configs):
                ax = axes_paper[row_idx, col_idx]
                
                for method in plot_order:
                    prefix = method_map[method]
                    x = np.array(res[f'{prefix}_iters'])
                    y = np.array(res[f'{prefix}_{metric_suffix}'])
                    
                    if skip_first and len(x) > 1:
                        x = x[1:]
                        y = y[1:] if len(y) == len(x) + 1 else y
                        
                    ax.plot(x, y, **method_styles[method])

                if metric_suffix == 'dice':
                    for method in plot_order:
                        prefix = method_map[method]
                        raw_x = np.array(res[f'{prefix}_iters'])
                        raw_y = np.array(res[f'{prefix}_dice'])
                        
                        if skip_first and len(raw_x) > 1:
                            plot_x = raw_x[1:]
                            plot_y = raw_y[1:]
                        else:
                            plot_x = raw_x
                            plot_y = raw_y
                        
                        is_stable_mask = plot_y >= (1.0 - 1e-5)
                        unstable_indices = np.where(~is_stable_mask)[0]
                        
                        if len(unstable_indices) > 0:
                            last_unstable_idx = unstable_indices[-1]
                            first_stable_idx = last_unstable_idx + 1
                        else:
                            first_stable_idx = 0
                            
                        if first_stable_idx < len(plot_y) and np.all(is_stable_mask[first_stable_idx:]):
                            stable_iter = plot_x[first_stable_idx]
                            color = method_styles[method]['color']
                            
                            ax.plot(stable_iter, 0, marker='s', color=color, 
                                    transform=ax.get_xaxis_transform(), markersize=8, 
                                    markeredgecolor='white', markeredgewidth=0.5, clip_on=False, zorder=10)
                # ------------------------------------------

                ax.set_xlabel('Iteration', fontsize=18)
                ax.set_ylabel(ylabel, fontsize=18)
                
                if metric_suffix == 'dice':
                    ax.set_ylim(0.95, 1.05)
                elif metric_suffix == 'rel_err_arr':
                    min_admm_err = np.min(res['admm_rel_err_arr'])
                    y_low = -10
                    if min_admm_err < -10:
                        y_low = int(np.ceil(min_admm_err)) - 1
                    ax.set_ylim(y_low, 1)
                    
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

                if metric_suffix == 'rel_err_arr':
                    admm_y = np.array(res['admm_rel_err_arr'])
                    admm_x = np.array(res['admm_iters'])
                    
                    admm_min_err = np.min(admm_y)
                    ax.axhline(y=admm_min_err, color='gray', linestyle='--', 
                               lw=1.5, alpha=0.7, zorder=0)
                    
                    ax.text(-0.08, admm_min_err, f'{admm_min_err:.2f}',
                            transform=ax.get_yaxis_transform(),
                            va='center', ha='right', fontsize=12, color='gray',
                            fontweight='bold', zorder=5)
                    
                    for k in range(1, 11):
                        threshold = -k
                        reach_indices = np.where(admm_y <= threshold)[0]
                        if len(reach_indices) == 0:
                            continue
                        idx = reach_indices[0]
                        ax.annotate(
                            str(admm_x[idx]),
                            xy=(admm_x[idx], admm_y[idx]),
                            xytext=(6, 6), textcoords='offset points',
                            fontsize=14, color=method_styles['ADMM']['color'],
                            ha='left', va='bottom',
                            arrowprops=dict(
                                arrowstyle='->', color=method_styles['ADMM']['color'],
                                lw=0.8, shrinkA=0, shrinkB=2
                            ),
                            bbox=dict(
                                boxstyle='round,pad=0.15', facecolor='white',
                                edgecolor=method_styles['ADMM']['color'],
                                alpha=0.85, linewidth=0.6
                            ),
                        )

        fig_paper.text(0.5, 0.02,
                       f'ROI={target_roi}; Columns correspond to Edge ratios: {", ".join(map(str, edge_ratios))}',
                       ha='center', va='center', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=(0, 0.04, 0.95, 0.95))
        filename_paper = conv_data_dir / f'convergence_ROI{target_roi}_main_text_3row.pdf'
        plt.savefig(filename_paper, dpi=300, bbox_inches='tight', format="pdf")
        plt.close(fig_paper)
        print(f"[OK] Main text figure saved: {filename_paper}")
    else:
        print(f"\n[WARN] ROI={target_roi} not found in results. Skipping main text figure.")

    # ===============================================================
    # Generate Table: Iterations to reach 10^-k
    # ===============================================================
    table_path = conv_data_dir / 'convergence_iterations_summary.tex'
    create_convergence_summary_tables(all_results, roi_counts, edge_ratios, table_path)



if __name__ == '__main__':
    main()
