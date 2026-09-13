import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GraphUtilities import (
    gsp_compute_graph_learning_theta_batch,
    sparsify_fc,
    calculate_signal_difference,
)
import NumbaGraph
from learn_graph_utility import get_record_points, run_model_multi_lam

# ---------- Environment Variables ----------
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


# ---------- Plotting Functions ----------
def plot_all_figures(all_results, model_name, output_dir, target_roi=500,
                     roi_list=[100, 500, 1000],
                     ratio_list=[0.04, 0.08, 0.16, 0.32, 0.64],
                     lambda_list=[0.02, 0.04, 0.06, 0.08]):
    """
    all_results: dict with keys ROI, each value is list of dict per ratio
    model_name: 'Entropy' or 'Variance' (used in filenames)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # ----- Color Scheme -----
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(lambda_list)))  # type: ignore
    lambda_colors = {lam: cmap[i] for i, lam in enumerate(lambda_list)}
    lambda_labels = {lam: f'λ₂={lam}' for lam in lambda_list}

    # ENLARGED Global font settings
    plt.rcParams.update({
        'font.size': 22,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 20,
        'figure.titlesize': 26
    })

    LOG_FLOOR = -10

    # ---------- Supplementary: Relative Error Overview (3x5) ----------
    print(f"Plotting: {model_name} supplementary relative error overview")
    fig1, axes1 = plt.subplots(3, 5, figsize=(24, 14))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    legend_elems = []
    for lam in lambda_list:
        legend_elems.append(plt.Line2D([0], [0], color=lambda_colors[lam], lw=1.5,  # type: ignore
                                       label=lambda_labels[lam]))
    fig1.legend(handles=legend_elems, loc='upper right',
                bbox_to_anchor=(0.99, 0.97), fontsize=18)

    for row_idx, roi in enumerate(roi_list):
        for col_idx, res in enumerate(all_results[roi]):
            ax = axes1[row_idx, col_idx]
            for lam in lambda_list:
                data = res[model_name][lam]
                iters = np.array(data['iters'])
                log_rel = np.clip(np.array(data['log10_rel']), LOG_FLOOR, None)
                ax.plot(iters, log_rel, color=lambda_colors[lam],
                        lw=1.5, alpha=0.8, marker='o', markersize=2)
            ax.axhline(y=LOG_FLOOR, color='gray', linestyle=':', lw=1.5, alpha=0.7)
            ax.set_ylim(LOG_FLOOR - 1, 1)
            ax.set_yticks([LOG_FLOOR, -5, 0])
            ax.set_yticklabels([r'$\leq -10$', '-5', '0'])
            ax.grid(True, alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel(f'ROI={roi}\nlog₁₀ Rel. Error', fontsize=20)
            ax.set_xlabel('Iteration', fontsize=20)
    fig1.text(0.5, 0.02, 'Rows: ROI; Columns: Edge ratios',
              ha='center', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=(0, 0.03, 0.92, 0.95))
    plt.savefig(output_dir / f'log10_relative_error_allROI_{model_name}.pdf',
                dpi=150, bbox_inches='tight', format='pdf')
    plt.close(fig1)

    # ---------- Supplementary: Consecutive Dice Overview (3x5) ----------
    print(f"Plotting: {model_name} supplementary dice coefficient overview")
    fig_dice, axes_dice = plt.subplots(3, 5, figsize=(24, 14))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    fig_dice.legend(handles=legend_elems, loc='upper right',
                    bbox_to_anchor=(0.99, 0.97), fontsize=18)

    for row_idx, roi in enumerate(roi_list):
        for col_idx, res in enumerate(all_results[roi]):
            ax = axes_dice[row_idx, col_idx]
            for lam in lambda_list:
                data = res[model_name][lam]
                iters = np.array(data['iters'])
                dice_seq = np.array(data['dices'])

                if len(dice_seq) == len(iters) - 1:
                    plot_x = iters[1:]
                else:
                    plot_x = iters[:len(dice_seq)]

                ax.plot(plot_x, dice_seq, color=lambda_colors[lam],
                        lw=1.5, alpha=0.8, marker='s', markersize=2)

            ax.axhline(y=1.0, color='gray', linestyle=':', lw=1.5, alpha=0.7)
            ax.set_ylim(0.95, 1.01)
            ax.set_yticks([0.95, 0.975, 1.0])
            ax.grid(True, alpha=0.3)

            if col_idx == 0:
                ax.set_ylabel(f'ROI={roi}\nDice Coefficient', fontsize=20)
            ax.set_xlabel('Iteration', fontsize=20)

    fig_dice.text(0.5, 0.02, 'Rows: ROI; Columns: Edge ratios',
                  ha='center', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=(0, 0.03, 0.92, 0.95))
    plt.savefig(output_dir / f'dice_coefficient_allROI_{model_name}.pdf',
                dpi=150, bbox_inches='tight', format='pdf')
    plt.close(fig_dice)

    # ---------- Supplementary: Iteration Heatmap ----------
    print(f"Plotting: {model_name} iteration heatmap")
    fig_heat, axes_heat = plt.subplots(1, 3, figsize=(18, 5))
    y_labels = [f'{lam}' for lam in lambda_list]
    for idx, roi in enumerate(roi_list):
        ax = axes_heat[idx]
        mat = np.zeros((len(lambda_list), len(ratio_list)))
        for j, res in enumerate(all_results[roi]):
            for i, lam in enumerate(lambda_list):
                mat[i, j] = res[model_name][lam]['iters'][-1]
        im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(ratio_list)))
        ax.set_xticklabels(ratio_list, fontsize=18)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=18)
        ax.set_xlabel('Edge Ratio', fontsize=20)
        ax.set_title(f'ROI={roi}', fontsize=20, fontweight='bold')
        for i in range(len(y_labels)):
            for j in range(len(ratio_list)):
                val = int(mat[i, j])
                color = 'white' if mat[i, j] > mat.max() * 0.6 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=18, color=color, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
    axes_heat[0].set_ylabel('λ₂', fontsize=20)
    plt.tight_layout()
    plt.savefig(output_dir / f'iterations_heatmap_{model_name}.pdf',
                dpi=150, bbox_inches='tight', format='pdf')
    plt.close(fig_heat)

    print(f"All {model_name} figures generated successfully.")


# ---------- Main Program ----------
def main():
    # Configuration
    data_dir = Path('Schaefer')
    roi_list = [100, 500, 1000]
    ratio_list = [0.04, 0.08, 0.16, 0.32, 0.64]
    lambda_list = [0.02 * i for i in range(1, 5)]
    max_iter = 1000
    tol = 1e-10
    cache_dir = Path('Cache')
    cache_dir.mkdir(exist_ok=True)
    output_dir = Path('Convergence_Data_Combined')
    output_dir.mkdir(exist_ok=True)

    record_points = get_record_points(max_iter)
    print(f"Recording {len(record_points)}/{max_iter} iterations "
          f"({len(record_points)/max_iter:.1%} storage)")

    all_results = {roi: [] for roi in roi_list}

    models_to_run = ['Entropy', 'Variance']

    for roi in roi_list:
        print(f"\n=== Processing ROI={roi} ===")
        file_path = data_dir / f'{roi}.csv'
        df = pd.read_csv(file_path, sep=',')
        ts = df[df.columns].to_numpy()
        norms = np.linalg.norm(ts, axis=0)
        valid = norms > 1e-10
        ts = ts[:, valid]
        ts /= norms[valid]
        n_nodes = ts.shape[1]
        graph_op = NumbaGraph.GraphOperator_Unsafe(n_nodes)
        signal_diff = calculate_signal_difference(ts)
        dist = graph_op.Q(signal_diff)

        for ratio in ratio_list:
            print(f"  Edge ratio {ratio} ...")
            expected_edges = max(int((n_nodes - 1) * ratio), 1)
            theta = gsp_compute_graph_learning_theta_batch(
                dist.squeeze(), [expected_edges],
                geom_mean=False, is_sorted=False
            )[0]
            scaled_signal_diff = theta * signal_diff
            init_edges = np.exp(-scaled_signal_diff / 2)
            init_edges = sparsify_fc(init_edges, [expected_edges])

            cache_file = cache_dir / f'ROI{roi}_ratio{ratio}.pkl'
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                config_results = {'ratio': ratio}
                for model in models_to_run:
                    if model in cached:
                        config_results[model] = cached[model]
                    else:
                        print(f"    Cache missing {model}, recomputing...")
                        res = run_model_multi_lam(
                            model, n_nodes, init_edges, scaled_signal_diff,
                            ratio, lambda_list, max_iter, tol,
                            record_points=record_points
                        )
                        config_results[model] = res
                        cached[model] = res
                        with open(cache_file, 'wb') as f:
                            pickle.dump(cached, f)
                all_results[roi].append(config_results)
                print(f"    Loaded from cache successfully.")
                continue

            config_results = {'ratio': ratio}

            print("    Computing EntropyProx ...")
            res_ent = run_model_multi_lam(
                'Entropy', n_nodes, init_edges, scaled_signal_diff,
                ratio, lambda_list, max_iter, tol,
                record_points=record_points
            )
            config_results['Entropy'] = res_ent

            print("    Computing VarianceProx ...")
            res_var = run_model_multi_lam(
                'Variance', n_nodes, init_edges, scaled_signal_diff,
                ratio, lambda_list, max_iter, tol,
                record_points=record_points
            )
            config_results['Variance'] = res_var

            with open(cache_file, 'wb') as f:
                pickle.dump(config_results, f)
            all_results[roi].append(config_results)
            print("    Computation complete and cached.")

    print("\n=== Starting Plotting ===")
    plot_all_figures(all_results, 'Entropy', output_dir, target_roi=500,
                     roi_list=roi_list, ratio_list=ratio_list,
                     lambda_list=lambda_list)
    plot_all_figures(all_results, 'Variance', output_dir, target_roi=500,
                     roi_list=roi_list, ratio_list=ratio_list,
                     lambda_list=lambda_list)

    print("\nAll tasks completed!")


if __name__ == '__main__':
    main()
    