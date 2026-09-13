import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



model = 'ent'
classifier_name = 'SVM_Linear'
reg_strength = '0.0080'

CONFIG = {
    'edge_ratio': 0.24,
    'feature_type': 'max_div_std_all',
    'reg_strength': reg_strength,
    'yeo_nets': 17,
    'group_name': 'ISM',
    'test_params_str': '0.05-0.05-False',
    'top_n': 50,
    'classifiers': [classifier_name],
    'model': model
}


# ==================== Parameter Settings ====================
feature_type = 'max_div_std_all'
yeo_nets = 17
group_name = 'ISM'
test_params_str = '0.05-0.05-False'
edge_ratios = [0.02 * i for i in range(5, 26)]
reg_strengths = (
    [0.002 * i for i in range(1, 5)]
    + [0.02 * i for i in range(1, 5)]
    + [0.2 * i for i in range(1, 5)]
)
stability_threshold = '0.9'
# ==========================================================

RESULT_ROOT = (
    Path(f'graphs-{model}')
    / 'FunImgARglobalCWSF'
    / f'bna-feat-out-{stability_threshold}'
)

auc_array = np.full((len(edge_ratios), len(reg_strengths)), np.nan)

for i, edge_ratio in enumerate(edge_ratios):
    for j, reg_strength in enumerate(reg_strengths):
        reg_str = f'{reg_strength:.04f}'
        prefix = (
            f'{group_name}-Yeo{yeo_nets}-{classifier_name}'
            f'-{edge_ratio:.02f}-{feature_type}-{test_params_str}'
        )
        roc_file = RESULT_ROOT / reg_str / f'{prefix}-rocdata.npy'

        if roc_file.exists():
            data = np.load(roc_file, allow_pickle=True).item()
            auc_array[i, j] = data['auc_mean']

# ==================== Global Font Size Settings ====================
plt.rcParams.update({
    'font.size': 14,           # Base font size (default is 10)
    'axes.titlesize': 18,      # Title font size
    'axes.labelsize': 16,      # X/Y axis label font size
    'xtick.labelsize': 12,     # X-axis tick label font size
    'ytick.labelsize': 12,     # Y-axis tick label font size
})
# ===================================================================

# ==================== Plotting ====================
fig, ax = plt.subplots(figsize=(12, 8))

vmin = np.nanmin(auc_array)
vmax = np.nanmax(auc_array)
im = ax.imshow(
    auc_array, cmap='viridis', origin='lower',
    aspect='auto', vmin=vmin, vmax=vmax,
)
cbar = plt.colorbar(im, ax=ax, label='AUC (Mean)')
cbar.ax.tick_params(labelsize=12)          # Colorbar tick label font size
cbar.set_label('AUC (Mean)', fontsize=16)  # Colorbar label font size

# ---------- X-axis labels: remove meaningless trailing zeros ----------
# The :g format automatically strips trailing zeros, e.g. 0.0020 -> 0.002, 0.2000 -> 0.2
x_labels = [f'{e:g}' for e in reg_strengths]
ax.set_xticks(np.arange(len(reg_strengths)))
ax.set_xticklabels(x_labels, rotation=45, ha='right')

ax.set_yticks(np.arange(len(edge_ratios)))
ax.set_yticklabels([f'{e:.2f}' for e in edge_ratios])

ax.set_xlabel('Reg Strength')
ax.set_ylabel('Edge Density')
ax.set_title(f'AUC Heatmap: {classifier_name} | Feature: {feature_type}')
ax.grid(False)

# ---------- Annotate AUC values > 0.7 on each cell ----------
# Viridis colormap brightness threshold is ~0.55; use white text below and black above
brightness_threshold = (vmin + vmax) / 2.0  # Simple adaptive threshold

for i in range(auc_array.shape[0]):
    for j in range(auc_array.shape[1]):
        val = auc_array[i, j]
        if not np.isnan(val) and val > 0.7:
            text_color = 'white' if val < brightness_threshold else 'black'
            ax.text(
                j, i,
                f'{val:.3f}',
                ha='center', va='center',
                fontsize=11,         # Increased annotation font size from 9 to 11
                color=text_color,
            )

plt.tight_layout()
save_path = f'{classifier_name}-AUCs_{model}.pdf'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
plt.show()

print("DONE")

