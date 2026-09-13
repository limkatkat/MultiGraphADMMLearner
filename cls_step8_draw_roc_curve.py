import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


model = 'var'
reg_strength = '0.0020'

# ==================== Configuration ====================
# Modify the parameter combination to plot here
CONFIG = {
    'clf_name': 'SVM_Linear',          # Classifier name: 'SVM_Linear' or 'LR_L2'
    'edge_ratio': 0.24,                # Edge ratio (float)
    'feature_type': 'max_div_std_all', # Feature type
    'reg_strength': reg_strength,      # Regularization strength (string, 4 decimal places)
    'yeo_nets': 17,                    # Number of brain networks
    'group_name': 'ISM',               # Group name
    'test_params_str': '0.05-0.05-False' # Test parameter suffix
}


# Base path setup
BASE_DIR = Path(f'graphs-{model}')
# Assume the threshold is 0.9; modify here if different
STABILITY_THRESHOLD = '0.9'
RESULT_ROOT = BASE_DIR / 'FunImgARglobalCWSF' / f'bna-feat-out-{STABILITY_THRESHOLD}'
# ================================================


def plot_roc_curve(config):
    # 1. Build the file path
    # Directory structure: RESULT_ROOT / {reg_strength} / .npy file
    reg_dir = RESULT_ROOT / f"{float(config['reg_strength']):.04f}"

    # Build the filename prefix
    # Format: ISM-Yeo17-SVM_Linear-0.24-max_div_std_all-0.05-0.05-False
    prefix = (f"{config['group_name']}-Yeo{config['yeo_nets']}-{config['clf_name']}"
              f"-{config['edge_ratio']:.02f}-{config['feature_type']}-{config['test_params_str']}")

    roc_file = reg_dir / f'{prefix}-rocdata.npy'

    if not roc_file.exists():
        print(f"Error: ROC data file not found:\n   {roc_file}")
        return

    print(f"Loading data: {roc_file.name}")
    data = np.load(roc_file, allow_pickle=True).item()

    # 2. Extract data
    fpr = data['fpr']
    tpr_mean = data['tpr_mean']
    # Handle possible boundary cases to keep bounds within [0, 1]
    tpr_upper = np.minimum(data['tpr_upper'], 1.0)
    tpr_lower = np.maximum(data['tpr_lower'], 0.0)
    auc_mean = data['auc_mean']
    auc_std = data['auc_std']

    # 3. Plot
    plt.figure(figsize=(8, 7))

    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3, label='Chance Level (AUC = 0.50)')

    # Mean ROC curve
    plt.plot(fpr, tpr_mean, color='#E74C3C', lw=3,
             label=f'ROC Curve (AUC = {auc_mean:.3f} $\\pm$ {auc_std:.3f})')

    # Standard deviation shaded area
    plt.fill_between(fpr, tpr_lower, tpr_upper, color='#E74C3C', alpha=0.2, label='1 Std. Dev.')

    # 4. Styling
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)

    title_str = (f"{config['clf_name']} | Feature: {config['feature_type']}\n"
                 f"Reg={config['reg_strength']}, Sparsity={config['edge_ratio']}")
    plt.title(title_str, fontsize=16, fontweight='bold')

    plt.legend(loc='lower right', fontsize=12, frameon=True, facecolor='white', edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the figure
    save_path = f'{prefix}-ROC_Curve-{model}.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format="pdf")
    plt.show()
    print(f"ROC curve saved to: {save_path}")


if __name__ == '__main__':
    plot_roc_curve(CONFIG)
    
    