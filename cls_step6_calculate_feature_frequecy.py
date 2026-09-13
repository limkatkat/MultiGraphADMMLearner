import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Union, Callable

model = 'ent'
reg_strength = '0.0080'
edge_ratio = 0.24

CONFIG = {
    'edge_ratio': edge_ratio,
    # Options: 'max_div_std', 'max_x_std', 'max_div_std_all', 'max_x_std_all', 'max', 'std', etc.
    'feature_type': 'max_div_std_all',
    'yeo_nets': 17,
    'reg_strength': reg_strength,
    'group_name': 'ISM',
    'test_params_str': '0.05-0.05-False',
    'top_n': 50,
    'classifiers': ['SVM_Linear'],
    'model': model
}

BASE_DIR = Path(f'graphs-{CONFIG['model']}')
STABILITY_THRESHOLD = '0.9'
RESULT_ROOT = BASE_DIR / 'FunImgARglobalCWSF' / f'bna-feat-out-{STABILITY_THRESHOLD}'
DATA_DIR = BASE_DIR / 'FunImgARglobalCWSF' / 'bna-feat'

# YEO 17 network name mapping (must be consistent with the main script)
YEO_17_NAMES = {
    0: 'Subcortical',
    1: 'VisPeri', 2: 'VisCent',
    3: 'SomMotA', 4: 'SomMotB',
    5: 'DorsAttnA', 6: 'DorsAttnB',
    7: 'SalVentAttnA', 8: 'SalVentAttnB',
    9: 'Limbic1', 10: 'Limbic2',
    11: 'ContC', 12: 'ContA', 13: 'ContB',
    14: 'DefaultD', 15: 'DefaultC',
    16: 'DefaultA', 17: 'DefaultB'
}

# Feature type definitions: includes pre-computed features and original composite features
FEATURE_TYPES_DEF: List[Union[str, Tuple[str, str, str, Callable, bool]]] = [
    # --- Pre-computed features (string type) ---
    'max_div_std',
    'max_x_std',
    # --- Original composite features ---
    ('max', 'std', 'max_div_std_all', lambda x, y: x / (y + 1e-8), False),
    ('max', 'std', 'max_x_std_all',   lambda x, y: x * y,           True),
    # --- Basic features ---
    'max', 'maxratio', 'std', 'mean'
]
# ================================================


def get_feature_names(feature_type_config, df_header):
    """Replicate the feature generation logic from the main script to ensure consistent column ordering."""
    final_feature_cols = []

    if isinstance(feature_type_config, tuple):
        suffix_1, suffix_2, new_name, func, use_new_only = feature_type_config

        all_cols = df_header.columns.tolist()
        cols_1 = [c for c in all_cols if c.endswith(f'_{suffix_1}')]
        cols_2 = [c for c in all_cols if c.endswith(f'_{suffix_2}')]

        dict_1 = {c.rsplit('_', 1)[0]: c for c in cols_1}
        dict_2 = {c.rsplit('_', 1)[0]: c for c in cols_2}
        common_prefixes = sorted(set(dict_1.keys()) & set(dict_2.keys()))

        interaction_cols = []
        base_cols = []

        for prefix in common_prefixes:
            interaction_cols.append(f'{prefix}_{new_name}')
            base_cols.extend([dict_1[prefix], dict_2[prefix]])

        base_cols = list(dict.fromkeys(base_cols))

        if use_new_only:
            final_feature_cols = interaction_cols
        else:
            final_feature_cols = base_cols + interaction_cols

    else:
        # String type: covers basic features AND pre-computed max_div_std / max_x_std
        suffix = feature_type_config
        final_feature_cols = [col for col in df_header.columns if col.endswith(f'_{suffix}')]

    return final_feature_cols


def map_feature_name(original_name, yeo_names):
    """Convert feature names from 'Nxx_Nyy_xxx' format to human-readable network names."""
    if original_name.startswith('N') and '_' in original_name:
        parts = original_name.split('_')
        net1_part = parts[0][1:]  # Remove 'N' prefix to get 'xx'

        # Handle 'intra' features (e.g., N03_intra_max_div_std)
        if 'intra' in original_name:
            net_num = int(net1_part)
            net_name = yeo_names.get(net_num, f'Net{net_num}')
            # intra feature format: N03_intra_max_div_std -> parts = ['N03', 'intra', 'max', 'div', 'std']
            suffix_part = '_'.join(parts[2:])
            return f"{net_name}_intra_{suffix_part}", original_name
        else:
            # inter feature format: N03_N07_max_div_std -> parts = ['N03', 'N07', 'max', 'div', 'std']
            net2_part = parts[1][1:]  # Remove 'N' prefix
            net1_num = int(net1_part)
            net2_num = int(net2_part)
            net1_name = yeo_names.get(net1_num, f'Net{net1_num}')
            net2_name = yeo_names.get(net2_num, f'Net{net2_num}')
            suffix_part = '_'.join(parts[2:])
            return f"{net1_name}_{net2_name}_{suffix_part}", original_name

    return original_name, original_name


def analyze_stability(config):
    if "reg_strength" in config:
        entropy_dir = RESULT_ROOT / f'{float(config["reg_strength"]):.04f}'
    else:
        entropy_dir = RESULT_ROOT

    # 1. Find the corresponding feature definition (tuple or string)
    feature_def = None
    for ft in FEATURE_TYPES_DEF:
        if (isinstance(ft, tuple) and ft[2] == config['feature_type']) or \
           (isinstance(ft, str) and ft == config['feature_type']):
            feature_def = ft
            break

    if feature_def is None:
        print(f"[ERROR] Definition for '{config['feature_type']}' not found in FEATURE_TYPES_DEF")
        print(f"        Available features: {[ft[2] if isinstance(ft, tuple) else ft for ft in FEATURE_TYPES_DEF]}")
        return

    # 2. Load raw data header to retrieve feature names
    if "reg_strength" in config:
        csv_path = DATA_DIR / f'{float(config["reg_strength"]):.04f}' / f"Yeo{config['yeo_nets']}-{config['edge_ratio']:.02f}.csv"
    else:
        csv_path = DATA_DIR / f"Yeo{config['yeo_nets']}-{config['edge_ratio']:.02f}.csv"
    if not csv_path.exists():
        print(f"[ERROR] Raw data file not found: {csv_path}")
        return

    df_header = pd.read_csv(csv_path, nrows=1)  # Read header only
    feature_names = get_feature_names(feature_def, df_header)

    if len(feature_names) == 0:
        print(f"[ERROR] No columns matching '_{config['feature_type']}' suffix found in CSV")
        print(f"        CSV column examples: {df_header.columns[:10].tolist()}")
        return

    # 3. Convert feature names to human-readable network names
    results = [map_feature_name(name, YEO_17_NAMES) for name in feature_names]
    readable_feature_names, _ = tuple(list(col) for col in zip(*results))

    print(f"[INFO] Parsed {len(readable_feature_names)} readable feature names")

    # 4. Iterate over classifiers, load frequency arrays and aggregate
    freq_accumulator = np.zeros(len(readable_feature_names))
    valid_clf_count = 0

    print(f"[INFO] Aggregating feature frequencies for classifiers: {config['classifiers']}")

    for clf_name in config['classifiers']:
        prefix = (f"{config['group_name']}-Yeo{config['yeo_nets']}-{clf_name}"
                  f"-{config['edge_ratio']:.02f}-{config['feature_type']}-{config['test_params_str']}")

        freq_file = entropy_dir / f'{prefix}-freq.npy'
        print (freq_file)

        if not freq_file.exists():
            print(f"  [WARN] Skipping {clf_name}: frequency file not found ({freq_file.name})")
            continue

        freq_data = np.load(freq_file)

        if len(freq_data) != len(readable_feature_names):
            print(f"  [WARN] Dimension mismatch for {clf_name} (data:{len(freq_data)} vs names:{len(readable_feature_names)})")
            continue

        freq_accumulator += freq_data
        valid_clf_count += 1
        print(f"  [OK] Loaded {clf_name}")

    if valid_clf_count == 0:
        print("[ERROR] No frequency data loaded successfully.")
        return

    # Suffix-level mapping table (not full name matching)
    SUFFIX_MAP = {
        '_max_div_std_all': '_max_div_std',
        '_max_x_std_all':   '_max_x_std',
    }

    def map_to_csv_col(name: str) -> str:
        """Convert composite feature full names back to actual CSV column names."""
        for old_suffix, new_suffix in SUFFIX_MAP.items():
            if name.endswith(old_suffix):
                return name[:-len(old_suffix)] + new_suffix
        return name  # Return as-is for basic / pre-computed features

    # 5. Compute average frequency and create DataFrame
    avg_freq = freq_accumulator / valid_clf_count

    mapped_feature_names = [map_to_csv_col(name) for name in feature_names]

    df_res = pd.DataFrame({
        'Feature': readable_feature_names,
        'Avg_Frequency': avg_freq,
        'OriginalName': mapped_feature_names  # Correct CSV column names
    })

    df_res = df_res.sort_values(by='Avg_Frequency', ascending=False).reset_index(drop=True)
    df_res.to_csv(f'frequency-{config['model']}.csv', encoding='utf-8', index=False)

    # 6. Visualization
    df_top = df_res.head(config['top_n'])

    plt.figure(figsize=(12, 10))

    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(df_top)))
    bars = plt.barh(df_top['Feature'][::-1], df_top['Avg_Frequency'][::-1],
                    color=colors[::-1], edgecolor='black', alpha=0.9)

    plt.xlabel('Average Stability Frequency (Cross-Classifier)', fontsize=14)
    plt.ylabel('Network Feature', fontsize=14)

    title_str = (f"Top {config['top_n']} Stable Network Features\n"
                 f"Feature: {config['feature_type']} | Sparsity: {config['edge_ratio']}")
    plt.title(title_str, fontsize=16, fontweight='bold')

    for bar, val in zip(bars, df_top['Avg_Frequency'][::-1]):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=10)

    plt.xlim(0, 1.05)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.yticks(fontsize=11)

    save_name = entropy_dir / f"Stability_Rank_{config['feature_type']}_E{config['edge_ratio']}-{config['model']}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"[OK] Stability rank plot saved: {save_name}")

    return df_res


if __name__ == '__main__':
    df_final = analyze_stability(CONFIG)

    if df_final is not None:
        print("\n[RESULT] Top stable network features ranking:")
        print(df_final.head(30).to_string(index=False))
        