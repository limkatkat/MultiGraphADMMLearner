import pandas as pd
import numpy as np
import re
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ==================== Parameter Settings ====================
edge_ratio = 0.24
entropy_strength = '0.0080'
yeo_nets = 17
model = 'ent'

STABILITY_CSV_PATH = f'frequency-{model}.csv'
DATA_DIR = f"graphs-{model}/FunImgARglobalCWSF/bna-feat"

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


def decode_feature_name(feature_csv_name: str) -> str:
    """Convert N01_N02_max_div_std format to VisPeri_VisCent_max_div_std"""
    def replace_net(match):
        # N01 -> extract number 1 -> lookup in dictionary; keep original if not found
        net_num = int(match.group(0)[1:])  # Remove 'N' prefix and convert to int
        return YEO_17_NAMES.get(net_num, match.group(0))
    return re.sub(r'N\d+', replace_net, feature_csv_name)


# 1. Load feature list with frequency = 1
df_stability = pd.read_csv(STABILITY_CSV_PATH)
freq_1_features = df_stability[df_stability['Avg_Frequency'] == 1]['OriginalName'].tolist()

if len(freq_1_features) == 0:
    print("Warning: No features with frequency = 1 found. Please check frequency.csv")
    exit(0)

print(f"Total {len(freq_1_features)} features with frequency = 1 to be tested")

# 2. Load raw data
raw_data_path = f"{DATA_DIR}/{entropy_strength}/Yeo{yeo_nets}-{edge_ratio:.02f}.csv"
df_raw = pd.read_csv(raw_data_path)

if 'Group' not in df_raw.columns:
    raise ValueError("Missing 'Group' column in raw data")

# Pre-check feature columns
missing = [f for f in freq_1_features if f not in df_raw.columns]
if missing:
    print(f"Error: The following feature columns do not exist in the raw CSV: {missing}")
    raise ValueError("Missing feature columns. Please ensure frequency.csv has been corrected with suffix mapping")

# 3. Welch's t-test
results = []
for feat in freq_1_features:
    hc = df_raw[df_raw['Group'] == 'HC'][feat].dropna()
    ism = df_raw[df_raw['Group'] == 'ISM'][feat].dropna()

    if len(hc) == 0 or len(ism) == 0:
        print(f"Warning: No data for one of the groups in '{feat}', skipping")
        continue

    t_stat, p_value = stats.ttest_ind(hc, ism, equal_var=False)

    hc_mean, hc_std = hc.mean(), hc.std()
    ism_mean, ism_std = ism.mean(), ism.std()
    pooled_std = np.sqrt((hc_std ** 2 + ism_std ** 2) / 2)
    cohen_d = (hc_mean - ism_mean) / pooled_std if pooled_std != 0 else 0.0

    results.append({
        'Feature_CSV': feat,
        'Feature_Readable': decode_feature_name(feat),  # Add readable name column
        'HC_Mean': hc_mean,
        'ISM_Mean': ism_mean,
        'HC_Std': hc_std,
        'ISM_Std': ism_std,
        'Welch_t': t_stat,
        'p_value': p_value,
        "Cohen's_d": cohen_d,
    })

# 4. FDR correction
results_df = pd.DataFrame(results)
if len(results_df) > 0:
    rejected, p_fdr, _, _ = multipletests(results_df['p_value'].values, method='fdr_bh')
    results_df['p_fdr'] = p_fdr
    results_df['Significant_FDR05'] = rejected
else:
    results_df['p_fdr'] = []
    results_df['Significant_FDR05'] = []

# 5. Save results
output_path = f"freq1_diff-{model}.csv"
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\nDone! Saved to: {output_path}")
print(f"   Valid features: {len(results_df)}, Significant (FDR < 0.05): {results_df['Significant_FDR05'].sum()}")
print("\nFirst 10 rows preview:")
print(results_df[['Feature_CSV', 'Feature_Readable', "Cohen's_d", 'p_fdr', 'Significant_FDR05']].head(10).to_string(index=False))

