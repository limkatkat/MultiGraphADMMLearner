import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Union, Callable

import numpy as np
import pandas as pd

# Assuming classify_utility contains the English interface defined previously
from classify_utility import ClassifierManager, process_single_feature_set


# Suppress feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# ==================== Model Configuration ====================

LINEAR_C_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100]

CLF_MANAGERS = [
    
    ClassifierManager(
        name='LR_L2', 
        param_grid={'C': LINEAR_C_GRID}, 
        fixed_kwargs={
            # 'l1_ratio': 0, 
            'solver': 'liblinear', 
            'max_iter': 2000, 
            'class_weight': 'balanced'
        }, 
        need_scale=True,
        has_weights=True
    ),
    
    ClassifierManager(
        name='SVM_Linear',
        param_grid={'C': LINEAR_C_GRID},
        fixed_kwargs={
           'max_iter': 10000, 
           'class_weight': 'balanced'
        },
        need_scale=True,
        has_weights=True
    )
]


if __name__ == '__main__':
    
    # Parse Arguments
    REMOVE_MEAN = sys.argv[1].upper() == 'TRUE' if len(sys.argv) > 1 else False
    SUFFIX = '-0' if REMOVE_MEAN else ''
    
    STABILITY_FREQ_THRESHOLD = 0.9
    
    GROUP_NAME = "ISM" if len(sys.argv) < 3 else sys.argv[2].upper()
    print(f'Remove Mean={REMOVE_MEAN}, Group={GROUP_NAME}')
    
    # Define Paths
    DATA_ROOT = Path('/Volumes/EXT-2/fc-results/20260803/graphs-l2')
    DATA_DIR = DATA_ROOT / f'FunImgARglobalCWSF{SUFFIX}' / 'bna-feat'
    
    OUTPUT_DIR = DATA_ROOT / f'FunImgARglobalCWSF{SUFFIX}' / f'bna-feat-out-{STABILITY_FREQ_THRESHOLD}'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    YEO_NETWORK_COUNTS = [17, 17]#, 7]
    
    # Feature Types: (suffix1, suffix2, output_name, func, use_new_only) or string
    FEATURE_TYPES: List[Union[str, Tuple[str, str, str, Callable, bool]]] = [
        ('max', 'std', 'max_div_std_all', lambda x, y: x / (y + 1e-8), False),
        ('max', 'std', 'max_x_std_all', lambda x, y: x * y, True),
        'max', 'maxratio', 'std', 'mean'
    ]
   
    EDGE_RATIOS = [0.02 * i for i in range(5, 26)]
    TOTAL_CONFIGS = len(EDGE_RATIOS) * len(CLF_MANAGERS)
    
    TEST_PARAMS_LIST = [
        {'p_threshold': 0.05, 'min_frequency': 0.05, 'need_correction': False},
        None,
        # {'p_threshold': 0.05, 'min_frequency': 0.05, 'need_correction': False}
    ]
    
    # ENTROPY_BARRIER_STRENGTHS = [0.002*i for i in range(1, 5)] + [0.02*i for i in range(2, 5)] + [0.2*i for i in range(1, 5)]
    # ENTROPY_BARRIER_STRENGTHS = [0.2*i for i in range(1, 5)]
    ENTROPY_BARRIER_STRENGTHS = [0.02]

    for entropy_strength in ENTROPY_BARRIER_STRENGTHS:
        
        entropy_str = f'{entropy_strength:.04f}'
        entropy_output_dir = OUTPUT_DIR / entropy_str
        entropy_output_dir.mkdir(exist_ok=True)
                
        for yeo_nets, test_params in zip(YEO_NETWORK_COUNTS, TEST_PARAMS_LIST):
            
            if test_params is not None:
                p_threshold = test_params.get('p_threshold')  # Assuming keys map correctly
                min_frequency = test_params.get('min_frequency')
                need_correction = test_params.get('need_correction')
            else:
                p_threshold = min_frequency = need_correction = None
                
            test_params_str = f'{p_threshold}-{min_frequency}-{need_correction}'
            
            for feature_type in FEATURE_TYPES:
                
                # Determine feature ID
                if isinstance(feature_type, tuple):
                    feature_id = feature_type[2]
                else:
                    feature_id = feature_type
                    
                result_csv_path = entropy_output_dir / f'{GROUP_NAME}-model_eval_results{SUFFIX}-Yeo{yeo_nets}-{feature_id}-{test_params_str}.csv'
                
                if result_csv_path.exists():
                    continue
                
                all_results = []
                current_task_count = 0
                
                for edge_ratio in EDGE_RATIOS:
                    print("\n" + "="*60)
                    print(f" Loading Data: Edge Ratio = {edge_ratio}, Entropy = {entropy_str}, Feature Type = {feature_type}")
                    print("="*60)
                    
                    # Load Data
                    feat_file_path = DATA_DIR / entropy_str / f'Yeo{yeo_nets}-{edge_ratio:.02f}.csv'
                    df_all_features = pd.read_csv(feat_file_path)
                    
                    # Create Masks
                    df_all_features['IsHC'] = df_all_features['Group'].str.upper() == 'HC'
                    # df_all_features['IsMDD'] = df_all_features['Group'].str.upper() == 'MDD'
                    df_all_features['IsISM'] = df_all_features['Group'].str.upper() == 'ISM'
                    
                    # Filter Group
                    mask_group = df_all_features[f"Is{GROUP_NAME}"] | df_all_features['IsHC']
                    df_all_features = df_all_features[mask_group].copy().reset_index(drop=True)
                    
                    labels = df_all_features[f"Is{GROUP_NAME}"].to_numpy()
                    
                    # Feature Engineering
                    if isinstance(feature_type, tuple):
                        suffix_1, suffix_2, new_name, func, use_new_only = feature_type
                        
                        all_cols = df_all_features.columns.tolist()
                        cols_1 = [c for c in all_cols if c.endswith(f'_{suffix_1}')]
                        cols_2 = [c for c in all_cols if c.endswith(f'_{suffix_2}')]
                        
                        dict_1 = {c.rsplit('_', 1)[0]: c for c in cols_1}
                        dict_2 = {c.rsplit('_', 1)[0]: c for c in cols_2}
                        common_prefixes = sorted(set(dict_1.keys()) & set(dict_2.keys()))
                        
                        interaction_list = []
                        interaction_cols = []
                        base_cols = []
                        is_concat_mode = False  
                        
                        for prefix in common_prefixes:
                            col_1 = dict_1[prefix]
                            col_2 = dict_2[prefix]
                            
                            x = df_all_features[col_1].values.astype(float)
                            y = df_all_features[col_2].values.astype(float)
                            
                            new_vals = func(x, y)
                            
                            if isinstance(new_vals, (tuple, list)):
                                is_concat_mode = True
                                x_clean = np.nan_to_num(new_vals[0], nan=0.0, posinf=0.0, neginf=0.0)
                                y_clean = np.nan_to_num(new_vals[1], nan=0.0, posinf=0.0, neginf=0.0)
                                interaction_list.append(x_clean)
                                interaction_list.append(y_clean)
                                interaction_cols.extend([col_1, col_2])
                            else:
                                vals_clean = np.nan_to_num(new_vals, nan=0.0, posinf=0.0, neginf=0.0)
                                interaction_list.append(vals_clean)
                                interaction_cols.append(f'{prefix}_{new_name}')
                                base_cols.extend([col_1, col_2])
                        
                        base_cols = list(dict.fromkeys(base_cols))
                        interaction_matrix = np.column_stack(interaction_list) if len(interaction_list) > 0 else np.empty((len(labels), 0))
                        
                        if is_concat_mode:
                            final_feature_matrix = interaction_matrix
                            final_feature_cols = interaction_cols
                        else:
                            if use_new_only:
                                final_feature_matrix = interaction_matrix
                                final_feature_cols = interaction_cols
                            else:
                                if len(base_cols) > 0:
                                    base_matrix = df_all_features[base_cols].to_numpy().astype(float)
                                    base_matrix = np.nan_to_num(base_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                                else:
                                    base_matrix = np.empty((len(labels), 0))
                                    
                                final_feature_matrix = np.hstack([base_matrix, interaction_matrix])
                                final_feature_cols = base_cols + interaction_cols
                                
                        df_all_features = pd.DataFrame(final_feature_matrix, columns=final_feature_cols, index=df_all_features.index)
                        n_features = len(final_feature_cols)
                            
                    else:
                        final_feature_cols = [col for col in df_all_features.columns if col.endswith(f'_{feature_type}')]
                        n_features = len(final_feature_cols)
                        df_all_features = df_all_features[final_feature_cols]
                        
                        df_all_features = df_all_features.astype(float)
                        df_all_features = df_all_features.apply(lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))

                    
                    # ==================== Classification Loop ====================
                    for current_clf in CLF_MANAGERS:
                        current_task_count += 1
                        
                        print("\n" + "#"*60)
                        print(f"[Global Progress {current_task_count}/{TOTAL_CONFIGS}]")
                        print(f"  Edge Ratio   : {edge_ratio}")
                        print(f"  Algorithm    : {current_clf.name}")
                        print(f"  N Features   : {n_features}")
                        print("#"*60)
                        
                        res = process_single_feature_set(
                            all_sample_features=df_all_features, 
                            all_sample_labels=labels, 
                            current_classifier=current_clf, 
                            all_stratify_labels=None,
                            test_params=test_params,
                            stability_freq_threshold=STABILITY_FREQ_THRESHOLD,
                            n_jobs=10
                        )
                        
                        
                        roc_data = res['roc_data']
                        avg_weights = res['avg_weights']
                        avg_freq = res['feat_freq']
                        
                        prefix = f'{GROUP_NAME}-Yeo{yeo_nets}-{current_clf.name}-{edge_ratio:.02f}-{feature_id}-{test_params_str}'
                        
                        np.save(entropy_output_dir / f'{prefix}-rocdata.npy', roc_data)
                        np.save(entropy_output_dir / f'{prefix}-weights.npy', avg_weights)
                        np.save(entropy_output_dir / f'{prefix}-freq.npy', avg_freq)
                        
                        del res['roc_data']
                        del res['avg_weights']
                        del res['feat_freq']
                        
                        print(f"\n{'-'*60}")
                        print(f"Summary [Edge Ratio:{edge_ratio} | Clf:{current_clf.name}]")
                        print("  [Overall Performance (Mean ± Std)]")
                        print(f"    Acc: {res['acc_mean']:.4f} ± {res['acc_std']:.4f}")
                        print(f"    AUC: {res['auc_mean']:.4f} ± {res['auc_std']:.4f}")
                        print(f"    F1 : {res['f1_mean']:.4f} ± {res['f1_std']:.4f}")
                        print(f"{'-'*60}")
                        
                        all_results.append({
                            'edge_ratio': edge_ratio,
                            'dim_reduction': 'None',
                            'algorithm': current_clf.name,
                            **res
                        })

                print("\n\n All tasks completed! Exporting results...")
                df_results = pd.DataFrame(all_results)
                df_results = df_results[['edge_ratio', 'dim_reduction', 'algorithm', 'acc_mean', 'acc_std', 'auc_mean', 'auc_std', 'f1_mean', 'f1_std']]
                df_results.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
                print(f" Results saved to: {result_csv_path}")

