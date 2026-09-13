import os
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Import custom Numba-optimized graph utility
import NumbaGraph


def process_single_ratio(
    edge_ratio_str: str,
    input_dir: str | Path,
    files_for_ratio: List[str],
    network_names: List[str],
    network_indices: dict,
    num_regions: int,
    eps_tolerance: float = 1e-10
) -> Tuple[pd.DataFrame, str]:
    """
    Process feature extraction for a single edge ratio.
    Returns the corresponding DataFrame and the ratio string.
    """
    # Rebuild graph operator in subprocess (Numba compiled objects cannot be serialized)
    graph = NumbaGraph.GraphOperator_Unsafe(num_regions)
    
    all_sample_features = []
    
    for fname in files_for_ratio:
        # Filename format: Group-SubID-Ratio
        items = os.path.splitext(os.path.basename(fname))[0].split('-')
        group = items[0]
        subject_id = '-'.join(items[1:-1])
        
        full_path = Path(input_dir) / fname
        fc_vectors = np.load(full_path)
        adjacent_matrices = graph.Q(fc_vectors)

        feature_dict: dict[str, str | float] = {
            'SubID': subject_id,
            'Group': group,
            'Edge_Ratio': edge_ratio_str
        }

        for i_net, network_1 in enumerate(network_names):
            index_1 = network_indices[network_1]
            for j_net, network_2 in enumerate(network_names):
                if i_net < j_net:
                    continue

                if i_net == j_net:
                    # Intra-network connection: take diagonal sub-block
                    adjmat = adjacent_matrices[np.ix_(index_1, index_1)]
                    max_svalue = f'{network_1}_intra_max'
                    max_sratio = f'{network_1}_intra_maxratio'
                    mean_svalue = f'{network_1}_intra_mean'
                    std_svalue = f'{network_1}_intra_std'
                    norm_coeff = len(index_1)
                else:  # i_net > j_net
                    index_2 = network_indices[network_2]
                    # Inter-network connection: take cross sub-block
                    adjmat = adjacent_matrices[np.ix_(index_1, index_2)]
                    max_svalue = f'{network_1}_{network_2}_max'
                    max_sratio = f'{network_1}_{network_2}_maxratio'
                    mean_svalue = f'{network_1}_{network_2}_mean'
                    std_svalue = f'{network_1}_{network_2}_std'
                    norm_coeff = math.sqrt(len(index_1) * len(index_2))

                                # Calculate and normalize singular values
                svalues = np.linalg.svd(adjmat, compute_uv=False)
                
                # Base features
                max_val = float(np.max(svalues) / norm_coeff)
                std_val = float(np.std(svalues) / norm_coeff)
                
                feature_dict[max_svalue] = max_val
                if np.sum(svalues) < eps_tolerance:
                    feature_dict[max_sratio] = 0.0
                else:
                    feature_dict[max_sratio] = float(np.max(svalues) / np.sum(svalues))
                feature_dict[mean_svalue] = float(np.mean(svalues) / norm_coeff)
                feature_dict[std_svalue] = std_val
                
                # Derived features: max_div_std & max_x_std
                div_name = f'{network_1}_intra_max_div_std' if i_net == j_net else f'{network_1}_{network_2}_max_div_std'
                mul_name = f'{network_1}_intra_max_x_std'   if i_net == j_net else f'{network_1}_{network_2}_max_x_std'
                
                feature_dict[div_name] = max_val / (std_val + 1e-8)
                feature_dict[mul_name] = max_val * std_val

        all_sample_features.append(feature_dict)

    df_results = pd.DataFrame(all_sample_features)
    return df_results, edge_ratio_str


if __name__ == '__main__':
    
    # --- Configuration ---
    num_yeo_networks = 17
    n_jobs = 12
    num_regions = 246
    
    # Entropy strengths to iterate over
    entropy_strengths = [0.002*i for i in range(1, 5)] + [0.02*i for i in range(1, 5)] + [0.2*i for i in range(1, 5)]
    
    # Edge ratios to process
    expected_edge_ratios = [i * 0.02 for i in range(5, 26)]
    ratio_strings = [f'{ratio:.02f}' for ratio in expected_edge_ratios]
    
    # --- Path Setup ---
    base_in_folder = Path('graphs-var') / 'FunImgARglobalCWSF' / 'bna'
    
    base_out_folder = Path('graphs-var') / 'FunImgARglobalCWSF' / 'bna-feat'

    os.makedirs(base_out_folder, exist_ok=True)
    
    # --- Network Mapping ---
    df_yeo = pd.read_csv('yeo.csv')
    
    grouped = df_yeo.groupby(f'Yeo_{num_yeo_networks}network')
    network_to_region_indices = {f'N{net:02d}': np.array(idx) for net, idx in grouped.indices.items()}
    network_names = sorted(list(network_to_region_indices.keys()))

    # --- Main Loop over entropy Strengths ---
    for ent_strength in entropy_strengths:
        # String formatting for path (4 decimal places)
        param_str = f'{ent_strength:.04f}'
        current_input_dir = base_in_folder / param_str
        
        # Optimization: Pre-group files by edge ratio to avoid scanning all files in every worker
        all_filenames = [f for f in os.listdir(current_input_dir) if f.endswith('.npy')]
        
        ratio_to_files = defaultdict(list)
        for fname in all_filenames:
            # Extract ratio from filename (assumes format: ...-Ratio.npy)
            ratio = os.path.splitext(fname)[0].split('-')[-1]
            ratio_to_files[ratio].append(fname)
        
        # Prepare parallel tasks
        list_for_jobs = []
        for ratio_str in ratio_strings:
            current_files = ratio_to_files.get(ratio_str, [])
            if not current_files:
                continue # Skip if no files for this ratio
            
            list_for_jobs.append(
                delayed(process_single_ratio)(
                    edge_ratio_str=ratio_str,
                    input_dir=current_input_dir,
                    files_for_ratio=current_files,
                    network_names=network_names,
                    network_indices=network_to_region_indices,
                    num_regions=num_regions
                )
            )
        
        # Execute Parallel Processing
        if n_jobs == 1:
            results = [job_func() for job_func in list_for_jobs]
        else:
            if list_for_jobs:
                with joblib.parallel_config(n_jobs=n_jobs, backend='loky', inner_max_num_threads=1):
                    results = Parallel(verbose=10)(list_for_jobs)
            else:
                results = []

        # Save results
        if not results:
            print(f"No results found for L2 strength {param_str}.")
            continue
            
        for df_rat, ratio_str in results: # type: ignore
            csv_dir = base_out_folder / param_str
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / f'Yeo{num_yeo_networks}-{ratio_str}.csv'
            df_rat.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f'Edge ratio {ratio_str} processed and saved.')

        print(f"All feature extraction completed for L2 strength {param_str}.")
