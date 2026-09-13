import pandas as pd
from pathlib import Path
import re

def analyze_top_auc_results(model: str):
    """
    Analyze all experimental results and extract records with AUC > 0.7.
    Sorts them from high to low and saves to a CSV file.
    """
    
    # ==================== Configuration Area ====================
    # Base path, corresponding to DATA_ROOT and output directory structure in the original script
    # Assumes the script is running in the project root directory; modify if necessary
    BASE_DIR = Path(f'graphs-{model}')
    
    # Root directory prefix for result files
    # Corresponds to original script: OUTPUT_DIR = DATA_ROOT / f'FunImgARglobalCWSF{SUFFIX}' / f'bna-feat-out-{STABILITY_FREQ_THRESHOLD}'
    # Stability threshold is set to 0.9 here; modify if your settings differ
    RESULT_ROOT = BASE_DIR / 'FunImgARglobalCWSF' / 'bna-feat-out-0.9'
    
    # Output filename
    OUTPUT_FILE = f'Top_AUC_{model}.csv'
    
    # AUC filtering threshold
    AUC_THRESHOLD = 0.6
    # ===========================================================

    all_results = []
    
    print(f"Scanning directory: {RESULT_ROOT.resolve()}")
    
    # Recursively find all CSV files matching the criteria
    # Pattern: ISM-model_eval_results*.csv
    # Using rglob for recursive search (will traverse all entropy subfolders)
    csv_files = list(RESULT_ROOT.rglob("ISM-model_eval_results*.csv"))
    
    if not csv_files:
        print(" No result files found, please check if the path configuration is correct.")
        return

    print(f"Found {len(csv_files)} result files, starting processing...")

    for csv_path in csv_files:
        if 'max_x_std' in str(csv_path):
            continue
        try:
            # 1. Parse metadata from filename
            # Filename format example: ISM-model_eval_results-Yeo17-max-None-None-None.csv
            # Or: ISM-model_eval_results-Yeo7-max_div_std_all-0.05-0.05-False.csv
            filename = csv_path.name
            
            # Use regex to extract key information
            # Matches number following Yeo, and subsequent feature name and parameters
            match = re.search(r'Yeo(\d+)-(.+)-(.+)-(.+)-(.+)\.csv', filename)
            
            if not match:
                print(f"   Filename format mismatch, skipping: {filename}")
                continue
            
            yeo_num = int(match.group(1))
            feature_type = match.group(2)
            # test_params part can be further parsed if needed, kept as a string for now
            
            # Get parent folder name (i.e., entropy_strength)
            entropy_strength = csv_path.parent.name
            
            # 2. Read CSV data
            df = pd.read_csv(csv_path)
            
            # 3. Add metadata columns
            df['entropy_strength'] = entropy_strength
            df['yeo_network'] = yeo_num
            df['feature_type'] = feature_type
            df['source_file'] = filename
            
            # Keep only necessary columns to prevent column name conflicts
            # Original columns contain: edge_ratio, algorithm, auc_mean, etc.
            all_results.append(df)
            
        except Exception as e:
            print(f"   Error processing file {csv_path.name}: {e}")

    if not all_results:
        print("No valid data read.")
        return

    # 4. Concatenate all data
    combined_df = pd.concat(all_results, ignore_index=True)
    
    print(f"Total of {len(combined_df)} experimental records read.")

    # 5. Filter AUC > 0.7
    # Assumes column name is 'auc_mean', no change needed if original script uses 'auc_mean'
    top_df = combined_df[combined_df['auc_mean'] > AUC_THRESHOLD].copy()
    
    if top_df.empty:
        print(f"No results found with AUC > {AUC_THRESHOLD}.")
        return

    # 6. Sort by AUC from high to low
    top_df.sort_values(by='auc_mean', ascending=False, inplace=True)

    # 7. Arrange column order for readability
    # Prioritize key metrics and parameters
    priority_cols = [
        'auc_mean', 'auc_std', 'acc_mean', 'f1_mean', 
        'algorithm', 'edge_ratio', 
        'feature_type', 'yeo_network', 'entropy_strength',
        'source_file'
    ]
    
    # Ensure all columns exist
    final_cols = [col for col in priority_cols if col in top_df.columns]
    # Append remaining columns
    remaining_cols = [col for col in top_df.columns if col not in final_cols]
    
    output_df = top_df[final_cols + remaining_cols]

    # 8. Save results
    output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f" Processing complete!")
    print(f"   Filtered {len(top_df)} records with AUC > {AUC_THRESHOLD}.")
    print(f"   Results saved to: {Path(OUTPUT_FILE).resolve()}")
    
    # Print preview of top 5
    print("\n Top 5 Records Preview:")
    print(output_df[['auc_mean', 'algorithm', 'edge_ratio', 'feature_type', 'yeo_network']].head(5).to_string(index=False))
    print("="*50)

if __name__ == '__main__':
    analyze_top_auc_results(model='ent')
