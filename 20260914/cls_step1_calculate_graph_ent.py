import os
# --- Environment Configuration ---
# Force single-threading for BLAS/MKL/OpenBLAS to prevent thread oversubscription 
# when using Python multiprocessing (joblib).
os.environ.update({
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1"
})



from pathlib import Path
from typing import List

import joblib

# Import custom modules
from learn_graph_utility import PrepareLogDegreeEntropyGraphLearner, SingleGraphLearnerWrapper


def process_graphs_by_ratio(
    infolder: str | Path,
    outfolder: str | Path,
    expected_edge_ratios: List[float],
    mask_folders: List[str],
    entropy_strengths: List[float],
    remove_mean: bool = False,
    n_jobs: int = 1
):
    """
    Process graph learning for specific entropy strength values across different mask folders.
    
    Parameters
    ----------
    infolder : str | Path
        Root directory containing input data.
    outfolder : str | Path
        Root directory for output results.
    expected_edge_ratios : list[float]
        List of edge ratios to compute.
    mask_folders : list[str]
        List of sub-folders (masks) to process.
    entropy_strengths : list[float]
        List of entropy regularization strengths to iterate over.
    remove_mean : bool
        Whether to remove mean from signals.
    n_jobs : int
        Number of parallel jobs. If 1, runs sequentially.
    """
    infolder = Path(infolder)
    outfolder = Path(outfolder)

    for ent_strength in entropy_strengths:
        all_jobs = []
        
        # 1. Prepare Jobs and Create Directories
        # Pre-creating directories avoids race conditions in multiprocessing.
        for mask_name in mask_folders:
            # Construct output path for this mask and strength (4 decimal places)
            strength_dir_name = f"{ent_strength:.04f}"
            current_out_dir = outfolder / mask_name / strength_dir_name
            current_out_dir.mkdir(parents=True, exist_ok=True)

            # Iterate over groups and files
            mask_data_dir = infolder / mask_name
            # Filter out hidden files and ensure it's a directory
            groups = [d.name for d in mask_data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            
            for group in groups:
                # Find all CSV files in the group folder
                group_path = mask_data_dir / group
                csv_files = list(group_path.glob("*.csv"))
                
                for data_file in csv_files:
                    # Construct output filename prefix
                    # Pattern: {Group}-{OriginalFilename}
                    out_prefix = f"{group}-{data_file.stem}"
                    full_out_prefix = current_out_dir / out_prefix
                    
                    # Add to job list
                    all_jobs.append((data_file, full_out_prefix))

        # 2. Execute Jobs (Parallel or Sequential)
        # Instantiate the learner maker for this specific entropy strength
        learner_maker = PrepareLogDegreeEntropyGraphLearner(ent_strength)
        
        if n_jobs > 1:
            with joblib.parallel_config(backend='loky', n_jobs=n_jobs, inner_max_num_threads=1):
                results = joblib.Parallel(verbose=10)(
                    joblib.delayed(SingleGraphLearnerWrapper)(
                        infile, outfile_prefix, expected_edge_ratios, 
                        learner_maker=learner_maker, 
                        remove_mean=remove_mean
                    )
                    for infile, outfile_prefix in all_jobs
                )
        else:
            results = []
            for infile, outfile_prefix in all_jobs:
                results.append(
                    SingleGraphLearnerWrapper(
                        infile, outfile_prefix, expected_edge_ratios, 
                        learner_maker=learner_maker, 
                        remove_mean=remove_mean
                    )
                )

        # 3. Error Reporting
        errors = [r for r in results if r is not True]
        if errors:
            print("\n============= The following errors occurred =============")
            for e in errors:
                print(e)
        else:
            print(f"\nAll completed! {len(all_jobs)} FC matrices saved to {current_out_dir}")


def main():
    # --- Configuration ---
    remove_mean = False # mean value removal is not required for graph learning
    
    # Define paths
    base_in_path = Path(r"../data-excel")
    base_out_path = Path("graphs-ent")
    
    # Select subfolder based on 'remove_mean' flag
    if not remove_mean:
        base_out_path = base_out_path / "FunImgARglobalCWSF"
    else:
        base_out_path = base_out_path / "FunImgARglobalCWSF-0"

    # Parameters
    expected_edge_ratios = [0.02 * i for i in range(5, 26)]
    mask_folders = ['bna']
    entropy_strengths = [0.002*i for i in range(1, 5)] + [0.02*i for i in range(1, 5)] + [0.2*i for i in range(1, 5)]
    
    # Ensure output directory exists
    base_out_path.mkdir(parents=True, exist_ok=True)

    # Run processing
    process_graphs_by_ratio(
        infolder=base_in_path,
        outfolder=base_out_path,
        expected_edge_ratios=expected_edge_ratios,
        mask_folders=mask_folders,
        remove_mean=remove_mean,
        entropy_strengths=entropy_strengths,
        n_jobs=12
    )


if __name__ == "__main__":
    main()
