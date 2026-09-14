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
from learn_graph_utility import PearsonCorrelationLearner, SingleGraphLearnerWrapper


def process_graphs_by_ratio(
    infolder: str | Path,
    outfolder: str | Path,
    expected_edge_ratios: List[float],
    mask_folders: List[str],
    remove_mean: bool = False,
    n_jobs: int = 1
):
    """
    Process graph learning (Pearson FC) across different mask folders.

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
    remove_mean : bool
        Whether to remove mean from signals.
    n_jobs : int
        Number of parallel jobs. If 1, runs sequentially.
    """
    infolder = Path(infolder)
    outfolder = Path(outfolder)

    # 1. Collect all jobs and pre-create directories
    # Pre-creating directories avoids race conditions in multiprocessing.
    all_jobs = []

    for mask_name in mask_folders:
        mask_out_dir = outfolder / mask_name
        mask_out_dir.mkdir(parents=True, exist_ok=True)

        mask_in_dir = infolder / mask_name
        
        # Skip if input mask directory doesn't exist
        if not mask_in_dir.exists():
            print(f"Warning: Input directory not found: {mask_in_dir}")
            continue

        # Iterate over groups (subdirectories), filtering out hidden ones
        groups = [d.name for d in mask_in_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        for group in groups:
            group_path = mask_in_dir / group
            
            # Find all CSV files in the group folder
            csv_files = list(group_path.glob("*.csv"))

            for data_file in csv_files:
                # Construct output filename prefix
                # Pattern: {Group}-{OriginalFilename}
                out_prefix = f"{group}-{data_file.stem}"
                full_out_prefix = mask_out_dir / out_prefix

                all_jobs.append((data_file, full_out_prefix))

    # 2. Execute Jobs (Parallel or Sequential)
    if n_jobs > 1:
        with joblib.parallel_config(backend='loky', n_jobs=n_jobs, inner_max_num_threads=1):
            results = joblib.Parallel(verbose=10)(
                joblib.delayed(SingleGraphLearnerWrapper)(
                    infile, outfile_prefix, expected_edge_ratios,
                    learner_maker=PearsonCorrelationLearner,
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
                    learner_maker=PearsonCorrelationLearner,
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
        print(f"\nAll completed! {len(all_jobs)} FC matrices saved to {outfolder}")


def main():
    # --- Configuration ---
    remove_mean = True  # Mean value removal is required for Pearsons's method

    # Define paths
    base_in_path = Path('..') / "data-excel"
    
    # Select subfolder based on 'remove_mean' flag
    if not remove_mean:
        base_out_path = Path("graphs-pearson") / "FunImgARglobalCWSF"
    else:
        base_out_path = Path("graphs-pearson") / "FunImgARglobalCWSF-0"

    expected_edge_ratios = [0.02 * i for i in range(5, 26)]
    mask_folders = ['bna']

    # Ensure output directory exists
    base_out_path.mkdir(parents=True, exist_ok=True)

    # Run processing
    process_graphs_by_ratio(
        infolder=base_in_path,
        outfolder=base_out_path,
        expected_edge_ratios=expected_edge_ratios,
        mask_folders=mask_folders,
        remove_mean=remove_mean,
        n_jobs=1
    )


if __name__ == "__main__":
    main()

