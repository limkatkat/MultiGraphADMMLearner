# MultiGraphADMMLearner

Scalable smoothness-based graph learning with ADMM for functional brain network inference and insomnia classification.

## Requirements

- Python 3.12
- numpy
- pandas
- numba
- statsmodels
- matplotlib

Install the dependencies with:

```bash
pip install numpy pandas numba statsmodels matplotlib


Repository Structure
exp0_*.py — Wall-clock experiments for the ADMM solver.

exp1_*.py — Experiments corresponding to Section 4.1 (convergence behavior comparison).

exp2_*.py — Experiments corresponding to Section 4.2.1 (proposed models for functional connectivity).

cls_step*_*.py — Full classification pipeline, including feature extraction, nested cross-validation, stability selection, and evaluation.

Experiments
exp0: Wall-clock experiments for ADMM
Measures the runtime of the proposed ADMM solver under different graph sizes and sparsity levels.

exp1: Section 4.1 — Convergence behavior comparison
Compares the convergence behavior of the proposed ADMM solver against baseline solvers for the log-degree model. The comparison is performed on resting-state fMRI data across different parcellation resolutions and target edge densities.

exp2: Section 4.2.1 — Proposed models for functional connectivity
Implements the Logdeg-Entropy and Logdeg-Variance models. The scripts construct functional connectivity graphs, extract topological features, and prepare the inputs for the classification pipeline.

cls_step_.py: Classification pipeline
The complete classification workflow consists of a sequence of scripts prefixed with cls_step. Each script corresponds to one stage of the pipeline, including:


Data
The rs-fMRI data used in this study cannot be made publicly available due to ethical and legal restrictions. The original data acquisition protocols were approved by the Medical Ethics Committee of The Third Hospital of Inner Mongolia Autonomous Region. The Institutional Review Board of the same hospital determined that this retrospective analysis of de-identified data was exempt from ethics review and from the requirement for informed consent. In addition, Chinese laws and regulations on information security and personal data protection prevent public sharing of human-subject data.

License
This project is licensed under the MIT License. See the LICENSE file for details.
