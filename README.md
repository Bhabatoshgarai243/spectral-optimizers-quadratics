# Constant Step Sizes for Muon-Type Spectral Optimizers on Strongly Convex Quadratics

Reproducibility code for the numerical experiments on strongly convex quadratics.

The repository contains:
* `paper.pdf`: Manuscript PDF
* `code/`: All experiment scripts and runners

---

## Structure

All scripts reside in `code/`:

| Script | Purpose | Output |
| :--- | :--- | :--- |
| `P1_2.py` | Step-size threshold and local rate experiments | `P1_P2_combined.pdf` |
| `P3.py` | Regime comparison on dense quadratics | `P3_updated_polargrad.pdf` |
| `P4.py` | Normalization offset verification | `P4_results.md` |
| `run_experiments.py` | Master runner for all experiments | `results/` |

---

## Quick Start

```bash
cd code

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .\.venv\Scripts\Activate.ps1

# Install requirements and run
pip install -r requirements.txt
python run_experiments.py