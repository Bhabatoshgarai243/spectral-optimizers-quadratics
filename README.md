# Reproducibility package: Muon on quadratics

This repository contains the numerical experiments used in the manuscript
**Constant Step Sizes for Muon-Type Spectral Optimizers on Strongly Convex
Quadratics**. The experiments compare spectral gradient shaping methods on
quadratic objectives and reproduce the paper's figures.

The code is intentionally small and self-contained:

| Experiment | Script | Main result |
| --- | --- | --- |
| P1/P2 combined figure | `P1_2.py` | `P1_P2_combined.pdf` |
| P3: Newton–Schulz, polar, PolarGrad, and GD comparison | `P3.py` | `P3_updated_polargrad.pdf` |
| P4: local rate and denominator constant | `P4.py` | `P4_results.md` |

All random experiments use the explicit seed `20260912`. The matrix dimensions,
condition number, iteration counts, step-size grids, and fitting windows are
defined in the corresponding scripts and are not supplied through hidden
configuration files.

## Quick start

### Option A: pip and a virtual environment

Python 3.10--3.13 is supported.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_experiments.py
```

On macOS/Linux, replace the activation command with
`source .venv/bin/activate`.

### Option B: conda/mamba

```powershell
conda env create -f environment.yml
conda activate muon-quadratics
python run_experiments.py
```

The runner uses Matplotlib's non-interactive `Agg` backend, so it works on
headless machines and in continuous integration. It executes each experiment
in a clean output directory and writes the generated files to `results/`.

To choose another output directory:

```powershell
python run_experiments.py --output-dir results\my-run
```

The run takes a few minutes on a typical laptop because the experiments use
long iteration loops and repeated SVDs.

## Expected outputs

After a successful run, `results/` contains:

```text
P1_P2_combined.pdf
P3_updated_polargrad.pdf
P4_results.md
```

P4 should report measured rates close to the theoretical rates:

```text
beta = 1.00  -> 0.966667
beta = 1.90  -> 0.936667
beta = 2.05  -> 1.050000
```

The measured rates and epsilon spreads are printed and saved by `P4.py`.
Small floating-point differences across operating systems and BLAS
implementations are expected; the qualitative conclusions and reported rates
should remain unchanged.

## Repository layout

```text
.
├── P1_2.py
├── P3.py
├── P4.py
├── run_experiments.py
├── requirements.txt
├── environment.yml
├── P1_P2_combined.pdf
├── P3_updated_polargrad.pdf
└── P4_results.md
```

The SVG figures and `P4_results.md` at the repository root are reference
artifacts from the supplied manuscript experiments. P3 now produces a PDF
figure because the updated experiment includes five curves and an inset.
New runs should be written under `results/` so that reference files are not
overwritten.

## Reproducibility notes

* NumPy is used for all linear algebra and random number generation.
* Matplotlib is used only for figure generation.
* The random generator is NumPy's `default_rng` with seed `20260912`.
* The quadratic Hessians have eigenvalues logarithmically spaced between
  `1/30` and `1`, hence condition number 30.
* P3 uses a dense orthogonally rotated Hessian; P1/P2 and P4 use diagonal
  Hessians.
* No data download, network access, GPU, or proprietary software is required.

## Citation

If you use this code, please cite the associated ICASSP 2027 manuscript:

> Bhabatosh [authors and final bibliographic information to be added],
> “Constant Step Sizes for Muon-Type Spectral Optimizers on Strongly Convex
> Quadratics.”

Please replace the placeholder author/bibliographic information above with the
final camera-ready citation before publishing the repository.

## License

This repository is released under the [MIT License](./LICENSE).
