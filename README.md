# JESPRIT-Sketching — Poisson mixture estimation from PGF sketches

Estimating the parameters of a multivariate **mixed Poisson distribution** from
a compressed *sketch* of the data, with a joint ESPRIT (JESPRIT) algorithm.
Mathematical-modelling *pôle projet* at CentraleSupélec, first semester
2025-2026.

## Problem

Given `n` samples of `d`-dimensional counts drawn from a mixture of `r`
independent Poisson components, recover the rate vectors `λ_k ∈ ℝ^d` and the
mixture weights `π_k`. Likelihood-based methods such as EM are iterative and
their cost grows with `n`. Here the data is summarised once, and the estimation
works only on that summary.

## Method

1. **PGF sketch.** The empirical probability-generating function
   `G(z) = mean(∏_j z_j^{x_j})` is sampled along `M` random directions, with
   `S` base points per direction and `N` evaluation points per line, on a grid
   scaled by `δ`. For a Poisson mixture the PGF is a sum of exponentials,
   `G(z) = Σ_k π_k exp(⟨λ_k, z − 1⟩)`. Estimating the rates therefore becomes a
   harmonic-retrieval problem.
2. **Global subspace.** The sketches from all directions are stacked and a
   single SVD gives a common signal subspace of dimension `r`.
3. **Rotational invariance (ESPRIT).** Each direction yields a shift-invariance
   matrix `Ψ_l` whose eigenvalues encode the projection of every rate vector on
   that direction.
4. **Joint diagonalisation.** All `Ψ_l` are diagonalised jointly. Because they
   share eigenvectors, the projections are paired across directions without a
   combinatorial matching step.
5. **Recovery.** Rate vectors come from a linear system in the projections. The
   weights come from least squares.

Numerical experiments studied the number of directions, samples per direction,
grid scale and total sample size. Two findings: the phase-unwrapping correction
considered at first could *hurt* the estimate, and the grid scale has to match
the order of magnitude of the rates. Recovery got harder mainly as the number of
components `r` grew, much less as the dimension `d` grew.

## Repository structure

```text
src/
├── dataset_gen.py      # sample from a mixed Poisson distribution
├── pgf.py              # empirical PGF and its sampling along random lines (sketch)
├── esprit.py           # 1D ESPRIT
├── joint_diag.py       # joint diagonalisation of commuting matrices
└── jesprit.py          # full JESPRIT estimator + error metrics
script/
├── test_*.py                    # unit-style checks of each block
├── check_subspaces.py           # sanity check of the estimated subspaces
├── evaluate_parameters.py       # sweep over M, S, N, δ  -> log/evaluate_parameters.*
└── analyze_sample_complexity.py # error vs n, d, r      -> log/sample_complexity.*
log/                    # experiment outputs (text + figures)
Sketching-Latex/        # theory notes: mixed Poisson, PGF, ESPRIT, JESPRIT, results
final-report/           # final report (LaTeX + PDF)
final-presentation/     # final presentation (Beamer + PDF)
ckim/                   # follow-up material: short-paper draft and a note on a later codebase
```

## Running

```bash
pip install -e .          # installs numpy, scipy, pandas, matplotlib, seaborn, joblib
python script/test_jesprit.py
python script/evaluate_parameters.py
python script/analyze_sample_complexity.py
```

Run the scripts from the repository root, because they write to `log/`. They
import the modules in `src/` directly (`from jesprit import jesprit`), so
either install the package as above or put `src/` on the path:

```bash
PYTHONPATH=src python script/test_jesprit.py
```

Minimal use:

```python
import numpy as np
from dataset_gen import generate_mixed_poisson_samples
from jesprit import jesprit

A = np.eye(3)                                # d x m mixing matrix (rates = A @ z_k)
z = np.array([[100, 30, 1], [1, 100, 1], [1, 1, 100]], float)  # m x r latent factors
pi = np.array([0.7, 0.2, 0.1])               # mixture weights
X, _ = generate_mixed_poisson_samples(A, pi, z, n_samples=10_000)

rates_hat, weights_hat = jesprit(X, r=3, M=20, S=18, N=18, delta=0.01)
```
