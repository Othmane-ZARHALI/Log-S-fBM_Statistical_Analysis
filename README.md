# Rough VS Multifractal Hypothesis Testing

This repository contains the Python implementation and numerical experiments associated with a statistical hypothesis-testing study for the Hurst exponent of a log-volatility field, it is mainly based on the papers:

**"From Rough to Multifractal volatility: the Log S-fBM model"**
Peng Wu, Jean-François Muzy, Emmanuel Bacry (2022)
arXiv: https://arxiv.org/abs/2201.09516

**"From rough to multifractal multidimensional volatility: A multidimensional Log S-fBM model"**
Othmane Zarhali, Emmanuel Bacry, Jean-François Muzy (2026)
arXiv: https://arxiv.org/abs/2601.10517

**"A GMM approach to estimate the roughness of stochastic volatility"**
Anine E. Bolko, Kim Christensen, Mikko S. Pakkanen, Bezirgen Veliyev (2020)
arXiv: https://arxiv.org/abs/2010.04610

The code allows the simulation of the log-volatility field, and the numerical study of two statistical tests for **H = 0 vs. H ≠ 0**, i.e. for distinguishing the *rough volatility* regime from its *multifractal* / log-correlated boundary case, together with a standalone Central Limit Theorem illustration used as a pedagogical warm-up.

---

## Overview

The log-volatility field $\omega(t)$ studied throughout this repository is a centred stationary Gaussian process with autocovariance

$
c_H(x) =
\begin{cases}
\lambda^2 \log\dfrac{T}{x}, & H=0,\ 0<x<T,\\[1ex]
\dfrac{\lambda^2}{2H(1-2H)}\Big(T^{2H}-x^{2H}\Big), & H\neq0,\ 0<x<T,\\[1ex]
0, & x\ge T,
\end{cases}
$

so that the same covariance family interpolates between:

- **Rough volatility regime:** $0 < H < \frac12$
- **Multifractal regime:** $H = 0$, the logarithmic / log-correlated boundary case

The main parameters are:

- $H$: Hurst exponent controlling roughness
- $T$: correlation (memory) scale
- $\lambda^2$: volatility-of-volatility (intermittency) coefficient
- $n, \Delta$: number and length of the aggregation windows used to build the test statistics

On each window $j$, two block quantities are defined:

$$
M_j := \int_{\text{window }j} e^{2\omega(t)}\,dt, \qquad
\Omega_j := \int_{\text{window }j} \big(\omega(t)-\overline\omega\big)\,dt,
$$

and the centred log-mass statistic $\Sigma_n := \sum_{j=1}^n(\log M_j-\mathbb E[\log M_j])$ satisfies, under both the null and the alternative,

$$
\Sigma_n/\lambda \;\xrightarrow{\ d\ }\; \mathcal N\big(0,V_n(H)\big),
$$

for a closed-form asymptotic variance $V_n(H)$. Two competing test statistics for $H_0: H=0$ vs. $H_1: H\neq0$ are studied and compared:

- a **regularized-proportion statistic** $T_n$ (Theorem 24), built from a Gaussian-CDF-smoothed indicator of the log-mass exceeding a threshold;
- a **direct log-mass statistic** $Z_n := \Sigma_n/(\lambda\sqrt{V_n(0)})$, which shares the same asymptotic power as $T_n$ but shows noticeably better finite-sample power and needs no threshold or bandwidth to choose.

---

## Repository layout

```
.
├── clt_demo/
│   └── clt_illustration.py             # Monte Carlo illustration of the CLT
├── hurst_test/
│   ├── reference_simulator.py          # FFT/circulant-embedding field simulator + independent-fields diagnostics
│   ├── core_simulation.py              # ⚠ interface stub, see "Missing dependencies" below
│   ├── efficient_field_sim.py          # ⚠ interface stub, see "Missing dependencies" below
│   ├── clt_check.py                    # CLT check on the log-mass statistic
│   ├── regularized_indicator.py        # regularized vs. hard indicator statistic comparison
│   ├── hypothesis_test_regularized.py  # T_n test (Theorem 24), size & power
│   ├── hypothesis_test_logM.py         # direct log-M test Z_n, size & power
│   └── compute_one_point_logM.py       # single (H, n) power point, for batch/cluster runs
├── docs/
│   ├── documentation.tex               # LaTeX source of the mathematical documentation
│   └── documentation.pdf               # compiled documentation
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### CLT illustration

```bash
cd clt_demo
python clt_illustration.py
```

Simulates $Z_n = (X_1 + \dots + X_n)/\sqrt n$ for i.i.d. $X_i \sim \mathrm{Uniform}(-\sqrt3,\sqrt3)$ (mean 0, variance 1) across $n \in \{1, 2, 5, 10, 20, 50, 100\}$, and overlays each empirical histogram against the $\mathcal N(0,1)$ density.

### Reference field simulator (self-contained, runs out of the box)

```bash
cd hurst_test
python reference_simulator.py
```

Simulates two *independent* realizations of $\omega(t)$ sharing the same covariance law, and reproduces the three independent-fields null-hypothesis diagnostics:

1. $\mathrm{std}(R)$ vs. $\lambda$ (log-log, weighted power-law fit) — `std_vs_lambda.{pdf,png}`
2. 4-panel histogram of $\log(\mathrm{IV})$ vs. $\lambda\Omega$ — `histogram_comparison.{pdf,png}`
3. Empirical QQ-plots of the same two quantities — `qq_plots.{pdf,png}`

### Hypothesis-testing scripts (require the Cholesky-based simulator)

```bash
cd hurst_test
python clt_check.py
python regularized_indicator.py
python hypothesis_test_regularized.py
python hypothesis_test_logM.py
python compute_one_point_logM.py <H> <n>
```

`hypothesis_test_regularized.py` and `hypothesis_test_logM.py` each report the empirical size at $H=0$ and the empirical vs. theoretical power curve across a grid of $H\neq0$ values, for the $T_n$ and $Z_n$ statistics respectively. `compute_one_point_logM.py` computes a single $(H, n)$ power point and appends it to `power_results_logM.jsonl`, for building a power curve via a job array over a large $(H, n)$ grid.

---

## ⚠ Missing dependencies

The source this repository was built from imported `build_chol`, `simulate_M`, `Vn_formula` (from a module `core_simulation`) and `simulate_Y_batch` (from a module `efficient_field_sim`) that were **referenced but never defined**. `hurst_test/core_simulation.py` and `hurst_test/efficient_field_sim.py` in this repo are **interface stubs**: their docstrings document the expected signature and behaviour (reconstructed from how each function is called), but the function bodies raise `NotImplementedError` rather than guessing at the Cholesky-based simulation internals.

`reference_simulator.py` (and hence `clt_demo/clt_illustration.py`, which has no such dependency) run correctly out of the box. The five other scripts under `hurst_test/` will raise `NotImplementedError` with a pointer back to the relevant docstring until the original implementations of these two modules are dropped in.

---

## Documentation

Full mathematical documentation — covariance model, both test statistics with their asymptotics, per-function parameter tables, algorithm steps, and implementation notes — is in [`docs/documentation.pdf`](docs/documentation.pdf) (built from [`docs/documentation.tex`](docs/documentation.tex)).

---

## References

1. Wu, P., Muzy, J.-F., Bacry, E. (2022). "From rough to multifractal volatility: The log S-fBM model." *Physica A: Statistical Mechanics and its Applications*, 604. arXiv: https://arxiv.org/abs/2201.09516
2. Zarhali, O., Bacry, E., Muzy, J.-F. (2026). "From rough to multifractal multidimensional volatility: A multidimensional Log S-fBM model." arXiv: https://arxiv.org/abs/2601.10517
3. Bolko, A. E., Christensen, K., Pakkanen, M. S., Veliyev, B. (2020). "A GMM approach to estimate the roughness of stochastic volatility." arXiv: https://arxiv.org/abs/2010.04610
4. Gatheral, J., Jaisson, T., Rosenbaum, M. (2018). "Volatility is rough." *Quantitative Finance*, 18(6), 933–949.
5. Wood, A. T. A., Chan, G. (1994). "Simulation of Stationary Gaussian Processes in $[0,1]^d$." *J. Comput. Graph. Statist.*, 3(4), 409–432.

---

## License

MIT — see individual file headers for provenance notes.
