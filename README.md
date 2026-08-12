# rough-vs-multifractal-hypothesis-testing

Monte Carlo code accompanying a statistical hypothesis-testing study of
**H = 0 vs. H ≠ 0** for a fractional/logarithmic log-volatility field —
i.e. distinguishing a *rough* (Hurst-type) volatility model from its
*multifractal* / log-correlated (H → 0) boundary case — together with a
standalone Central Limit Theorem illustration used as a pedagogical
warm-up.

Full mathematical documentation is in [`docs/documentation.pdf`](docs/documentation.pdf)
(built from [`docs/documentation.tex`](docs/documentation.tex)).

## Repository layout

```
.
├── clt_demo/
│   └── clt_illustration.py        # Monte Carlo illustration of the CLT
├── hurst_test/
│   ├── reference_simulator.py     # FFT/circulant-embedding field simulator + Figs 1-3
│   ├── core_simulation.py         # ⚠ interface stub, see "Missing dependencies" below
│   ├── efficient_field_sim.py     # ⚠ interface stub, see "Missing dependencies" below
│   ├── clt_check.py               # CLT check on the log-mass statistic
│   ├── regularized_indicator.py   # regularized vs. hard indicator statistic comparison
│   ├── hypothesis_test_regularized.py  # T_n test (Theorem 24), size & power
│   ├── hypothesis_test_logM.py         # direct log-M test Z_n, size & power
│   └── compute_one_point_logM.py       # single (H, n) power point, for batch/cluster runs
├── docs/
│   ├── documentation.tex          # LaTeX source of the mathematical documentation
│   └── documentation.pdf          # compiled documentation
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### CLT illustration

```bash
cd clt_demo
python clt_illustration.py
```

Simulates `Z_n = (X_1 + ... + X_n)/sqrt(n)` for i.i.d.
`X_i ~ Uniform(-sqrt(3), sqrt(3))` (mean 0, variance 1) across
`n ∈ {1, 2, 5, 10, 20, 50, 100}`, and overlays each empirical histogram
against the `N(0,1)` density.

### Reference field simulator (self-contained, runs out of the box)

```bash
cd hurst_test
python reference_simulator.py
```

Reproduces the three independent-fields diagnostics:

1. `std(R)` vs. `λ` (log-log, weighted power-law fit) — `std_vs_lambda.{pdf,png}`
2. 4-panel histogram of `log(IV)` vs. `λΩ` — `histogram_comparison.{pdf,png}`
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

## ⚠ Missing dependencies

The uploaded source imported `build_chol`, `simulate_M`, `Vn_formula`
(from a module `core_simulation`) and `simulate_Y_batch` (from a module
`efficient_field_sim`) that were **referenced but never defined** in the
material provided. `hurst_test/core_simulation.py` and
`hurst_test/efficient_field_sim.py` in this repo are **interface stubs**:
their docstrings document the expected signature and behaviour
(reconstructed from how each function is called at the various call
sites), but the function bodies raise `NotImplementedError` rather than
guessing at the Cholesky-based simulation internals — filling those in
incorrectly would silently corrupt a hypothesis-testing pipeline.

`reference_simulator.py` (and hence `clt_demo/clt_illustration.py`,
which has no such dependency) run correctly out of the box. The five
other scripts under `hurst_test/` will raise `NotImplementedError` with
a pointer back to the relevant docstring until the original
implementations of these two modules are dropped in.

## Method summary

See `docs/documentation.pdf` for full derivations. In brief:

- The log-volatility field has covariance transitioning from a
  power-law (fractional, Hurst exponent `H`) kernel to a **logarithmic**
  kernel as `H → 0` — the boundary between rough and multifractal
  behaviour.
- Two competing test statistics for `H = 0` vs. `H ≠ 0` are studied:
  a regularized-proportion statistic `T_n` (Theorem 24) and a direct
  log-mass statistic `Z_n`; both are asymptotically `N(0,1)` under `H_0`
  after normalization by the theoretical variance `V_n(·)`, with
  `Z_n` showing better finite-sample power.
- `reference_simulator.py` validates, under the null (statistically
  independent fields), that the residual `std(R)` scales linearly in
  `λ = √(λ²)` as predicted.

## License

MIT — see individual file headers for provenance notes.
