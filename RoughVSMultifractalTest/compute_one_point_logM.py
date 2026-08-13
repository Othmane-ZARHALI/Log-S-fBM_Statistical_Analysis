"""
===========================================================
File        : compute_one_point_logM.py
Project     : rough-vs-multifractal-hypothesis-testing
Authors     : Othmane Zarhali
Created     : 2026
Description :
    Computes the empirical power of the DIRECT log-M test for a SINGLE
    (H, n) pair, and appends the result as one JSON line to
    power_results_logM.jsonl. Intended to be run as a fresh process per
    data point -- established as necessary (memory / stability) for
    large n in the earlier p_hat^eps version of this study.
    This module defines:
      - main, the command-line entry point computing and appending a
        single (H, n) power point.

Mathematical background
------------------------
The empirical power reported is that of the direct log-M test statistic

    Z_n := Sigma_n / (lambda * sqrt(V_n(0))),
    Sigma_n = sum_{j=1}^n ln(M_j / Delta),

as defined and studied at fixed n in hypothesis_test_logM.py. This script
computes a single power point pi_hat(H, n) := P(|Z_n| > z_crit | H) at a
given alternative H and sample size n, using the chunked, memory-efficient
simulator simulate_Y_batch (see efficient_field_sim.py) rather than the
dense Cholesky simulator, to handle the large T = 20000 regime used here.

Requires ``efficient_field_sim.py`` (simulate_Y_batch) and
``core_simulation.py`` (Vn_formula) in the same directory -- see those
modules' docstrings if they still raise NotImplementedError.

Usage
-----
    python compute_one_point_logM.py <H> <n>

Example
-------
    python compute_one_point_logM.py 0.2 10

References
----------
Bolko, A. E., Christensen, K., Pakkanen, M. S., Veliyev, B. (2020). "A GMM
    approach to estimate the roughness of stochastic volatility."
    arXiv:2010.04610.
Zarhali, O., Bacry, E., Muzy, J.-F. (2026). "From rough to multifractal
    multidimensional volatility: A multidimensional Log S-fBM model."
    arXiv:2601.10517.
===========================================================
"""

import sys
import json
import numpy as np
from scipy.stats import norm

from efficient_field_sim import simulate_Y_batch
from core_simulation import Vn_formula

# ----------------------------------------------------------------------
# Fixed parameters
# ----------------------------------------------------------------------
DELTA = 1.0
T = 20_000.0
LAM = 0.1
ALPHA = 0.05
Z_CRIT = norm.ppf(1 - ALPHA / 2)
H_PROXY = 0.05  # corrected: H=0.01 has a genuine numerical conditioning
                 # issue (verified in both the Cholesky and FFT-based
                 # simulators), fixed by using H=0.05.
PTS_PER_WINDOW = 50
SIGMA2 = 1.0
N_CALIB = 500
N_REPS = 300
CHUNK = 25

OUTPUT_FILE = 'power_results_logM.jsonl'


def main():
    if len(sys.argv) != 3:
        sys.exit(f"Usage: python {sys.argv[0]} <H> <n>")

    H_alt = float(sys.argv[1])
    n = int(sys.argv[2])

    Ln = n * DELTA
    V0 = Vn_formula(H_PROXY, Ln, T)

    # --- calibration batch: estimate the centering mean under H_PROXY ---
    Y_calib = simulate_Y_batch(n, DELTA, T, H_PROXY, LAM, n_reps=N_CALIB,
                                pts_per_window=PTS_PER_WINDOW, sigma2=SIGMA2,
                                seed=10_000 + n, chunk_size=CHUNK)
    mu_hat = Y_calib.mean()
    del Y_calib

    # --- main batch: evaluate the test statistic at the alternative H ---
    Y_main = simulate_Y_batch(n, DELTA, T, H_alt, LAM, n_reps=N_REPS,
                               pts_per_window=PTS_PER_WINDOW, sigma2=SIGMA2,
                               seed=20_000 + n, chunk_size=CHUNK)
    Y = Y_main - mu_hat
    Sigma_n = Y.sum(axis=1)
    Zn = Sigma_n / (LAM * np.sqrt(V0))
    power_emp = float(np.mean(np.abs(Zn) > Z_CRIT))

    result = {'H': H_alt, 'n': n, 'power': power_emp}
    with open(OUTPUT_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')

    print(f"H={H_alt}, n={n}: power={power_emp:.4f}  [saved to {OUTPUT_FILE}]")


if __name__ == "__main__":
    main()
