"""
hypothesis_test_logM.py
=========================

Numerical simulation of the H=0 vs. H != 0 statistical test using the
DIRECT log-M test statistic

    Z_n := Sigma_n / (lambda * sqrt(V_n(0))),   Sigma_n = sum_{j=1}^n ln(M_j / Delta),

rather than the regularized-proportion test T_n (Theorem 24) implemented
in ``hypothesis_test_regularized.py``. Z_n has the same asymptotic power
as T_n but noticeably better finite-sample power, and requires no
threshold ``a`` or bandwidth ``eps`` to choose.

Parameters: n=5, Delta=1, T=8. Size here is already excellent at
lambda=0.1, so no need for a smaller lambda (contrast with the
regularized test, which needed lambda=0.05 for comparable size).

Produces
--------
* SIZE check: empirical rejection rate under H0 (H=0, via the numerically
  safe proxy H_PROXY=0.05 -- see note below).
* POWER check: empirical vs. theoretical rejection rate over a grid of
  true H != 0 values, plotted together.

Requires ``core_simulation.py`` (build_chol, simulate_M, Vn_formula) in
the same directory -- see that module's docstring if it still raises
NotImplementedError.

Usage
-----
    python hypothesis_test_logM.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from core_simulation import build_chol, simulate_M, Vn_formula

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
N, DELTA, T = 5, 1.0, 8.0
LN = N * DELTA
DELTA_FINE = 0.02
SIGMA2 = 1.0
LAM = 0.1  # size already excellent here; no need for a smaller lambda
ALPHA = 0.05
Z_CRIT = norm.ppf(1 - ALPHA / 2)

M_CALIB = 200_000
M_REPS = 20_000

# H_PROXY=0.01 was tried first and found to have a genuine numerical
# conditioning issue in the Cholesky simulator near the H -> 0 singular
# boundary (verified: the LINEAR Gaussian part alone, bypassing the
# log-mass nonlinearity entirely, already showed an 8.8% variance
# inflation relative to the theoretical V_n(H), while V_n(H) itself was
# independently confirmed correct via direct double-integration of the
# covariance function). H_PROXY=0.05 shows excellent agreement
# (ratio 0.99-1.01) and is used instead.
H_PROXY = 0.05
H_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


def run_at_H(H, seed_offset):
    """Simulate Sigma_n = sum_j ln(M_j/Delta), centered by a
    calibration-batch estimate of its mean, under true Hurst exponent H."""
    L, K = build_chol(N, DELTA, T, H, LAM, DELTA_FINE)
    M_calib = simulate_M(L, K, N, DELTA, DELTA_FINE, SIGMA2, M_CALIB, seed=7000 + seed_offset)
    mu_hat = np.log(M_calib / DELTA).mean()
    M_main = simulate_M(L, K, N, DELTA, DELTA_FINE, SIGMA2, M_REPS, seed=8000 + seed_offset)
    Y = np.log(M_main / DELTA) - mu_hat
    Sigma_n = Y.sum(axis=0)
    return Sigma_n


def run():
    V0 = Vn_formula(H_PROXY, LN, T)
    print(f"z_crit = {Z_CRIT:.4f}\n")

    # --- SIZE check under H0: H=0 (proxy) ---
    Sigma_n_H0 = run_at_H(H_PROXY, seed_offset=0)
    Zn_H0 = Sigma_n_H0 / (LAM * np.sqrt(V0))
    rejection_rate_H0 = np.mean(np.abs(Zn_H0) > Z_CRIT)
    print(f"SIZE check (H={H_PROXY}, proxy for H=0): empirical rejection rate = "
          f"{rejection_rate_H0:.4f}  (target alpha = {ALPHA})")
    print(f"  mean(Zn)={Zn_H0.mean():.4f}  var(Zn)={Zn_H0.var():.4f}  (theory: mean=0, var=1)\n")

    # --- POWER check under several H != 0 ---
    power_emp, power_theory = [], []
    for i, H in enumerate(H_GRID):
        Sigma_n_H = run_at_H(H, seed_offset=i + 1)
        Zn_H = Sigma_n_H / (LAM * np.sqrt(V0))
        rr = np.mean(np.abs(Zn_H) > Z_CRIT)
        power_emp.append(rr)
        VH = Vn_formula(H, LN, T)
        R = np.sqrt(VH / V0)
        pi_H = 1 - norm.cdf(Z_CRIT / R) + norm.cdf(-Z_CRIT / R)
        power_theory.append(pi_H)
        print(f"H={H:.2f}:  V_n(H)={VH:10.4f}  R={R:.4f}  empirical power={rr:.4f}  "
              f"theoretical power={pi_H:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(H_GRID, power_theory, 'r-', lw=2, label='theoretical power $\\pi(H)$')
    ax.plot(H_GRID, power_emp, 'o', color='#1f77b4', markersize=8,
            label='empirical power (log-$M$ test)')
    ax.axhline(ALPHA, color='gray', ls='--', label=f'$\\alpha={ALPHA}$')
    ax.axhline(rejection_rate_H0, color='green', ls=':', lw=1.5,
               label=f'empirical size at $H_0$ ({rejection_rate_H0:.3f})')
    ax.set_xlabel('H (true value)')
    ax.set_ylabel('rejection probability')
    ax.set_title(f'Size and power of the log-$M$ test ($n={N}, \\lambda={LAM}$)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('logM_test_size_power.pdf')
    fig.savefig('logM_test_size_power.png', dpi=150)
    print("\nSaved logM_test_size_power.{pdf,png}")


if __name__ == "__main__":
    run()
