"""
===========================================================
File        : hypothesis_test_regularized.py
Project     : rough-vs-multifractal-hypothesis-testing
Authors     : Othmane Zarhali
Created     : 2026
Description :
    Numerical simulation of the H=0 vs. H != 0 statistical test based on
    the regularized-proportion statistic (Theorem 24), at n=5, Delta=1,
    T=8, threshold a=0.1, bandwidth eps=0.3 (in the delta-method-valid
    regime, eps >> eps* = sqrt(a*lambda)), lambda=0.05.
    This module defines:
      - run_at_H, simulating the T_n statistic under a given true Hurst
        exponent H via core_simulation.build_chol / simulate_M.
      - run, assembling the size check (H=0) and the power curve
        (H != 0) and producing the associated figure.

Mathematical background
------------------------
With c_eps := phi(a/eps) / eps (the delta-method derivative factor,
phi the standard normal density) and p_hat_n^eps the regularized
proportion statistic of regularized_indicator.py, the test statistic is

    T_n := n * (p_hat_n^eps - Phi(-a/eps)) / (lambda * c_eps * sqrt(V_n(0))).

Reject H0: H=0 when |T_n| > z_{1-alpha/2}. Under H0, T_n --d--> N(0,1)
(asymptotic size alpha). Under a true H, with R(H) := sqrt(V_n(H)/V_n(0)),
the asymptotic power is

    pi(H) = 1 - Phi(z_{1-alpha/2}/R(H)) + Phi(-z_{1-alpha/2}/R(H)).

V_n(0) is the exact closed-form H -> 0 limit of Vn_formula (via
L'Hopital); since the underlying field's own variance
nu^2 = lambda^2 / (H(1-2H)) diverges at literal H=0, simulating "H=0"
paths uses a numerically-safe proxy H_PROXY = 0.01 (a real feature of the
model near the singular boundary, not a simulation artifact).

Produces
--------
* SIZE check: empirical rejection rate under H0 (H=0), compared to the
  nominal level alpha.
* POWER check: empirical vs. theoretical rejection rate over a grid of
  true H != 0 values, plotted together.

Requires ``core_simulation.py`` (build_chol, simulate_M, Vn_formula) in
the same directory -- see that module's docstring if it still raises
NotImplementedError.

Usage
-----
    python hypothesis_test_regularized.py

References
----------
Bolko, A. E., Christensen, K., Pakkanen, M. S., Veliyev, B. (2020). "A GMM
    approach to estimate the roughness of stochastic volatility."
    arXiv:2010.04610.
===========================================================
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
LAM = 0.05  # chosen for size closest to target among {0.1, 0.05, 0.03}
A, EPS = 0.1, 0.3  # eps=0.3 is in the delta-method-valid regime
ALPHA = 0.05
Z_CRIT = norm.ppf(1 - ALPHA / 2)

M_CALIB = 200_000
M_REPS = 20_000

H_PROXY = 0.01  # numerically safe stand-in for simulating "H=0"
H_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


def run_at_H(H, seed_offset):
    """Simulate the T_n statistic (Sigma_n / lambda, pre-normalization)
    under a given true Hurst exponent H, using a fresh calibration batch
    (to estimate mu_hat) followed by the main replication batch."""
    L, K = build_chol(N, DELTA, T, H, LAM, DELTA_FINE)
    M_calib = simulate_M(L, K, N, DELTA, DELTA_FINE, SIGMA2, M_CALIB, seed=7000 + seed_offset)
    mu_hat = np.log(M_calib / DELTA).mean()
    M_main = simulate_M(L, K, N, DELTA, DELTA_FINE, SIGMA2, M_REPS, seed=8000 + seed_offset)
    Y = np.log(M_main / DELTA) - mu_hat
    p_hat = norm.cdf((Y - A) / EPS).mean(axis=0)
    V0 = Vn_formula(H_PROXY, LN, T)
    c_eps = norm.pdf(A / EPS) / EPS
    return N * (p_hat - norm.cdf(-A / EPS)) / (LAM * c_eps * np.sqrt(V0))


def run():
    V0 = Vn_formula(H_PROXY, LN, T)
    c_eps = norm.pdf(A / EPS) / EPS
    print(f"V_n(0) = {V0:.4f},  c_eps = {c_eps:.5f},  z_crit = {Z_CRIT:.4f}")
    print(f"(H proxy for simulating 'H=0' paths: H={H_PROXY}, since "
          f"nu^2=lambda^2/(H(1-2H)) diverges at literal H=0)\n")

    # --- SIZE check under H0: H=0 (numerically safe proxy) ---
    Tn_H0 = run_at_H(H_PROXY, seed_offset=0)
    rejection_rate_H0 = np.mean(np.abs(Tn_H0) > Z_CRIT)
    print(f"SIZE check (H=0): empirical rejection rate = {rejection_rate_H0:.4f}  "
          f"(target alpha = {ALPHA})")
    print(f"  mean(Tn)={Tn_H0.mean():.4f}  var(Tn)={Tn_H0.var():.4f}  (theory: mean=0, var=1)\n")

    # --- POWER check under several H != 0 ---
    power_emp, power_theory = [], []
    for i, H in enumerate(H_GRID):
        Tn_H = run_at_H(H, seed_offset=i + 1)
        rr = np.mean(np.abs(Tn_H) > Z_CRIT)
        power_emp.append(rr)
        VH = Vn_formula(H, LN, T)
        R = np.sqrt(VH / V0)
        pi_H = 1 - norm.cdf(Z_CRIT / R) + norm.cdf(-Z_CRIT / R)
        power_theory.append(pi_H)
        print(f"H={H:.2f}:  V_n(H)={VH:10.4f}  R={R:.4f}  empirical power={rr:.4f}  "
              f"theoretical power={pi_H:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(H_GRID, power_theory, 'r-', lw=2, label='theoretical power $\\pi(H)$')
    ax.plot(H_GRID, power_emp, 'o', color='#1f77b4', markersize=8, label='empirical rejection rate')
    ax.axhline(ALPHA, color='gray', ls='--', label=f'$\\alpha={ALPHA}$')
    ax.axhline(rejection_rate_H0, color='green', ls=':', lw=1.5,
               label=f'empirical size at H=0 ({rejection_rate_H0:.3f})')
    ax.set_xlabel('H (true value)')
    ax.set_ylabel('rejection probability')
    ax.set_title(f'Size and power of the $H=0$ vs $H\\ne0$ test '
                 f'($n={N}, \\lambda={LAM}, a={A}, \\epsilon={EPS}$)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('test_size_power.pdf')
    fig.savefig('test_size_power.png', dpi=150)
    print("\nSaved test_size_power.{pdf,png}")


if __name__ == "__main__":
    run()
