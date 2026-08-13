"""
clt_check.py
============

CLT check on the log-mass statistic ln(M) at a small vol-of-vol
lambda^2 = 1e-2 (lambda = 0.1), for a single window (n=1), memory-length
T = 1.05 deliberately close to Delta = 1 (weak within-window correlation,
rho = 1 - (Delta/T)^(2H) ~ 0.019 at H=0.2). Included for completeness even
though at this small lambda the Gaussian fit is expected to be good
regardless of T -- this is the theorem's own asymptotic regime.

Produces
--------
Figure A : histogram of W := (Sigma_n - mean) / lambda vs. the fitted
           N(mean, std^2) density.
Figure B : QQ-plot of the standardized W against the standard normal.

Requires ``core_simulation.py`` (build_chol, simulate_M, Vn_formula) in
the same directory -- see that module's docstring if it still raises
NotImplementedError.

Usage
-----
    python clt_check.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest, skew, kurtosis, probplot

from core_simulation import build_chol, simulate_M, Vn_formula

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
N, DELTA, H, T = 1, 1.0, 0.2, 1.05
LAM = 0.1  # lambda^2 = 1e-2
DELTA_FINE = 0.02
M_REPS = 30_000
SEED = 1


def run():
    V_n_theory = Vn_formula(H, N * DELTA, T)
    rho_window = 1 - (DELTA / T) ** (2 * H)

    # --- simulate ---
    L, K = build_chol(N, DELTA, T, H, LAM, DELTA_FINE)
    M = simulate_M(L, K, N, DELTA, DELTA_FINE, sigma2=1.0, M_reps=M_REPS, seed=SEED)
    Y = np.log(M / DELTA)
    Sigma_n = Y.sum(axis=0)
    W = (Sigma_n - Sigma_n.mean()) / LAM  # W := Sigma_n / lambda, centered

    mean_, std_ = W.mean(), W.std()
    ks_stat, ks_p = kstest((W - mean_) / std_, 'norm')

    print(f"n={N}, Delta={DELTA}, T={T} (rho_window={rho_window:.4f}), H={H}, "
          f"lambda={LAM} (lambda^2={LAM ** 2:.0e})")
    print(f"Theoretical V_n = {V_n_theory:.4f}  (sqrt = {np.sqrt(V_n_theory):.4f})")
    print(f"Empirical: mean(W)={mean_:.4f}  std(W)={std_:.4f}  skew={skew(W):.4f}  "
          f"kurt={kurtosis(W):.4f}  KS p={ks_p:.4f}")

    # ---- Figure A: histogram vs. fitted Gaussian ----
    fig1, ax1 = plt.subplots(figsize=(7.5, 5.5))
    ax1.hist(W, bins=60, density=True, alpha=0.6, color='#1f77b4', label='Simulated')
    xs = np.linspace(W.min(), W.max(), 300)
    ax1.plot(xs, norm.pdf(xs, mean_, std_), 'r-', lw=2,
              label=fr'fitted $N({mean_:.3f},{std_:.3f}^2)$')
    ax1.set_xlabel(r'$W = \Sigma_n/\lambda$')
    ax1.set_ylabel('density')
    ax1.set_title(fr'Histogram: $\lambda^2=10^{{-2}}$, $n={N}$, $T={T}$, $H={H}$'
                  '\n' fr'std$={std_:.3f}$ (theory $\sqrt{{V_n}}={np.sqrt(V_n_theory):.3f}$), '
                  fr'skew$={skew(W):.3f}$, kurt$={kurtosis(W):.3f}$, KS $p={ks_p:.3f}$')
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig('clt_hist_lambda2_1em2.pdf')
    fig1.savefig('clt_hist_lambda2_1em2.png', dpi=150)
    print("Saved clt_hist_lambda2_1em2.{pdf,png}")

    # ---- Figure B: QQ-plot against the standard normal ----
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))
    W_std = (W - mean_) / std_
    probplot(W_std, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markersize(3)
    ax2.get_lines()[0].set_alpha(0.4)
    ax2.set_title(fr'QQ-plot: $\lambda^2=10^{{-2}}$, $n={N}$, $T={T}$, $H={H}$')
    ax2.set_xlabel('Theoretical quantiles (N(0,1))')
    ax2.set_ylabel(r'Ordered $W$ (standardized)')
    fig2.tight_layout()
    fig2.savefig('clt_qq_lambda2_1em2.pdf')
    fig2.savefig('clt_qq_lambda2_1em2.png', dpi=150)
    print("Saved clt_qq_lambda2_1em2.{pdf,png}")


if __name__ == "__main__":
    run()
