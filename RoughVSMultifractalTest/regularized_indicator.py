"""
regularized_indicator.py
=========================

Compares the smooth (Gaussian-CDF-regularized) proportion statistic

    p_hat_n^eps := (1/n) * sum_{j=1}^n Phi( (Y_j - a) / eps )

against the hard indicator statistic

    p_hat_n := (1/n) * sum_{j=1}^n 1{Y_j >= a}

at n=5, T=8, Delta=1, H=0.2, lambda=0.1, threshold a=0.1, for a grid of
bandwidths eps in {1, 0.1, 0.01, 1e-4}. As eps -> 0, p_hat_n^eps -> p_hat_n
pointwise; the regularized statistic is the one whose asymptotic normality
is used to build the T_n test in ``hypothesis_test_regularized.py``, with
the hard indicator eps=0 limit shown here purely as the target it
approximates.

Produces
--------
* A 4-panel figure overlaying, for each eps, the histogram of p_hat_n^eps
  against the (discrete) pmf of the hard indicator p_hat_n.
* Summary statistics (mean, variance, exact-match fraction) at each eps.
* A nearest-point pmf comparison table between the two statistics at the
  smallest eps tested (1e-4).

Requires ``core_simulation.py`` (build_chol, simulate_M) in the same
directory -- see that module's docstring if it still raises
NotImplementedError.

Usage
-----
    python regularized_indicator.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from core_simulation import build_chol, simulate_M

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
N, DELTA, H, T = 5, 1.0, 0.2, 8.0  # T >= n*Delta=5 required for validity
LAM = 0.1  # lambda^2 = 1e-2
DELTA_FINE = 0.02
M_REPS = 30_000
SEED = 1
A = 0.1
EPS_LIST = [1.0, 0.1, 0.01, 1e-4]


def run():
    # --- simulate once, shared across all eps (paired comparison) ---
    L, K = build_chol(N, DELTA, T, H, LAM, DELTA_FINE)
    M = simulate_M(L, K, N, DELTA, DELTA_FINE, sigma2=1.0, M_reps=M_REPS, seed=SEED)
    Y = np.log(M / DELTA)
    Y = Y - Y.mean()  # centered (pooled mean across windows and replications)

    # --- hard indicator: p_hat_n = (1/n) sum_j 1{Y_j >= a}, support {0,1/n,...,1} ---
    p_hat_hard = (Y >= A).mean(axis=0)
    vals, counts = np.unique(p_hat_hard, return_counts=True)
    hard_pmf = counts / counts.sum()
    print("Hard indicator (n=5): support =", vals, " pmf =", hard_pmf)

    # ---- 4-panel figure: regularized histogram vs. hard indicator pmf ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)
    bins = np.linspace(-0.05, 1.05, 100)
    bar_width = bins[1] - bins[0]  # match histogram bin width for comparable heights

    for ax, eps in zip(axes, EPS_LIST):
        p_hat_eps = norm.cdf((Y - A) / eps).mean(axis=0)
        ax.hist(p_hat_eps, bins=bins, density=True, alpha=0.55, color='#1f77b4',
                label=r'$\hat p_n^\epsilon$')
        ax.bar(vals, hard_pmf / bar_width, width=bar_width, alpha=0.85, color='red',
               label=r'$\hat p_n$ (hard)')
        ax.set_xlabel(r'$\hat p_n$ or $\hat p_n^\epsilon$')
        ax.set_title(fr'$\epsilon={eps:g}$')
        ax.legend()
    axes[0].set_ylabel('density')

    fig.suptitle(r'Empirical distribution of $\hat p_n^\epsilon$ vs. the hard indicator '
                 fr'$\hat p_n$, for $\epsilon=1,\,0.1,\,0.01,\,10^{{-4}}$ '
                 fr'($n={N},\lambda={LAM},a={A}$)')
    fig.tight_layout()
    fig.savefig('phat_vs_hard_n5.pdf')
    fig.savefig('phat_vs_hard_n5.png', dpi=150)
    print("Saved phat_vs_hard_n5.{pdf,png}")

    # ---- moments and exact-match fraction at each eps ----
    print(f"\n{'eps':>10} {'mean':>10} {'var':>12} {'exact_match_frac':>18}")
    for eps in EPS_LIST:
        p_hat_eps = norm.cdf((Y - A) / eps).mean(axis=0)
        exact_match = np.mean(p_hat_eps == p_hat_hard)
        print(f"{eps:10.2e} {p_hat_eps.mean():10.5f} {p_hat_eps.var():12.6f} {exact_match:18.5f}")

    # ---- nearest-point pmf comparison table at the smallest eps ----
    eps_smallest = 1e-4
    p_hat_eps_smallest = norm.cdf((Y - A) / eps_smallest).mean(axis=0)
    nearest_idx = np.argmin(np.abs(p_hat_eps_smallest[:, None] - vals[None, :]), axis=1)
    blue_pmf_approx = np.array([np.mean(nearest_idx == k) for k in range(len(vals))])

    print(f"\nNearest-point pmf comparison at eps={eps_smallest:.0e}:")
    print(f"{'value':>8} {'red (hard pmf)':>16} {'blue (nearest-point pmf)':>26} {'diff':>10}")
    for v, hp, bp in zip(vals, hard_pmf, blue_pmf_approx):
        print(f"{v:8.1f} {hp:16.5f} {bp:26.5f} {bp - hp:10.6f}")

    print(f"\ntotal blue mass accounted for: {blue_pmf_approx.sum():.5f}")
    print(f"mean(p_hat_eps)={p_hat_eps_smallest.mean():.6f}  "
          f"mean(p_hat_hard)={p_hat_hard.mean():.6f}")
    print(f"var(p_hat_eps)={p_hat_eps_smallest.var():.6f}  "
          f"var(p_hat_hard)={p_hat_hard.var():.6f}")


if __name__ == "__main__":
    run()
