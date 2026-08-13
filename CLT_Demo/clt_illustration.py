"""
===========================================================
File        : clt_illustration.py
Project     : rough-vs-multifractal-hypothesis-testing
Authors     : Othmane Zarhali
Created     : 2026
Description :
    Monte Carlo illustration of the Central Limit Theorem (CLT).
    This module defines:
      - simulate_normalized_sum, drawing Monte Carlo replications of the
        CLT-normalized partial sum Z_n.
      - plot_clt_grid, building the multi-panel histogram-vs-Gaussian
        comparison figure across a grid of sample sizes n.

Mathematical background
-----------------------
We draw i.i.d. samples X_1, ..., X_n from a Uniform(-sqrt(3), sqrt(3))
distribution (mean 0, variance 1), and form the normalized partial sum

    Z_n = (X_1 + ... + X_n) / sqrt(n).

By the Lindeberg-Levy Central Limit Theorem, as n -> infinity,

    Z_n --d--> N(0, 1),

regardless of the (non-Gaussian) shape of the underlying uniform law. This
module compares the empirical distribution of Z_n, over many Monte Carlo
repetitions, against the standard normal density N(0, 1) for a grid of
sample sizes n, giving a direct visual demonstration of the convergence
rate.

Usage
-----
    python clt_illustration.py

Output
------
Displays (and saves) a grid of histograms, one per value of n, each
overlaid with the standard normal density.

References
----------
Billingsley, P. (1995). "Probability and Measure," 3rd ed., Theorem 27.1
    (Lindeberg-Levy Central Limit Theorem). Wiley.
===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

# Number of Monte Carlo repetitions used to build each empirical histogram.
N_MC = 50_000

# Grid of sample sizes n over which convergence to N(0,1) is illustrated.
N_VALUES = [1, 2, 5, 10, 20, 50, 100]


def simulate_normalized_sum(n: int, n_mc: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate n_mc independent draws of the CLT-normalized sum

        Z_n = (X_1 + ... + X_n) / sqrt(n),   X_i ~ iid Uniform(-sqrt(3), sqrt(3)).

    The support (-sqrt(3), sqrt(3)) is chosen so that each X_i has mean 0
    and variance 1, which is the standard normalization required for the
    classical (Lindeberg-Levy) CLT statement Z_n -> N(0, 1) in distribution.

    Parameters
    ----------
    n : int
        Number of terms summed (the CLT sample size).
    n_mc : int
        Number of independent Monte Carlo replications of Z_n.
    rng : np.random.Generator
        Random number generator (for reproducibility).

    Returns
    -------
    np.ndarray of shape (n_mc,)
        Monte Carlo draws of Z_n.
    """
    X = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(n_mc, n))
    return X.sum(axis=1) / np.sqrt(n)


def plot_clt_grid(n_values, n_mc: int, seed: int = 0):
    """
    Build the multi-panel figure: one histogram of Z_n per n in n_values,
    each overlaid with the standard normal density N(0, 1).

    Parameters
    ----------
    n_values : list[int]
        Sample sizes to display, one subplot each.
    n_mc : int
        Monte Carlo repetitions per subplot.
    seed : int
        Seed for the random number generator, for reproducibility.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng = np.random.default_rng(seed)

    # Layout: fixed 2x4 grid: 7 panels used + 1 removed (kept from the
    # original figure for compatibility with N_VALUES having 7 entries).
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    x_grid = np.linspace(-4, 4, 400)

    for k, n in enumerate(n_values):
        Z = simulate_normalized_sum(n, n_mc, rng)
        ax = axes[k]

        ax.hist(
            Z,
            bins=50,
            density=True,
            alpha=0.7,
            edgecolor="black",
            label="Simulation",
        )
        ax.plot(x_grid, norm.pdf(x_grid), "r", lw=2, label="N(0,1)")

        ax.set_title(f"$n={n}$")
        ax.set_xlim(-4, 4)

    # Remove any unused trailing subplot(s) if n_values has fewer entries
    # than the number of grid cells.
    for extra_ax in axes[len(n_values):]:
        fig.delaxes(extra_ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Numerical illustration of the Central Limit Theorem", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def main():
    fig = plot_clt_grid(N_VALUES, N_MC, seed=0)
    fig.savefig("clt_illustration.pdf")
    fig.savefig("clt_illustration.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
