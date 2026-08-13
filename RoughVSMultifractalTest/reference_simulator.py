"""
===========================================================
File        : reference_simulator.py
Project     : rough-vs-multifractal-hypothesis-testing
Authors     : Othmane Zarhali
Created     : 2026
Description :
    Reference FFT-based (circulant embedding) simulator for the
    log-integrated volatility field used throughout the H=0 vs. H!=0
    (rough vs. multifractal) hypothesis-testing study.
    This module defines:
      - sfbm_om_corr, building the fractional/logarithmic autocovariance
        sequence of the driving log-volatility field.
      - gaussprocess, an exact stationary Gaussian process simulator via
        Wood-Chan circulant embedding (FFT method).
      - genlogVol_independent, drawing the log-mass surrogate zz2 and the
        raw block integral lamOmega from two independent field
        realizations sharing the same covariance law.
      - make_figure_std_vs_lambda, make_figure_histograms, make_figure_qq,
        the three independent-fields null-hypothesis diagnostics.

Mathematical background
------------------------
Let omega(t) be a centered Gaussian field with autocovariance

    c_H(x) = lambda^2 * log(T/x)                          for H = 0,
    c_H(x) = lambda^2/[2H(1-2H)] * (T^{2H} - x^{2H})       for H != 0,

for 0 < x < T (Hurst-type roughness exponent H, volatility-of-volatility
lambda^2, integral scale T). On window j, define

    zz2      = log( integral_block e^{2 omega(t)} dt ), re-centered
               (the theoretical log-integrated-variance surrogate,
               computed stably via ``scipy.special.logsumexp``);
    lamOmega = integral_block ( omega(t) - mean(omega) ) dt
               (the raw centered block integral of the field itself).

These are generated from two *independent* realizations sharing the same
covariance law (genlogVol_independent), which is the null-hypothesis
benchmark used in the three diagnostic figures below: under independence,
the residual R := zz2 - (2/M) * lamOmega should have std(R) growing like
lambda as lambda^2 -> infinity, i.e. slope ~ 1 in the log-log std(R) vs.
lambda fit performed by make_figure_std_vs_lambda.

Usage
-----
    python reference_simulator.py

runs all three diagnostic figures (std(R) vs. lambda, 4-panel histograms,
and QQ-plots) and saves them as PDF/PNG in the current directory.

References
----------
Wood, A. T. A., Chan, G. (1994). "Simulation of Stationary Gaussian
    Processes in [0,1]^d." J. Comput. Graph. Statist., 3(4), 409-432.
Gatheral, J., Jaisson, T., Rosenbaum, M. (2018). "Volatility is rough."
    Quantitative Finance, 18(6), 933-949.
Zarhali, O., Bacry, E., Muzy, J.-F. (2026). "From rough to multifractal
    multidimensional volatility: A multidimensional Log S-fBM model."
    arXiv:2601.10517.
===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp


# ============================================================================
# Covariance construction and Gaussian field generator
# ============================================================================

def sfbm_om_corr(size: int, dt: float = 0.1, T: float = 200, lambda2: float = 0.02,
                  H: float = 0):
    """
    Build the autocovariance sequence of the driving log-volatility field
    omega(t) and its stationary-shift mean m, at lag grid k*dt, k=0..size-1.

    Two regimes:
      * H == 0 : a *logarithmic*-memory kernel,
                     cc(x) = lambda2 * log(T / x) for x < T
                 (the H -> 0 limiting case of fractional covariance,
                 corresponding to the boundary of the rough-volatility
                 family used as the "H=0" null hypothesis in the tests
                 below).
      * H != 0 : the fractional (power-law) kernel
                     cc(x) = K * ((T*dt)^(2H) - x^(2H)) for x < T*dt,
                     K = lambda2 / (2H(1-2H)),
                 i.e. the covariance of a (shifted) fractional Brownian
                 motion with Hurst exponent H, scaled by lambda2.

    Parameters
    ----------
    size : int
        Number of covariance lags to generate.
    dt : float
        Time-grid spacing.
    T : float
        Memory-length / cutoff parameter of the kernel.
    lambda2 : float
        Volatility-of-volatility parameter (variance scale).
    H : float
        Hurst-type roughness exponent. H == 0 selects the logarithmic
        (rough / "H=0") kernel; H != 0 selects the fractional kernel.

    Returns
    -------
    m : float
        Mean-shift so that the simulated field has the correct
        stationary mean (m = -cc[0]).
    cc : np.ndarray
        Autocovariance sequence, length ``size``.
    """
    xx = dt * np.arange(1, size)
    if H == 0:
        cc = lambda2 * np.log(T / xx) * (xx < T)
        cc = np.append(lambda2 * (1 + np.log(T / dt)), cc)
    else:
        xx = np.append(0, xx)
        K = lambda2 / (2 * H * (1 - 2 * H))
        cc = K * ((T * dt) ** (2 * H) - xx ** (2 * H)) * (xx < T * dt)
    m = -cc[0]
    return m, cc


def gaussprocess(covariance: np.ndarray, size: int) -> np.ndarray:
    """
    Simulate a stationary centered Gaussian process with the given
    autocovariance sequence via circulant embedding (FFT method):
    embed the covariance in a symmetric circulant matrix, take its
    (real, non-negative) spectral density via FFT, and color complex
    white noise by its square root before inverse-FFT-ing back to the
    time domain.

    Parameters
    ----------
    covariance : np.ndarray
        Target autocovariance sequence (lags 0, 1, 2, ...).
    size : int
        Length of the process path to return.

    Returns
    -------
    np.ndarray of shape (size,)
        One realization of the Gaussian process.
    """
    m = int(2 ** np.floor(np.log2(size)))
    M = 2 * m
    covariance = np.concatenate([covariance, np.zeros(m + 1 - len(covariance))])
    thecorr = np.concatenate([covariance, np.flip(covariance[1:-1])])
    fftcorr = np.real(np.fft.fft(thecorr))
    u = np.random.normal(size=M)
    v = np.random.normal(size=M) * 1j
    fftcorr = np.sqrt(fftcorr + 0j) * (u + v) / np.sqrt(2)
    corr = np.real(np.fft.ifft(fftcorr))
    return corr[:size] * np.sqrt(2 * M)


def _corr_and_mean(size, subsample, T, lambda2, H, M):
    """
    Shared helper that builds the (mean, covariance) pair once per call to
    ``genlogVol_independent``, so that the independent-fields version can
    call ``gaussprocess`` twice with separately-seeded noise but the SAME
    underlying covariance structure (i.e. two draws from the same law).
    """
    dt = 1 / subsample
    fact = 1 if H == 0 else M ** (-2 * H) / 4
    lambda2_eff = lambda2 * fact
    T_eff = T * M
    N = size * M - 1
    N = 2 ** np.ceil(np.log2(N))
    m, corr = sfbm_om_corr(size=int(N * subsample), dt=dt, T=T_eff * subsample,
                            lambda2=lambda2_eff, H=H)
    return m, corr


def genlogVol_independent(size: int = 4096, subsample: int = 8, T: float = 4096,
                           lambda2: float = 0.025, H: float = 0, M: int = 32,
                           seed_zz2: int = None, seed_lamOmega: int = None):
    """
    Draw zz2 and lamOmega from TWO INDEPENDENT underlying field
    realizations that share the same covariance law but no randomness
    (i.e. the null-hypothesis / independence benchmark).

        zz2      = log( integral_block e^{2 omega(t)} dt ), re-centered
                   -- the theoretical log-integrated-variance surrogate;
        lamOmega = integral_block ( omega(t) - mean(omega) ) dt
                   -- the raw centered block integral of the field.

    Parameters
    ----------
    size : int
        Number of blocks (output length) requested.
    subsample : int
        Fine-grid points per unit time.
    T : float
        Memory-length parameter passed to the covariance kernel.
    lambda2 : float
        Volatility-of-volatility parameter.
    H : float
        Hurst-type roughness exponent (0 = logarithmic/"rough" kernel).
    M : int
        Number of fine-grid points aggregated per output block.
    seed_zz2, seed_lamOmega : int, optional
        Independent RNG seeds for each field draw (for reproducibility).

    Returns
    -------
    zz2 : np.ndarray of shape (n_blocks,)
    lamOmega : np.ndarray of shape (n_blocks,)
    """
    dt = 1 / subsample
    m, corr = _corr_and_mean(size, subsample, T, lambda2, H, M)

    # --- field #1: drives zz2 (log block-integral of e^{2 omega}) ---
    if seed_zz2 is not None:
        np.random.seed(seed_zz2)
    om1 = gaussprocess(corr, size * M * subsample) + m
    n_blocks = (len(om1) // subsample) // M
    om1_blocks = om1[:n_blocks * M * subsample].reshape(n_blocks, M * subsample)
    # logsumexp(2*omega) + log(dt) = log( sum_i e^{2 omega_i} dt ) ~ log integral
    log_mm_blocks = logsumexp(2 * om1_blocks, axis=1) + np.log(dt)
    zz2 = log_mm_blocks - log_mm_blocks.mean()

    # --- field #2: INDEPENDENT draw, same covariance law -> drives lamOmega ---
    if seed_lamOmega is not None:
        np.random.seed(seed_lamOmega)
    om2 = gaussprocess(corr, size * M * subsample) + m
    om2_blocks = om2[:n_blocks * M * subsample].reshape(n_blocks, M * subsample)
    lamOmega = (om2_blocks - om2_blocks.mean()).sum(axis=1) * dt

    return zz2, lamOmega


# ============================================================================
# Figure 1: std(R) vs. lambda (log-log), independent fields
# ============================================================================

def make_figure_std_vs_lambda(lambda_sq_list=(1e16, 1e2, 1e-2), n_reps: int = 8):
    """
    For each lambda^2 in ``lambda_sq_list``, simulate independent (zz2,
    lamOmega) pairs, form the residual R = zz2 - (2/M) * lamOmega, and
    report/plot std(R) vs. lambda = sqrt(lambda^2) on log-log axes,
    together with a weighted-least-squares power-law fit
    std(R) ~ C * lambda^slope. Under independence the expected slope is
    ~ 1 in the large-lambda2 regime.

    Saves ``std_vs_lambda.{pdf,png}`` in the current directory.
    """
    size, subsample, T, M, H = 4096, 8, 4096, 32, 0
    scale = 2 / M

    results = []
    for i, lam_sq in enumerate(lambda_sq_list):
        stdRs = []
        for r in range(n_reps):
            zz2, lamO = genlogVol_independent(
                size=size, subsample=subsample, T=T, lambda2=lam_sq, H=H, M=M,
                seed_zz2=500_000 + i * 1000 + r, seed_lamOmega=600_000 + i * 1000 + r)
            R = zz2 - scale * lamO
            stdRs.append(R.std())
        results.append({'lambda_sq': lam_sq, 'lam': np.sqrt(lam_sq),
                         'mean_std_R': np.mean(stdRs),
                         'sem_std_R': np.std(stdRs) / np.sqrt(n_reps)})

    print(f"{'lambda^2':>10} {'lambda':>12} {'mean std(R)':>14} {'SEM':>10} {'std(R)/lambda':>14}")
    for r in results:
        print(f"{r['lambda_sq']:10.1e} {r['lam']:12.4e} {r['mean_std_R']:14.6e} "
              f"{r['sem_std_R']:10.4e} {r['mean_std_R'] / r['lam']:14.4f}")

    lam_arr = np.array([r['lam'] for r in results])
    std_arr = np.array([r['mean_std_R'] for r in results])
    sem_arr = np.array([r['sem_std_R'] for r in results])

    # Weighted least squares fit of log(std) on log(lambda), weights = 1/SEM^2.
    ll, ls = np.log(lam_arr), np.log(std_arr)
    w = (std_arr / sem_arr) ** 2
    A = np.vstack([ll, np.ones_like(ll)]).T
    W = np.diag(w)
    coef, *_ = np.linalg.lstsq(np.sqrt(W) @ A, np.sqrt(W) @ ls, rcond=None)
    slope, intercept = coef
    se_slope = np.sqrt(np.linalg.inv(A.T @ W @ A)[0, 0])
    print(f"\nFitted slope across these {len(lambda_sq_list)} points: {slope:.4f} +/- "
          f"{se_slope:.4f} (independent fields: expect ~1)")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(lam_arr, std_arr, yerr=sem_arr, fmt='o', color='#9467bd', capsize=4,
                markersize=8, label='reference simulator, INDEPENDENT fields')
    xs = np.linspace(lam_arr.min(), lam_arr.max(), 200)
    ax.plot(xs, np.exp(intercept) * xs ** slope, 'k-', lw=2, label=f'fit: slope={slope:.3f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda=\sqrt{\lambda^2}$ (log scale)')
    ax.set_ylabel(r'std$(R)$ (log scale)')
    ax.set_title(r'std$(R)$ vs. $\lambda$: independent fields')
    ax.legend()
    fig.tight_layout()
    fig.savefig('std_vs_lambda.pdf')
    fig.savefig('std_vs_lambda.png', dpi=150)
    print("Saved std_vs_lambda.{pdf,png}")
    return results, slope, se_slope


# ============================================================================
# Figure 2: 4-panel histogram comparison, independent fields
# ============================================================================

def make_figure_histograms(lambda_sq_list=(1e16, 1e2, 1e-2, 1e-4)):
    """
    For each lambda^2, overlay the empirical histograms of zz2 (log
    integrated variance) and of the scaled lamOmega, and report std(R)
    and corr(zz2, lamOmega_scaled) in each panel's title.

    Saves ``histogram_comparison.{pdf,png}``.
    """
    size, subsample, T, M, H = 4096, 8, 4096, 32, 0
    scale = 2 / M

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    summary = []
    for i, (ax, lam_sq) in enumerate(zip(axes, lambda_sq_list)):
        zz2, lamO = genlogVol_independent(
            size=size, subsample=subsample, T=T, lambda2=lam_sq, H=H, M=M,
            seed_zz2=700_000 + i, seed_lamOmega=800_000 + i)
        lamO_scaled = scale * lamO
        R = zz2 - lamO_scaled
        corr = np.corrcoef(zz2, lamO_scaled)[0, 1]
        summary.append({'lambda_sq': lam_sq, 'std_R': R.std(), 'corr': corr})

        lo, hi = min(zz2.min(), lamO_scaled.min()), max(zz2.max(), lamO_scaled.max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(zz2, bins=bins, alpha=0.55, color='#1f77b4', label=r'$\ln(\mathrm{M})$')
        ax.hist(lamO_scaled, bins=bins, alpha=0.55, color='#ff7f0e', label=r'$\lambda\Omega$')
        ax.set_title(fr'$\lambda^2={lam_sq:.0e}$' + '\n' +
                     fr'std$(R)={R.std():.3g}$, corr$={corr:.4f}$')
        ax.set_xlabel('value')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('count')

    fig.suptitle(r'Reference simulator, independent fields only')
    fig.tight_layout()
    fig.savefig('histogram_comparison.pdf')
    fig.savefig('histogram_comparison.png', dpi=150)
    print("Saved histogram_comparison.{pdf,png}")

    print(f"\n{'lambda^2':>10} {'std(R)':>14} {'corr':>10}")
    for s in summary:
        print(f"{s['lambda_sq']:10.1e} {s['std_R']:14.4g} {s['corr']:10.4f}")
    return summary


# ============================================================================
# Figure 3: empirical QQ-plots, independent fields
# ============================================================================

def make_figure_qq(lambda_sq_list=(1e16, 1e2, 1e-2, 1e-4)):
    """
    QQ-plots of the empirical quantiles of the scaled lamOmega against
    the quantiles of zz2 (log integrated variance), one panel per
    lambda^2, with the 45-degree reference line for visual calibration.

    Saves ``qq_plots.{pdf,png}``.
    """
    size, subsample, T, M, H = 4096, 8, 4096, 32, 0
    scale = 2 / M

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (ax, lam_sq) in enumerate(zip(axes, lambda_sq_list)):
        zz2, lamO = genlogVol_independent(
            size=size, subsample=subsample, T=T, lambda2=lam_sq, H=H, M=M,
            seed_zz2=700_000 + i, seed_lamOmega=800_000 + i)
        lamO_scaled = scale * lamO

        q = np.linspace(0.001, 0.999, 200)
        q_lamO = np.quantile(lamO_scaled, q)
        q_zz2 = np.quantile(zz2, q)

        ax.plot(q_lamO, q_zz2, 'o', markersize=3, alpha=0.7)

        lo = min(q_lamO.min(), q_zz2.min())
        hi = max(q_lamO.max(), q_zz2.max())
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)

        ax.set_title(rf'$\lambda^2={lam_sq:.0e}$')
        ax.set_xlabel(r'Quantiles of $\lambda\Omega$')
        ax.set_ylabel(r'Quantiles of $\log(\mathrm{IV})$')
        ax.grid(alpha=0.3)

    fig.suptitle(r'Empirical QQ plots: $\log(\mathrm{IV})$ vs $\lambda\Omega$', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('qq_plots.pdf')
    fig.savefig('qq_plots.png', dpi=150)
    print("Saved qq_plots.{pdf,png}")


if __name__ == "__main__":
    print("=" * 70)
    print("FIGURE 1: std(R) vs. lambda")
    print("=" * 70)
    make_figure_std_vs_lambda()

    print("\n" + "=" * 70)
    print("FIGURE 2: 4-panel histogram")
    print("=" * 70)
    make_figure_histograms()

    print("\n" + "=" * 70)
    print("FIGURE 3: empirical QQ plots")
    print("=" * 70)
    make_figure_qq()
