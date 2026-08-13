"""
===========================================================
File        : core_simulation.py
Project     : rough-vs-multifractal-hypothesis-testing
Authors     : Othmane Zarhali
Created     : 2026
Description :
    *** EXTERNAL DEPENDENCY -- NOT PART OF THE SUPPLIED SOURCE ***
    Several scripts in this package (clt_check.py,
    regularized_indicator.py, hypothesis_test_regularized.py,
    hypothesis_test_logM.py) import three functions from a module named
    core_simulation that was referenced, but never defined, in the
    original uploaded script. This file documents the *interface* those
    callers rely on (reconstructed purely from how the functions are
    called and from the surrounding comments), so the package imports
    cleanly and the call sites are self-documenting. It intentionally
    raises NotImplementedError rather than guessing at the Cholesky-based
    simulation internals -- filling those in silently would risk putting
    incorrect mathematics into a hypothesis-testing pipeline.
    Replace the function bodies below with your original implementations
    (or send them over and this stub can be completed).

Expected interface
-------------------
build_chol(n, Delta, T, H, lam, delta_fine) -> (L, K)
    Builds a Cholesky factor ``L`` (and companion matrix/array ``K``) of the
    covariance of a fine-grid (spacing ``delta_fine``) Gaussian log-volatility
    path over ``n`` windows of length ``Delta``, memory-length parameter
    ``T``, Hurst exponent ``H``, and volatility-of-volatility ``lam``
    (so that ``lam`` plays the role of ``lambda`` = sqrt(lambda^2) elsewhere
    in this package). Used to simulate correlated log-vol paths windowed
    into ``n`` blocks of size ``Delta``.

simulate_M(L, K, n, Delta, delta_fine, sigma2, M_reps, seed) -> M
    Given the Cholesky factor ``L`` / ``K`` from ``build_chol``, simulates
    ``M_reps`` independent realizations of the windowed integrated-variance
    proxy M_j (j = 1..n), analogous to the block quantity
    ``exp(zz2)`` in ``reference_simulator.py``, using base variance
    ``sigma2`` and RNG ``seed``. Returns an array of shape (n, M_reps) (row j
    = window j across replications), consistent with calls such as
    ``Y = np.log(M / Delta)`` and ``Sigma_n = Y.sum(axis=0)``.

Vn_formula(H, Ln, T) -> float
    Closed-form asymptotic variance V_n(H) of the (centered, lambda-
    normalized) log-mass statistic Sigma_n / lambda over a window of total
    length ``Ln = n * Delta`` and memory parameter ``T``, used both to
    calibrate the test statistics (``T_n``, ``Z_n``) and to compute the
    theoretical power curve ``pi(H) = 1 - Phi(z_crit/R) + Phi(-z_crit/R)``,
    ``R = sqrt(V_n(H)/V_n(0))``.
===========================================================
"""


def build_chol(n, Delta, T, H, lam, delta_fine):
    """See module docstring for the expected interface. Not implemented:
    the Cholesky-based fine-grid simulator was not part of the supplied
    source and must be restored from the original implementation."""
    raise NotImplementedError(
        "build_chol: original implementation not included in the supplied "
        "source -- see the module docstring for the expected interface."
    )


def simulate_M(L, K, n, Delta, delta_fine, sigma2, M_reps, seed):
    """See module docstring for the expected interface. Not implemented:
    depends on the Cholesky factors produced by build_chol."""
    raise NotImplementedError(
        "simulate_M: original implementation not included in the supplied "
        "source -- see the module docstring for the expected interface."
    )


def Vn_formula(H, Ln, T):
    """See module docstring for the expected interface. Not implemented:
    the closed-form asymptotic variance formula was not part of the
    supplied source."""
    raise NotImplementedError(
        "Vn_formula: original implementation not included in the supplied "
        "source -- see the module docstring for the expected interface."
    )
