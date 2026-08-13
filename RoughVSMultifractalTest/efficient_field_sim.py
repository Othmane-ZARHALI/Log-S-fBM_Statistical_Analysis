"""
===========================================================
File        : efficient_field_sim.py
Project     : rough-vs-multifractal-hypothesis-testing
Authors     : Othmane Zarhali
Created     : 2026
Description :
    *** EXTERNAL DEPENDENCY -- NOT PART OF THE SUPPLIED SOURCE ***
    compute_one_point_logM.py imports simulate_Y_batch from this module.
    It was referenced but never defined in the original uploaded script;
    this file documents the interface reconstructed from the call site so
    the package imports cleanly. See core_simulation.py for the same
    caveat.

Expected interface
-------------------
simulate_Y_batch(n, Delta, T, H, lam, n_reps, pts_per_window, sigma2,
                  seed, chunk_size) -> Y

    A memory-efficient (chunked) counterpart to
    ``core_simulation.simulate_M`` + ``np.log(M/Delta)``: simulates
    ``n_reps`` replications of the centered log-mass increments
    Y_j = log(M_j / Delta) over ``n`` windows of length ``Delta``, at Hurst
    exponent ``H`` and volatility-of-volatility ``lam``, using
    ``pts_per_window`` fine-grid points per window and processing the
    ``n_reps`` replications in batches of ``chunk_size`` to control peak
    memory (needed for the large ``n`` / ``T`` regimes used in
    ``compute_one_point_logM.py``, e.g. T=20000). Returns an array of shape
    (n_reps, n), consistent with
    ``Sigma_n = Y.sum(axis=1)`` at the call site.
===========================================================
"""


def simulate_Y_batch(n, Delta, T, H, lam, n_reps, pts_per_window, sigma2,
                      seed, chunk_size):
    """See module docstring for the expected interface. Not implemented:
    the chunked field simulator was not part of the supplied source and
    must be restored from the original implementation."""
    raise NotImplementedError(
        "simulate_Y_batch: original implementation not included in the "
        "supplied source -- see the module docstring for the expected "
        "interface."
    )
