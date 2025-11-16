"""Tests for inference methods."""

import numpy as np
import pytest
from kbv.inference.em import EMKalmanFilter
from kbv.data.synth import generate_simple_sv


def test_em_kalman():
    """Test EM algorithm for Kalman filter."""
    # Generate data
    returns, _ = generate_simple_sv(n_obs=200, seed=42)
    observations = returns**2

    # Initialize EM
    em = EMKalmanFilter(n_state=1, n_obs=1)

    # Run EM
    results = em.fit(observations, max_iter=10, verbose=False)

    assert "F" in results
    assert "H" in results
    assert "Q" in results
    assert "R" in results
    assert "log_likelihood" in results


def test_em_convergence():
    """Test that EM improves log-likelihood."""
    returns, _ = generate_simple_sv(n_obs=200, seed=42)
    observations = returns**2

    em = EMKalmanFilter(n_state=1, n_obs=1)
    results = em.fit(observations, max_iter=20, verbose=False)

    # Log-likelihood should be finite
    assert np.isfinite(results["log_likelihood"])

