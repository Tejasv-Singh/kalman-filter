"""Tests for Kalman filter implementations."""

import numpy as np
import pytest
from kbv.models.kalman import KalmanFilter
from kbv.models.ukf import UnscentedKalmanFilter
from kbv.models.switching_kf import SwitchingKalmanFilter


def test_kalman_filter_basic():
    """Test basic Kalman filter functionality."""
    kf = KalmanFilter(
        F=np.array([[1.0]]),
        H=np.array([[1.0]]),
        Q=np.array([[0.1]]),
        R=np.array([[1.0]]),
    )

    observations = np.random.randn(100)
    filtered_states, filtered_covs = kf.filter(observations)

    assert len(filtered_states) == 100
    assert filtered_covs.shape == (100, 1, 1)


def test_kalman_smooth():
    """Test Kalman smoother."""
    kf = KalmanFilter(
        F=np.array([[0.9]]),
        H=np.array([[1.0]]),
        Q=np.array([[0.1]]),
        R=np.array([[1.0]]),
    )

    observations = np.random.randn(50)
    smoothed_states, smoothed_covs = kf.smooth(observations)

    assert len(smoothed_states) == 50
    assert smoothed_covs.shape == (50, 1, 1)


def test_kalman_log_likelihood():
    """Test log-likelihood computation."""
    kf = KalmanFilter(
        F=np.array([[1.0]]),
        H=np.array([[1.0]]),
        Q=np.array([[0.1]]),
        R=np.array([[1.0]]),
    )

    observations = np.random.randn(50)
    log_lik = kf.log_likelihood(observations)

    assert isinstance(log_lik, float)
    assert not np.isnan(log_lik)
    assert not np.isinf(log_lik)


def test_ukf_basic():
    """Test Unscented Kalman Filter."""
    def f(x):
        return 0.9 * x

    def h(x):
        return x

    ukf = UnscentedKalmanFilter(
        f=f, h=h, Q=np.array([[0.1]]), R=np.array([[1.0]]),
        n_state=1, n_obs=1
    )

    observations = np.random.randn(50).reshape(-1, 1)
    filtered_states, _ = ukf.filter(observations)

    assert len(filtered_states) == 50


def test_switching_kf():
    """Test Switching Kalman Filter."""
    from kbv.models.kalman import KalmanFilter

    kf1 = KalmanFilter(
        F=np.array([[0.9]]), H=np.array([[1.0]]),
        Q=np.array([[0.1]]), R=np.array([[1.0]])
    )
    kf2 = KalmanFilter(
        F=np.array([[0.9]]), H=np.array([[1.0]]),
        Q=np.array([[0.2]]), R=np.array([[1.0]])
    )

    transition_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])

    skf = SwitchingKalmanFilter(
        n_regimes=2,
        transition_matrix=transition_matrix,
        filters=[kf1, kf2],
    )

    observations = np.random.randn(50)
    filtered_states, filtered_covs, regime_probs = skf.filter(observations)

    assert len(filtered_states) == 50
    assert regime_probs.shape == (50, 2)
    assert np.allclose(regime_probs.sum(axis=1), 1.0)

