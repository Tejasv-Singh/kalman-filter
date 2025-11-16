"""Tests for synthetic data generation."""

import numpy as np
import pytest
from kbv.data.synth import (
    generate_heston_path,
    generate_regime_switching_sv,
    generate_simple_sv,
    generate_garch_data,
)


def test_heston_path():
    """Test Heston path generation."""
    times, prices, variances = generate_heston_path(n_steps=100, seed=42)

    assert len(times) == 101
    assert len(prices) == 101
    assert len(variances) == 101
    assert np.all(prices > 0)
    assert np.all(variances >= 0)


def test_regime_switching_sv():
    """Test regime-switching SV generation."""
    returns, log_vol, regimes = generate_regime_switching_sv(n_obs=100, seed=42)

    assert len(returns) == 100
    assert len(log_vol) == 100
    assert len(regimes) == 100
    assert np.all(regimes >= 0)
    assert np.all(regimes < 2)


def test_simple_sv():
    """Test simple SV generation."""
    returns, log_vol = generate_simple_sv(n_obs=100, seed=42)

    assert len(returns) == 100
    assert len(log_vol) == 100


def test_garch_data():
    """Test GARCH data generation."""
    returns, volatility = generate_garch_data(n_obs=100, seed=42)

    assert len(returns) == 100
    assert len(volatility) == 100
    assert np.all(volatility > 0)


def test_reproducibility():
    """Test that seeds produce reproducible results."""
    returns1, _ = generate_simple_sv(n_obs=100, seed=42)
    returns2, _ = generate_simple_sv(n_obs=100, seed=42)

    np.testing.assert_array_equal(returns1, returns2)

