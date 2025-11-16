"""Evaluation metrics for volatility models."""

import numpy as np
from typing import Dict, Optional
from scipy.stats import pearsonr


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    if len(y_true) < 2:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return r


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Quasi-likelihood loss for volatility forecasts.

    QLIKE = mean(log(y_pred) + y_true / y_pred)
    """
    return np.mean(np.log(y_pred + 1e-8) + y_true / (y_pred + 1e-8))


def evaluate_volatility_forecast(
    true_vol: np.ndarray,
    pred_vol: np.ndarray,
    returns: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of volatility forecasts.

    Parameters
    ----------
    true_vol : array-like
        True volatility values
    pred_vol : array-like
        Predicted volatility values
    returns : array-like, optional
        Returns for realized volatility computation

    Returns
    -------
    dict
        Dictionary of metrics
    """
    metrics = {
        "rmse": rmse(true_vol, pred_vol),
        "mae": mae(true_vol, pred_vol),
        "mape": mape(true_vol, pred_vol),
        "correlation": correlation(true_vol, pred_vol),
        "qlike": qlike_loss(true_vol, pred_vol),
    }

    if returns is not None:
        # Realized volatility
        realized_vol = np.abs(returns)
        metrics["correlation_realized"] = correlation(realized_vol, pred_vol)

    return metrics


def directional_accuracy(
    true_vol: np.ndarray, pred_vol: np.ndarray
) -> Dict[str, float]:
    """
    Compute directional accuracy of volatility forecasts.

    Parameters
    ----------
    true_vol : array-like
        True volatility
    pred_vol : array-like
        Predicted volatility

    Returns
    -------
    dict
        Directional accuracy metrics
    """
    true_direction = np.diff(true_vol) > 0
    pred_direction = np.diff(pred_vol) > 0
    correct = (true_direction == pred_direction).sum()
    total = len(true_direction)

    return {
        "directional_accuracy": correct / total if total > 0 else 0.0,
        "correct": int(correct),
        "total": int(total),
    }


def log_score(
    returns: np.ndarray, pred_vol: np.ndarray, pred_mean: Optional[np.ndarray] = None
) -> float:
    """
    Log score for probabilistic forecasts.

    Assumes returns ~ N(pred_mean, pred_vol^2)
    """
    if pred_mean is None:
        pred_mean = np.zeros_like(returns)

    log_scores = -0.5 * (
        np.log(2 * np.pi * pred_vol**2)
        + ((returns - pred_mean) / pred_vol) ** 2
    )
    return np.mean(log_scores)


def coverage_probability(
    returns: np.ndarray,
    pred_vol: np.ndarray,
    pred_mean: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> float:
    """
    Compute coverage probability of prediction intervals.

    Parameters
    ----------
    returns : array-like
        Observed returns
    pred_vol : array-like
        Predicted volatility
    pred_mean : array-like, optional
        Predicted mean (default: 0)
    alpha : float
        Significance level (default: 0.05 for 95% interval)

    Returns
    -------
    float
        Coverage probability
    """
    if pred_mean is None:
        pred_mean = np.zeros_like(returns)

    from scipy.stats import norm

    z_score = norm.ppf(1 - alpha / 2)
    lower = pred_mean - z_score * pred_vol
    upper = pred_mean + z_score * pred_vol
    covered = (returns >= lower) & (returns <= upper)
    return covered.mean()

