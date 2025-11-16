"""Utility functions for the KBV package."""

import numpy as np
from typing import Tuple, Optional
import jax.numpy as jnp
from jax import jit


def ensure_2d(x: np.ndarray) -> np.ndarray:
    """Ensure array is 2D."""
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def log_likelihood_normal(
    y: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> float:
    """
    Compute log-likelihood of multivariate normal distribution.

    Parameters
    ----------
    y : array-like
        Observed values
    mean : array-like
        Mean vector
    cov : array-like
        Covariance matrix

    Returns
    -------
    float
        Log-likelihood
    """
    y = ensure_2d(y)
    mean = ensure_2d(mean)
    n = y.shape[0]
    diff = y - mean
    try:
        L = np.linalg.cholesky(cov)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        quad_form = np.linalg.solve(L, diff)
        return -0.5 * (n * np.log(2 * np.pi) + log_det + np.sum(quad_form**2))
    except np.linalg.LinAlgError:
        return -np.inf


@jit
def jax_cholesky_update(L: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Rank-1 Cholesky update using JAX.

    Computes Cholesky decomposition of L @ L.T + v @ v.T
    """
    n = L.shape[0]
    L_new = L.copy()
    for i in range(n):
        r = jnp.sqrt(L_new[i, i] ** 2 + v[i] ** 2)
        c = r / L_new[i, i] if L_new[i, i] != 0 else 1.0
        s = v[i] / L_new[i, i] if L_new[i, i] != 0 else 0.0
        L_new = L_new.at[i, i].set(r)
        if i < n - 1:
            L_new = L_new.at[i + 1 :, i].set(
                (L_new[i + 1 :, i] + s * v[i + 1 :]) / c
            )
            v = v.at[i + 1 :].set(c * v[i + 1 :] - s * L_new[i + 1 :, i])
    return L_new


def normalize_probs(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize probabilities to sum to 1."""
    probs = np.asarray(probs)
    sums = np.sum(probs, axis=axis, keepdims=True)
    sums = np.where(sums == 0, 1.0, sums)
    return probs / sums


def resample_particles(
    particles: np.ndarray, weights: np.ndarray, method: str = "systematic"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample particles based on weights.

    Parameters
    ----------
    particles : array-like
        Particle states (n_particles, state_dim)
    weights : array-like
        Normalized weights (n_particles,)
    method : str
        Resampling method: 'systematic', 'multinomial', or 'stratified'

    Returns
    -------
    particles : array
        Resampled particles
    weights : array
        Uniform weights after resampling
    """
    n_particles = len(particles)
    weights = np.asarray(weights)
    weights = weights / weights.sum()

    if method == "systematic":
        u = np.random.rand() / n_particles
        cumsum = np.cumsum(weights)
        indices = np.zeros(n_particles, dtype=int)
        j = 0
        for i in range(n_particles):
            while cumsum[j] < u + i / n_particles:
                j += 1
            indices[i] = j
    elif method == "multinomial":
        indices = np.random.choice(n_particles, size=n_particles, p=weights)
    elif method == "stratified":
        u = (np.arange(n_particles) + np.random.rand(n_particles)) / n_particles
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, u)
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    return particles[indices], np.ones(n_particles) / n_particles


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute effective sample size (ESS) from particle weights.

    ESS = 1 / sum(w^2)
    """
    weights = np.asarray(weights)
    weights = weights / weights.sum()
    return 1.0 / np.sum(weights**2)


def check_positive_definite(matrix: np.ndarray, name: str = "matrix") -> None:
    """Check if matrix is positive definite."""
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name} is not positive definite")


def make_positive_definite(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Make matrix positive definite by adding small diagonal."""
    matrix = np.asarray(matrix)
    eigenvals = np.linalg.eigvals(matrix)
    min_eigenval = np.min(eigenvals)
    if min_eigenval <= 0:
        matrix = matrix + (eps - min_eigenval) * np.eye(matrix.shape[0])
    return matrix

