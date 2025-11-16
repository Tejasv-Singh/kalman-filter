"""Stochastic Variational Inference (SVI) for Bayesian models."""

import numpy as np
import jax.numpy as jnp
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import numpyro
from typing import Dict, Optional
from tqdm import tqdm


def run_svi(
    model,
    data: np.ndarray,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    guide: Optional[object] = None,
    rng_key: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run Stochastic Variational Inference (SVI).

    Parameters
    ----------
    model : callable
        NumPyro model function
    data : array
        Observed data
    num_iterations : int
        Number of optimization iterations
    learning_rate : float
        Learning rate for Adam optimizer
    guide : object, optional
        Custom guide (default: AutoNormal)
    rng_key : int, optional
        Random key
    verbose : bool
        Show progress bar

    Returns
    -------
    dict
        Dictionary with optimized parameters and samples
    """
    import jax.random as random

    if rng_key is None:
        rng_key = random.PRNGKey(42)

    # Create guide if not provided
    if guide is None:
        guide = AutoNormal(model)

    # Setup SVI
    optimizer = numpyro.optim.Adam(step_size=learning_rate)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Initialize
    svi_state = svi.init(rng_key, data)

    # Training loop
    losses = []
    iterator = tqdm(range(num_iterations)) if verbose else range(num_iterations)

    for i in iterator:
        svi_state, loss = svi.update(svi_state, data)
        losses.append(loss)

        if verbose and (i + 1) % 100 == 0:
            iterator.set_description(f"Loss: {loss:.4f}")

    # Get optimized parameters
    params = svi.get_params(svi_state)

    # Sample from posterior
    predictive = numpyro.infer.Predictive(guide, params=params, num_samples=1000)
    samples = predictive(rng_key, data)

    return {
        "params": params,
        "samples": samples,
        "losses": np.array(losses),
    }


def run_svi_with_schedule(
    model,
    data: np.ndarray,
    num_iterations: int = 1000,
    initial_lr: float = 0.01,
    final_lr: float = 0.001,
    guide: Optional[object] = None,
    rng_key: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run SVI with learning rate schedule.

    Parameters
    ----------
    model : callable
        NumPyro model function
    data : array
        Observed data
    num_iterations : int
        Number of optimization iterations
    initial_lr : float
        Initial learning rate
    final_lr : float
        Final learning rate
    guide : object, optional
        Custom guide
    rng_key : int, optional
        Random key
    verbose : bool
        Show progress bar

    Returns
    -------
    dict
        Results dictionary
    """
    import jax.random as random

    if rng_key is None:
        rng_key = random.PRNGKey(42)

    if guide is None:
        guide = AutoNormal(model)

    # Learning rate schedule
    lr_schedule = np.linspace(initial_lr, final_lr, num_iterations)

    optimizer = numpyro.optim.Adam(step_size=initial_lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    svi_state = svi.init(rng_key, data)
    losses = []
    iterator = tqdm(range(num_iterations)) if verbose else range(num_iterations)

    for i in iterator:
        # Update learning rate
        optimizer = numpyro.optim.Adam(step_size=lr_schedule[i])
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        svi_state, loss = svi.update(svi_state, data)
        losses.append(loss)

        if verbose and (i + 1) % 100 == 0:
            iterator.set_description(f"Loss: {loss:.4f}, LR: {lr_schedule[i]:.6f}")

    params = svi.get_params(svi_state)
    predictive = numpyro.infer.Predictive(guide, params=params, num_samples=1000)
    samples = predictive(rng_key, data)

    return {
        "params": params,
        "samples": samples,
        "losses": np.array(losses),
    }

