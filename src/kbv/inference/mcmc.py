"""MCMC inference using NumPyro NUTS sampler."""

import numpy as np
import jax.random as random
from numpyro.infer import MCMC, NUTS
from typing import Dict, Optional
from tqdm import tqdm


def run_mcmc(
    model,
    data: np.ndarray,
    num_samples: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 1,
    rng_key: Optional[int] = None,
    progress_bar: bool = True,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Run MCMC using NUTS sampler.

    Parameters
    ----------
    model : callable
        NumPyro model function
    data : array
        Observed data
    num_samples : int
        Number of posterior samples
    num_warmup : int
        Number of warmup samples
    num_chains : int
        Number of chains
    rng_key : int, optional
        Random key
    progress_bar : bool
        Show progress bar
    **kwargs
        Additional arguments passed to NUTS

    Returns
    -------
    dict
        Dictionary with MCMC samples
    """
    if rng_key is None:
        rng_key = random.PRNGKey(42)

    # Setup NUTS sampler
    kernel = NUTS(model, **kwargs)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    # Run MCMC
    mcmc.run(rng_key, data)

    # Get samples
    samples = mcmc.get_samples()

    # Convert JAX arrays to numpy
    samples_np = {k: np.asarray(v) for k, v in samples.items()}

    return {
        "samples": samples_np,
        "mcmc": mcmc,  # Keep MCMC object for diagnostics
    }


def run_mcmc_multiple_chains(
    model,
    data: np.ndarray,
    num_samples: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 4,
    rng_key: Optional[int] = None,
    progress_bar: bool = True,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Run MCMC with multiple chains for diagnostics.

    Parameters
    ----------
    model : callable
        NumPyro model function
    data : array
        Observed data
    num_samples : int
        Number of posterior samples per chain
    num_warmup : int
        Number of warmup samples per chain
    num_chains : int
        Number of chains
    rng_key : int, optional
        Random key
    progress_bar : bool
        Show progress bar
    **kwargs
        Additional arguments passed to NUTS

    Returns
    -------
    dict
        Dictionary with MCMC samples and diagnostics
    """
    if rng_key is None:
        rng_key = random.PRNGKey(42)

    # Generate keys for each chain
    rng_keys = random.split(rng_key, num_chains)

    kernel = NUTS(model, **kwargs)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    mcmc.run(rng_keys, data)
    samples = mcmc.get_samples()

    # Convert to numpy
    samples_np = {k: np.asarray(v) for k, v in samples.items()}

    # Compute diagnostics
    diagnostics = {}
    for key, value in samples_np.items():
        if value.ndim >= 2:  # Has chain dimension
            # R-hat (Gelman-Rubin statistic)
            n_chains, n_samples = value.shape[:2]
            chain_means = np.mean(value, axis=1)
            chain_vars = np.var(value, axis=1)

            between_chain_var = np.var(chain_means, axis=0, ddof=1)
            within_chain_var = np.mean(chain_vars, axis=0)

            var_hat = (
                (n_samples - 1) / n_samples * within_chain_var
                + 1 / n_samples * between_chain_var
            )
            r_hat = np.sqrt(var_hat / (within_chain_var + 1e-8))

            diagnostics[f"{key}_rhat"] = r_hat
            diagnostics[f"{key}_ess"] = compute_ess(value)

    return {
        "samples": samples_np,
        "diagnostics": diagnostics,
        "mcmc": mcmc,
    }


def compute_ess(samples: np.ndarray) -> np.ndarray:
    """
    Compute Effective Sample Size (ESS).

    Parameters
    ----------
    samples : array
        MCMC samples (n_chains, n_samples, ...)

    Returns
    -------
    ess : array
        Effective sample size
    """
    # Simple ESS computation using autocorrelation
    if samples.ndim < 2:
        return len(samples)

    n_chains, n_samples = samples.shape[:2]
    ess_per_chain = []

    for chain in range(n_chains):
        chain_samples = samples[chain].flatten()
        # Compute autocorrelation
        autocorr = np.correlate(chain_samples, chain_samples, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / autocorr[0]

        # Find first negative autocorrelation
        first_neg = np.where(autocorr < 0)[0]
        if len(first_neg) > 0:
            tau = 1 + 2 * np.sum(autocorr[1 : first_neg[0]])
        else:
            tau = 1 + 2 * np.sum(autocorr[1:])

        ess = n_samples / tau
        ess_per_chain.append(ess)

    return np.mean(ess_per_chain)

