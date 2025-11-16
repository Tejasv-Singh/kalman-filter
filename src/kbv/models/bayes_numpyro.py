"""Bayesian Stochastic Volatility model using NumPyro."""

import numpy as np
import jax.numpy as jnp
import jax
from numpyro import distributions as dist
from numpyro.contrib.control_flow import scan
import numpyro
from typing import Dict, Optional


class BayesianStochasticVolatility:
    """
    Bayesian Stochastic Volatility model.

    Model:
        y_t ~ N(0, exp(h_t))
        h_t = mu + phi * (h_{t-1} - mu) + sigma * eta_t
        h_0 ~ N(mu, sigma^2 / (1 - phi^2))

    Priors:
        mu ~ N(0, 1)
        phi ~ Uniform(-1, 1)  (or transformed to ensure stationarity)
        sigma ~ HalfNormal(0.1)

    This uses a non-centered parameterization for better MCMC sampling.
    """

    def __init__(self):
        pass

    def model(self, returns: np.ndarray) -> None:
        """
        NumPyro model definition.

        Parameters
        ----------
        returns : array
            Observed returns (T,)
        """
        T = len(returns)
        returns = jnp.asarray(returns)

        # Priors
        mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
        phi_raw = numpyro.sample("phi_raw", dist.Normal(0.0, 1.0))
        # Transform to (-1, 1) for stationarity
        phi = numpyro.deterministic("phi", jnp.tanh(phi_raw))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))

        # Non-centered parameterization for log-volatility
        # h_t = mu + sigma_h * h_tilde_t
        # where h_tilde follows AR(1) with unit variance
        h_tilde_0 = numpyro.sample(
            "h_tilde_0", dist.Normal(0.0, 1.0 / jnp.sqrt(1 - phi**2 + 1e-6))
        )

        def transition(carry, _):
            h_tilde_prev = carry
            h_tilde = numpyro.sample(
                "h_tilde", dist.Normal(phi * h_tilde_prev, 1.0)
            )
            return h_tilde, h_tilde

        # Scan over time
        _, h_tilde_seq = scan(
            transition, h_tilde_0, None, length=T - 1
        )
        h_tilde_seq = jnp.concatenate([[h_tilde_0], h_tilde_seq])

        # Transform to log-volatility
        h_seq = numpyro.deterministic("h", mu + sigma * h_tilde_seq)
        volatility = numpyro.deterministic("volatility", jnp.exp(h_seq / 2))

        # Likelihood
        with numpyro.plate("time", T):
            numpyro.sample("returns", dist.Normal(0.0, volatility), obs=returns)

    def predict_volatility(
        self, samples: Dict[str, np.ndarray], n_steps: int = 1
    ) -> np.ndarray:
        """
        Predict future volatility from posterior samples.

        Parameters
        ----------
        samples : dict
            Posterior samples with keys 'mu', 'phi', 'sigma', 'h'
        n_steps : int
            Number of steps ahead to predict

        Returns
        -------
        pred_vol : array
            Predicted volatility (n_samples, n_steps)
        """
        mu = samples["mu"]
        phi = samples["phi"]
        sigma = samples["sigma"]
        h_last = samples["h"][:, -1]  # Last log-volatility

        n_samples = len(mu)
        pred_vol = np.zeros((n_samples, n_steps))

        for i in range(n_samples):
            h = h_last[i]
            for t in range(n_steps):
                h = mu[i] + phi[i] * (h - mu[i]) + sigma[i] * np.random.randn()
                pred_vol[i, t] = np.exp(h / 2)

        return pred_vol

