"""Particle Filter (Sequential Monte Carlo) implementation."""

import numpy as np
from typing import Callable, Tuple, Optional
from kbv.utils import resample_particles, effective_sample_size, normalize_probs


class ParticleFilter:
    """
    Particle Filter (Sequential Monte Carlo) for non-linear/non-Gaussian models.

    State-space model:
        x_t ~ p(x_t | x_{t-1})  (state transition)
        y_t ~ p(y_t | x_t)       (observation likelihood)

    Parameters
    ----------
    n_particles : int
        Number of particles
    n_state : int
        State dimension
    transition_fn : callable
        Function that samples from p(x_t | x_{t-1})
        Signature: x_t = transition_fn(x_{t-1}, t)
    likelihood_fn : callable
        Function that computes p(y_t | x_t)
        Signature: log_prob = likelihood_fn(y_t, x_t)
    initial_dist : callable, optional
        Function that samples initial particles
        Signature: x_0 = initial_dist()
    resample_method : str
        Resampling method: 'systematic', 'multinomial', or 'stratified'
    resample_threshold : float
        Effective sample size threshold for resampling (default: 0.5)
    """

    def __init__(
        self,
        n_particles: int,
        n_state: int,
        transition_fn: Callable,
        likelihood_fn: Callable,
        initial_dist: Optional[Callable] = None,
        resample_method: str = "systematic",
        resample_threshold: float = 0.5,
    ):
        self.n_particles = n_particles
        self.n_state = n_state
        self.transition_fn = transition_fn
        self.likelihood_fn = likelihood_fn
        self.resample_method = resample_method
        self.resample_threshold = resample_threshold

        if initial_dist is None:
            self.initial_dist = lambda: np.zeros(n_state)
        else:
            self.initial_dist = initial_dist

    def filter(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run particle filter.

        Parameters
        ----------
        observations : array
            Observations (T, n_obs) or (T,)

        Returns
        -------
        filtered_states : array
            Filtered state means (T, n_state)
        filtered_covs : array
            Filtered state covariances (T, n_state, n_state)
        ess : array
            Effective sample size at each time step (T,)
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        T = len(observations)
        filtered_states = np.zeros((T, self.n_state))
        filtered_covs = np.zeros((T, self.n_state, self.n_state))
        ess = np.zeros(T)

        # Initialize particles
        particles = np.array([self.initial_dist() for _ in range(self.n_particles)])
        weights = np.ones(self.n_particles) / self.n_particles

        for t in range(T):
            # Resample if needed
            ess_current = effective_sample_size(weights)
            ess[t] = ess_current

            if ess_current < self.resample_threshold * self.n_particles:
                particles, weights = resample_particles(
                    particles, weights, method=self.resample_method
                )

            # Predict: propagate particles through transition
            particles = np.array(
                [self.transition_fn(particles[i], t) for i in range(self.n_particles)]
            )

            # Update: compute likelihood weights
            log_weights = np.array(
                [self.likelihood_fn(observations[t], particles[i]) for i in range(self.n_particles)]
            )

            # Normalize weights (log-space for numerical stability)
            log_weights = log_weights - np.max(log_weights)
            weights = np.exp(log_weights)
            weights = normalize_probs(weights)

            # Compute filtered state estimate
            filtered_states[t] = np.sum(weights[:, None] * particles, axis=0)

            # Compute filtered covariance
            diff = particles - filtered_states[t]
            filtered_covs[t] = np.sum(
                weights[:, None, None] * np.einsum("ij,ik->ijk", diff, diff), axis=0
            )

        return filtered_states, filtered_covs, ess

    def smooth(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run particle smoother (backward pass).

        Note: This is a simple backward smoother. For more advanced
        methods, see forward-backward particle smoother.

        Parameters
        ----------
        observations : array
            Observations

        Returns
        -------
        smoothed_states : array
            Smoothed state means (T, n_state)
        smoothed_covs : array
            Smoothed state covariances (T, n_state, n_state)
        """
        # For now, return filtered estimates
        # Full backward smoothing requires storing full particle history
        return self.filter(observations)[:2]

