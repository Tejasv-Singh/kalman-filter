"""Expectation-Maximization algorithm for parameter estimation."""

import numpy as np
from typing import Tuple, Optional, Dict
from kbv.models.kalman import KalmanFilter
from scipy.optimize import minimize


class EMKalmanFilter:
    """
    Expectation-Maximization for Kalman Filter parameter estimation.

    Estimates F, H, Q, R from observations using EM algorithm.
    """

    def __init__(
        self,
        n_state: int,
        n_obs: int,
        F_init: Optional[np.ndarray] = None,
        H_init: Optional[np.ndarray] = None,
        Q_init: Optional[np.ndarray] = None,
        R_init: Optional[np.ndarray] = None,
    ):
        self.n_state = n_state
        self.n_obs = n_obs

        # Initialize parameters
        if F_init is None:
            self.F = np.eye(n_state)
        else:
            self.F = np.asarray(F_init)

        if H_init is None:
            self.H = np.random.randn(n_obs, n_state) * 0.1
        else:
            self.H = np.asarray(H_init)

        if Q_init is None:
            self.Q = np.eye(n_state) * 0.1
        else:
            self.Q = np.asarray(Q_init)

        if R_init is None:
            self.R = np.eye(n_obs) * 0.1
        else:
            self.R = np.asarray(R_init)

    def e_step(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        E-step: Run Kalman smoother to get state estimates.

        Returns
        -------
        smoothed_states : array
            Smoothed state means
        smoothed_covs : array
            Smoothed state covariances
        smoothed_cross_covs : array
            Smoothed cross-covariances E[x_t x_{t-1}^T]
        """
        kf = KalmanFilter(self.F, self.H, self.Q, self.R)
        smoothed_states, smoothed_covs = kf.smooth(observations)

        T = len(observations)
        smoothed_cross_covs = np.zeros((T - 1, self.n_state, self.n_state))

        # Compute cross-covariances
        filtered_states, filtered_covs = kf.filter(observations)
        pred_states = np.zeros((T, self.n_state))
        pred_covs = np.zeros((T, self.n_state, self.n_state))

        x = kf.x0.copy()
        P = kf.P0.copy()

        for t in range(T):
            x_pred, P_pred = kf.predict(x, P)
            pred_states[t] = x_pred
            pred_covs[t] = P_pred
            x, P, _ = kf.update(x_pred, P_pred, observations[t])

        # Backward pass for cross-covariances
        for t in range(T - 2, -1, -1):
            J = filtered_covs[t] @ self.F.T @ np.linalg.inv(pred_covs[t + 1])
            smoothed_cross_covs[t] = (
                filtered_covs[t] @ J.T
                + J
                @ (smoothed_covs[t + 1] - filtered_covs[t + 1])
                @ J.T
            )

        return smoothed_states, smoothed_covs, smoothed_cross_covs

    def m_step(
        self,
        observations: np.ndarray,
        smoothed_states: np.ndarray,
        smoothed_covs: np.ndarray,
        smoothed_cross_covs: np.ndarray,
    ) -> None:
        """
        M-step: Update parameters to maximize expected log-likelihood.
        """
        T = len(observations)
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        # Update F
        sum1 = np.sum(smoothed_cross_covs, axis=0)
        sum2 = np.sum(smoothed_covs[:-1], axis=0)
        self.F = sum1 @ np.linalg.inv(sum2)

        # Update Q
        sum_q = np.zeros((self.n_state, self.n_state))
        for t in range(1, T):
            sum_q += (
                smoothed_covs[t]
                - self.F @ smoothed_cross_covs[t - 1].T
                - smoothed_cross_covs[t - 1] @ self.F.T
                + self.F @ smoothed_covs[t - 1] @ self.F.T
            )
        self.Q = sum_q / (T - 1)
        self.Q = (self.Q + self.Q.T) / 2  # Ensure symmetry

        # Update H
        sum_h1 = np.sum(
            observations[:, :, None] @ smoothed_states[:, None, :], axis=0
        )
        sum_h2 = np.sum(
            smoothed_states[:, :, None] @ smoothed_states[:, None, :]
            + smoothed_covs,
            axis=0,
        )
        self.H = sum_h1 @ np.linalg.inv(sum_h2)

        # Update R
        sum_r = np.zeros((self.n_obs, self.n_obs))
        for t in range(T):
            y_pred = self.H @ smoothed_states[t]
            diff = observations[t] - y_pred
            sum_r += np.outer(diff, diff) + self.H @ smoothed_covs[t] @ self.H.T
        self.R = sum_r / T
        self.R = (self.R + self.R.T) / 2  # Ensure symmetry

    def fit(
        self,
        observations: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Run EM algorithm.

        Parameters
        ----------
        observations : array
            Observations
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        verbose : bool
            Print progress

        Returns
        -------
        dict
            Fitted parameters
        """
        prev_log_lik = -np.inf

        for iteration in range(max_iter):
            # E-step
            smoothed_states, smoothed_covs, smoothed_cross_covs = self.e_step(
                observations
            )

            # Compute log-likelihood
            kf = KalmanFilter(self.F, self.H, self.Q, self.R)
            log_lik = kf.log_likelihood(observations)

            # M-step
            self.m_step(observations, smoothed_states, smoothed_covs, smoothed_cross_covs)

            # Check convergence
            if abs(log_lik - prev_log_lik) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

            prev_log_lik = log_lik

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: log-likelihood = {log_lik:.4f}")

        return {
            "F": self.F,
            "H": self.H,
            "Q": self.Q,
            "R": self.R,
            "log_likelihood": log_lik,
        }

