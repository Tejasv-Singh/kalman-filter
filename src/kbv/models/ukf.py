"""Unscented Kalman Filter for non-linear state-space models."""

import numpy as np
from typing import Callable, Tuple, Optional
from kbv.utils import ensure_2d, make_positive_definite


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter (UKF) for non-linear state-space models.

    State-space model:
        x_t = f(x_{t-1}) + w_t,  w_t ~ N(0, Q)
        y_t = h(x_t) + v_t,      v_t ~ N(0, R)

    Parameters
    ----------
    f : callable
        State transition function: x_t = f(x_{t-1})
    h : callable
        Observation function: y_t = h(x_t)
    Q : array-like
        Process noise covariance (n_state, n_state)
    R : array-like
        Observation noise covariance (n_obs, n_obs)
    n_state : int
        State dimension
    n_obs : int
        Observation dimension
    x0 : array-like, optional
        Initial state mean
    P0 : array-like, optional
        Initial state covariance
    alpha : float
        UKF parameter (default: 1e-3)
    beta : float
        UKF parameter (default: 2.0)
    kappa : float
        UKF parameter (default: 0.0)
    """

    def __init__(
        self,
        f: Callable,
        h: Callable,
        Q: np.ndarray,
        R: np.ndarray,
        n_state: int,
        n_obs: int,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.f = f
        self.h = h
        self.Q = make_positive_definite(np.asarray(Q))
        self.R = make_positive_definite(np.asarray(R))
        self.n_state = n_state
        self.n_obs = n_obs

        if x0 is None:
            self.x0 = np.zeros(n_state)
        else:
            self.x0 = np.asarray(x0).flatten()

        if P0 is None:
            self.P0 = np.eye(n_state)
        else:
            self.P0 = make_positive_definite(np.asarray(P0))

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (n_state + kappa) - n_state

        # Weights
        self.n_sigma = 2 * n_state + 1
        W0_m = self.lambda_ / (n_state + self.lambda_)
        W0_c = W0_m + (1 - alpha**2 + beta)
        Wi = 1 / (2 * (n_state + self.lambda_))

        self.Wm = np.array([W0_m] + [Wi] * (2 * n_state))
        self.Wc = np.array([W0_c] + [Wi] * (2 * n_state))

    def _compute_sigma_points(
        self, x: np.ndarray, P: np.ndarray
    ) -> np.ndarray:
        """
        Compute sigma points.

        Parameters
        ----------
        x : array
            State mean
        P : array
            State covariance

        Returns
        -------
        sigma_points : array
            Sigma points (n_sigma, n_state)
        """
        x = ensure_2d(x).flatten()
        n = len(x)

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky((n + self.lambda_) * P)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(make_positive_definite((n + self.lambda_) * P))

        sigma_points = np.zeros((self.n_sigma, n))
        sigma_points[0] = x

        for i in range(n):
            sigma_points[i + 1] = x + L[:, i]
            sigma_points[i + n + 1] = x - L[:, i]

        return sigma_points

    def _predict_sigma_points(self, sigma_points: np.ndarray) -> np.ndarray:
        """Propagate sigma points through state transition."""
        return np.array([self.f(sp) for sp in sigma_points])

    def _update_sigma_points(self, sigma_points: np.ndarray) -> np.ndarray:
        """Propagate sigma points through observation function."""
        return np.array([self.h(sp) for sp in sigma_points])

    def predict(
        self, x: np.ndarray, P: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step.

        Parameters
        ----------
        x : array
            Current state mean
        P : array
            Current state covariance

        Returns
        -------
        x_pred : array
            Predicted state mean
        P_pred : array
            Predicted state covariance
        """
        # Generate sigma points
        sigma_points = self._compute_sigma_points(x, P)

        # Propagate through state transition
        sigma_points_pred = self._predict_sigma_points(sigma_points)

        # Compute predicted mean
        x_pred = np.sum(self.Wm[:, None] * sigma_points_pred, axis=0)

        # Compute predicted covariance
        diff = sigma_points_pred - x_pred
        P_pred = np.sum(
            self.Wc[:, None, None] * np.einsum("ij,ik->ijk", diff, diff), axis=0
        ) + self.Q

        return x_pred, P_pred

    def update(
        self, x_pred: np.ndarray, P_pred: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step.

        Parameters
        ----------
        x_pred : array
            Predicted state mean
        P_pred : array
            Predicted state covariance
        y : array
            Observation

        Returns
        -------
        x_upd : array
            Updated state mean
        P_upd : array
            Updated state covariance
        """
        # Generate sigma points from prediction
        sigma_points = self._compute_sigma_points(x_pred, P_pred)

        # Propagate through observation function
        sigma_points_obs = self._update_sigma_points(sigma_points)

        # Compute predicted observation mean
        y_pred = np.sum(self.Wm[:, None] * sigma_points_obs, axis=0)

        # Compute innovation covariance
        diff_obs = sigma_points_obs - y_pred
        S = (
            np.sum(
                self.Wc[:, None, None]
                * np.einsum("ij,ik->ijk", diff_obs, diff_obs),
                axis=0,
            )
            + self.R
        )

        # Compute cross-covariance
        diff_state = sigma_points - x_pred
        Pxy = np.sum(
            self.Wc[:, None, None]
            * np.einsum("ij,ik->ijk", diff_state, diff_obs),
            axis=0,
        )

        # Kalman gain
        K = Pxy @ np.linalg.inv(S)

        # Update
        v = y - y_pred
        x_upd = x_pred + K @ v
        P_upd = P_pred - K @ S @ K.T

        return x_upd, P_upd

    def filter(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run UKF filter (forward pass).

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
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        T = len(observations)
        filtered_states = np.zeros((T, self.n_state))
        filtered_covs = np.zeros((T, self.n_state, self.n_state))

        x = self.x0.copy()
        P = self.P0.copy()

        for t in range(T):
            # Predict
            x_pred, P_pred = self.predict(x, P)

            # Update
            x, P = self.update(x_pred, P_pred, observations[t])

            filtered_states[t] = x
            filtered_covs[t] = P

        return filtered_states, filtered_covs

