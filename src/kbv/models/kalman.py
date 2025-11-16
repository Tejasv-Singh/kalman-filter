"""Classical Kalman Filter implementation."""

import numpy as np
from typing import Tuple, Optional
from kbv.utils import ensure_2d, check_positive_definite, make_positive_definite


class KalmanFilter:
    """
    Linear Kalman Filter for state-space models.

    State-space model:
        x_t = F * x_{t-1} + w_t,  w_t ~ N(0, Q)
        y_t = H * x_t + v_t,      v_t ~ N(0, R)

    Parameters
    ----------
    F : array-like
        State transition matrix (n_state, n_state)
    H : array-like
        Observation matrix (n_obs, n_state)
    Q : array-like
        Process noise covariance (n_state, n_state)
    R : array-like
        Observation noise covariance (n_obs, n_obs)
    x0 : array-like, optional
        Initial state mean (n_state,)
    P0 : array-like, optional
        Initial state covariance (n_state, n_state)
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ):
        self.F = np.asarray(F)
        self.H = np.asarray(H)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)

        n_state = self.F.shape[0]
        n_obs = self.H.shape[0]

        # Ensure matrices are 2D
        self.F = ensure_2d(self.F)
        self.H = ensure_2d(self.H)
        self.Q = ensure_2d(self.Q)
        self.R = ensure_2d(self.R)

        # Validate dimensions
        assert self.F.shape == (n_state, n_state), "F must be square"
        assert self.H.shape[1] == n_state, "H columns must match state dimension"
        assert self.Q.shape == (n_state, n_state), "Q must be square"
        assert self.R.shape == (n_obs, n_obs), "R must be square"

        # Make covariance matrices positive definite
        self.Q = make_positive_definite(self.Q)
        self.R = make_positive_definite(self.R)

        # Initial state
        if x0 is None:
            self.x0 = np.zeros(n_state)
        else:
            self.x0 = np.asarray(x0).flatten()

        if P0 is None:
            self.P0 = np.eye(n_state)
        else:
            self.P0 = make_positive_definite(np.asarray(P0))

        self.n_state = n_state
        self.n_obs = n_obs

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
        x = ensure_2d(x)
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred.flatten(), P_pred

    def update(
        self, x_pred: np.ndarray, P_pred: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        K : array
            Kalman gain
        """
        x_pred = ensure_2d(x_pred)
        y = ensure_2d(y)

        # Innovation
        y_pred = self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
        v = y - y_pred  # Innovation

        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update
        x_upd = x_pred + K @ v
        P_upd = P_pred - K @ self.H @ P_pred

        return x_upd.flatten(), P_upd, K

    def filter(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter (forward pass).

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
            x, P, _ = self.update(x_pred, P_pred, observations[t])

            filtered_states[t] = x
            filtered_covs[t] = P

        return filtered_states, filtered_covs

    def smooth(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman smoother (forward-backward pass).

        Parameters
        ----------
        observations : array
            Observations (T, n_obs) or (T,)

        Returns
        -------
        smoothed_states : array
            Smoothed state means (T, n_state)
        smoothed_covs : array
            Smoothed state covariances (T, n_state, n_state)
        """
        # Forward pass
        filtered_states, filtered_covs = self.filter(observations)

        # Backward pass
        T = len(observations)
        smoothed_states = np.zeros((T, self.n_state))
        smoothed_covs = np.zeros((T, self.n_state, self.n_state))

        # Initialize with last filtered state
        smoothed_states[-1] = filtered_states[-1]
        smoothed_covs[-1] = filtered_covs[-1]

        # Run predictions for backward pass
        pred_states = np.zeros((T, self.n_state))
        pred_covs = np.zeros((T, self.n_state, self.n_state))
        x = self.x0.copy()
        P = self.P0.copy()

        for t in range(T):
            x_pred, P_pred = self.predict(x, P)
            pred_states[t] = x_pred
            pred_covs[t] = P_pred
            x, P, _ = self.update(x_pred, P_pred, observations[t])

        # Backward recursion
        for t in range(T - 2, -1, -1):
            # Smoothing gain
            J = filtered_covs[t] @ self.F.T @ np.linalg.inv(pred_covs[t + 1])

            # Smoothed state
            smoothed_states[t] = (
                filtered_states[t]
                + J @ (smoothed_states[t + 1] - pred_states[t + 1])
            )

            # Smoothed covariance
            smoothed_covs[t] = (
                filtered_covs[t]
                + J @ (smoothed_covs[t + 1] - pred_covs[t + 1]) @ J.T
            )

        return smoothed_states, smoothed_covs

    def log_likelihood(self, observations: np.ndarray) -> float:
        """
        Compute log-likelihood of observations.

        Parameters
        ----------
        observations : array
            Observations

        Returns
        -------
        float
            Log-likelihood
        """
        from kbv.utils import log_likelihood_normal

        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        T = len(observations)
        log_lik = 0.0

        x = self.x0.copy()
        P = self.P0.copy()

        for t in range(T):
            # Predict
            x_pred, P_pred = self.predict(x, P)

            # Innovation
            y_pred = self.H @ ensure_2d(x_pred)
            S = self.H @ P_pred @ self.H.T + self.R

            # Log-likelihood contribution
            log_lik += log_likelihood_normal(observations[t], y_pred.flatten(), S)

            # Update
            x, P, _ = self.update(x_pred, P_pred, observations[t])

        return log_lik

