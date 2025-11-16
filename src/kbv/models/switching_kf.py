"""Switching Kalman Filter with Hidden Markov Model."""

import numpy as np
from typing import Tuple, Optional
from kbv.models.kalman import KalmanFilter
from kbv.utils import ensure_2d, normalize_probs, make_positive_definite


class SwitchingKalmanFilter:
    """
    Switching Kalman Filter (SKF) with Hidden Markov Model.

    The filter maintains multiple Kalman filters, one per regime, and
    mixes their predictions based on HMM transition probabilities.

    Parameters
    ----------
    n_regimes : int
        Number of regimes
    transition_matrix : array-like
        (n_regimes, n_regimes) transition probability matrix
    initial_probs : array-like, optional
        Initial regime probabilities (n_regimes,)
    filters : list, optional
        List of KalmanFilter objects, one per regime
    F_list : list, optional
        List of state transition matrices (one per regime)
    H_list : list, optional
        List of observation matrices (one per regime)
    Q_list : list, optional
        List of process noise covariances (one per regime)
    R_list : list, optional
        List of observation noise covariances (one per regime)
    """

    def __init__(
        self,
        n_regimes: int,
        transition_matrix: np.ndarray,
        initial_probs: Optional[np.ndarray] = None,
        filters: Optional[list] = None,
        F_list: Optional[list] = None,
        H_list: Optional[list] = None,
        Q_list: Optional[list] = None,
        R_list: Optional[list] = None,
    ):
        self.n_regimes = n_regimes

        # Transition matrix
        self.transition_matrix = np.asarray(transition_matrix)
        assert self.transition_matrix.shape == (
            n_regimes,
            n_regimes,
        ), "Transition matrix must be (n_regimes, n_regimes)"
        self.transition_matrix = normalize_probs(
            self.transition_matrix, axis=1
        )

        # Initial probabilities
        if initial_probs is None:
            self.initial_probs = np.ones(n_regimes) / n_regimes
        else:
            self.initial_probs = normalize_probs(np.asarray(initial_probs))

        # Initialize filters
        if filters is not None:
            assert len(filters) == n_regimes, "Number of filters must match n_regimes"
            self.filters = filters
        elif F_list is not None:
            self.filters = []
            for i in range(n_regimes):
                kf = KalmanFilter(
                    F=F_list[i],
                    H=H_list[i] if H_list else H_list[0],
                    Q=Q_list[i] if Q_list else Q_list[0],
                    R=R_list[i] if R_list else R_list[0],
                )
                self.filters.append(kf)
        else:
            raise ValueError("Must provide either filters or F_list/H_list/Q_list/R_list")

        self.n_state = self.filters[0].n_state
        self.n_obs = self.filters[0].n_obs

    def filter(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run switching Kalman filter.

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
        regime_probs : array
            Regime probabilities (T, n_regimes)
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        T = len(observations)
        filtered_states = np.zeros((T, self.n_state))
        filtered_covs = np.zeros((T, self.n_state, self.n_state))
        regime_probs = np.zeros((T, self.n_regimes))

        # Initialize regime probabilities
        mu = self.initial_probs.copy()

        # Initialize filter states
        filter_states = {}
        filter_covs = {}
        for r in range(self.n_regimes):
            filter_states[r] = self.filters[r].x0.copy()
            filter_covs[r] = self.filters[r].P0.copy()

        for t in range(T):
            # Predict step for each regime
            pred_states = {}
            pred_covs = {}
            pred_obs = {}
            pred_obs_covs = {}
            likelihoods = np.zeros(self.n_regimes)

            for r in range(self.n_regimes):
                # Predict
                x_pred, P_pred = self.filters[r].predict(
                    filter_states[r], filter_covs[r]
                )
                pred_states[r] = x_pred
                pred_covs[r] = P_pred

                # Predicted observation
                y_pred = self.filters[r].H @ ensure_2d(x_pred)
                S = (
                    self.filters[r].H
                    @ P_pred
                    @ self.filters[r].H.T
                    + self.filters[r].R
                )
                pred_obs[r] = y_pred.flatten()
                pred_obs_covs[r] = S

                # Likelihood of observation under this regime
                from kbv.utils import log_likelihood_normal

                likelihoods[r] = np.exp(
                    log_likelihood_normal(observations[t], pred_obs[r], S)
                )

            # Update regime probabilities (HMM update)
            # mu_{t|t} propto likelihood * transition^T * mu_{t-1|t-1}
            mu_pred = self.transition_matrix.T @ mu
            mu = normalize_probs(likelihoods * mu_pred)

            # Mix predictions
            x_mixed = np.zeros(self.n_state)
            P_mixed = np.zeros((self.n_state, self.n_state))

            for r in range(self.n_regimes):
                x_mixed += mu[r] * pred_states[r]
                P_mixed += mu[r] * (
                    pred_covs[r]
                    + np.outer(
                        pred_states[r] - x_mixed,
                        pred_states[r] - x_mixed,
                    )
                )

            # Update each filter
            for r in range(self.n_regimes):
                filter_states[r], filter_covs[r], _ = self.filters[r].update(
                    pred_states[r], pred_covs[r], observations[t]
                )

            # Mix updated states
            x_upd = np.zeros(self.n_state)
            P_upd = np.zeros((self.n_state, self.n_state))

            for r in range(self.n_regimes):
                x_upd += mu[r] * filter_states[r]
                P_upd += mu[r] * (
                    filter_covs[r]
                    + np.outer(filter_states[r] - x_upd, filter_states[r] - x_upd)
                )

            filtered_states[t] = x_upd
            filtered_covs[t] = P_upd
            regime_probs[t] = mu

            # Update mu for next iteration
            mu = mu.copy()

        return filtered_states, filtered_covs, regime_probs

    def get_most_likely_regimes(self, regime_probs: np.ndarray) -> np.ndarray:
        """
        Get most likely regime at each time step.

        Parameters
        ----------
        regime_probs : array
            Regime probabilities (T, n_regimes)

        Returns
        -------
        regimes : array
            Most likely regime indices (T,)
        """
        return np.argmax(regime_probs, axis=1)

