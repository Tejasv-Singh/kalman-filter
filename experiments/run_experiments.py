"""Experiment runner for volatility model comparisons."""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any
import argparse
from tqdm import tqdm

from kbv.models.kalman import KalmanFilter
from kbv.models.switching_kf import SwitchingKalmanFilter
from kbv.models.bayes_numpyro import BayesianStochasticVolatility
from kbv.inference.mcmc import run_mcmc
from kbv.data.synth import generate_simple_sv, generate_regime_switching_sv
from kbv.metrics import evaluate_volatility_forecast


def run_kalman_experiment(returns: np.ndarray, true_vol: np.ndarray) -> Dict[str, Any]:
    """Run Kalman filter experiment."""
    observations = returns**2

    kf = KalmanFilter(
        F=np.array([[0.95]]),
        H=np.array([[1.0]]),
        Q=np.array([[0.01]]),
        R=np.array([[0.1]]),
    )

    filtered_states, _ = kf.filter(observations)
    filtered_vol = np.exp(filtered_states.flatten() / 2)

    metrics = evaluate_volatility_forecast(true_vol, filtered_vol, returns)
    return {"method": "KalmanFilter", "metrics": metrics, "volatility": filtered_vol.tolist()}


def run_switching_kf_experiment(
    returns: np.ndarray, true_vol: np.ndarray, true_regimes: np.ndarray
) -> Dict[str, Any]:
    """Run switching Kalman filter experiment."""
    observations = returns**2

    from kbv.models.kalman import KalmanFilter

    kf_low = KalmanFilter(
        F=np.array([[0.95]]), H=np.array([[1.0]]),
        Q=np.array([[0.01]]), R=np.array([[0.1]])
    )
    kf_high = KalmanFilter(
        F=np.array([[0.95]]), H=np.array([[1.0]]),
        Q=np.array([[0.05]]), R=np.array([[0.1]])
    )

    transition_matrix = np.array([[0.98, 0.02], [0.02, 0.98]])

    skf = SwitchingKalmanFilter(
        n_regimes=2,
        transition_matrix=transition_matrix,
        filters=[kf_low, kf_high],
    )

    filtered_states, _, regime_probs = skf.filter(observations)
    filtered_vol = np.exp(filtered_states.flatten() / 2)
    predicted_regimes = skf.get_most_likely_regimes(regime_probs)

    metrics = evaluate_volatility_forecast(true_vol, filtered_vol, returns)
    regime_accuracy = (predicted_regimes == true_regimes).mean()

    return {
        "method": "SwitchingKalmanFilter",
        "metrics": metrics,
        "regime_accuracy": float(regime_accuracy),
        "volatility": filtered_vol.tolist(),
    }


def run_bayesian_experiment(returns: np.ndarray, true_vol: np.ndarray) -> Dict[str, Any]:
    """Run Bayesian SV experiment."""
    model = BayesianStochasticVolatility()

    mcmc_results = run_mcmc(
        model.model, returns, num_samples=500, num_warmup=250, progress_bar=False
    )

    samples = mcmc_results["samples"]
    vol_samples = samples["volatility"]
    filtered_vol = vol_samples.mean(axis=0)

    metrics = evaluate_volatility_forecast(true_vol, filtered_vol, returns)

    return {
        "method": "BayesianSV",
        "metrics": metrics,
        "volatility": filtered_vol.tolist(),
        "posterior_mean": {
            "mu": float(samples["mu"].mean()),
            "phi": float(samples["phi"].mean()),
            "sigma": float(samples["sigma"].mean()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run volatility model experiments")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--n-obs", type=int, default=500, help="Number of observations")
    parser.add_argument("--output", type=str, default="results.json", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results = []

    for trial in tqdm(range(args.n_trials)):
        seed = args.seed + trial
        np.random.seed(seed)

        # Generate simple SV data
        returns, true_log_vol = generate_simple_sv(n_obs=args.n_obs, seed=seed)
        true_vol = np.exp(true_log_vol / 2)

        # Run experiments
        kf_result = run_kalman_experiment(returns, true_vol)
        kf_result["trial"] = trial
        results.append(kf_result)

        bayes_result = run_bayesian_experiment(returns, true_vol)
        bayes_result["trial"] = trial
        results.append(bayes_result)

        # Generate regime-switching data
        returns_rs, log_vol_rs, true_regimes = generate_regime_switching_sv(
            n_obs=args.n_obs, seed=seed
        )
        true_vol_rs = np.exp(log_vol_rs / 2)

        skf_result = run_switching_kf_experiment(returns_rs, true_vol_rs, true_regimes)
        skf_result["trial"] = trial
        results.append(skf_result)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

    # Print summary
    methods = ["KalmanFilter", "BayesianSV", "SwitchingKalmanFilter"]
    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        if method_results:
            avg_rmse = np.mean([r["metrics"]["rmse"] for r in method_results])
            avg_corr = np.mean([r["metrics"]["correlation"] for r in method_results])
            print(f"{method}: RMSE={avg_rmse:.4f}, Correlation={avg_corr:.4f}")


if __name__ == "__main__":
    main()

