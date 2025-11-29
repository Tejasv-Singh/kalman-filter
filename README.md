# Kalman–Bayesian Volatility Filtering for Regime-Switching Stochastic Models

[![Build Status](https://github.com/your_username/kalman-bayes-volatility/actions/workflows/ci.yml/badge.svg)](https://github.com/your_username/kalman-bayes-volatility/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage](https://img.shields.io/badge/coverage-85%25-green.svg)](https://github.com/your_username/kalman-bayes-volatility)

A production-grade Python framework for advanced volatility modeling in financial markets. This project implements and compares various state-space models, from classical Kalman Filters to modern Bayesian inference and regime-switching models, for tracking latent stochastic volatility.

## Abstract

We present a comprehensive framework for filtering and estimating latent volatility in financial time series. The core challenge in quantitative finance is that volatility is not directly observable. This project addresses this by formulating volatility as a latent state in a state-space model. We implement several powerful filtering techniques:

1. **Classical Filtering:** Linear Kalman Filter (KF) and Unscented Kalman Filter (UKF) for non-linear models.
2. **Regime-Switching Models:** A Switching Kalman Filter (SKF) based on the Hamilton (1989) filter, which allows model parameters (like volatility of volatility) to switch between discrete, unobserved regimes (e.g., "high-vol" and "low-vol" states) governed by a Hidden Markov Model (HMM).
3. **Bayesian / SMC Methods:** A fully Bayesian Stochastic Volatility (SV) model implemented in NumPyro (using both NUTS for MCMC and SVI for fast approximation) and a Sequential Monte Carlo (Particle Filter) implementation for non-Gaussian, non-linear state estimation.

The project provides synthetic data generators (Heston, Regime-Switching SV), a high-performance filtering library (accelerated with JAX), and comparative notebooks that benchmark these models against standards like GARCH.

## 🚀 Features

* **High-Performance Filters:** Implementations of Kalman, Unscented, and Switching Kalman Filters accelerated with `jax`.
* **Bayesian Inference:** A non-centered Bayesian Stochastic Volatility model with `numpyro`, solvable with both MCMC (NUTS) and SVI.
* **Sequential Monte Carlo:** A robust Particle Filter for general-purpose non-linear/non-Gaussian filtering.
* **Regime-Switching:** A core `SwitchingKalmanFilter` that dynamically mixes filter states based on HMM probabilities, providing real-time regime classification.
* **Data Pipelines:** Includes synthetic data generators for Heston and regime-switching models, plus a `yfinance` data fetcher.
* **MLOps Ready:** Complete with `pyproject.toml`, `Dockerfile` for containerization, and a full CI pipeline using GitHub Actions for testing and linting.
* **Visualization:** A `streamlit` dashboard for interactive analysis and teaching-quality notebooks for research.

## 📂 Project Structure

```
kalman-bayes-volatility/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── docker/
│   └── Dockerfile
├── src/
│   └── kbv/
│       ├── __init__.py
│       ├── data/
│       │   ├── fetch.py
│       │   └── synth.py
│       ├── models/
│       │   ├── kalman.py
│       │   ├── ukf.py
│       │   ├── switching_kf.py
│       │   ├── particle_filter.py
│       │   └── bayes_numpyro.py
│       ├── inference/
│       │   ├── em.py
│       │   ├── svi.py
│       │   └── mcmc.py
│       ├── metrics.py
│       └── utils.py
├── notebooks/
│   ├── 01_synthetic_data.ipynb
│   ├── 02_kalman_demo.ipynb
│   ├── 03_bayesian_sv_numpyro.ipynb
│   └── 04_benchmarks.ipynb
├── experiments/
│   └── run_experiments.py
├── dashboards/
│   └── app_streamlit.py
├── tests/
│   ├── test_synth.py
│   ├── test_kalman.py
│   └── test_inference.py
├── .github/
│   └── workflows/
│       └── ci.yml
└── docs/
    ├── index.md
    └── api.md
```

## 📦 Installation

### From Source

```bash
git clone https://github.com/your_username/kalman-bayes-volatility.git
cd kalman-bayes-volatility
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -e .
```

### Using Docker

```bash
docker build -t kbv:latest -f docker/Dockerfile .
docker run -p 8501:8501 kbv:latest streamlit run dashboards/app_streamlit.py
```

## 🎯 Quickstart

### Basic Kalman Filter

```python
from kbv.models.kalman import KalmanFilter
import numpy as np

# Simple random walk + noise model
kf = KalmanFilter(
    F=np.array([[1.0]]),  # state transition
    H=np.array([[1.0]]),  # observation matrix
    Q=np.array([[0.1]]),  # process noise
    R=np.array([[1.0]])   # observation noise
)

# Simulate data
T = 100
true_states = np.cumsum(np.random.randn(T) * 0.1)
observations = true_states + np.random.randn(T)

# Filter
filtered_states, covariances = kf.filter(observations)
smoothed_states, smoothed_cov = kf.smooth(observations)
```

### Bayesian Stochastic Volatility

```python
from kbv.models.bayes_numpyro import BayesianStochasticVolatility
from kbv.inference.mcmc import run_mcmc

model = BayesianStochasticVolatility()
returns = np.random.randn(200) * 0.02  # daily returns

# Run MCMC
mcmc_samples = run_mcmc(model, returns, num_samples=1000, num_warmup=500)
print(f"Posterior mean volatility: {mcmc_samples['volatility'].mean()}")
```

### Switching Kalman Filter

```python
from kbv.models.switching_kf import SwitchingKalmanFilter

# Two-regime model: low-vol and high-vol
skf = SwitchingKalmanFilter(
    n_regimes=2,
    transition_matrix=np.array([[0.95, 0.05], [0.05, 0.95]]),
    initial_probs=np.array([0.5, 0.5])
)

# Filter with regime probabilities
filtered, regimes = skf.filter(observations)
print(f"Regime at t=50: {regimes[50].argmax()}")  # Most likely regime
```

## 📊 Notebooks

1. **01_synthetic_data.ipynb:** Generate synthetic data from Heston and regime-switching models.
2. **02_kalman_demo.ipynb:** Visualize Kalman filtering on synthetic and real data.
3. **03_bayesian_sv_numpyro.ipynb:** Bayesian inference with traceplots and posterior diagnostics.
4. **04_benchmarks.ipynb:** Compare all methods against GARCH and evaluate forecasting performance.

## 🔬 Research Aims

* **Robust Volatility Estimation:** Compare filtering methods under different market conditions.
* **Regime Detection:** Real-time identification of volatility regimes using HMM-based switching models.
* **Bayesian Uncertainty Quantification:** Full posterior distributions for volatility, not just point estimates.
* **Computational Efficiency:** JAX-accelerated implementations suitable for high-frequency data.

## 🛣️ Roadmap

- [ ] Add support for multivariate volatility (VECM, BEKK models)
- [ ] Implement online learning for regime transition probabilities
- [ ] Add GPU acceleration for large-scale particle filters
- [ ] Integration with `arch` library for GARCH comparisons
- [ ] Real-time data streaming from market APIs

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{kalman_bayes_volatility,
  title = {Kalman–Bayesian Volatility Filtering for Regime-Switching Stochastic Models},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your_username/kalman-bayes-volatility}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
