# Kalman-Bayesian Volatility Filtering Documentation

## Introduction

This project provides a comprehensive framework for filtering and estimating latent volatility in financial time series using state-space models. Volatility is a key concept in quantitative finance, representing the degree of variation in asset prices. However, volatility is not directly observable and must be inferred from observed returns.

## Mathematical Framework

### State-Space Models

A state-space model consists of two equations:

**State Equation:**
\[
x_t = f(x_{t-1}) + w_t, \quad w_t \sim \mathcal{N}(0, Q)
\]

**Observation Equation:**
\[
y_t = h(x_t) + v_t, \quad v_t \sim \mathcal{N}(0, R)
\]

where:
- \(x_t\) is the latent state (e.g., log-volatility)
- \(y_t\) is the observed variable (e.g., returns)
- \(f\) and \(h\) are transition and observation functions
- \(w_t\) and \(v_t\) are process and observation noise

### Kalman Filter

For linear Gaussian models, the Kalman Filter provides optimal filtering. The algorithm consists of:

1. **Prediction:**
   \[
   \hat{x}_{t|t-1} = F \hat{x}_{t-1|t-1}
   \]
   \[
   P_{t|t-1} = F P_{t-1|t-1} F^T + Q
   \]

2. **Update:**
   \[
   K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}
   \]
   \[
   \hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (y_t - H \hat{x}_{t|t-1})
   \]
   \[
   P_{t|t} = (I - K_t H) P_{t|t-1}
   \]

### Stochastic Volatility Model

The stochastic volatility (SV) model is:

\[
y_t = \exp(h_t/2) \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1)
\]
\[
h_t = \mu + \phi (h_{t-1} - \mu) + \sigma \eta_t, \quad \eta_t \sim \mathcal{N}(0, 1)
\]

where \(h_t\) is the log-volatility.

### Regime-Switching Models

In regime-switching models, parameters can change according to a Hidden Markov Model (HMM):

\[
s_t \sim \text{Markov}(\Pi)
\]
\[
h_t = \mu_{s_t} + \phi_{s_t} (h_{t-1} - \mu_{s_t}) + \sigma_{s_t} \eta_t
\]

where \(s_t\) is the regime at time \(t\) and \(\Pi\) is the transition matrix.

## Architecture

The project is organized into several modules:

- **`models/`**: Core filtering algorithms (KF, UKF, SKF, Particle Filter, Bayesian SV)
- **`data/`**: Data fetching and synthetic data generation
- **`inference/`**: Parameter estimation methods (EM, MCMC, SVI)
- **`metrics/`**: Evaluation metrics for volatility forecasts

## How to Extend This Project

### Adding a New Filter

1. Create a new class in `src/kbv/models/`
2. Implement `filter()` and optionally `smooth()` methods
3. Add to `__init__.py` exports
4. Write tests in `tests/`

### Adding a New Data Source

1. Add fetching function to `src/kbv/data/fetch.py`
2. Follow the pattern of existing functions
3. Update `__init__.py`

### Adding a New Inference Method

1. Create module in `src/kbv/inference/`
2. Implement parameter estimation logic
3. Add to `__init__.py`

## References

- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series. *Econometrica*, 57(2), 357-384.
- Harvey, A. C. (1990). *Forecasting, structural time series models and the Kalman filter*. Cambridge University Press.
- Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: likelihood inference and comparison with ARCH models. *Review of Economic Studies*, 65(3), 361-393.

