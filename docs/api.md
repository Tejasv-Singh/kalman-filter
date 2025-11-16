# API Reference

## Models

### KalmanFilter

```python
from kbv.models.kalman import KalmanFilter

kf = KalmanFilter(F, H, Q, R, x0=None, P0=None)
```

**Parameters:**
- `F`: State transition matrix
- `H`: Observation matrix
- `Q`: Process noise covariance
- `R`: Observation noise covariance
- `x0`: Initial state mean (optional)
- `P0`: Initial state covariance (optional)

**Methods:**
- `filter(observations)`: Run forward filter
- `smooth(observations)`: Run forward-backward smoother
- `log_likelihood(observations)`: Compute log-likelihood

### UnscentedKalmanFilter

```python
from kbv.models.ukf import UnscentedKalmanFilter

ukf = UnscentedKalmanFilter(f, h, Q, R, n_state, n_obs, ...)
```

**Parameters:**
- `f`: State transition function
- `h`: Observation function
- `Q`: Process noise covariance
- `R`: Observation noise covariance
- `n_state`: State dimension
- `n_obs`: Observation dimension

### SwitchingKalmanFilter

```python
from kbv.models.switching_kf import SwitchingKalmanFilter

skf = SwitchingKalmanFilter(
    n_regimes, transition_matrix, filters=[...]
)
```

**Parameters:**
- `n_regimes`: Number of regimes
- `transition_matrix`: HMM transition matrix
- `filters`: List of KalmanFilter objects (one per regime)

### ParticleFilter

```python
from kbv.models.particle_filter import ParticleFilter

pf = ParticleFilter(
    n_particles, n_state, transition_fn, likelihood_fn
)
```

**Parameters:**
- `n_particles`: Number of particles
- `n_state`: State dimension
- `transition_fn`: Function to sample from p(x_t | x_{t-1})
- `likelihood_fn`: Function to compute p(y_t | x_t)

### BayesianStochasticVolatility

```python
from kbv.models.bayes_numpyro import BayesianStochasticVolatility

model = BayesianStochasticVolatility()
```

**Methods:**
- `model(returns)`: NumPyro model definition
- `predict_volatility(samples, n_steps)`: Forecast future volatility

## Data

### Synthetic Data

```python
from kbv.data.synth import (
    generate_heston_path,
    generate_regime_switching_sv,
    generate_simple_sv,
    generate_garch_data
)
```

### Real Data

```python
from kbv.data.fetch import fetch_stock_data, fetch_and_prepare

data = fetch_stock_data("AAPL", period="1y")
prices, returns = fetch_and_prepare("AAPL", period="1y")
```

## Inference

### EM Algorithm

```python
from kbv.inference.em import EMKalmanFilter

em = EMKalmanFilter(n_state, n_obs)
results = em.fit(observations, max_iter=100)
```

### MCMC

```python
from kbv.inference.mcmc import run_mcmc

results = run_mcmc(model, data, num_samples=1000, num_warmup=500)
```

### SVI

```python
from kbv.inference.svi import run_svi

results = run_svi(model, data, num_iterations=1000)
```

## Metrics

```python
from kbv.metrics import evaluate_volatility_forecast

metrics = evaluate_volatility_forecast(true_vol, pred_vol, returns)
```

Returns dictionary with:
- `rmse`: Root mean squared error
- `mae`: Mean absolute error
- `mape`: Mean absolute percentage error
- `correlation`: Pearson correlation
- `qlike`: Quasi-likelihood loss

