"""Streamlit dashboard for volatility filtering."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from kbv.models.kalman import KalmanFilter
from kbv.models.switching_kf import SwitchingKalmanFilter
from kbv.data.synth import generate_simple_sv, generate_regime_switching_sv
from kbv.data.fetch import fetch_and_prepare
from kbv.metrics import evaluate_volatility_forecast

sns.set_style("whitegrid")
st.set_page_config(page_title="Volatility Filtering Dashboard", layout="wide")


def main():
    st.title("Kalman-Bayesian Volatility Filtering Dashboard")

    # Sidebar
    st.sidebar.header("Configuration")
    data_source = st.sidebar.selectbox(
        "Data Source", ["Synthetic (Simple SV)", "Synthetic (Regime-Switching)", "Real Data (Yahoo Finance)"]
    )

    if data_source == "Real Data (Yahoo Finance)":
        symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
        period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

        try:
            _, returns = fetch_and_prepare(symbol, period=period)
            returns = returns.values
            true_vol = None
            true_regimes = None
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return
    else:
        n_obs = st.sidebar.slider("Number of Observations", 100, 1000, 500)
        seed = st.sidebar.number_input("Random Seed", value=42)

        if data_source == "Synthetic (Simple SV)":
            returns, true_log_vol = generate_simple_sv(n_obs=n_obs, seed=seed)
            true_vol = np.exp(true_log_vol / 2)
            true_regimes = None
        else:  # Regime-switching
            returns, log_vol, true_regimes = generate_regime_switching_sv(
                n_obs=n_obs, seed=seed
            )
            true_vol = np.exp(log_vol / 2)

    # Model selection
    st.sidebar.header("Models")
    use_kf = st.sidebar.checkbox("Kalman Filter", value=True)
    use_skf = st.sidebar.checkbox("Switching Kalman Filter", value=False)
    use_bayes = st.sidebar.checkbox("Bayesian SV (slow)", value=False)

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Returns")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(returns)
        ax.set_title("Returns")
        ax.set_xlabel("Time")
        ax.set_ylabel("Return")
        st.pyplot(fig)

    with col2:
        st.subheader("Squared Returns (Volatility Proxy)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(returns**2)
        ax.set_title("Squared Returns")
        ax.set_xlabel("Time")
        ax.set_ylabel("Squared Return")
        st.pyplot(fig)

    # Run models
    results = {}

    if use_kf:
        with st.spinner("Running Kalman Filter..."):
            observations = returns**2
            kf = KalmanFilter(
                F=np.array([[0.95]]),
                H=np.array([[1.0]]),
                Q=np.array([[0.01]]),
                R=np.array([[0.1]]),
            )
            filtered_states, _ = kf.filter(observations)
            kf_vol = np.exp(filtered_states.flatten() / 2)
            results["Kalman Filter"] = kf_vol

    if use_skf:
        with st.spinner("Running Switching Kalman Filter..."):
            observations = returns**2
            # from kbv.models.kalman import KalmanFilter

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
            skf_vol = np.exp(filtered_states.flatten() / 2)
            results["Switching KF"] = skf_vol
            results["_regimes"] = skf.get_most_likely_regimes(regime_probs)

    if use_bayes:
        with st.spinner("Running Bayesian SV (this may take a while)..."):
            from kbv.models.bayes_numpyro import BayesianStochasticVolatility
            from kbv.inference.mcmc import run_mcmc

            model = BayesianStochasticVolatility()
            mcmc_results = run_mcmc(
                model.model, returns, num_samples=500, num_warmup=250, progress_bar=False
            )
            samples = mcmc_results["samples"]
            bayes_vol = samples["volatility"].mean(axis=0)
            results["Bayesian SV"] = bayes_vol

    # Plot results
    st.subheader("Volatility Estimates")
    fig, ax = plt.subplots(figsize=(12, 6))

    if true_vol is not None:
        ax.plot(true_vol, label="True Volatility", linewidth=2, color="black")

    for name, vol in results.items():
        if not name.startswith("_"):
            ax.plot(vol, label=name, alpha=0.7)

    ax.set_title("Volatility Estimation")
    ax.set_xlabel("Time")
    ax.set_ylabel("Volatility")
    ax.legend()
    st.pyplot(fig)

    # Metrics
    if true_vol is not None and results:
        st.subheader("Evaluation Metrics")
        metrics_data = []

        for name, vol in results.items():
            if not name.startswith("_"):
                metrics = evaluate_volatility_forecast(true_vol, vol, returns)
                metrics["Method"] = name
                metrics_data.append(metrics)

        if metrics_data:
            df = pd.DataFrame(metrics_data)
            df = df.set_index("Method")
            st.dataframe(df)

    # Regime detection
    if "_regimes" in results and true_regimes is not None:
        st.subheader("Regime Detection")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(true_regimes, label="True Regime", marker="o", linestyle="None", markersize=3)
        ax.plot(results["_regimes"], label="Predicted Regime", alpha=0.7)
        ax.set_title("Regime Detection")
        ax.set_ylabel("Regime")
        ax.set_xlabel("Time")
        ax.legend()
        ax.set_ylim(-0.5, 1.5)
        st.pyplot(fig)

        accuracy = (results["_regimes"] == true_regimes).mean()
        st.metric("Regime Detection Accuracy", f"{accuracy:.2%}")


if __name__ == "__main__":
    main()

