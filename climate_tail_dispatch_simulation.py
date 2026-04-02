#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate-resilient battery dispatch simulation
Implements a simulation study for the manuscript:
A Tail-Regime Stochastic Programming Framework for Climate-Resilient Battery Dispatch

Designed for Google Colab or local Python.

What this script does:
1. Simulates heavy-tailed climate-driven residual net load with latent weather regimes.
2. Fits the proposed EVT-regime model and benchmark models on training data.
3. Generates day-ahead scenarios for each method.
4. Solves a scenario-based CVaR battery reserve planning problem.
5. Evaluates realized dispatch on out-of-sample data in a rolling-horizon experiment.
6. Repeats over Monte Carlo replications and stress levels.
7. Saves all plots/tables to disk and zips the outputs.

Honest note:
The proposed EVT-tail method is designed to dominate mainly on tail-risk, blackout,
and reserve-coverage metrics under heavy-tailed/extreme-weather data-generating processes.
It may not always have the lowest mean cost in every replication, because added protection
can increase ordinary-period operating cost.
"""

import os
import sys
import json
import math
import time
import shutil
import zipfile
import warnings
import textwrap
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# =========================
# Optional dependency install helper for Colab
# =========================
def ensure_package(pkg_name: str, import_name: Optional[str] = None):
    import importlib
    name = import_name or pkg_name
    try:
        importlib.import_module(name)
    except ImportError:
        import subprocess
        print(f"Installing missing package: {pkg_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])

# Core scientific stack
for pkg, imp in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("statsmodels", "statsmodels"),
    ("cvxpy", "cvxpy"),
    ("scikit-learn", "sklearn"),
]:
    ensure_package(pkg, imp)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import genpareto
from scipy.linalg import toeplitz
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm
import cvxpy as cp
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.30
plt.rcParams["font.size"] = 11


# ============================================================
# Configuration
# ============================================================
@dataclass
class SimConfig:
    seed: int = 1234
    n_replications: int = 8
    stress_levels: Tuple[str, ...] = ("mild", "severe", "extreme")

    train_days: int = 240
    test_days: int = 36
    horizon_hours: int = 24

    n_scenarios_proposed: int = 120
    n_scenarios_benchmark: int = 120
    reserve_simulations: int = 2000

    quantile_q: float = 0.95
    cvar_alpha: float = 0.95
    risk_lambda: float = 2.5
    reserve_horizon: int = 24
    reserve_eps: float = 0.05

    battery_capacity: float = 220.0
    battery_charge_limit: float = 55.0
    battery_discharge_limit: float = 55.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_min: float = 20.0
    soc_max: float = 220.0
    soc_init: float = 120.0

    thermal_cap: float = 160.0
    thermal_cost: float = 30.0
    degradation_cost: float = 1.5
    curtailment_cost: float = 1.0
    voll: float = 1000.0
    reserve_slack_cost: float = 100.0
    reserve_cost: float = 0.8

    show_plots_inline: bool = True
    save_dir: str = "climate_tail_dispatch_outputs"

    # runtime control
    run_stress_sensitivity: bool = True


REGIME_NAMES = {
    0: "normal",
    1: "freeze",
    2: "heatwave",
    3: "wind_drought",
    4: "compound",
}
EXTREME_REGIMES = [1, 2, 3, 4]


# ============================================================
# Utility helpers
# ============================================================
def make_output_dirs(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    subdirs = [
        "plots",
        "tables",
        "logs",
        "artifacts",
        "daily_results",
    ]
    paths = {}
    for s in subdirs:
        p = os.path.join(base_dir, s)
        os.makedirs(p, exist_ok=True)
        paths[s] = p
    return paths


def print_banner(msg: str):
    print("\n" + "=" * 100)
    print(msg)
    print("=" * 100)


def save_dataframe(df: pd.DataFrame, path: str, index: bool = False, title: Optional[str] = None):
    df.to_csv(path, index=index)
    if title:
        print(f"\n{title}")
        print(df.to_string(index=index))
    else:
        print(df.to_string(index=index))


def safe_cvar(x: np.ndarray, alpha: float = 0.95) -> float:
    x = np.asarray(x)
    if len(x) == 0:
        return np.nan
    q = np.quantile(x, alpha)
    tail = x[x >= q]
    if len(tail) == 0:
        return q
    return float(np.mean(tail))


def paired_summary_stat(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = np.asarray(a) - np.asarray(b)
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    se_d = sd_d / np.sqrt(len(d)) if len(d) > 0 else np.nan
    t_stat = mean_d / se_d if se_d > 0 else np.nan
    return {"mean_diff": mean_d, "sd_diff": sd_d, "se_diff": se_d, "t_like": t_stat}


# ============================================================
# Data-generating process
# ============================================================
def get_stress_params(level: str) -> Dict:
    if level == "mild":
        return {
            "tail_scale_mult": 0.9,
            "tail_shape_shift": -0.05,
            "transition_boost": 0.92,
            "load_shock_mult": 0.8,
            "thermal_derate_mult": 0.7,
        }
    elif level == "severe":
        return {
            "tail_scale_mult": 1.20,
            "tail_shape_shift": 0.05,
            "transition_boost": 1.00,
            "load_shock_mult": 1.0,
            "thermal_derate_mult": 1.0,
        }
    elif level == "extreme":
        return {
            "tail_scale_mult": 1.55,
            "tail_shape_shift": 0.12,
            "transition_boost": 1.07,
            "load_shock_mult": 1.22,
            "thermal_derate_mult": 1.20,
        }
    else:
        raise ValueError(f"Unknown stress level: {level}")


def simulate_regime_path(n_hours: int, rng: np.random.Generator, stress_level: str) -> np.ndarray:
    p = get_stress_params(stress_level)
    # Row-stochastic transition matrix with persistent adverse regimes
    P = np.array([
        [0.86, 0.04, 0.04, 0.05, 0.01],  # normal
        [0.16, 0.70, 0.03, 0.03, 0.08],  # freeze
        [0.18, 0.02, 0.69, 0.04, 0.07],  # heatwave
        [0.14, 0.03, 0.03, 0.73, 0.07],  # wind_drought
        [0.14, 0.07, 0.07, 0.08, 0.64],  # compound
    ], dtype=float)

    # amplify persistence in adverse regimes according to stress level
    boost = p["transition_boost"]
    for r in EXTREME_REGIMES:
        P[r, r] = min(0.92, P[r, r] * boost)
        row_rest = np.sum(P[r]) - P[r, r]
        if row_rest > 0:
            scale = (1.0 - P[r, r]) / row_rest
            for j in range(P.shape[1]):
                if j != r:
                    P[r, j] *= scale

    # normalize
    P = P / P.sum(axis=1, keepdims=True)
    pi0 = np.array([0.84, 0.04, 0.04, 0.06, 0.02])
    states = np.zeros(n_hours, dtype=int)
    states[0] = rng.choice(np.arange(5), p=pi0)
    for t in range(1, n_hours):
        states[t] = rng.choice(np.arange(5), p=P[states[t-1]])
    return states


def simulate_climate_grid_series(config: SimConfig, rng: np.random.Generator, stress_level: str) -> pd.DataFrame:
    n_hours = 24 * (config.train_days + config.test_days)
    regimes = simulate_regime_path(n_hours, rng, stress_level)
    hours = np.arange(n_hours)
    hod = hours % 24
    day = hours // 24
    dow = day % 7
    doy = day % 365

    stress = get_stress_params(stress_level)

    # Seasonal and diurnal bases
    year_phase = 2 * np.pi * doy / 365.0
    day_phase = 2 * np.pi * hod / 24.0

    # Latent meteorology
    temp_base = 22 + 10 * np.sin(year_phase - 0.8) + 4 * np.sin(day_phase - 0.5)
    wind_base = 0.55 + 0.16 * np.sin(year_phase + 0.9) + 0.08 * np.sin(day_phase + 0.2)
    solar_shape = np.maximum(0, np.sin(np.pi * (hod - 6) / 12.0))
    solar_base = solar_shape * (0.65 + 0.18 * np.sin(year_phase - 0.1))

    temp_noise = rng.normal(0, 2.2, size=n_hours)
    wind_noise = np.zeros(n_hours)
    wind_noise[0] = rng.normal(0, 0.06)
    for t in range(1, n_hours):
        wind_noise[t] = 0.82 * wind_noise[t-1] + rng.normal(0, 0.05)

    temp = temp_base + temp_noise
    wind_index = np.clip(wind_base + wind_noise, 0.02, 1.2)
    solar_index = np.clip(solar_base + rng.normal(0, 0.05, size=n_hours), 0, 1.2)

    # Regime impacts
    temp_shift = np.zeros(n_hours)
    wind_mult = np.ones(n_hours)
    solar_mult = np.ones(n_hours)
    load_shock = np.zeros(n_hours)
    thermal_avail = np.ones(n_hours)

    freeze = regimes == 1
    heatwave = regimes == 2
    wind_drought = regimes == 3
    compound = regimes == 4

    temp_shift[freeze] -= 14
    temp_shift[heatwave] += 10
    temp_shift[compound] -= 7 + 5 * rng.random(np.sum(compound))

    wind_mult[freeze] *= 0.78
    wind_mult[wind_drought] *= 0.38
    wind_mult[compound] *= 0.34
    wind_mult[heatwave] *= 0.82

    solar_mult[freeze] *= 0.62
    solar_mult[compound] *= 0.56
    solar_mult[heatwave] *= 0.88

    load_shock[freeze] += 18 * stress["load_shock_mult"]
    load_shock[heatwave] += 14 * stress["load_shock_mult"]
    load_shock[wind_drought] += 4 * stress["load_shock_mult"]
    load_shock[compound] += 26 * stress["load_shock_mult"]

    thermal_avail[freeze] -= 0.10 * stress["thermal_derate_mult"]
    thermal_avail[compound] -= 0.15 * stress["thermal_derate_mult"]
    thermal_avail = np.clip(thermal_avail, 0.6, 1.0)

    temp = temp + temp_shift
    wind_avail = 68 * np.clip(wind_index * wind_mult, 0, None)
    solar_avail = 72 * np.clip(solar_index * solar_mult, 0, None)

    # Load model: baseline + weather sensitivity + daily/weekly effects + AR piece
    load = (
        105
        + 12 * np.sin(day_phase - 1.0)
        + 6 * np.cos(2 * day_phase)
        + 7 * np.isin(dow, [0, 1, 2, 3, 4]).astype(float)
        + 0.58 * np.maximum(18 - temp, 0)
        + 0.50 * np.maximum(temp - 26, 0)
        + load_shock
        + rng.normal(0, 4.0, size=n_hours)
    )

    # Heavy-tailed weather stress shock on residual net load
    tail_shape_by_regime = {
        0: 0.05,
        1: 0.22,
        2: 0.18,
        3: 0.28,
        4: 0.36,
    }
    tail_scale_by_regime = {
        0: 6.0,
        1: 13.0,
        2: 11.0,
        3: 14.0,
        4: 19.0,
    }
    exceed_prob_by_regime = {
        0: 0.05,
        1: 0.16,
        2: 0.12,
        3: 0.20,
        4: 0.25,
    }

    residual_no_tail = load - wind_avail - solar_avail
    tail_shock = np.zeros(n_hours)
    for t in range(n_hours):
        r = regimes[t]
        p_exc = exceed_prob_by_regime[r]
        if rng.uniform() < p_exc:
            xi = tail_shape_by_regime[r] + stress["tail_shape_shift"]
            scale = tail_scale_by_regime[r] * stress["tail_scale_mult"]
            xi = max(-0.2, xi)
            y = genpareto.rvs(c=xi, loc=0, scale=scale, random_state=rng)
            # persistence kicker for clustered extremes
            if t > 0 and regimes[t-1] == r and r in EXTREME_REGIMES:
                y *= (1.0 + 0.15 * rng.random())
            tail_shock[t] = y

    residual_net_load = residual_no_tail + tail_shock

    # allow moderate export / renewable surplus in benign periods
    surplus_noise = rng.normal(0, 2.5, size=n_hours)
    residual_net_load = residual_net_load + surplus_noise

    # realized effective renewable total implied by residual net load
    renewable_total = load - residual_net_load
    renewable_total = np.clip(renewable_total, 0, None)

    df = pd.DataFrame({
        "t": np.arange(n_hours),
        "day": day,
        "hour": hod,
        "dow": dow,
        "doy": doy,
        "regime": regimes,
        "regime_name": [REGIME_NAMES[r] for r in regimes],
        "temp": temp,
        "wind_index": wind_index,
        "solar_index": solar_index,
        "load": load,
        "wind_avail": wind_avail,
        "solar_avail": solar_avail,
        "renewable_total": renewable_total,
        "thermal_avail_factor": thermal_avail,
        "tail_shock": tail_shock,
        "residual_net_load": residual_net_load,
    })
    return df


# ============================================================
# Feature engineering and estimation
# ============================================================
def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    X["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    X["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    X["sin_year"] = np.sin(2 * np.pi * df["doy"] / 365.0)
    X["cos_year"] = np.cos(2 * np.pi * df["doy"] / 365.0)
    X["temp"] = df["temp"].values
    X["temp_cold"] = np.maximum(18 - df["temp"].values, 0)
    X["temp_hot"] = np.maximum(df["temp"].values - 26, 0)
    X["wind_index"] = df["wind_index"].values
    X["solar_index"] = df["solar_index"].values
    X["is_weekend"] = df["dow"].isin([5, 6]).astype(float).values
    return X


@dataclass
class FittedModels:
    mean_model: object
    q_model: object
    X_columns: List[str]
    threshold_q: float
    pooled_non_exc_resid: np.ndarray
    regime_non_exc_resid: Dict[int, np.ndarray]
    pooled_exc_resid: np.ndarray
    regime_exc_prob: Dict[int, float]
    regime_gpd: Dict[int, Dict[str, float]]
    transition_matrix: np.ndarray
    reserve_stats: Dict[int, Dict[str, float]]
    train_df: pd.DataFrame



def fit_models(train_df: pd.DataFrame, config: SimConfig) -> FittedModels:
    X = build_design_matrix(train_df)
    y = train_df["residual_net_load"].values

    mean_model = sm.OLS(y, X).fit()
    q_model = QuantReg(y, X).fit(q=config.quantile_q, max_iter=5000)

    mu_hat = mean_model.predict(X)
    u_hat = q_model.predict(X)
    resid = y - mu_hat
    exceed = y > u_hat
    excess = np.maximum(y - u_hat, 0)

    pooled_non_exc = resid[~exceed]
    if len(pooled_non_exc) == 0:
        pooled_non_exc = resid.copy()

    regime_non_exc = {}
    regime_exc_prob = {}
    regime_gpd = {}
    reserve_stats = {}

    for r in sorted(train_df["regime"].unique()):
        idx_r = train_df["regime"].values == r
        idx_non_exc = idx_r & (~exceed)
        idx_exc = idx_r & exceed
        regime_non_exc[r] = resid[idx_non_exc] if np.sum(idx_non_exc) >= 30 else pooled_non_exc
        regime_exc_prob[r] = float(np.mean(exceed[idx_r])) if np.sum(idx_r) > 0 else float(np.mean(exceed))

        exc_vals = excess[idx_exc]
        if len(exc_vals) >= 25 and np.std(exc_vals) > 1e-8:
            try:
                c_hat, loc_hat, scale_hat = genpareto.fit(exc_vals, floc=0)
                regime_gpd[r] = {"shape": float(c_hat), "scale": float(scale_hat)}
            except Exception:
                regime_gpd[r] = {"shape": 0.15, "scale": float(np.std(exc_vals) + 1.0)}
        else:
            fallback = excess[exceed]
            if len(fallback) >= 25 and np.std(fallback) > 1e-8:
                c_hat, loc_hat, scale_hat = genpareto.fit(fallback, floc=0)
                regime_gpd[r] = {"shape": float(c_hat), "scale": float(scale_hat)}
            else:
                regime_gpd[r] = {"shape": 0.10, "scale": 8.0}

        # Regime-specific cluster statistics for reserve simulations
        exc_series = pd.Series(excess[idx_r])
        pos = exc_series.values > 0
        cluster_sums = []
        cluster_lens = []
        running_sum, running_len = 0.0, 0
        for flag, val in zip(pos, exc_series.values):
            if flag:
                running_sum += float(val)
                running_len += 1
            else:
                if running_len > 0:
                    cluster_sums.append(running_sum)
                    cluster_lens.append(running_len)
                    running_sum, running_len = 0.0, 0
        if running_len > 0:
            cluster_sums.append(running_sum)
            cluster_lens.append(running_len)

        if len(cluster_sums) == 0:
            cluster_sums = [0.0]
            cluster_lens = [1]

        reserve_stats[r] = {
            "cluster_sum_mean": float(np.mean(cluster_sums)),
            "cluster_sum_std": float(np.std(cluster_sums, ddof=0)),
            "cluster_len_mean": float(np.mean(cluster_lens)),
            "cluster_len_std": float(np.std(cluster_lens, ddof=0)),
        }

    # pooled exceedances for fallback / diagnostics
    pooled_exc = excess[exceed]
    if len(pooled_exc) == 0:
        pooled_exc = np.array([0.0])

    # Empirical transition matrix
    n_reg = len(REGIME_NAMES)
    Tmat = np.ones((n_reg, n_reg))  # Laplace smoothing
    reg_arr = train_df["regime"].values
    for i in range(1, len(reg_arr)):
        Tmat[reg_arr[i-1], reg_arr[i]] += 1
    Tmat = Tmat / Tmat.sum(axis=1, keepdims=True)

    return FittedModels(
        mean_model=mean_model,
        q_model=q_model,
        X_columns=list(X.columns),
        threshold_q=config.quantile_q,
        pooled_non_exc_resid=np.asarray(pooled_non_exc),
        regime_non_exc_resid=regime_non_exc,
        pooled_exc_resid=np.asarray(pooled_exc),
        regime_exc_prob=regime_exc_prob,
        regime_gpd=regime_gpd,
        transition_matrix=Tmat,
        reserve_stats=reserve_stats,
        train_df=train_df.copy(),
    )


# ============================================================
# Scenario generation
# ============================================================
def simulate_markov_regimes(start_regime: int, n_steps: int, Tmat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    regs = np.zeros(n_steps, dtype=int)
    regs[0] = start_regime
    for k in range(1, n_steps):
        regs[k] = rng.choice(np.arange(Tmat.shape[0]), p=Tmat[regs[k-1]])
    return regs


def proposed_scenarios(
    models: FittedModels,
    day_df: pd.DataFrame,
    start_regime: int,
    n_scenarios: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = len(day_df)
    X_day = build_design_matrix(day_df)[models.X_columns]
    mu_hat = np.asarray(models.mean_model.predict(X_day))
    u_hat = np.asarray(models.q_model.predict(X_day))

    netload_scen = np.zeros((n_scenarios, H))
    thermal_factor_scen = np.zeros((n_scenarios, H))
    reserve_floor_samples = np.zeros((n_scenarios, H))

    for s in range(n_scenarios):
        regs = simulate_markov_regimes(start_regime=start_regime, n_steps=H, Tmat=models.transition_matrix, rng=rng)
        thermal_factor_scen[s, :] = np.array([
            0.90 if r == 1 else (0.85 if r == 4 else 1.0) for r in regs
        ])
        series = np.zeros(H)
        exc_series = np.zeros(H)
        for h in range(H):
            r = int(regs[h])
            p_exc = models.regime_exc_prob.get(r, np.mean(list(models.regime_exc_prob.values())))
            if rng.uniform() < p_exc:
                pars = models.regime_gpd.get(r, {"shape": 0.10, "scale": 8.0})
                excess = genpareto.rvs(c=pars["shape"], loc=0, scale=max(pars["scale"], 1e-3), random_state=rng)
                y = u_hat[h] + excess
                exc_series[h] = excess
            else:
                pool = models.regime_non_exc_resid.get(r, models.pooled_non_exc_resid)
                eps = rng.choice(pool)
                y = min(mu_hat[h] + eps, u_hat[h] - 1e-6)
            series[h] = y
        netload_scen[s, :] = series
        reserve_floor_samples[s, :] = np.cumsum(exc_series)
    return netload_scen, thermal_factor_scen, reserve_floor_samples



def gaussian_scenarios(
    models: FittedModels,
    day_df: pd.DataFrame,
    n_scenarios: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = len(day_df)
    X_day = build_design_matrix(day_df)[models.X_columns]
    mu_hat = np.asarray(models.mean_model.predict(X_day))
    train_pred = np.asarray(models.mean_model.predict(build_design_matrix(models.train_df)[models.X_columns]))
    resid = models.train_df["residual_net_load"].values - train_pred
    sigma = float(np.std(resid, ddof=1))
    sigma = max(sigma, 1.0)

    # weak AR(1)-style correlation structure for smoothness
    rho = 0.55
    cov = sigma**2 * toeplitz(rho**np.arange(H))
    netload_scen = rng.multivariate_normal(mu_hat, cov, size=n_scenarios)
    thermal_factor_scen = np.repeat(day_df["thermal_avail_factor"].values.reshape(1, -1), n_scenarios, axis=0)

    q_hat = np.asarray(models.q_model.predict(X_day))
    exc = np.maximum(netload_scen - q_hat.reshape(1, -1), 0.0)
    reserve_floor_samples = np.cumsum(exc, axis=1)
    return netload_scen, thermal_factor_scen, reserve_floor_samples



def empirical_bootstrap_scenarios(
    models: FittedModels,
    day_df: pd.DataFrame,
    n_scenarios: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = len(day_df)
    X_day = build_design_matrix(day_df)[models.X_columns]
    mu_hat = np.asarray(models.mean_model.predict(X_day))
    train_X = build_design_matrix(models.train_df)[models.X_columns]
    resid = models.train_df["residual_net_load"].values - np.asarray(models.mean_model.predict(train_X))

    # moving block bootstrap from residual series to preserve dependence
    block_len = 6
    netload_scen = np.zeros((n_scenarios, H))
    q_hat = np.asarray(models.q_model.predict(X_day))
    thermal_factor_scen = np.repeat(day_df["thermal_avail_factor"].values.reshape(1, -1), n_scenarios, axis=0)

    for s in range(n_scenarios):
        sampled_resid = []
        while len(sampled_resid) < H:
            start = rng.integers(0, max(1, len(resid) - block_len))
            sampled_resid.extend(resid[start:start+block_len].tolist())
        sampled_resid = np.array(sampled_resid[:H])
        path = mu_hat + sampled_resid
        netload_scen[s, :] = path

    exc = np.maximum(netload_scen - q_hat.reshape(1, -1), 0.0)
    reserve_floor_samples = np.cumsum(exc, axis=1)
    return netload_scen, thermal_factor_scen, reserve_floor_samples



def deterministic_mean_scenario(
    models: FittedModels,
    day_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H = len(day_df)
    X_day = build_design_matrix(day_df)[models.X_columns]
    mu_hat = np.asarray(models.mean_model.predict(X_day))
    netload_scen = mu_hat.reshape(1, -1)
    thermal_factor_scen = day_df["thermal_avail_factor"].values.reshape(1, -1)
    q_hat = np.asarray(models.q_model.predict(X_day))
    exc = np.maximum(netload_scen - q_hat.reshape(1, -1), 0.0)
    reserve_floor_samples = np.cumsum(exc, axis=1)
    return netload_scen, thermal_factor_scen, reserve_floor_samples


# ============================================================
# Optimization layer
# ============================================================
def solve_reserve_planning_problem(
    netload_scen: np.ndarray,
    thermal_factor_scen: np.ndarray,
    reserve_floor_samples: np.ndarray,
    config: SimConfig,
    soc_init: float,
    method_name: str,
) -> Dict:
    """
    Two-stage style scenario LP.
    First-stage variable: reserve floor r_t shared across scenarios.
    Recourse variables: per-scenario battery and dispatch variables.
    """
    n_scen, H = netload_scen.shape
    probs = np.ones(n_scen) / n_scen

    # Scenario-implied tail reserve lower bound
    B = np.quantile(reserve_floor_samples, 1.0 - config.reserve_eps, axis=0)
    B = np.clip(B, 0, config.battery_capacity)

    # Decision variables
    r = cp.Variable(H)
    eta = cp.Variable()
    z = cp.Variable(n_scen, nonneg=True)

    g = cp.Variable((n_scen, H), nonneg=True)
    c = cp.Variable((n_scen, H), nonneg=True)
    d = cp.Variable((n_scen, H), nonneg=True)
    shed = cp.Variable((n_scen, H), nonneg=True)
    curtail = cp.Variable((n_scen, H), nonneg=True)
    soc = cp.Variable((n_scen, H))
    zeta = cp.Variable((n_scen, H), nonneg=True)

    constraints = []
    constraints += [r >= B, r >= 0, r <= config.battery_capacity]

    for s in range(n_scen):
        for h in range(H):
            prev_soc = soc_init if h == 0 else soc[s, h-1]
            constraints += [
                soc[s, h] == prev_soc + config.eta_charge * c[s, h] - d[s, h] / config.eta_discharge,
                soc[s, h] >= config.soc_min,
                soc[s, h] <= config.soc_max,
                c[s, h] <= config.battery_charge_limit,
                d[s, h] <= config.battery_discharge_limit,
                g[s, h] <= config.thermal_cap * thermal_factor_scen[s, h],
                # residual net load balance: positive net load must be supplied;
                # negative net load can be absorbed by charging/curtailment
                g[s, h] + d[s, h] + shed[s, h] - c[s, h] - curtail[s, h] == netload_scen[s, h],
                soc[s, h] >= r[h] - zeta[s, h],
            ]

    scenario_costs = []
    for s in range(n_scen):
        cost_s = (
            config.thermal_cost * cp.sum(g[s, :])
            + config.degradation_cost * cp.sum(c[s, :] + d[s, :])
            + config.curtailment_cost * cp.sum(curtail[s, :])
            + config.voll * cp.sum(shed[s, :])
            + config.reserve_slack_cost * cp.sum(zeta[s, :])
        )
        scenario_costs.append(cost_s)
        constraints += [z[s] >= cost_s - eta]

    expected_cost = cp.sum(cp.multiply(probs, cp.hstack(scenario_costs)))
    cvar_term = eta + (1.0 / (1.0 - config.cvar_alpha)) * cp.sum(cp.multiply(probs, z))
    reserve_term = config.reserve_cost * cp.sum(r)

    objective = cp.Minimize(reserve_term + expected_cost + config.risk_lambda * cvar_term)
    problem = cp.Problem(objective, constraints)

    # Try a few solvers in sequence
    solved = False
    last_err = None
    for solver in [cp.ECOS, cp.CLARABEL, cp.SCS, cp.OSQP]:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in ["optimal", "optimal_inaccurate"]:
                solved = True
                break
        except Exception as e:
            last_err = e
            continue

    if not solved:
        raise RuntimeError(f"Planning LP failed for {method_name}. Last solver error: {last_err}")

    return {
        "reserve_floor": np.asarray(r.value).reshape(-1),
        "tail_buffer": B,
        "planning_objective": float(problem.value),
        "expected_cost_in_sample": float(expected_cost.value),
        "cvar_in_sample": float(cvar_term.value),
        "status": problem.status,
    }



def solve_realized_dispatch(
    realized_netload: np.ndarray,
    realized_thermal_factor: np.ndarray,
    reserve_floor: np.ndarray,
    config: SimConfig,
    soc_init: float,
    method_name: str,
) -> Dict:
    H = len(realized_netload)

    g = cp.Variable(H, nonneg=True)
    c = cp.Variable(H, nonneg=True)
    d = cp.Variable(H, nonneg=True)
    shed = cp.Variable(H, nonneg=True)
    curtail = cp.Variable(H, nonneg=True)
    soc = cp.Variable(H)
    zeta = cp.Variable(H, nonneg=True)

    constraints = []
    for h in range(H):
        prev_soc = soc_init if h == 0 else soc[h-1]
        constraints += [
            soc[h] == prev_soc + config.eta_charge * c[h] - d[h] / config.eta_discharge,
            soc[h] >= config.soc_min,
            soc[h] <= config.soc_max,
            c[h] <= config.battery_charge_limit,
            d[h] <= config.battery_discharge_limit,
            g[h] <= config.thermal_cap * realized_thermal_factor[h],
            g[h] + d[h] + shed[h] - c[h] - curtail[h] == realized_netload[h],
            soc[h] >= reserve_floor[h] - zeta[h],
        ]

    obj = cp.Minimize(
        config.thermal_cost * cp.sum(g)
        + config.degradation_cost * cp.sum(c + d)
        + config.curtailment_cost * cp.sum(curtail)
        + config.voll * cp.sum(shed)
        + config.reserve_slack_cost * cp.sum(zeta)
    )
    prob = cp.Problem(obj, constraints)

    solved = False
    last_err = None
    for solver in [cp.ECOS, cp.CLARABEL, cp.SCS, cp.OSQP]:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                solved = True
                break
        except Exception as e:
            last_err = e
            continue

    if not solved:
        raise RuntimeError(f"Realized dispatch failed for {method_name}. Last solver error: {last_err}")

    result = {
        "realized_cost": float(prob.value),
        "soc_end": float(soc.value[-1]),
        "thermal_energy": float(np.sum(g.value)),
        "battery_charge": float(np.sum(c.value)),
        "battery_discharge": float(np.sum(d.value)),
        "battery_throughput": float(np.sum(c.value + d.value)),
        "load_shed": float(np.sum(shed.value)),
        "curtailment": float(np.sum(curtail.value)),
        "reserve_violation": float(np.sum(zeta.value)),
        "blackout": int(np.sum(shed.value) > 1e-6),
        "blackout_hours": int(np.sum(np.array(shed.value) > 1e-6)),
        "soc_path": np.array(soc.value).reshape(-1),
        "g_path": np.array(g.value).reshape(-1),
        "c_path": np.array(c.value).reshape(-1),
        "d_path": np.array(d.value).reshape(-1),
        "shed_path": np.array(shed.value).reshape(-1),
        "curtail_path": np.array(curtail.value).reshape(-1),
        "zeta_path": np.array(zeta.value).reshape(-1),
    }
    return result


# ============================================================
# Evaluation metrics
# ============================================================
def reserve_coverage_metric(day_df: pd.DataFrame, reserve_floor: np.ndarray, models: FittedModels) -> float:
    X = build_design_matrix(day_df)[models.X_columns]
    u_hat = np.asarray(models.q_model.predict(X))
    exc = np.maximum(day_df["residual_net_load"].values - u_hat, 0.0)
    cum_exc = np.cumsum(exc)
    return float(np.mean(cum_exc <= reserve_floor + 1e-8))


# ============================================================
# Plotting
# ============================================================
def save_and_show(fig, path: str, config: SimConfig):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if config.show_plots_inline:
        plt.show()
    plt.close(fig)


def plot_regime_segment(df: pd.DataFrame, out_path: str, config: SimConfig, title_suffix: str = ""):
    seg = df.iloc[:24 * 12].copy()
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(seg["t"].values, seg["residual_net_load"].values, label="Residual net load", linewidth=2)
    ax1.plot(seg["t"].values, seg["load"].values, label="Load", linewidth=1.3, alpha=0.8)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("MW equivalent")
    ax1.set_title(f"Simulated climate-grid series: first 12 days {title_suffix}")

    ax2 = ax1.twinx()
    ax2.step(seg["t"].values, seg["regime"].values, where="post", label="Regime", alpha=0.5)
    ax2.set_ylabel("Regime code")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    save_and_show(fig, out_path, config)


def plot_tail_fit(train_df: pd.DataFrame, models: FittedModels, out_path: str, config: SimConfig):
    X = build_design_matrix(train_df)[models.X_columns]
    y = train_df["residual_net_load"].values
    u_hat = np.asarray(models.q_model.predict(X))
    excess = np.maximum(y - u_hat, 0)
    excess = excess[excess > 0]
    excess = np.sort(excess)
    if len(excess) < 10:
        excess = np.linspace(0.1, 5, 20)
    pars = genpareto.fit(excess, floc=0)
    c_hat, _, scale_hat = pars

    probs = (np.arange(1, len(excess)+1) - 0.5) / len(excess)
    fitted_q = genpareto.ppf(probs, c=c_hat, loc=0, scale=scale_hat)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(fitted_q, excess, alpha=0.7)
    lim = max(np.max(fitted_q), np.max(excess))
    ax.plot([0, lim], [0, lim], linestyle="--")
    ax.set_xlabel("Fitted GPD quantiles")
    ax.set_ylabel("Empirical excess quantiles")
    ax.set_title("Tail QQ diagnostic for exceedances")
    save_and_show(fig, out_path, config)


def plot_scenario_fan(scenarios: np.ndarray, realized: np.ndarray, out_path: str, config: SimConfig, title: str):
    H = scenarios.shape[1]
    x = np.arange(H)
    q10 = np.quantile(scenarios, 0.10, axis=0)
    q50 = np.quantile(scenarios, 0.50, axis=0)
    q90 = np.quantile(scenarios, 0.90, axis=0)
    q97 = np.quantile(scenarios, 0.975, axis=0)
    q03 = np.quantile(scenarios, 0.025, axis=0)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(x, q03, q97, alpha=0.25, label="95% scenario band")
    ax.fill_between(x, q10, q90, alpha=0.30, label="80% scenario band")
    ax.plot(x, q50, linewidth=2, label="Median scenario")
    ax.plot(x, realized, linewidth=2, label="Realized path")
    ax.set_xlabel("Hour ahead")
    ax.set_ylabel("Residual net load")
    ax.set_title(title)
    ax.legend(loc="upper left")
    save_and_show(fig, out_path, config)


def plot_method_boxplot(results_df: pd.DataFrame, value_col: str, out_path: str, config: SimConfig, title: str):
    methods = list(results_df["method"].unique())
    data = [results_df.loc[results_df["method"] == m, value_col].values for m in methods]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=methods, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(value_col)
    plt.xticks(rotation=20)
    save_and_show(fig, out_path, config)


def plot_metric_bar(summary_df: pd.DataFrame, metric: str, out_path: str, config: SimConfig, title: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary_df["method"], summary_df[metric])
    ax.set_title(title)
    ax.set_ylabel(metric)
    plt.xticks(rotation=20)
    save_and_show(fig, out_path, config)


def plot_stress_heatmap(stress_summary: pd.DataFrame, metric: str, out_path: str, config: SimConfig):
    pivot = stress_summary.pivot(index="stress_level", columns="method", values=metric)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Stress sensitivity heatmap: {metric}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    save_and_show(fig, out_path, config)


# ============================================================
# Monte Carlo experiment core
# ============================================================
def evaluate_single_replication(
    rep_id: int,
    stress_level: str,
    config: SimConfig,
    rng: np.random.Generator,
    out_dirs: Dict[str, str],
    make_example_plots: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = simulate_climate_grid_series(config=config, rng=rng, stress_level=stress_level)
    split_hour = config.train_days * 24
    train_df = df.iloc[:split_hour].copy().reset_index(drop=True)
    test_df = df.iloc[split_hour:].copy().reset_index(drop=True)

    models = fit_models(train_df, config)

    if make_example_plots:
        plot_regime_segment(
            train_df,
            os.path.join(out_dirs["plots"], f"rep{rep_id}_{stress_level}_regime_segment.png"),
            config,
            title_suffix=f"({stress_level})",
        )
        plot_tail_fit(
            train_df,
            models,
            os.path.join(out_dirs["plots"], f"rep{rep_id}_{stress_level}_tail_qq.png"),
            config,
        )

    methods = ["Proposed_EVT_Regime", "Gaussian_CVaR", "Empirical_Bootstrap", "Deterministic_Mean"]
    daily_records = []
    example_payload = {}

    # Each rolling step is one day
    soc_states = {m: config.soc_init for m in methods}

    for d in range(config.test_days):
        start = d * config.horizon_hours
        end = start + config.horizon_hours
        day_df = test_df.iloc[start:end].copy().reset_index(drop=True)
        start_regime = int(test_df.iloc[max(start - 1, 0)]["regime"])
        realized_netload = day_df["residual_net_load"].values
        realized_thermal = day_df["thermal_avail_factor"].values

        scenario_dict = {}
        scenario_dict["Proposed_EVT_Regime"] = proposed_scenarios(
            models=models,
            day_df=day_df,
            start_regime=start_regime,
            n_scenarios=config.n_scenarios_proposed,
            rng=rng,
        )
        scenario_dict["Gaussian_CVaR"] = gaussian_scenarios(
            models=models,
            day_df=day_df,
            n_scenarios=config.n_scenarios_benchmark,
            rng=rng,
        )
        scenario_dict["Empirical_Bootstrap"] = empirical_bootstrap_scenarios(
            models=models,
            day_df=day_df,
            n_scenarios=config.n_scenarios_benchmark,
            rng=rng,
        )
        scenario_dict["Deterministic_Mean"] = deterministic_mean_scenario(
            models=models,
            day_df=day_df,
        )

        for method in methods:
            nl_scen, therm_scen, reserve_samples = scenario_dict[method]
            # deterministic benchmark gets zero reserve penalty from tail floor if desired,
            # but we still solve same planning problem so comparison stays coherent.
            plan = solve_reserve_planning_problem(
                netload_scen=nl_scen,
                thermal_factor_scen=therm_scen,
                reserve_floor_samples=reserve_samples,
                config=config,
                soc_init=soc_states[method],
                method_name=method,
            )

            realized = solve_realized_dispatch(
                realized_netload=realized_netload,
                realized_thermal_factor=realized_thermal,
                reserve_floor=plan["reserve_floor"],
                config=config,
                soc_init=soc_states[method],
                method_name=method,
            )
            soc_states[method] = realized["soc_end"]
            coverage = reserve_coverage_metric(day_df, plan["reserve_floor"], models)

            record = {
                "replication": rep_id,
                "stress_level": stress_level,
                "day_index": d,
                "method": method,
                "planning_objective": plan["planning_objective"],
                "expected_cost_in_sample": plan["expected_cost_in_sample"],
                "cvar_in_sample": plan["cvar_in_sample"],
                "tail_buffer_mean": float(np.mean(plan["tail_buffer"])),
                "reserve_floor_mean": float(np.mean(plan["reserve_floor"])),
                "reserve_floor_max": float(np.max(plan["reserve_floor"])),
                "realized_cost": realized["realized_cost"],
                "thermal_energy": realized["thermal_energy"],
                "battery_charge": realized["battery_charge"],
                "battery_discharge": realized["battery_discharge"],
                "battery_throughput": realized["battery_throughput"],
                "load_shed": realized["load_shed"],
                "curtailment": realized["curtailment"],
                "reserve_violation": realized["reserve_violation"],
                "blackout": realized["blackout"],
                "blackout_hours": realized["blackout_hours"],
                "soc_end": realized["soc_end"],
                "reserve_coverage": coverage,
            }
            daily_records.append(record)

            if make_example_plots and d == 0:
                example_payload[method] = {
                    "scenarios": nl_scen.copy(),
                    "realized": realized_netload.copy(),
                    "reserve_floor": plan["reserve_floor"].copy(),
                    "soc_path": realized["soc_path"].copy(),
                    "shed_path": realized["shed_path"].copy(),
                }

    daily_df = pd.DataFrame(daily_records)
    return daily_df, example_payload


# ============================================================
# Aggregation and reporting
# ============================================================
def aggregate_daily_results(all_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        all_daily
        .groupby(["stress_level", "method"], as_index=False)
        .agg(
            mean_realized_cost=("realized_cost", "mean"),
            sd_realized_cost=("realized_cost", "std"),
            cvar95_realized_cost=("realized_cost", lambda x: safe_cvar(np.array(x), 0.95)),
            mean_load_shed=("load_shed", "mean"),
            blackout_rate=("blackout", "mean"),
            mean_blackout_hours=("blackout_hours", "mean"),
            mean_reserve_violation=("reserve_violation", "mean"),
            mean_reserve_coverage=("reserve_coverage", "mean"),
            mean_throughput=("battery_throughput", "mean"),
            mean_tail_buffer=("tail_buffer_mean", "mean"),
            mean_reserve_floor=("reserve_floor_mean", "mean"),
            mean_thermal_energy=("thermal_energy", "mean"),
        )
        .sort_values(["stress_level", "mean_realized_cost"]) 
        .reset_index(drop=True)
    )

    by_rep = (
        all_daily
        .groupby(["replication", "stress_level", "method"], as_index=False)
        .agg(
            mean_realized_cost=("realized_cost", "mean"),
            cvar95_realized_cost=("realized_cost", lambda x: safe_cvar(np.array(x), 0.95)),
            mean_load_shed=("load_shed", "mean"),
            blackout_rate=("blackout", "mean"),
            mean_reserve_violation=("reserve_violation", "mean"),
            mean_reserve_coverage=("reserve_coverage", "mean"),
        )
    )

    paired_rows = []
    for stress in by_rep["stress_level"].unique():
        sub = by_rep[by_rep["stress_level"] == stress].copy()
        prop = sub[sub["method"] == "Proposed_EVT_Regime"].sort_values("replication")
        for benchmark in ["Gaussian_CVaR", "Empirical_Bootstrap", "Deterministic_Mean"]:
            b = sub[sub["method"] == benchmark].sort_values("replication")
            joined = prop.merge(b, on=["replication", "stress_level"], suffixes=("_prop", "_bench"))
            for metric in [
                "mean_realized_cost",
                "cvar95_realized_cost",
                "mean_load_shed",
                "blackout_rate",
                "mean_reserve_violation",
            ]:
                stat = paired_summary_stat(joined[f"{metric}_bench"].values, joined[f"{metric}_prop"].values)
                paired_rows.append({
                    "stress_level": stress,
                    "benchmark": benchmark,
                    "metric": metric,
                    **stat,
                    "positive_mean_diff_favors_proposed": True,
                })
    paired_df = pd.DataFrame(paired_rows)
    return summary, by_rep, paired_df


# ============================================================
# Main experiment driver
# ============================================================
def run_experiment(config: SimConfig):
    base_dir = config.save_dir
    out_dirs = make_output_dirs(base_dir)

    with open(os.path.join(out_dirs["logs"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    print_banner("STARTING CLIMATE-TAIL BATTERY DISPATCH SIMULATION")
    print(json.dumps(asdict(config), indent=2))

    all_daily_frames = []
    first_example_payload = None

    rep_counter = 0
    if config.run_stress_sensitivity:
        stress_levels = config.stress_levels
    else:
        stress_levels = (config.stress_levels[1],) if len(config.stress_levels) > 1 else (config.stress_levels[0],)

    for stress_level in stress_levels:
        print_banner(f"Stress level: {stress_level}")
        for rep in range(config.n_replications):
            rep_counter += 1
            rep_seed = config.seed + 1000 * rep_counter + 31
            rng = np.random.default_rng(rep_seed)
            print(f"Running replication {rep + 1}/{config.n_replications} for stress={stress_level}, seed={rep_seed}")
            daily_df, example_payload = evaluate_single_replication(
                rep_id=rep,
                stress_level=stress_level,
                config=config,
                rng=rng,
                out_dirs=out_dirs,
                make_example_plots=(rep == 0),
            )
            all_daily_frames.append(daily_df)
            daily_df.to_csv(os.path.join(out_dirs["daily_results"], f"daily_rep{rep}_{stress_level}.csv"), index=False)
            if first_example_payload is None and example_payload:
                first_example_payload = (stress_level, example_payload)

    all_daily = pd.concat(all_daily_frames, ignore_index=True)
    all_daily.to_csv(os.path.join(out_dirs["tables"], "all_daily_results.csv"), index=False)

    summary_df, by_rep_df, paired_df = aggregate_daily_results(all_daily)
    save_dataframe(
        summary_df,
        os.path.join(out_dirs["tables"], "summary_by_stress_and_method.csv"),
        index=False,
        title="Summary by stress level and method",
    )
    save_dataframe(
        by_rep_df.head(30),
        os.path.join(out_dirs["tables"], "summary_by_replication.csv"),
        index=False,
        title="Replication-level summary (first 30 rows shown)",
    )
    save_dataframe(
        paired_df,
        os.path.join(out_dirs["tables"], "paired_comparison_proposed_vs_benchmarks.csv"),
        index=False,
        title="Paired comparison table: positive mean difference favors Proposed_EVT_Regime",
    )

    # Overall summary across all stress levels
    overall = (
        all_daily.groupby("method", as_index=False)
        .agg(
            mean_realized_cost=("realized_cost", "mean"),
            cvar95_realized_cost=("realized_cost", lambda x: safe_cvar(np.array(x), 0.95)),
            mean_load_shed=("load_shed", "mean"),
            blackout_rate=("blackout", "mean"),
            mean_reserve_violation=("reserve_violation", "mean"),
            mean_reserve_coverage=("reserve_coverage", "mean"),
            mean_throughput=("battery_throughput", "mean"),
        )
        .sort_values("mean_realized_cost")
        .reset_index(drop=True)
    )
    save_dataframe(
        overall,
        os.path.join(out_dirs["tables"], "overall_summary.csv"),
        index=False,
        title="Overall summary across all stress levels",
    )

    # Plots
    one_stress = stress_levels[0]
    plot_data = all_daily[all_daily["stress_level"] == one_stress].copy()
    plot_method_boxplot(
        plot_data,
        "realized_cost",
        os.path.join(out_dirs["plots"], f"boxplot_realized_cost_{one_stress}.png"),
        config,
        title=f"Daily realized cost by method ({one_stress})",
    )
    plot_method_boxplot(
        plot_data,
        "load_shed",
        os.path.join(out_dirs["plots"], f"boxplot_load_shed_{one_stress}.png"),
        config,
        title=f"Daily load shed by method ({one_stress})",
    )

    stress_summary = summary_df.copy()
    for metric in ["blackout_rate", "mean_load_shed", "cvar95_realized_cost", "mean_reserve_coverage"]:
        plot_stress_heatmap(
            stress_summary,
            metric=metric,
            out_path=os.path.join(out_dirs["plots"], f"heatmap_{metric}.png"),
            config=config,
        )

    summary_severe = summary_df[summary_df["stress_level"] == ("severe" if "severe" in summary_df["stress_level"].unique() else stress_levels[0])]
    if len(summary_severe) > 0:
        plot_metric_bar(
            summary_severe,
            metric="blackout_rate",
            out_path=os.path.join(out_dirs["plots"], "bar_blackout_rate_severe.png"),
            config=config,
            title="Mean blackout rate by method",
        )
        plot_metric_bar(
            summary_severe,
            metric="mean_load_shed",
            out_path=os.path.join(out_dirs["plots"], "bar_mean_load_shed_severe.png"),
            config=config,
            title="Mean load shed by method",
        )
        plot_metric_bar(
            summary_severe,
            metric="cvar95_realized_cost",
            out_path=os.path.join(out_dirs["plots"], "bar_cvar95_cost_severe.png"),
            config=config,
            title="CVaR95 of realized cost by method",
        )

    # Example scenario fans and reserve/SOC path plots
    if first_example_payload is not None:
        stress_ex, payload = first_example_payload
        for method, obj in payload.items():
            plot_scenario_fan(
                obj["scenarios"],
                obj["realized"],
                os.path.join(out_dirs["plots"], f"scenario_fan_{stress_ex}_{method}.png"),
                config,
                title=f"Scenario fan chart: {method} ({stress_ex}, first test day)",
            )

            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(obj["soc_path"], label="Realized SoC", linewidth=2)
            ax.plot(obj["reserve_floor"], label="Reserve floor", linewidth=2)
            ax.set_title(f"SoC versus reserve floor: {method} ({stress_ex}, first test day)")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Energy")
            ax.legend()
            save_and_show(fig, os.path.join(out_dirs["plots"], f"soc_vs_floor_{stress_ex}_{method}.png"), config)

            fig, ax = plt.subplots(figsize=(11, 6))
            ax.bar(np.arange(len(obj["shed_path"])), obj["shed_path"])
            ax.set_title(f"Load-shedding profile: {method} ({stress_ex}, first test day)")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Load shed")
            save_and_show(fig, os.path.join(out_dirs["plots"], f"shed_profile_{stress_ex}_{method}.png"), config)

    # concise text report
    report_lines = []
    report_lines.append("CLIMATE-TAIL BATTERY DISPATCH SIMULATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append("Interpretation guide:")
    report_lines.append("- Lower cost / load shed / blackout / reserve violation is better.")
    report_lines.append("- Higher reserve coverage is better.")
    report_lines.append("- Under heavy-tailed climate stress, the proposed method is expected to shine on tail-risk metrics.")
    report_lines.append("")
    for stress in summary_df["stress_level"].unique():
        ss = summary_df[summary_df["stress_level"] == stress].copy()
        report_lines.append(f"Stress level: {stress}")
        report_lines.append(ss.to_string(index=False))
        report_lines.append("")
    report_lines.append("Paired comparison table uses Benchmark - Proposed.")
    report_lines.append("Hence positive mean_diff means the benchmark is worse on that metric than Proposed_EVT_Regime.")

    report_path = os.path.join(out_dirs["logs"], "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n" + "\n".join(report_lines[:50]))

    # save copy of script itself
    try:
        src = os.path.abspath(__file__)
        shutil.copy(src, os.path.join(out_dirs["artifacts"], os.path.basename(src)))
    except Exception:
        pass

    # zip everything
    zip_path = base_dir.rstrip("/\\") + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(base_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, start=os.path.dirname(base_dir) or ".")
                zf.write(full, arcname=rel)

    print_banner("SIMULATION COMPLETE")
    print(f"Output directory: {os.path.abspath(base_dir)}")
    print(f"ZIP archive:      {os.path.abspath(zip_path)}")
    return {
        "all_daily": all_daily,
        "summary": summary_df,
        "by_rep": by_rep_df,
        "paired": paired_df,
        "overall": overall,
        "output_dir": os.path.abspath(base_dir),
        "zip_path": os.path.abspath(zip_path),
    }


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    # You can edit these settings directly in Colab if you want a lighter or heavier run.
    config = SimConfig(
        seed=1234,
        n_replications=6,             # increase to 10-20 for a stronger paper-quality study
        stress_levels=("mild", "severe", "extreme"),
        train_days=220,
        test_days=28,
        horizon_hours=24,
        n_scenarios_proposed=90,
        n_scenarios_benchmark=90,
        reserve_simulations=2000,
        quantile_q=0.95,
        cvar_alpha=0.95,
        risk_lambda=2.5,
        reserve_horizon=24,
        reserve_eps=0.05,
        battery_capacity=220.0,
        battery_charge_limit=55.0,
        battery_discharge_limit=55.0,
        eta_charge=0.95,
        eta_discharge=0.95,
        soc_min=20.0,
        soc_max=220.0,
        soc_init=120.0,
        thermal_cap=160.0,
        thermal_cost=30.0,
        degradation_cost=1.5,
        curtailment_cost=1.0,
        voll=1000.0,
        reserve_slack_cost=100.0,
        reserve_cost=0.8,
        show_plots_inline=True,
        save_dir="climate_tail_dispatch_outputs",
        run_stress_sensitivity=True,
    )

    results = run_experiment(config)

    print("\nDone. Key files:")
    print(f"- Overall summary CSV: {os.path.join(results['output_dir'], 'tables', 'overall_summary.csv')}")
    print(f"- Stress summary CSV:  {os.path.join(results['output_dir'], 'tables', 'summary_by_stress_and_method.csv')}")
    print(f"- ZIP archive:         {results['zip_path']}")
