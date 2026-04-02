#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-data analysis for:
A Tail-Regime Stochastic Programming Framework for Climate-Resilient Battery Dispatch

Dataset strategy (truthful and operationally relevant)
------------------------------------------------------
Primary grid data source:
    - CAISO actual system load (historical)
    - CAISO historical fuel mix (solar + wind actuals)
    - CAISO historical day-ahead load forecast
Accessed through the Python library `gridstatus`.

Weather covariates:
    - Open-Meteo historical archive API for hourly weather variables
    - weighted across major California load centers

Why this is suitable:
    - CAISO publishes demand, day-ahead demand forecast, and renewable generation information.
    - CAISO explicitly reports / discusses net demand = demand minus wind and solar.
    - This aligns naturally with the paper's residual net-load concept.

What this script does
---------------------
1. Downloads real CAISO load, load forecast, and fuel mix data.
2. Downloads weighted historical weather covariates for California load centers.
3. Constructs hourly residual net load = actual load - actual wind - actual solar.
4. Builds weather-state proxy regimes (normal / cold_stress / heatwave / wind_drought / compound).
5. Fits:
      - OLS mean model for residual net load
      - quantile-regression dynamic threshold model
      - regime-specific EVT/GPD tail models
6. Runs an expanding-window rolling backtest on real data.
7. Compares four methods:
      - Proposed EVT-tail scenario + reserve floor
      - Gaussian scenario benchmark
      - Empirical block-bootstrap benchmark
      - Deterministic mean-only benchmark
8. Solves real-time deterministic battery dispatch each day with the method-specific reserve floor.
9. Saves all tables/plots and creates a downloadable ZIP.

Important honesty note
----------------------
This code is designed as a *real-data verification / pseudo-operational backtest*.
The weather covariates merged here are historical realized weather covariates, not archived
vendor-specific NWP forecasts. This keeps the script reproducible and free of API keys,
while still allowing a rigorous real-data evaluation of the EVT reserve methodology.
For journal submission, one can later replace these with archived day-ahead weather forecasts.
"""

import os
import sys
import json
import math
import time
import shutil
import zipfile
import warnings
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")


def ensure_package(pkg_name: str, import_name: Optional[str] = None):
    import importlib
    name = import_name or pkg_name
    try:
        importlib.import_module(name)
    except ImportError:
        import subprocess
        print(f"Installing missing package: {pkg_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])


for pkg, imp in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("statsmodels", "statsmodels"),
    ("cvxpy", "cvxpy"),
    ("requests", "requests"),
    ("gridstatus", "gridstatus"),
    ("tqdm", "tqdm"),
]:
    ensure_package(pkg, imp)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto, norm
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import cvxpy as cp
import requests
from tqdm import tqdm
import gridstatus

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.30
plt.rcParams["font.size"] = 11


# ============================================================
# Configuration
# ============================================================
@dataclass
class RealDataConfig:
    # CAISO + weather data window
    start_date: str = "2021-09-01"
    end_date: str = "2022-12-31"
    train_end_date: str = "2022-08-31"
    test_start_date: str = "2022-09-01"
    test_end_date: str = "2022-12-31"

    # hourly backtest settings
    horizon_hours: int = 24
    reserve_horizon_hours: int = 6
    reserve_eps: float = 0.05
    quantile_q: float = 0.95
    n_scenarios: int = 400
    refit_every_days: int = 7

    # dispatch economics (stylized but data-scaled)
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    thermal_cost: float = 120.0        # $ / GWh-equivalent (stylized)
    degradation_cost: float = 12.0     # $ / GWh throughput
    curtailment_cost: float = 5.0      # $ / GWh spill
    voll: float = 15000.0              # value of lost load
    reserve_slack_cost: float = 2000.0 # violating reserve floor is very costly

    show_plots_inline: bool = True
    save_dir: str = "climate_tail_dispatch_real_data_outputs"

    # weather footprint: weighted CAISO load centers / representative cities
    weather_points: Tuple[Tuple[str, float, float, float], ...] = (
        ("Los_Angeles", 34.0522, -118.2437, 0.32),
        ("Sacramento", 38.5816, -121.4944, 0.23),
        ("Fresno", 36.7378, -119.7871, 0.18),
        ("San_Francisco", 37.7749, -122.4194, 0.17),
        ("San_Diego", 32.7157, -117.1611, 0.10),
    )

    timezone: str = "America/Los_Angeles"


STATE_NAMES = {
    0: "normal",
    1: "cold_stress",
    2: "heatwave",
    3: "wind_drought",
    4: "compound",
}


# ============================================================
# Output helpers
# ============================================================
def make_output_dirs(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    subdirs = ["plots", "tables", "logs", "artifacts", "daily_results", "cache"]
    out = {}
    for s in subdirs:
        p = os.path.join(base_dir, s)
        os.makedirs(p, exist_ok=True)
        out[s] = p
    return out


def print_banner(msg: str):
    print("\n" + "=" * 110)
    print(msg)
    print("=" * 110)


def save_dataframe(df: pd.DataFrame, path: str, index: bool = False, title: Optional[str] = None, max_rows: int = 40):
    df.to_csv(path, index=index)
    if title is not None:
        print(f"\n{title}")
        show_df = df.head(max_rows) if len(df) > max_rows else df
        print(show_df.to_string(index=index))


def finalize_zip(source_dir: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.relpath(full, source_dir)
                zf.write(full, arc)


def maybe_show():
    if plt.get_fignums():
        plt.tight_layout()
        plt.show()


# ============================================================
# Time / column helpers
# ============================================================
def month_chunks(start_date: str, end_date: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []
    cursor = start.replace(day=1)
    while cursor <= end:
        chunk_start = max(cursor, start)
        chunk_end = min(cursor + pd.offsets.MonthEnd(1), end)
        chunks.append((pd.Timestamp(chunk_start), pd.Timestamp(chunk_end)))
        cursor = (cursor + pd.offsets.MonthBegin(1))
    return chunks


def standardize_time_col(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    df = df.copy()
    time_candidates = [
        "Interval Start", "Time", "Timestamp", "datetime", "DATE", "Date",
    ]
    time_col = None
    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError(f"Could not find time column in columns: {list(df.columns)}")
    dt = pd.to_datetime(df[time_col])
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert(timezone).dt.tz_localize(None)
    except Exception:
        pass
    df["time"] = dt
    return df


def find_column_case_insensitive(df: pd.DataFrame, keyword: str) -> Optional[str]:
    key = keyword.lower().strip()
    for c in df.columns:
        if key == c.lower().strip():
            return c
    for c in df.columns:
        if key in c.lower().strip():
            return c
    return None


def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# ============================================================
# Data acquisition: CAISO via gridstatus
# ============================================================
def safe_fetch(fn, *args, max_tries: int = 3, sleep_sec: float = 3.0, **kwargs):
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            print(f"Attempt {attempt}/{max_tries} failed: {e}")
            time.sleep(sleep_sec * attempt)
    raise last_err


def fetch_caiso_load(config: RealDataConfig, out_paths: Dict[str, str]) -> pd.DataFrame:
    cache_path = os.path.join(out_paths["cache"], "caiso_load_hourly.csv")
    if os.path.exists(cache_path):
        print(f"Loading cached CAISO load from {cache_path}")
        return pd.read_csv(cache_path, parse_dates=["time"])

    iso = gridstatus.CAISO()
    pieces = []
    print_banner("Downloading CAISO actual load via gridstatus")
    for start, end in tqdm(month_chunks(config.start_date, config.end_date), desc="CAISO load"):
        df = safe_fetch(iso.get_load, start, end=end, verbose=False)
        df = standardize_time_col(df, config.timezone)
        load_col = find_column_case_insensitive(df, "load")
        if load_col is None:
            nums = numeric_cols(df)
            if not nums:
                raise ValueError("No numeric load column found in CAISO load data.")
            load_col = nums[0]
        keep = df[["time", load_col]].rename(columns={load_col: "load_mw"})
        pieces.append(keep)

    load = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["time"])
    load = load.sort_values("time")
    load = load.set_index("time").resample("1H").mean().reset_index()
    load.to_csv(cache_path, index=False)
    return load


def fetch_caiso_fuel_mix(config: RealDataConfig, out_paths: Dict[str, str]) -> pd.DataFrame:
    cache_path = os.path.join(out_paths["cache"], "caiso_fuel_mix_hourly.csv")
    if os.path.exists(cache_path):
        print(f"Loading cached CAISO fuel mix from {cache_path}")
        return pd.read_csv(cache_path, parse_dates=["time"])

    iso = gridstatus.CAISO()
    pieces = []
    print_banner("Downloading CAISO fuel mix via gridstatus")
    for start, end in tqdm(month_chunks(config.start_date, config.end_date), desc="CAISO fuel mix"):
        df = safe_fetch(iso.get_fuel_mix, start, end=end, verbose=False)
        df = standardize_time_col(df, config.timezone)

        lower_map = {c.lower(): c for c in df.columns}
        solar_cols = [c for c in df.columns if "solar" in c.lower()]
        wind_cols = [c for c in df.columns if "wind" in c.lower()]
        if len(solar_cols) == 0 or len(wind_cols) == 0:
            raise ValueError(
                f"Could not identify solar/wind columns in fuel mix. Columns: {list(df.columns)}"
            )

        keep = pd.DataFrame({
            "time": df["time"],
            "solar_mw": df[solar_cols].sum(axis=1),
            "wind_mw": df[wind_cols].sum(axis=1),
        })
        pieces.append(keep)

    fuel = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["time"])
    fuel = fuel.sort_values("time")
    fuel = fuel.set_index("time").resample("1H").mean().reset_index()
    fuel.to_csv(cache_path, index=False)
    return fuel


def fetch_caiso_load_forecast(config: RealDataConfig, out_paths: Dict[str, str]) -> pd.DataFrame:
    cache_path = os.path.join(out_paths["cache"], "caiso_load_forecast_hourly.csv")
    if os.path.exists(cache_path):
        print(f"Loading cached CAISO load forecast from {cache_path}")
        return pd.read_csv(cache_path, parse_dates=["time"])

    iso = gridstatus.CAISO()
    pieces = []
    print_banner("Downloading CAISO historical load forecast via gridstatus")
    for start, end in tqdm(month_chunks(config.start_date, config.end_date), desc="CAISO load forecast"):
        df = safe_fetch(iso.get_load_forecast, start, end=end, verbose=False)
        df = standardize_time_col(df, config.timezone)
        fc_col = find_column_case_insensitive(df, "load forecast")
        if fc_col is None:
            nums = numeric_cols(df)
            if not nums:
                raise ValueError("No numeric forecast column found in CAISO load forecast data.")
            fc_col = nums[0]

        # If multiple TAC areas are present, pick the maximum across areas per interval as a robust
        # proxy for system-level forecast. This avoids double-counting when both zonal and total rows exist.
        keep = df[["time", fc_col]].rename(columns={fc_col: "load_forecast_mw"})
        keep = keep.groupby("time", as_index=False)["load_forecast_mw"].max()
        pieces.append(keep)

    forecast = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["time"])
    forecast = forecast.sort_values("time")
    forecast = forecast.set_index("time").resample("1H").mean().reset_index()
    forecast.to_csv(cache_path, index=False)
    return forecast


# ============================================================
# Data acquisition: weather via Open-Meteo archive
# ============================================================
def fetch_open_meteo_hourly(lat: float, lon: float, start_date: str, end_date: str, timezone: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "shortwave_radiation",
        ]),
        "timezone": timezone,
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    js = r.json()
    hourly = js["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly.get("temperature_2m", []),
        "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
        "wind_speed_10m": hourly.get("wind_speed_10m", []),
        "shortwave_radiation": hourly.get("shortwave_radiation", []),
    })
    return df


def fetch_weighted_weather(config: RealDataConfig, out_paths: Dict[str, str]) -> pd.DataFrame:
    cache_path = os.path.join(out_paths["cache"], "weighted_weather_hourly.csv")
    if os.path.exists(cache_path):
        print(f"Loading cached weather from {cache_path}")
        return pd.read_csv(cache_path, parse_dates=["time"])

    print_banner("Downloading weighted historical weather from Open-Meteo")
    pieces = []
    for name, lat, lon, wt in config.weather_points:
        print(f"Fetching weather for {name} ({lat}, {lon}) with weight={wt}")
        df = safe_fetch(fetch_open_meteo_hourly, lat, lon, config.start_date, config.end_date, config.timezone)
        df = df.rename(columns={
            "temperature_2m": f"temp_{name}",
            "relative_humidity_2m": f"rh_{name}",
            "wind_speed_10m": f"windspd_{name}",
            "shortwave_radiation": f"swrad_{name}",
        })
        pieces.append((df, wt, name))

    weather = pieces[0][0][["time"]].copy()
    for df, _, _ in pieces:
        weather = weather.merge(df, on="time", how="outer")
    weather = weather.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    # weighted average features
    for prefix in ["temp", "rh", "windspd", "swrad"]:
        val = 0.0
        for _, wt, name in pieces:
            val = val + wt * weather[f"{prefix}_{name}"]
        weather[prefix] = val

    keep = weather[["time", "temp", "rh", "windspd", "swrad"]].copy()
    keep.to_csv(cache_path, index=False)
    return keep


# ============================================================
# Merge + preprocessing
# ============================================================
def build_real_dataset(config: RealDataConfig, out_paths: Dict[str, str]) -> pd.DataFrame:
    load = fetch_caiso_load(config, out_paths)
    fuel = fetch_caiso_fuel_mix(config, out_paths)
    forecast = fetch_caiso_load_forecast(config, out_paths)
    weather = fetch_weighted_weather(config, out_paths)

    df = load.merge(fuel, on="time", how="inner")
    df = df.merge(forecast, on="time", how="left")
    df = df.merge(weather, on="time", how="left")
    df = df.sort_values("time").reset_index(drop=True)

    # fill / interpolate modest gaps
    for c in ["load_mw", "solar_mw", "wind_mw", "load_forecast_mw", "temp", "rh", "windspd", "swrad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].interpolate(limit_direction="both")
            df[c] = df[c].fillna(method="bfill").fillna(method="ffill")

    # convert MW to GW for numerical stability / interpretability
    for c in ["load_mw", "solar_mw", "wind_mw", "load_forecast_mw"]:
        df[c.replace("_mw", "_gw")] = df[c] / 1000.0

    df["renewable_gw"] = df["solar_gw"] + df["wind_gw"]
    df["residual_net_load_gw"] = df["load_gw"] - df["renewable_gw"]
    df["renewable_share"] = np.where(df["load_gw"] > 0, df["renewable_gw"] / df["load_gw"], np.nan)
    df["forecast_error_gw"] = df["load_gw"] - df["load_forecast_gw"]

    df["date"] = df["time"].dt.date.astype(str)
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["doy"] = df["time"].dt.dayofyear
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    # lags on residual net load
    df["rnl_lag_1"] = df["residual_net_load_gw"].shift(1)
    df["rnl_lag_24"] = df["residual_net_load_gw"].shift(24)
    df["rnl_roll_24_mean"] = df["residual_net_load_gw"].rolling(24).mean().shift(1)
    df["rnl_roll_24_std"] = df["residual_net_load_gw"].rolling(24).std().shift(1)

    # solar and wind generation features
    df["solar_lag_24"] = df["solar_gw"].shift(24)
    df["wind_lag_24"] = df["wind_gw"].shift(24)

    df = df.dropna().reset_index(drop=True)

    data_path = os.path.join(out_paths["artifacts"], "merged_real_dataset_hourly.csv")
    df.to_csv(data_path, index=False)
    print(f"Merged dataset saved to: {data_path}")
    return df


# ============================================================
# Regime proxies
# ============================================================
def assign_regime_proxies(df: pd.DataFrame, ref_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    out = df.copy()
    ref = out if ref_df is None else ref_df

    q_temp_low = ref["temp"].quantile(0.10)
    q_temp_high = ref["temp"].quantile(0.90)
    q_load_high = ref["load_gw"].quantile(0.80)
    q_rnl_high = ref["residual_net_load_gw"].quantile(0.80)
    q_wind_low = ref["wind_gw"].quantile(0.20)
    q_rad_low_day = ref.loc[ref["hour"].between(8, 17), "swrad"].quantile(0.20)

    cold = (out["temp"] <= q_temp_low) & (out["load_gw"] >= q_load_high)
    heat = (out["temp"] >= q_temp_high) & (out["load_gw"] >= q_load_high)
    wind_drought = (out["wind_gw"] <= q_wind_low) & (out["residual_net_load_gw"] >= q_rnl_high)
    solar_dull = (out["hour"].between(8, 17)) & (out["swrad"] <= q_rad_low_day)
    compound = ((heat | cold) & wind_drought) | ((heat | cold) & solar_dull & (out["residual_net_load_gw"] >= q_rnl_high))

    state = np.zeros(len(out), dtype=int)
    state[cold] = 1
    state[heat] = 2
    state[wind_drought] = 3
    state[compound] = 4

    # compound should dominate all others
    state[compound] = 4

    out["state"] = state
    out["state_name"] = pd.Series(state).map(STATE_NAMES).values
    return out


# ============================================================
# Modeling helpers
# ============================================================
def build_design_matrix(df: pd.DataFrame, include_forecast: bool = True) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    X["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    X["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    X["sin_year"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    X["cos_year"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    X["temp"] = df["temp"].values
    X["temp_cold"] = np.maximum(18 - df["temp"].values, 0)
    X["temp_hot"] = np.maximum(df["temp"].values - 28, 0)
    X["rh"] = df["rh"].values
    X["windspd"] = df["windspd"].values
    X["swrad"] = df["swrad"].values
    X["is_weekend"] = df["is_weekend"].values
    X["rnl_lag_1"] = df["rnl_lag_1"].values
    X["rnl_lag_24"] = df["rnl_lag_24"].values
    X["rnl_roll_24_mean"] = df["rnl_roll_24_mean"].values
    X["rnl_roll_24_std"] = df["rnl_roll_24_std"].fillna(df["rnl_roll_24_std"].median()).values
    X["solar_lag_24"] = df["solar_lag_24"].values
    X["wind_lag_24"] = df["wind_lag_24"].values
    if include_forecast and "load_forecast_gw" in df.columns:
        X["load_forecast_gw"] = df["load_forecast_gw"].values
        X["forecast_error_lag1"] = df["forecast_error_gw"].shift(1).fillna(0.0).values
    return X


@dataclass
class FittedRealDataModels:
    mean_model: object
    q_model: object
    x_columns: List[str]
    train_df: pd.DataFrame
    resid_train: np.ndarray
    u_train: np.ndarray
    regime_non_exc_resid: Dict[int, np.ndarray]
    regime_exc_prob: Dict[int, float]
    regime_gpd: Dict[int, Dict[str, float]]
    regime_probs: Dict[int, float]
    daily_residual_matrix: np.ndarray
    daily_dates: List[str]
    thermal_cap: float
    battery_capacity: float
    battery_power_cap: float



def fit_real_models(train_df: pd.DataFrame, config: RealDataConfig) -> FittedRealDataModels:
    train_df = train_df.copy()
    train_df = assign_regime_proxies(train_df, ref_df=train_df)

    X = build_design_matrix(train_df)
    y = train_df["residual_net_load_gw"].values

    mean_model = sm.OLS(y, X).fit()
    q_model = QuantReg(y, X).fit(q=config.quantile_q, max_iter=5000)

    mu_hat = mean_model.predict(X)
    u_hat = q_model.predict(X)
    resid = y - mu_hat
    exceed = y > u_hat
    excess = np.maximum(y - u_hat, 0)

    regime_non_exc_resid = {}
    regime_exc_prob = {}
    regime_gpd = {}
    regime_probs = {}

    pooled_non_exc = resid[~exceed]
    if len(pooled_non_exc) < 100:
        pooled_non_exc = resid.copy()

    for s in sorted(train_df["state"].unique()):
        idx = train_df["state"].values == s
        idx_non = idx & (~exceed)
        idx_exc = idx & exceed

        regime_probs[s] = float(np.mean(idx))
        regime_non_exc_resid[s] = resid[idx_non] if np.sum(idx_non) >= 30 else pooled_non_exc
        regime_exc_prob[s] = float(np.mean(exceed[idx])) if np.sum(idx) > 0 else float(np.mean(exceed))

        exc_vals = excess[idx_exc]
        if len(exc_vals) >= 25 and np.std(exc_vals) > 1e-8:
            try:
                c_hat, _, scale_hat = genpareto.fit(exc_vals, floc=0)
                regime_gpd[s] = {"shape": float(c_hat), "scale": float(scale_hat)}
            except Exception:
                regime_gpd[s] = {"shape": 0.10, "scale": max(0.05, float(np.std(exc_vals)))}
        else:
            fallback = excess[exceed]
            if len(fallback) >= 25 and np.std(fallback) > 1e-8:
                c_hat, _, scale_hat = genpareto.fit(fallback, floc=0)
                regime_gpd[s] = {"shape": float(c_hat), "scale": float(scale_hat)}
            else:
                regime_gpd[s] = {"shape": 0.10, "scale": 0.20}

    # build daily residual matrix for bootstrap and covariance estimation
    tmp = train_df[["date", "hour"]].copy()
    tmp["resid"] = resid
    daily_wide = tmp.pivot(index="date", columns="hour", values="resid").dropna()
    daily_residual_matrix = daily_wide.values.astype(float)
    daily_dates = daily_wide.index.tolist()

    # data-driven system sizing: thermal threshold + 4-hour battery
    thermal_cap = float(train_df["residual_net_load_gw"].quantile(0.80))
    hourly_excess = np.maximum(train_df["residual_net_load_gw"].values - thermal_cap, 0)
    battery_power_cap = float(max(np.quantile(hourly_excess, 0.95), 0.30))
    battery_capacity = float(max(4.0 * battery_power_cap, 1.2))

    return FittedRealDataModels(
        mean_model=mean_model,
        q_model=q_model,
        x_columns=list(X.columns),
        train_df=train_df,
        resid_train=resid,
        u_train=u_hat,
        regime_non_exc_resid=regime_non_exc_resid,
        regime_exc_prob=regime_exc_prob,
        regime_gpd=regime_gpd,
        regime_probs=regime_probs,
        daily_residual_matrix=daily_residual_matrix,
        daily_dates=daily_dates,
        thermal_cap=thermal_cap,
        battery_capacity=battery_capacity,
        battery_power_cap=battery_power_cap,
    )


# ============================================================
# Scenario generation
# ============================================================
def predict_mu_u(models: FittedRealDataModels, df_day: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df_tmp = df_day.copy()
    df_tmp = assign_regime_proxies(df_tmp, ref_df=models.train_df)
    X_day = build_design_matrix(df_tmp)
    X_day = X_day[models.x_columns]
    mu = np.asarray(models.mean_model.predict(X_day), dtype=float)
    u = np.asarray(models.q_model.predict(X_day), dtype=float)
    return mu, u, df_tmp



def simulate_evt_scenarios(models: FittedRealDataModels, df_day: pd.DataFrame, n_scenarios: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu, u, df_tmp = predict_mu_u(models, df_day)
    state_vec = df_tmp["state"].values.astype(int)
    scenarios = np.zeros((n_scenarios, len(df_day)))

    for s in range(n_scenarios):
        path = np.zeros(len(df_day))
        prev_exc = False
        for h in range(len(df_day)):
            st_id = int(state_vec[h])
            p_exc = models.regime_exc_prob.get(st_id, np.mean(list(models.regime_exc_prob.values())))

            # slight persistence kicker if previous hour was extreme exceedance and current state is also adverse
            if prev_exc and st_id in [1, 2, 3, 4]:
                p_exc = min(0.98, 1.25 * p_exc)

            if rng.uniform() < p_exc:
                pars = models.regime_gpd.get(st_id, {"shape": 0.10, "scale": 0.20})
                exc = genpareto.rvs(c=pars["shape"], loc=0, scale=max(pars["scale"], 1e-4), random_state=rng)
                path[h] = max(u[h], mu[h]) + exc
                prev_exc = True
            else:
                resid_pool = models.regime_non_exc_resid.get(st_id, models.resid_train)
                draw = rng.choice(resid_pool)
                candidate = mu[h] + draw
                path[h] = min(candidate, u[h])
                prev_exc = False
        scenarios[s, :] = path
    return scenarios



def simulate_gaussian_scenarios(models: FittedRealDataModels, df_day: pd.DataFrame, n_scenarios: int, seed: int = 456) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu, _, _ = predict_mu_u(models, df_day)
    R = models.daily_residual_matrix
    if R.shape[0] >= 20:
        Sigma = np.cov(R, rowvar=False)
    else:
        Sigma = np.diag(np.repeat(np.var(models.resid_train), len(df_day)))
    Sigma = Sigma + 1e-5 * np.eye(Sigma.shape[0])
    draws = rng.multivariate_normal(mean=np.zeros(len(df_day)), cov=Sigma, size=n_scenarios)
    return mu[None, :] + draws



def simulate_bootstrap_scenarios(models: FittedRealDataModels, df_day: pd.DataFrame, n_scenarios: int, seed: int = 789) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu, _, _ = predict_mu_u(models, df_day)
    R = models.daily_residual_matrix
    if R.shape[0] == 0:
        boot = rng.choice(models.resid_train, size=(n_scenarios, len(df_day)), replace=True)
    else:
        idx = rng.integers(0, R.shape[0], size=n_scenarios)
        boot = R[idx, :]
    return mu[None, :] + boot



def simulate_deterministic_scenario(models: FittedRealDataModels, df_day: pd.DataFrame) -> np.ndarray:
    mu, _, _ = predict_mu_u(models, df_day)
    return mu.reshape(1, -1)


# ============================================================
# Reserve floor from scenarios
# ============================================================
def reserve_floor_from_scenarios(
    scenarios: np.ndarray,
    thermal_cap: float,
    battery_capacity: float,
    reserve_horizon_hours: int,
    reserve_eps: float,
) -> np.ndarray:
    n_scen, H = scenarios.shape
    reserve = np.zeros(H)
    for t in range(H):
        needs = []
        end = min(H, t + reserve_horizon_hours)
        for s in range(n_scen):
            need = float(np.sum(np.maximum(scenarios[s, t:end] - thermal_cap, 0)))
            needs.append(need)
        reserve[t] = min(np.quantile(needs, 1.0 - reserve_eps), battery_capacity)
    return reserve


# ============================================================
# Deterministic dispatch under realized path + reserve floor
# ============================================================
def solve_dispatch_realized(
    realized_rnl: np.ndarray,
    reserve_floor: np.ndarray,
    thermal_cap: float,
    battery_capacity: float,
    battery_power_cap: float,
    config: RealDataConfig,
    soc_init: Optional[float] = None,
):
    H = len(realized_rnl)
    soc0 = battery_capacity * 0.55 if soc_init is None else soc_init

    g = cp.Variable(H, nonneg=True)
    d = cp.Variable(H, nonneg=True)
    c = cp.Variable(H, nonneg=True)
    shed = cp.Variable(H, nonneg=True)
    spill = cp.Variable(H, nonneg=True)
    e = cp.Variable(H)
    zeta = cp.Variable(H, nonneg=True)

    constraints = []
    for t in range(H):
        constraints += [g[t] + d[t] + shed[t] - c[t] - spill[t] == realized_rnl[t]]
        constraints += [g[t] <= thermal_cap]
        constraints += [c[t] <= battery_power_cap]
        constraints += [d[t] <= battery_power_cap]
        constraints += [e[t] >= 0, e[t] <= battery_capacity]
        constraints += [e[t] >= reserve_floor[t] - zeta[t]]
        if t == 0:
            constraints += [e[t] == soc0 + config.eta_charge * c[t] - d[t] / config.eta_discharge]
        else:
            constraints += [e[t] == e[t-1] + config.eta_charge * c[t] - d[t] / config.eta_discharge]

    objective = cp.Minimize(
        config.thermal_cost * cp.sum(g)
        + config.degradation_cost * cp.sum(c + d)
        + config.voll * cp.sum(shed)
        + config.curtailment_cost * cp.sum(spill)
        + config.reserve_slack_cost * cp.sum(zeta)
    )

    prob = cp.Problem(objective, constraints)
    solved = False
    for solver in [cp.ECOS, cp.OSQP, cp.SCS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                solved = True
                break
        except Exception:
            continue

    if not solved:
        raise RuntimeError(f"Dispatch LP failed. Status = {prob.status}")

    result = {
        "objective": float(prob.value),
        "g": np.asarray(g.value).ravel(),
        "d": np.asarray(d.value).ravel(),
        "c": np.asarray(c.value).ravel(),
        "shed": np.asarray(shed.value).ravel(),
        "spill": np.asarray(spill.value).ravel(),
        "soc": np.asarray(e.value).ravel(),
        "zeta": np.asarray(zeta.value).ravel(),
        "soc_end": float(np.asarray(e.value).ravel()[-1]),
    }
    return result


# ============================================================
# Plots and tables
# ============================================================
def plot_data_overview(df: pd.DataFrame, out_paths: Dict[str, str], config: RealDataConfig):
    fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    sample = df[(df["time"] >= pd.Timestamp("2022-08-15")) & (df["time"] <= pd.Timestamp("2022-09-20"))].copy()
    if len(sample) == 0:
        sample = df.tail(24 * 30).copy()

    ax[0].plot(sample["time"], sample["load_gw"], label="Load (GW)")
    ax[0].plot(sample["time"], sample["solar_gw"], label="Solar (GW)")
    ax[0].plot(sample["time"], sample["wind_gw"], label="Wind (GW)")
    ax[0].set_title("CAISO real data overview: load and renewable generation")
    ax[0].legend(loc="upper right")

    ax[1].plot(sample["time"], sample["residual_net_load_gw"], label="Residual net load (GW)")
    ax[1].plot(sample["time"], sample["load_forecast_gw"], label="Day-ahead load forecast (GW)")
    ax[1].set_title("Residual net load and day-ahead load forecast")
    ax[1].legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(out_paths["plots"], "01_real_data_overview.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    if config.show_plots_inline:
        plt.show()
    plt.close()



def plot_weather_overview(df: pd.DataFrame, out_paths: Dict[str, str], config: RealDataConfig):
    sample = df[(df["time"] >= pd.Timestamp("2022-08-15")) & (df["time"] <= pd.Timestamp("2022-09-20"))].copy()
    if len(sample) == 0:
        sample = df.tail(24 * 30).copy()

    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax[0].plot(sample["time"], sample["temp"])
    ax[0].set_title("Weighted temperature")
    ax[1].plot(sample["time"], sample["windspd"])
    ax[1].set_title("Weighted wind speed")
    ax[2].plot(sample["time"], sample["swrad"])
    ax[2].set_title("Weighted shortwave radiation")
    plt.tight_layout()
    path = os.path.join(out_paths["plots"], "02_weather_overview.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    if config.show_plots_inline:
        plt.show()
    plt.close()



def plot_regime_counts(df: pd.DataFrame, out_paths: Dict[str, str], config: RealDataConfig):
    counts = df["state_name"].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("Weather-state proxy counts in the real dataset")
    plt.ylabel("Hours")
    plt.tight_layout()
    path = os.path.join(out_paths["plots"], "03_regime_counts.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    if config.show_plots_inline:
        plt.show()
    plt.close()



def plot_threshold_and_exceedances(models: FittedRealDataModels, out_paths: Dict[str, str], config: RealDataConfig):
    train = models.train_df.copy()
    X = build_design_matrix(train)[models.x_columns]
    mu = np.asarray(models.mean_model.predict(X))
    u = np.asarray(models.q_model.predict(X))
    sample = train[(train["time"] >= pd.Timestamp("2022-08-15")) & (train["time"] <= pd.Timestamp("2022-09-20"))].copy()
    if len(sample) == 0:
        sample = train.tail(24 * 30).copy()
    idx = sample.index.values

    plt.figure(figsize=(14, 6))
    plt.plot(sample["time"], train.loc[idx, "residual_net_load_gw"], label="Observed residual net load")
    plt.plot(sample["time"], mu[idx], label="Mean model")
    plt.plot(sample["time"], u[idx], label=f"Dynamic threshold q={config.quantile_q:.2f}")
    mask_exc = train.loc[idx, "residual_net_load_gw"].values > u[idx]
    plt.scatter(sample.loc[mask_exc, "time"], train.loc[idx, "residual_net_load_gw"].values[mask_exc], s=18, label="Exceedances")
    plt.legend(loc="upper right")
    plt.title("Dynamic threshold and exceedances on real CAISO data")
    plt.tight_layout()
    path = os.path.join(out_paths["plots"], "04_dynamic_threshold_exceedances.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    if config.show_plots_inline:
        plt.show()
    plt.close()



def plot_gpd_diagnostics(models: FittedRealDataModels, out_paths: Dict[str, str], config: RealDataConfig):
    rows = []
    for s, pars in models.regime_gpd.items():
        rows.append({"state": s, "state_name": STATE_NAMES[s], **pars})
    gpd_df = pd.DataFrame(rows).sort_values("state")

    # pooled exceedances for a simple QQ-style diagnostic
    train = models.train_df.copy()
    X = build_design_matrix(train)[models.x_columns]
    u = np.asarray(models.q_model.predict(X))
    y = train["residual_net_load_gw"].values
    exc = np.maximum(y - u, 0)
    exc = exc[exc > 0]
    if len(exc) > 20:
        c_hat, _, scale_hat = genpareto.fit(exc, floc=0)
        p = (np.arange(1, len(exc) + 1) - 0.5) / len(exc)
        theo = genpareto.ppf(p, c=c_hat, loc=0, scale=scale_hat)
        obs = np.sort(exc)
        plt.figure(figsize=(6, 6))
        plt.scatter(theo, obs, s=18)
        mx = max(np.max(theo), np.max(obs))
        plt.plot([0, mx], [0, mx], linestyle="--")
        plt.xlabel("Theoretical GPD quantiles")
        plt.ylabel("Observed exceedances")
        plt.title("Pooled exceedance GPD QQ diagnostic")
        plt.tight_layout()
        path = os.path.join(out_paths["plots"], "05_gpd_qq_plot.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        if config.show_plots_inline:
            plt.show()
        plt.close()

    return gpd_df



def plot_daily_metric(backtest_df: pd.DataFrame, metric: str, title: str, filename: str, out_paths: Dict[str, str], config: RealDataConfig):
    pivot = backtest_df.pivot_table(index="date", columns="method", values=metric, aggfunc="mean")
    plt.figure(figsize=(14, 6))
    for c in pivot.columns:
        plt.plot(pd.to_datetime(pivot.index), pivot[c], label=c)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(out_paths["plots"], filename)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    if config.show_plots_inline:
        plt.show()
    plt.close()



def plot_summary_bars(summary_df: pd.DataFrame, metric: str, title: str, filename: str, out_paths: Dict[str, str], config: RealDataConfig):
    tmp = summary_df[["method", metric]].sort_values(metric)
    plt.figure(figsize=(10, 5))
    plt.bar(tmp["method"], tmp[metric])
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    path = os.path.join(out_paths["plots"], filename)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    if config.show_plots_inline:
        plt.show()
    plt.close()



def plot_extreme_day_dispatch(extreme_day_df: pd.DataFrame, out_paths: Dict[str, str], config: RealDataConfig):
    if len(extreme_day_df) == 0:
        return
    day = extreme_day_df["date"].iloc[0]
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for m, sub in extreme_day_df.groupby("method"):
        ax[0].plot(sub["hour"], sub["reserve_floor"], label=f"reserve: {m}")
        ax[1].plot(sub["hour"], sub["soc"], label=f"soc: {m}")
    ax[0].plot(extreme_day_df["hour"].unique(), extreme_day_df.groupby("hour")["realized_rnl"].first().values,
               linestyle="--", label="realized residual net load")
    ax[0].set_title(f"Extreme day reserve floors and realized residual net load: {day}")
    ax[1].set_title("State of charge trajectories")
    ax[0].legend(loc="upper right", ncol=2)
    ax[1].legend(loc="upper right", ncol=2)
    plt.tight_layout()
    path = os.path.join(out_paths["plots"], "09_extreme_day_dispatch.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    if config.show_plots_inline:
        plt.show()
    plt.close()


# ============================================================
# Backtest
# ============================================================
def summarize_method_metrics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    grouped = backtest_df.groupby("method")
    rows = []
    for method, sub in grouped:
        rows.append({
            "method": method,
            "n_days": sub["date"].nunique(),
            "mean_total_cost": sub.groupby("date")["total_cost"].first().mean(),
            "median_total_cost": sub.groupby("date")["total_cost"].first().median(),
            "cvar95_total_cost": sub.groupby("date")["total_cost"].first().pipe(lambda x: x[x >= x.quantile(0.95)].mean()),
            "mean_load_shed": sub.groupby("date")["load_shed"].first().mean(),
            "mean_load_shed_hours": sub.groupby("date")["load_shed_hours"].first().mean(),
            "mean_curtailment": sub.groupby("date")["curtailment"].first().mean(),
            "mean_reserve_slack": sub.groupby("date")["reserve_slack"].first().mean(),
            "mean_battery_throughput": sub.groupby("date")["battery_throughput"].first().mean(),
            "mean_reserve_floor": sub.groupby("date")["reserve_floor_mean"].first().mean(),
            "max_daily_shed": sub.groupby("date")["load_shed"].first().max(),
            "blackout_day_rate": np.mean(sub.groupby("date")["load_shed"].first().values > 1e-8),
        })
    return pd.DataFrame(rows).sort_values("mean_total_cost")



def paired_improvement_table(backtest_df: pd.DataFrame, baseline_method: str = "Proposed_EVT") -> pd.DataFrame:
    daily = backtest_df.groupby(["date", "method"], as_index=False).agg({
        "total_cost": "first",
        "load_shed": "first",
        "reserve_slack": "first",
        "curtailment": "first",
    })
    wide = daily.pivot(index="date", columns="method")
    methods = sorted(set(backtest_df["method"]) - {baseline_method})
    rows = []
    for m in methods:
        if ("total_cost", baseline_method) not in wide.columns or ("total_cost", m) not in wide.columns:
            continue
        base_cost = wide[("total_cost", baseline_method)].dropna()
        comp_cost = wide[("total_cost", m)].reindex(base_cost.index)
        common = base_cost.index.intersection(comp_cost.dropna().index)
        if len(common) == 0:
            continue
        rows.append({
            "comparison": f"{baseline_method} vs {m}",
            "mean_cost_difference_comp_minus_base": float((wide.loc[common, ("total_cost", m)] - wide.loc[common, ("total_cost", baseline_method)]).mean()),
            "mean_shed_difference_comp_minus_base": float((wide.loc[common, ("load_shed", m)] - wide.loc[common, ("load_shed", baseline_method)]).mean()),
            "mean_slack_difference_comp_minus_base": float((wide.loc[common, ("reserve_slack", m)] - wide.loc[common, ("reserve_slack", baseline_method)]).mean()),
            "base_wins_on_cost_days": int(np.sum(wide.loc[common, ("total_cost", baseline_method)] < wide.loc[common, ("total_cost", m)])),
            "base_wins_on_shed_days": int(np.sum(wide.loc[common, ("load_shed", baseline_method)] < wide.loc[common, ("load_shed", m)])),
            "n_common_days": int(len(common)),
        })
    return pd.DataFrame(rows)



def run_backtest(df: pd.DataFrame, config: RealDataConfig, out_paths: Dict[str, str]):
    test_dates = pd.date_range(config.test_start_date, config.test_end_date, freq="D")
    all_daily_records = []
    extreme_day_panel = []

    fitted_models = None
    last_refit_date = None

    for i, day in enumerate(tqdm(test_dates, desc="Rolling backtest")):
        day_str = day.strftime("%Y-%m-%d")
        train_df = df[df["time"] < day].copy()
        day_df = df[(df["time"] >= day) & (df["time"] < day + pd.Timedelta(days=1))].copy()
        if len(day_df) < 24 or len(train_df) < 24 * 180:
            continue

        need_refit = (
            fitted_models is None
            or last_refit_date is None
            or (day - last_refit_date).days >= config.refit_every_days
        )
        if need_refit:
            print(f"\nRefitting models using data up to {day_str} ...")
            fitted_models = fit_real_models(train_df, config)
            last_refit_date = day

        methods = {
            "Proposed_EVT": simulate_evt_scenarios(fitted_models, day_df, config.n_scenarios, seed=1000 + i),
            "Gaussian_Scen": simulate_gaussian_scenarios(fitted_models, day_df, config.n_scenarios, seed=2000 + i),
            "Bootstrap_Scen": simulate_bootstrap_scenarios(fitted_models, day_df, config.n_scenarios, seed=3000 + i),
            "Deterministic": simulate_deterministic_scenario(fitted_models, day_df),
        }

        realized_rnl = day_df["residual_net_load_gw"].values
        daily_method_outputs = []

        for method_name, scen in methods.items():
            reserve_floor = reserve_floor_from_scenarios(
                scenarios=scen,
                thermal_cap=fitted_models.thermal_cap,
                battery_capacity=fitted_models.battery_capacity,
                reserve_horizon_hours=config.reserve_horizon_hours,
                reserve_eps=config.reserve_eps,
            )
            dispatch = solve_dispatch_realized(
                realized_rnl=realized_rnl,
                reserve_floor=reserve_floor,
                thermal_cap=fitted_models.thermal_cap,
                battery_capacity=fitted_models.battery_capacity,
                battery_power_cap=fitted_models.battery_power_cap,
                config=config,
                soc_init=fitted_models.battery_capacity * 0.55,
            )

            rec = {
                "date": day_str,
                "method": method_name,
                "total_cost": float(dispatch["objective"]),
                "load_shed": float(np.sum(dispatch["shed"])),
                "load_shed_hours": int(np.sum(dispatch["shed"] > 1e-6)),
                "curtailment": float(np.sum(dispatch["spill"])),
                "reserve_slack": float(np.sum(dispatch["zeta"])),
                "battery_throughput": float(np.sum(dispatch["c"] + dispatch["d"])),
                "thermal_generation": float(np.sum(dispatch["g"])),
                "reserve_floor_mean": float(np.mean(reserve_floor)),
                "reserve_floor_max": float(np.max(reserve_floor)),
                "soc_min": float(np.min(dispatch["soc"])),
                "soc_end": float(dispatch["soc_end"]),
                "realized_rnl_mean": float(np.mean(realized_rnl)),
                "realized_rnl_max": float(np.max(realized_rnl)),
                "thermal_cap": float(fitted_models.thermal_cap),
                "battery_capacity": float(fitted_models.battery_capacity),
                "battery_power_cap": float(fitted_models.battery_power_cap),
            }
            daily_method_outputs.append(rec)

            # keep an hourly panel for the most stressful day for later plotting
            daily_panel = pd.DataFrame({
                "date": day_str,
                "method": method_name,
                "hour": np.arange(24),
                "realized_rnl": realized_rnl,
                "reserve_floor": reserve_floor,
                "soc": dispatch["soc"],
                "shed": dispatch["shed"],
                "spill": dispatch["spill"],
                "thermal": dispatch["g"],
                "charge": dispatch["c"],
                "discharge": dispatch["d"],
            })
            extreme_day_panel.append(daily_panel)

        all_daily_records.extend(daily_method_outputs)

    backtest_df = pd.DataFrame(all_daily_records)
    if backtest_df.empty:
        raise RuntimeError("Backtest produced no results. Check data download or date window.")

    # Identify the worst realized-stress day based on Proposed_EVT load shed + cost + reserve max
    extreme_dates = backtest_df.groupby("date").agg(
        proposed_cost=("total_cost", lambda x: x[backtest_df.loc[x.index, "method"] == "Proposed_EVT"].iloc[0] if np.any(backtest_df.loc[x.index, "method"] == "Proposed_EVT") else np.nan),
        proposed_shed=("load_shed", lambda x: x[backtest_df.loc[x.index, "method"] == "Proposed_EVT"].iloc[0] if np.any(backtest_df.loc[x.index, "method"] == "Proposed_EVT") else np.nan),
    ).reset_index()
    extreme_dates["score"] = extreme_dates["proposed_cost"].fillna(0) + 1000 * extreme_dates["proposed_shed"].fillna(0)
    selected_extreme_date = extreme_dates.sort_values("score", ascending=False)["date"].iloc[0]
    extreme_panel_df = pd.concat(extreme_day_panel, ignore_index=True)
    extreme_panel_df = extreme_panel_df[extreme_panel_df["date"] == selected_extreme_date].copy()

    return backtest_df, extreme_panel_df


# ============================================================
# Main
# ============================================================
def main():
    config = RealDataConfig()
    out_paths = make_output_dirs(config.save_dir)

    # Save config
    with open(os.path.join(out_paths["artifacts"], "run_config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    print_banner("STEP 1: Downloading and building the real CAISO + weather dataset")
    df = build_real_dataset(config, out_paths)
    df = assign_regime_proxies(df, ref_df=df)

    # basic dataset summary
    dataset_summary = pd.DataFrame({
        "metric": [
            "n_hours", "start_time", "end_time", "mean_load_gw", "mean_solar_gw", "mean_wind_gw",
            "mean_residual_net_load_gw", "p95_residual_net_load_gw", "max_residual_net_load_gw",
            "mean_temperature", "mean_wind_speed", "mean_shortwave_radiation"
        ],
        "value": [
            len(df), str(df["time"].min()), str(df["time"].max()), df["load_gw"].mean(), df["solar_gw"].mean(), df["wind_gw"].mean(),
            df["residual_net_load_gw"].mean(), df["residual_net_load_gw"].quantile(0.95), df["residual_net_load_gw"].max(),
            df["temp"].mean(), df["windspd"].mean(), df["swrad"].mean()
        ]
    })
    save_dataframe(dataset_summary, os.path.join(out_paths["tables"], "dataset_summary.csv"), index=False, title="Dataset summary")

    regime_counts = df["state_name"].value_counts().rename_axis("state_name").reset_index(name="hours")
    save_dataframe(regime_counts, os.path.join(out_paths["tables"], "regime_counts.csv"), index=False, title="Regime proxy counts")

    # Plots before backtest
    plot_data_overview(df, out_paths, config)
    plot_weather_overview(df, out_paths, config)
    plot_regime_counts(df, out_paths, config)

    # initial fit on declared train period for diagnostics
    train_initial = df[df["time"] <= pd.Timestamp(config.train_end_date) + pd.Timedelta(hours=23)].copy()
    print_banner("STEP 2: Fitting initial EVT / threshold models on the declared training period")
    models_init = fit_real_models(train_initial, config)

    gpd_df = plot_gpd_diagnostics(models_init, out_paths, config)
    save_dataframe(gpd_df, os.path.join(out_paths["tables"], "evt_gpd_parameters_by_regime.csv"), index=False, title="Estimated regime-specific GPD parameters")
    plot_threshold_and_exceedances(models_init, out_paths, config)

    system_sizing = pd.DataFrame({
        "quantity": ["thermal_cap_gw", "battery_capacity_gwh", "battery_power_cap_gw"],
        "value": [models_init.thermal_cap, models_init.battery_capacity, models_init.battery_power_cap],
    })
    save_dataframe(system_sizing, os.path.join(out_paths["tables"], "data_driven_system_sizing.csv"), index=False, title="Data-driven thermal and battery sizing")

    # Rolling backtest
    print_banner("STEP 3: Running the real-data rolling backtest")
    backtest_df, extreme_panel_df = run_backtest(df, config, out_paths)
    save_dataframe(backtest_df.head(50), os.path.join(out_paths["tables"], "daily_backtest_results_preview.csv"), index=False, title="Preview of daily backtest results (first 50 rows)")
    backtest_df.to_csv(os.path.join(out_paths["daily_results"], "daily_backtest_results_full.csv"), index=False)
    extreme_panel_df.to_csv(os.path.join(out_paths["daily_results"], "selected_extreme_day_hourly_panel.csv"), index=False)

    summary_df = summarize_method_metrics(backtest_df)
    save_dataframe(summary_df, os.path.join(out_paths["tables"], "backtest_summary_by_method.csv"), index=False, title="Backtest summary by method")

    improvement_df = paired_improvement_table(backtest_df, baseline_method="Proposed_EVT")
    if len(improvement_df) > 0:
        save_dataframe(improvement_df, os.path.join(out_paths["tables"], "paired_improvement_vs_proposed.csv"), index=False, title="Paired comparison against Proposed_EVT")

    # plots from backtest
    plot_daily_metric(backtest_df, "total_cost", "Daily realized total cost by method", "06_daily_total_cost.png", out_paths, config)
    plot_daily_metric(backtest_df, "load_shed", "Daily realized load shed by method", "07_daily_load_shed.png", out_paths, config)
    plot_daily_metric(backtest_df, "reserve_slack", "Daily reserve-floor slack by method", "08_daily_reserve_slack.png", out_paths, config)

    plot_summary_bars(summary_df, "mean_total_cost", "Mean daily total cost by method", "10_mean_total_cost_bar.png", out_paths, config)
    plot_summary_bars(summary_df, "mean_load_shed", "Mean daily load shed by method", "11_mean_load_shed_bar.png", out_paths, config)
    plot_summary_bars(summary_df, "blackout_day_rate", "Blackout-day rate by method", "12_blackout_day_rate_bar.png", out_paths, config)

    plot_extreme_day_dispatch(extreme_panel_df, out_paths, config)

    # Markdown notes for later paper-writing
    notes_path = os.path.join(out_paths["artifacts"], "real_data_notes_for_paper.txt")
    with open(notes_path, "w") as f:
        f.write(
            "Real-data analysis notes\n"
            "========================\n"
            "1. Dataset: CAISO actual hourly load, solar and wind actual generation (from fuel mix), day-ahead load forecast, and weighted weather covariates.\n"
            "2. Core response variable: residual net load = load - solar - wind.\n"
            "3. Regime proxy states are heuristic but operationally meaningful: normal, cold_stress, heatwave, wind_drought, compound.\n"
            "4. Proposed method uses a dynamic threshold quantile regression + regime-specific EVT tails to generate stress scenarios.\n"
            "5. Benchmarks: Gaussian day scenario generation, empirical block bootstrap, deterministic mean-only.\n"
            "6. Dispatch evaluation uses realized next-day residual net load and method-specific reserve floors.\n"
        )

    # zip everything
    zip_path = f"{config.save_dir}.zip"
    finalize_zip(config.save_dir, zip_path)

    print_banner("DONE")
    print(f"All outputs saved to directory: {config.save_dir}")
    print(f"ZIP created at: {zip_path}")
    print("\nKey tables:")
    for fn in [
        "dataset_summary.csv",
        "evt_gpd_parameters_by_regime.csv",
        "data_driven_system_sizing.csv",
        "backtest_summary_by_method.csv",
        "paired_improvement_vs_proposed.csv",
    ]:
        p = os.path.join(out_paths["tables"], fn)
        if os.path.exists(p):
            print(" -", p)

    print("\nKey plots:")
    for fn in [
        "01_real_data_overview.png",
        "04_dynamic_threshold_exceedances.png",
        "05_gpd_qq_plot.png",
        "06_daily_total_cost.png",
        "07_daily_load_shed.png",
        "08_daily_reserve_slack.png",
        "09_extreme_day_dispatch.png",
    ]:
        p = os.path.join(out_paths["plots"], fn)
        if os.path.exists(p):
            print(" -", p)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR in real-data pipeline:")
        print(str(e))
        print("\nFull traceback:")
        traceback.print_exc()
        raise
