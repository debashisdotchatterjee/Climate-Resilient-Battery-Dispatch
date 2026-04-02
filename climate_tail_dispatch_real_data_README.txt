REAL-DATA BACKTEST CODE FOR THE EVT BATTERY-DISPATCH PAPER
===========================================================

Main file
---------
climate_tail_dispatch_real_data.py

What dataset this uses
----------------------
1. CAISO historical actual load (via the Python library gridstatus)
2. CAISO historical fuel mix to obtain actual solar and wind generation (via gridstatus)
3. CAISO historical load forecast (via gridstatus)
4. Historical weather covariates from Open-Meteo archive API, aggregated over major California load centers

Why this dataset is appropriate
-------------------------------
This combination gives a genuine operational power-system dataset with:
- actual demand,
- actual renewable generation,
- a day-ahead demand forecast,
- weather covariates,
which allows construction of residual net load = load - wind - solar and makes the EVT tail reserve methodology defensible on real data.

What the code does
------------------
- Downloads and merges the real data
- Constructs residual net load and weather-state proxy regimes
- Fits OLS mean + quantile-regression threshold + regime-specific GPD tails
- Runs a rolling backtest with four methods:
  * Proposed_EVT
  * Gaussian_Scen
  * Bootstrap_Scen
  * Deterministic
- Computes method-specific reserve floors from day-ahead scenarios
- Solves realized daily battery dispatch with those reserve floors
- Saves many tables and plots, and creates a ZIP of outputs

Run in Colab
------------
1. Upload climate_tail_dispatch_real_data.py
2. Run:
   !python climate_tail_dispatch_real_data.py

Expected output folder
----------------------
climate_tail_dispatch_real_data_outputs/
with subfolders:
- plots/
- tables/
- daily_results/
- artifacts/
- cache/

Main result files to inspect after the run
------------------------------------------
- tables/backtest_summary_by_method.csv
- tables/paired_improvement_vs_proposed.csv
- daily_results/daily_backtest_results_full.csv
- plots/06_daily_total_cost.png
- plots/07_daily_load_shed.png
- plots/08_daily_reserve_slack.png
- plots/09_extreme_day_dispatch.png

Honest modeling note
--------------------
This is a strong real-data pseudo-operational backtest, but it uses historical realized weather covariates rather than archived day-ahead NWP forecasts. That is acceptable for method verification and reproducibility, but for final journal submission one could upgrade the weather side to archived forecast inputs.
