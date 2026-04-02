Files:
1. climate_tail_dispatch_simulation.py
   Full Colab-ready simulation code.
2. climate_tail_dispatch_README.txt
   Quick usage notes.

Quick Colab usage:
---------------------------------
# Cell 1: upload or copy the script, then run
!python climate_tail_dispatch_simulation.py

# Optional: heavier paper-quality run
# Edit the config block at the bottom of the script:
# - n_replications = 10 to 20
# - n_scenarios_proposed = 150 to 300
# - n_scenarios_benchmark = 150 to 300
# - train_days = 365 or more
# - test_days = 60 to 120

Outputs created automatically:
---------------------------------
- climate_tail_dispatch_outputs/
  - plots/
  - tables/
  - logs/
  - daily_results/
  - artifacts/
- climate_tail_dispatch_outputs.zip

If the blackouts are too rare and methods look too similar:
---------------------------------
Use a harsher stress-test by editing config:
- thermal_cap = 120.0 or 130.0
- battery_capacity = 160.0 to 180.0
- battery_discharge_limit = 40.0 to 45.0
- stress_levels = ("severe", "extreme")
- risk_lambda = 3.0 to 4.0

Interpretation note:
---------------------------------
The proposed EVT-tail method is meant to outperform mainly on tail-risk metrics
(CVaR, blackout frequency, load shed, reserve coverage) under heavy-tailed stress.
It may not always minimize average operating cost because it deliberately carries
more protection against rare but damaging events.
