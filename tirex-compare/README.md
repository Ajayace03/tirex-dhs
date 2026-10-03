# TiRex comparison

This folder compares the three independent projects on the same 161 test windows:
- `tirex-base`: plain TiRex-2 with all 100 covariates.
- `tirex-aggressive-gather`: TiRex-DHS with hard selection; unneeded covariates are removed.
- `tirex-formal-selector`: TiRex-DHS with soft gating; covariates are down-weighted but still computed.

The question it answers is the one in the research plan: does dynamic, horizon-aware, sparse covariate selection change TiRex-2's accuracy, and does it save compute?

## Run

The three project folders must sit next to this one and must have been run (each has its own `run.ps1`). Then:

```powershell
.\run.ps1                          # reads ..\tirex-*\outputs and writes results\
.\run.ps1 --projects D:\elsewhere  # if the projects live somewhere else
.\run_all.ps1                      # runs the three projects one after another, then the comparison (about 3 hours)
```

`run.ps1` builds its own small `.venv` (numpy, scipy, matplotlib) when it is missing or broken.

The projects must not be timed at the same time. Run them one after another, as `run_all.ps1` does. Otherwise they compete for the CPU and the seconds-per-window numbers become meaningless. Training (`run.ps1` in each project) can overlap. If it did, redo only the timed test pass with `.\run.ps1 --test-only` in each selector project, one at a time.

## What is compared

| Kind | Measures |
|---|---|
| Accuracy | MAE, RMSE, CRPS, MAPE, bias, and 80% interval coverage at 7, 30, and 60 days; change against the baseline |
| Compute | seconds per window (all three horizons), GFLOPs per window, peak memory added during the test loop, series pushed through the 12 TiRex-2 blocks, covariates kept per horizon, one-off selector training time |
| Significance | for each method against the baseline, horizon, and metric: Diebold–Mariano test on per-window losses, Holm-adjusted over all tests (two-sided, α = 0.05), with a Wilcoxon signed-rank test as a cross-check |

The Diebold–Mariano test uses a Newey–West variance with h − 1 lags, because test windows one day apart share h − 1 forecast days. It also applies the Harvey–Leybourne–Newbold small-sample correction. The Wilcoxon test ignores that overlap, so it overstates significance; it is only a cross-check.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `results.md` | all tables in one page |
| `accuracy_by_horizon.csv` | accuracy of each method and horizon, and the change against the baseline |
| `efficiency.csv` | compute of each method, speed-up and FLOPs saved against the baseline |
| `significance.csv` | DM statistic, p-value, Holm-adjusted p-value, Wilcoxon p-value, verdict |
| `tuning_gather.csv`, `tuning_formal.csv` | the validation results each selector project chose from |
| `figures/accuracy_vs_time.png` | accuracy against seconds per window |
| `figures/accuracy_by_horizon.png` | CRPS and MAE per horizon |
| `figures/compute_cost.png` | time, FLOPs, memory, and series processed |
| `figures/crps_per_window.png` | CRPS of every test window over time |
| `figures/forecasts_one_day_ahead.png` | one-day-ahead median forecasts against the actual load |

## Results

Run of 2026-10-03, 161 daily test windows (2014-05-26 to 2014-11-02), 8-thread laptop CPU. Each project was timed alone.

| | Base TiRex-2 | **DHS hard selection (gather)** | DHS soft gating (formal) |
|---|---|---|---|
| Seconds per window | 1.85 | **0.23 (8.1x faster)** | 5.17 (2.8x slower) |
| GFLOPs per window | 255.1 | **19.3 (−92.4%)** | 733.6 (+188%) |
| Series through the 12 blocks | 101 | **3** | 303 |
| Covariates used (of 100) | 100 | **0** | 0 effective (all still computed) |
| CRPS 7 / 30 / 60 days (MWh) | 177.1 / 337.8 / 449.2 | 178.4 / 340.4 / 454.3 | 176.7 / 339.0 / 456.0 |
| CRPS vs. base | — | +0.8% / +0.8% / +1.1% | −0.2% / +0.4% / +1.5% |
| Any accuracy difference significant? (DM, Holm, α = 0.05) | — | **no** | no |
| Time difference significant? | — | **yes, faster** (p < 1e-150) | yes, slower |

**Answer to the research hypotheses:**
- **Efficiency:** H0 is rejected. Dynamic sparse selection applied as hard removal cuts time by 87.6% and FLOPs by 92.4% against TiRex-2.
- **Accuracy:** H0 is not rejected. No method's accuracy differs significantly from TiRex-2's at any horizon.

The selectors learned that the 100 client covariates add nothing to this forecast. The covariate check shows why: TiRex-2 gains at most 1.8% even from an oracle leading indicator. The full tables are in `results/results.md`.
