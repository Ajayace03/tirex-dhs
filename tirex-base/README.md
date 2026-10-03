# TiRex-2 baseline (all covariates)

This is the reference point for the TiRex-DHS study. It forecasts daily electricity demand with the frozen NX-AI/TiRex-2 model. The target and all 100 covariate series go through all 12 TiRex-2 blocks in one forward pass per forecast window. Nothing is selected and nothing is trained.

This project is what the research hypotheses call "TiRex-2 without the proposed mechanism". The two TiRex-DHS projects are compared against it on accuracy and on compute: time, FLOPs, and memory.

## The task

| | |
|---|---|
| Data | UCI ElectricityLoadDiagrams20112014: 370 clients, 15-minute readings, 2011–2014 |
| Panel | the 320 clients that already record on 2012-01-01; 1,096 days (2012-01-01 to 2014-12-31) |
| Target | daily energy (kWh) of 220 of those clients |
| Covariates | daily energy of the other 100 clients (a seeded random draw). None of them is part of the target sum. |
| Context | 192 days |
| Horizons | 7, 30, and 60 days, each scored on every day up to the horizon |
| Test windows | 161 daily origins from day 876 (2014-05-26) onward. Each window's 60 days fall inside the last 20% of the data. |
| Quantiles | 0.1 … 0.9 (TiRex-2's nine) |

The baseline makes one 60-day forecast per window. The 7- and 30-day scores use its first 7 and 30 days.

## Layout

```
data/              daily_load.csv (target), client_daily.npz (100 covariates), data_card.json
model/             TiRex-2 checkpoint (model.ckpt, model-config.yaml), so no download is needed
src/tirex_base/
  prepare_data.py  rebuilds data/ from the raw LD2011_2014.txt
  config.py        every setting in one place
  data.py          panel loading and the chronological train/validation/test windows
  forecast.py      the TiRex-2 forward pass
  metrics.py       MAE, RMSE, CRPS, bias, MAPE, interval coverage
  runtime.py       model loading, timing, FLOP counting, memory sampling
  evaluation.py    scoring and the files under outputs/
  pipeline.py      the run
tests/             metric and data checks
outputs/           results of the last run (see below)
setup.ps1          creates .venv and installs the pinned packages
run.ps1            runs the project in its own .venv
```

## Run

In PowerShell, from this folder:

```powershell
.\run.ps1            # full test run, about 5 minutes on an 8-core CPU
.\run.ps1 --quick    # 5 test windows, to check the setup
```

On a PC with an NVIDIA GPU and the CUDA build of PyTorch, add `--device cuda`. TiRex-2's pure-PyTorch kernels run on the GPU; no Triton is needed. The published results are CPU runs. On the development laptop, Windows Application Control blocked PyTorch's CUDA library, so the GPU path is untested there.

`run.ps1` uses `.venv`. If `.venv` is missing or doesn't start (for example after the folder was copied to another PC), it runs `setup.ps1` first. That needs Python 3.11 or newer and an internet connection once.

Run the tests:

```powershell
$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m pytest
```

Rebuild the data from the raw UCI file (optional; the built files are already in `data/`):

```powershell
$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m tirex_base.prepare_data C:\path\to\LD2011_2014.txt
```

## Outputs

| File | Contents |
|---|---|
| `metrics_by_horizon.csv` | MAE, RMSE, CRPS, bias, MAPE, 80% interval coverage and width, covariates used |
| `window_metrics.csv` | MAE and CRPS of every test window and horizon |
| `window_timing.csv` | wall-clock seconds of every test window |
| `efficiency.json` | seconds per window, GFLOPs per window, peak memory added during the test loop, series processed |
| `interval_calibration.csv` | observed coverage of the 20/40/60/80% central intervals (60-day horizon) |
| `forecasts.npz` | every quantile forecast and the matching actuals |
| `run_report.json`, `run.log` | settings, dataset, all metrics |
| `figures/` | metrics by horizon, MAE per window, interval calibration |

How the measurements are made:
- **Time:** wall-clock seconds for everything one forecast window needs, on 8 CPU threads. The model is warmed up first.
- **FLOPs:** counted with PyTorch's operator-level counter on 3 windows.
- **Memory:** the peak resident memory during the test loop, minus the memory before it.
- **CRPS:** twice the mean pinball loss over the nine quantile levels.

## Results

Run of 2026-10-03 on an 8-thread laptop CPU, 161 test windows (2014-05-26 to 2014-11-02).

| Horizon | MAE (MWh/day) | RMSE (MWh/day) | CRPS (MWh/day) | MAPE | 80% interval coverage |
|---|---|---|---|---|---|
| 7 days | 217.6 | 302.8 | 177.1 | 5.70% | 76.1% |
| 30 days | 429.8 | 532.9 | 337.8 | 11.49% | 64.8% |
| 60 days | 574.7 | 686.8 | 449.2 | 15.83% | 63.0% |

| Cost per forecast window (all three horizons) | |
|---|---|
| Time | 1.85 s (median 1.83 s) |
| Compute | 255.1 GFLOPs |
| Series through the 12 blocks | 101 (target + 100 covariates) |
| Peak memory added during the test loop | 159 MB |

The 80% intervals cover less than 80% at 30 and 60 days, so TiRex-2's bands are somewhat narrow on this test period. The comparison of all three projects is in `../tirex-comparison/results/results.md`.
