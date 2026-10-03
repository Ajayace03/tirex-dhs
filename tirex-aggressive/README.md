# TiRex-DHS with hard selection (aggressive gather)

This is TiRex-DHS (Dynamic Horizon-aware Sparse covariate selection) built for **lower compute**. A small learned selector scores every covariate for the current window and for each forecast horizon. Covariates it does not need are removed before TiRex-2's cross-series attention, so all 12 blocks only process the target and the covariates it kept. When no covariate is worth it, the target runs alone.

TiRex-2 stays frozen; only the selector is trained (about 67k parameters). The task, data, and test windows are the same as in `tirex-base`, so the results can be compared directly.

## How it works

```
192-day context ──► scale + patch + embed ──► first xLSTM (time mixer), all 101 series
                                                      │
                     ┌────────────────────────────────┤
                     ▼                                ▼
     selector: g_ih = σ(v·tanh(W_x x_i + W_h h_t + W_e e_h + b) + c)
          x_i  covariate i after the first xLSTM (mean over context patches)
          h_t  the target's current state (last context patch)
          e_h  learned embedding of horizon h ∈ {7, 30, 60}
                     │
                     ▼
     keep covariate i for horizon h only if g_ih > τ   (g̃ = g · 1[g > τ])
                     │
                     ▼
     target + kept covariates ──► 12 × [variate attention (+log g̃) → xLSTM] ──► quantile head
     (the three horizons run as three attention groups in one batched pass)
```

**Where the gate acts.** In TiRex-2, a covariate reaches the target only through the variate (cross-series) attention. So the gate is applied there: covariate i adds `log g_ih` to its attention logit in every block.
- With g = 1, the model is exactly TiRex-2.
- As g → 0, the result is identical to deleting the series. `tests/test_backbone.py` checks both.

Simply scaling the covariate (x̃ = g·x) would not work, because each attention block RMS-normalises its input and cancels the scale.

**Why a tanh layer.** In the purely linear score σ(W_x x_i + W_h h_t + W_e e_h + b), the context term and the horizon term are the same for every covariate. They could move all gates up or down together, but never change which covariates rank first. The tanh layer lets the current context and the horizon change the ranking. That is the "dynamic, horizon-aware" property.

**Training** (TiRex-2 frozen, Adam, 2 epochs over 74 training windows):

```
L = L_forecast + λ1 · L_sparse + λ2 · L_stable
L_forecast  pinball loss, all 9 quantiles, every day 1..h of each horizon, divided by the window's mean target level
L_sparse    mean gate over all covariates and horizons (L1, since g ≥ 0)
L_stable    w_t · mean|g_t − g_{t−1}| with w_t = exp(−‖h_t − h_{t−1}‖): gates should change only when the context does
```

- Gates start near 0.9, so training begins close to plain TiRex-2.
- λ1 ramps up over the first 30% of steps so the gates don't collapse early.
- Each step trains on a random subset of 32 covariates, so backpropagating through the 12-block stack stays affordable on a CPU. The selector is shared across covariates, so it learns to score any of them.

**Tuning on validation only:**
- One selector is trained for each λ1 in {0, 0.001, 0.01, 0.1}, with λ2 = 0.01.
- Each is tried with τ in {0.3, 0.5, 0.7} on the 50 validation windows (days 767–875).
- Among the configurations whose validation CRPS is within 2% of the best, the one that keeps the fewest covariates wins.
- The test windows are used once, at the end.

## Layout

```
data/  model/          same data and TiRex-2 checkpoint as tirex-base
src/tirex_gather/
  backbone.py          TiRex-2 forward pass split into stages, with the gated variate attention
  selector.py          the DHS selector and the context weight w_t
  training.py          the loss above and the training loop
  forecast.py          soft path (training) and hard selection (inference)
  pipeline.py          train → tune on validation → timed test run → reports
  config.py data.py metrics.py runtime.py evaluation.py reporting.py prepare_data.py
tests/                 metrics, data splits, selector, and staged-model equivalence checks
outputs/               results of the last run
setup.ps1  run.ps1     environment and launcher
```

## Run

```powershell
.\run.ps1               # train, tune, and test: about 1.5 hours on an 8-core CPU
.\run.ps1 --test-only   # reuse outputs\selector.pt and redo only the timed test run (about 2 minutes)
.\run.ps1 --quick       # tiny run to check the setup
$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m pytest
```

On a PC with an NVIDIA GPU and the CUDA build of PyTorch, add `--device cuda`. TiRex-2's pure-PyTorch kernels run on the GPU; no Triton is needed. The published results are CPU runs. On the development laptop, Windows Application Control blocked PyTorch's CUDA library, so the GPU path is untested there.

`run.ps1` builds `.venv` with `setup.ps1` when it is missing or broken (needs Python 3.11+ and internet once).

## Outputs

These are the same files as `tirex-base`, plus:

| File | Contents |
|---|---|
| `tuning.csv` | validation CRPS, covariates kept, and speed for every λ1 and τ; the chosen row is marked |
| `training_history.csv` | loss terms and mean gate per epoch for every λ1 |
| `selector.pt` | the trained selector and its chosen configuration |
| `gates.npz` | gate of every covariate, horizon, and test window, and which covariates were kept |
| `figures/gate_heatmap.png` | mean gate per horizon for the 30 highest-scoring covariates |
| `figures/active_covariates.png` | how many covariates each test window kept, per horizon |

`efficiency.json` also records:
- the number of series pushed through the stack per window;
- how often the selection changes from one day to the next;
- the one-off selector training time.

## Results

Run of 2026-10-03 on an 8-thread laptop CPU, same 161 test windows as `tirex-base`. Timings come from `--test-only`, run alone on the machine.

**What the selector learned.** For every λ1 in {0, 0.001, 0.01, 0.1}, training drove all 100 gates to about 0. That includes λ1 = 0, where the only pressure is forecast accuracy. So no covariate is kept for any window or horizon, and the target runs through TiRex-2 alone. The training loss agrees: it was lower with the covariates switched off (0.0358) than at the start with them on (0.0366). On this data, the 100 client series cost compute and buy no accuracy. The selector learns to skip them.

| Validation (50 windows) | Covariates kept | Mean CRPS (MWh) |
|---|---|---|
| every λ1 and τ | 0 | 90.3 |

All configurations tie. The rule (fewest covariates within 2% of the best CRPS, then lowest CRPS) took the first one: λ1 = 0, τ = 0.3.

**Test results**

| Horizon | MAE (MWh/day) | CRPS (MWh/day) | CRPS vs. base | 80% coverage | Covariates kept |
|---|---|---|---|---|---|
| 7 days | 219.0 | 178.4 | +0.75% | 75.0% | 0 of 100 |
| 30 days | 431.3 | 340.4 | +0.77% | 63.0% | 0 of 100 |
| 60 days | 579.5 | 454.3 | +1.14% | 61.7% | 0 of 100 |

| Cost per forecast window | Gather | Base TiRex-2 |
|---|---|---|
| Time | **0.229 s** | 1.850 s (**8.1x faster**, 87.6% less time) |
| Compute | **19.3 GFLOPs** | 255.1 GFLOPs (**92.4% fewer**) |
| Series through the 12 blocks | 3 (the target once per horizon) | 101 |
| Peak memory added | 153 MB | 159 MB |

None of the accuracy differences from the baseline is statistically significant (Diebold–Mariano, Holm-adjusted p ≥ 0.9). The time saving is (p < 1e-150). Selector training and tuning took 88 minutes once, while sharing the CPU with the other project.

The memory figure is resident memory of the whole process, so it barely moves. The removed covariates' activations are small next to PyTorch's allocator and the 380 MB model.

**Why the covariates are dropped.** `python -m tirex_gather.covariate_probe` (output in `outputs/covariate_probe.csv`) gives plain TiRex-2 an oracle covariate: the target's own value 60 days ahead, with 2% noise. Even that improves the 60-day CRPS by only 1.8%, and the 100 real clients by 0.9%. TiRex-2's attention compares series only at the same time patch, so it gets little out of past-only covariates. A selector that removes them gives up very little accuracy and saves almost all of their compute.
