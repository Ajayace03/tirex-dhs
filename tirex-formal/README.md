# TiRex-DHS with soft gating (formal selector)

This is TiRex-DHS (Dynamic Horizon-aware Sparse covariate selection) exactly as the method description states it: **soft selection, not hard deletion**. A small learned selector gives every covariate a gate g ∈ [0, 1] for the current window and for each forecast horizon. The gate down-weights that covariate inside TiRex-2's cross-series attention. No series is removed, so a "suppressed" covariate can still contribute a little.

This project answers the question: does learned, horizon-aware weighting of covariates change TiRex-2's accuracy? Its sister project `tirex-aggressive-gather` uses the same selector but removes covariates to save compute. Both are compared with `tirex-base` on the same test windows.

TiRex-2 stays frozen; only the selector is trained (about 67k parameters).

## How it works

```
192-day context ──► scale + patch + embed ──► first xLSTM (time mixer), all 101 series
                                                      │
     selector: g_ih = σ(v·tanh(W_x x_i + W_h h_t + W_e e_h + b) + c)
          x_i  covariate i after the first xLSTM (mean over context patches)
          h_t  the target's current state (last context patch)
          e_h  learned embedding of horizon h ∈ {7, 30, 60}
                     │
                     ▼
     all 101 series ──► 12 × [variate attention (+log g_ih per covariate) → xLSTM] ──► quantile head
     one gated pass per horizon (batched as three attention groups)
```

**Where the gate acts.** A covariate reaches the target only through TiRex-2's variate attention. So covariate i adds `log g_ih` to its attention logit in every block, which multiplies its attention weight by g_ih.
- With g = 1, the model is exactly TiRex-2.
- With g → 0, the series is silenced (identical to removing it; `tests/test_backbone.py` checks both).

Scaling the covariate itself (x̃ = g·x) would be cancelled by the RMS normalisation in front of each attention block.

**Why a tanh layer.** In the linear score σ(W_x x_i + W_h h_t + W_e e_h + b), the context term and the horizon term are the same for every covariate. They can only move all gates together, not change which covariates rank first. The tanh layer makes the ranking depend on the context and the horizon.

**Training** (TiRex-2 frozen, Adam, 2 epochs over 74 training windows):

```
L = L_forecast + λ1 · L_sparse + λ2 · L_stable
L_forecast  pinball loss, all 9 quantiles, every day 1..h of each horizon, divided by the window's mean target level
L_sparse    mean gate over all covariates and horizons (L1, since g ≥ 0)
L_stable    w_t · mean|g_t − g_{t−1}| with w_t = exp(−‖h_t − h_{t−1}‖): gates should change only when the context does
```

- Gates start near 0.9.
- λ1 ramps up over the first 30% of steps.
- Each step uses a random subset of 32 covariates, to keep backpropagation through the 12 blocks affordable on a CPU.
- One selector is trained for each λ1 in {0, 0.001, 0.01, 0.1}, with λ2 = 0.01. The one with the lowest CRPS on the 50 validation windows (days 767–875) is kept.
- The test windows are used once, at the end.

**The cost.** Soft gating keeps every series in every block, and the gates differ per horizon. So each window runs the full model once per horizon: about 3× the baseline's compute. That is the price of the exact soft formulation. Hard removal (`tirex-aggressive-gather`) is how the method saves compute.

## Layout

```
data/  model/          same data and TiRex-2 checkpoint as tirex-base
src/tirex_formal/
  backbone.py          TiRex-2 forward pass split into stages, with the gated variate attention
  selector.py          the DHS selector and the context weight w_t
  training.py          the loss above and the training loop
  forecast.py          the soft-gated forecast
  pipeline.py          train → tune on validation → timed test run → reports
  config.py data.py metrics.py runtime.py evaluation.py reporting.py prepare_data.py
tests/                 metrics, data splits, selector, and staged-model equivalence checks
outputs/               results of the last run
setup.ps1  run.ps1     environment and launcher
```

## Run

```powershell
.\run.ps1               # train, tune, and test: about 2 hours on an 8-core CPU
.\run.ps1 --test-only   # reuse outputs\selector.pt and redo only the timed test run (about 15 minutes)
.\run.ps1 --quick       # tiny run to check the setup
$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m pytest
```

On a PC with an NVIDIA GPU and the CUDA build of PyTorch, add `--device cuda`. TiRex-2's pure-PyTorch kernels run on the GPU; no Triton is needed. The published results are CPU runs. On the development laptop, Windows Application Control blocked PyTorch's CUDA library, so the GPU path is untested there.

`run.ps1` builds `.venv` with `setup.ps1` when it is missing or broken (needs Python 3.11+ and internet once).

## Outputs

These are the same files as `tirex-base`, plus:
- `tuning.csv`, `training_history.csv`, and `selector.pt`;
- `gates.npz`: the gate of every covariate, horizon, and window;
- `figures/gate_heatmap.png` and `figures/active_covariates.png`.

Here "active" means g > 0.5. Every covariate is still computed either way.

## Results

Run of 2026-10-03 on an 8-thread laptop CPU, same 161 test windows as `tirex-base`. Timings come from `--test-only`, run alone on the machine.

**What the selector learned.** For every λ1 in {0, 0.001, 0.01, 0.1}, the gates went to about 0 for all 100 covariates (no gate above 0.5 in any window). The selector judged the covariates unhelpful, exactly as the sister project's selector did.

| Validation (50 windows) | λ1 = 0 | 0.001 | 0.01 | 0.1 |
|---|---|---|---|---|
| Mean CRPS (MWh) | **91.7** (chosen) | 91.7 | 91.7 | 91.7 |

**Test results**

| Horizon | MAE (MWh/day) | CRPS (MWh/day) | CRPS vs. base | 80% coverage |
|---|---|---|---|---|
| 7 days | 217.0 | 176.7 | −0.20% | 76.6% |
| 30 days | 428.3 | 339.0 | +0.37% | 64.5% |
| 60 days | 581.1 | 456.0 | +1.52% | 61.4% |

| Cost per forecast window | Soft gating | Base TiRex-2 |
|---|---|---|
| Time | 5.17 s | 1.85 s (2.8x **slower**) |
| Compute | 733.6 GFLOPs | 255.1 GFLOPs (2.9x **more**) |
| Series through the 12 blocks | 303 (all 101, once per horizon) | 101 |
| Peak memory added | 444 MB | 159 MB |

None of the accuracy differences from the baseline is statistically significant after Holm adjustment. The closest is 60-day CRPS, +1.5%, Holm p = 0.07.

**What this shows.** Soft, horizon-aware gating as the method describes it does work as a selector: it finds that the covariates are not needed. It cannot turn that finding into savings, because a soft gate still computes every series, and horizon-specific gates need one pass per horizon. The saving only appears when the gates are applied as hard removal, which is what `tirex-aggressive-gather` does: 8.1x faster at the same accuracy. Selector training and tuning took 100 minutes once, while sharing the CPU with the other project.
