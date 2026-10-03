"""Train the selector, tune it on validation, run hard selection on the test windows, and report."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import Settings, load_settings
from .data import load_panel
from .evaluation import efficiency_summary, score, write_outputs
from .forecast import DHSModel
from .reporting import write_gate_charts, write_rows
from .runtime import (MemorySampler, configure_process, count_flops, gpu_peak_mb, load_backbone,
                      reset_gpu_peak, synchronize)
from .selector import DHSSelector
from .training import train_selector

TITLE = "TiRex-DHS hard selection"


def run_windows(model: DHSModel, panel, settings: Settings, origins, tau: float):
    """Hard-selection forecasts. The timer covers everything a deployment would run per window."""
    forecasts = {h: [] for h in settings.horizons}
    gates, kept, seconds = [], [], []
    with torch.no_grad():
        for origin in origins:
            started = time.perf_counter()
            prepared, embedded = model.window(*panel.context(origin, settings.context_days))
            logits = model.logits(embedded)
            predictions, chosen = model.hard(prepared, embedded, logits, tau)
            synchronize(settings.device)
            seconds.append(time.perf_counter() - started)
            for h, prediction in zip(settings.horizons, predictions):
                forecasts[h].append(prediction.cpu().numpy())
            g = torch.sigmoid(logits).cpu().numpy()
            mask = np.zeros_like(g, dtype=np.int8)
            for h_index, c in enumerate(chosen):
                mask[h_index, c.cpu().numpy()] = 1
            gates.append(g)
            kept.append(mask)
    return {h: np.stack(v) for h, v in forecasts.items()}, np.stack(gates), np.stack(kept), seconds


def validation_crps(panel, origins, forecasts, quantiles, horizons) -> list[float]:
    rows, _, _ = score(panel, origins, forecasts, quantiles, horizons)
    return [row["crps_kwh"] for row in rows]


def tune(backbone, panel, settings: Settings, quantiles, logger):
    """Fit one selector per lambda_sparse, try every tau, and pick a configuration on validation only.

    Rule: among configurations whose mean validation CRPS is within accuracy_tolerance of the best,
    take the one that keeps the fewest covariates (then the lower CRPS).
    """
    val_origins = panel.val_origins(settings)
    rows, selectors, histories = [], {}, []
    for lambda_sparse in settings.lambda_sparse_grid:
        selector, info = train_selector(backbone, panel, settings, quantiles, lambda_sparse, logger)
        selectors[lambda_sparse] = selector
        histories.extend(info["history"])
        model = DHSModel(backbone, selector, settings.horizons)
        for tau in settings.tau_grid:
            forecasts, _, kept, seconds = run_windows(model, panel, settings, val_origins, tau)
            crps = validation_crps(panel, val_origins, forecasts, quantiles, settings.horizons)
            row = {
                "lambda_sparse": lambda_sparse,
                "lambda_stable": settings.lambda_stable,
                "tau": tau,
                **{f"val_crps_h{h}": c for h, c in zip(settings.horizons, crps)},
                "val_crps_mean": float(np.mean(crps)),
                "val_active_covariates": float(kept.sum(axis=2).mean()),
                "val_seconds_per_window": float(np.mean(seconds)),
                "train_seconds": info["seconds"],
            }
            rows.append(row)
            logger.info("validation lambda_sparse=%s tau=%s CRPS=%.0f active=%.1f %.3fs/window",
                        lambda_sparse, tau, row["val_crps_mean"], row["val_active_covariates"],
                        row["val_seconds_per_window"])
    best = min(row["val_crps_mean"] for row in rows)
    eligible = [row for row in rows if row["val_crps_mean"] <= best * (1 + settings.accuracy_tolerance)]
    chosen = min(eligible, key=lambda row: (row["val_active_covariates"], row["val_crps_mean"]))
    for row in rows:
        row["chosen"] = int(row is chosen)
    return chosen, selectors[chosen["lambda_sparse"]], rows, histories, len(val_origins)


def run(output_dir: Path | None = None, quick: bool = False, test_only: bool = False, device: str = "cpu") -> Path:
    settings = load_settings(output_dir, quick, device)
    reports = settings.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    logger = configure_process(reports / ("test_run.log" if test_only else "run.log"), settings.threads)
    panel = load_panel(settings)
    test_origins = panel.test_origins(settings)
    logger.info("panel days=%s covariates=%s train=%s val=%s test=%s windows",
                panel.n_days, panel.n_covariates, len(panel.train_origins(settings)),
                len(panel.val_origins(settings)), len(test_origins))

    backbone, quantiles = load_backbone(settings.model_source, settings.device)
    if test_only:
        saved = torch.load(reports / "selector.pt", weights_only=False, map_location="cpu")
        selector = DHSSelector(backbone.embedding_dim, len(settings.horizons), settings.initial_gate)
        selector.load_state_dict(saved["state_dict"])
        selector.to(settings.device)
        selector.eval()
        chosen, tuning_seconds, n_val = saved["config"], saved["tuning_seconds"], saved["validation_windows"]
        logger.info("test only: reusing %s", reports / "selector.pt")
    else:
        tuning_started = time.perf_counter()
        chosen, selector, tuning_rows, histories, n_val = tune(backbone, panel, settings, quantiles, logger)
        tuning_seconds = time.perf_counter() - tuning_started
        write_rows(reports / "tuning.csv", tuning_rows)
        write_rows(reports / "training_history.csv", histories)
        torch.save({"state_dict": {k: v.cpu() for k, v in selector.state_dict().items()}, "config": chosen, "tuning_seconds": tuning_seconds,
                    "validation_windows": n_val}, reports / "selector.pt")
    tau = chosen["tau"]
    logger.info("chosen lambda_sparse=%s tau=%s", chosen["lambda_sparse"], tau)

    model = DHSModel(backbone, selector, settings.horizons)
    run_windows(model, panel, settings, test_origins[:1], tau)
    reset_gpu_peak(settings.device)
    with MemorySampler() as memory:
        forecasts, gates, kept, seconds = run_windows(model, panel, settings, test_origins, tau)

    def one_window(origin):
        prepared, embedded = model.window(*panel.context(origin, settings.context_days))
        model.hard(prepared, embedded, model.logits(embedded), tau)

    memory.gpu_peak_mb = gpu_peak_mb(settings.device)
    flops = [count_flops(lambda o=o: one_window(o)) for o in test_origins[:settings.flop_windows]]
    active = kept.sum(axis=2)
    horizon_rows, window_rows, intervals = score(panel, test_origins, forecasts, quantiles, settings.horizons, active)
    for row in horizon_rows:
        logger.info("h=%s MAE=%.0f CRPS=%.0f coverage80=%.1f active=%.1f", row["horizon_days"], row["mae_kwh"],
                    row["crps_kwh"], row["coverage_80_pct"], row["active_covariates"])

    changed = np.abs(np.diff(kept.astype(int), axis=0)).sum(axis=2).mean(axis=0) if len(kept) > 1 else np.zeros(len(settings.horizons))
    efficiency = efficiency_summary(
        seconds, flops, memory, panel, active, settings.horizons,
        {
            "series_through_stack_per_window": float((active + 1).sum(axis=1).mean()),
            "selector_tuning_seconds": tuning_seconds,
            "selector_parameters": int(sum(p.numel() for p in selector.parameters())),
            **{f"selection_changes_per_day_h{h}": float(c) for h, c in zip(settings.horizons, changed)},
        },
    )
    np.savez(reports / "gates.npz", gates=gates, kept=kept, names=np.array(panel.client_names),
             horizons=np.array(settings.horizons), origin_dates=panel.dates[test_origins].astype(str))
    report = {
        "project": "tirex-aggressive-gather",
        "method": "TiRex-DHS with hard selection: covariates with g <= tau are removed before every variate "
                  "attention, so the 12 TiRex-2 blocks only process the target and the kept covariates",
        "model": settings.model_source,
        "device": settings.device,
        "settings": {
            "context_days": settings.context_days,
            "horizons_days": list(settings.horizons),
            "lambda_sparse_grid": list(settings.lambda_sparse_grid),
            "lambda_stable": settings.lambda_stable,
            "tau_grid": list(settings.tau_grid),
            "chosen": chosen,
            "epochs": settings.epochs,
            "covariate_subset": settings.covariate_subset,
            "threads": settings.threads,
            "validation_windows": n_val,
            "quick": settings.quick,
        },
        "dataset": {"days": panel.n_days, "covariates": panel.n_covariates,
                    "start": str(panel.dates[0]), "end": str(panel.dates[-1])},
    }
    write_outputs(reports, TITLE, panel, test_origins, forecasts, quantiles, settings.horizons,
                  horizon_rows, window_rows, intervals, seconds, efficiency, report)
    write_gate_charts(reports, gates, kept, panel.client_names, settings.horizons,
                      panel.dates[test_origins], TITLE)
    logger.info("test %.3fs/window, %.1f GFLOPs/window, +%.0f MB, wrote %s",
                efficiency["seconds_per_window_mean"], efficiency["gflops_per_window"],
                efficiency["peak_added_memory_mb"], reports)
    return reports
