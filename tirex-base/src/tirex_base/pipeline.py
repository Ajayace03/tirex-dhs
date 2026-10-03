"""Run plain TiRex-2 on the test windows and write the performance report."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import load_settings
from .data import load_panel
from .evaluation import efficiency_summary, score, write_outputs
from .forecast import BaseForecaster
from .runtime import (MemorySampler, configure_process, count_flops, gpu_peak_mb, load_backbone,
                      reset_gpu_peak, synchronize)

TITLE = "TiRex-2 baseline"


def run(output_dir: Path | None = None, quick: bool = False, device: str = "cpu") -> Path:
    settings = load_settings(output_dir, quick, device)
    reports = settings.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    logger = configure_process(reports / "run.log", settings.threads)
    panel = load_panel(settings)
    origins = panel.test_origins(settings)
    logger.info("panel days=%s covariates=%s test windows=%s (%s to %s)", panel.n_days, panel.n_covariates,
                len(origins), panel.dates[origins[0]], panel.dates[origins[-1]])

    backbone, quantiles = load_backbone(settings.model_source, settings.device)
    forecaster = BaseForecaster(backbone, settings.max_horizon)

    def one_window(origin):
        return forecaster.forecast(*panel.context(origin, settings.context_days))

    with torch.no_grad():
        one_window(origins[0])
        forecasts, seconds = [], []
        reset_gpu_peak(settings.device)
        with MemorySampler() as memory:
            for origin in origins:
                started = time.perf_counter()
                prediction = one_window(origin)
                synchronize(settings.device)
                seconds.append(time.perf_counter() - started)
                forecasts.append(prediction.cpu().numpy())
    stacked = np.stack(forecasts)
    by_horizon = {h: stacked[:, :, :h] for h in settings.horizons}
    memory.gpu_peak_mb = gpu_peak_mb(settings.device)
    flops = [count_flops(lambda o=o: one_window(o)) for o in origins[:settings.flop_windows]]

    horizon_rows, window_rows, intervals = score(panel, origins, by_horizon, quantiles, settings.horizons)
    for row in horizon_rows:
        logger.info("h=%s MAE=%.0f CRPS=%.0f coverage80=%.1f", row["horizon_days"], row["mae_kwh"],
                    row["crps_kwh"], row["coverage_80_pct"])
    efficiency = efficiency_summary(
        seconds, flops, memory, panel, None, settings.horizons,
        {"series_through_stack_per_window": float(panel.n_series), "selector_tuning_seconds": 0.0},
    )
    report = {
        "project": "tirex-base",
        "method": "TiRex-2 forward pass with the target and every covariate in all 12 blocks, "
                  "one pass per window for all horizons",
        "model": settings.model_source,
        "device": settings.device,
        "settings": {
            "context_days": settings.context_days,
            "horizons_days": list(settings.horizons),
            "threads": settings.threads,
            "quick": settings.quick,
        },
        "dataset": {"days": panel.n_days, "covariates": panel.n_covariates,
                    "start": str(panel.dates[0]), "end": str(panel.dates[-1])},
    }
    write_outputs(reports, TITLE, panel, origins, by_horizon, quantiles, settings.horizons,
                  horizon_rows, window_rows, intervals, seconds, efficiency, report)
    logger.info("test %.3fs/window, %.1f GFLOPs/window, +%.0f MB, wrote %s",
                efficiency["seconds_per_window_mean"], efficiency["gflops_per_window"],
                efficiency["peak_added_memory_mb"], reports)
    return reports
