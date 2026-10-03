"""Score test forecasts and write the files the comparison folder reads."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import interval_table, score_block
from .reporting import write_charts, write_json, write_rows


def score(panel, origins, forecasts: dict[int, np.ndarray], quantiles, horizons, active=None):
    """forecasts[h] is [windows, Q, >=h]. active (optional) is [windows, horizons] covariate counts."""
    horizon_rows, window_rows = [], []
    for h_index, horizon in enumerate(horizons):
        actual = np.stack([panel.actuals(o, horizon) for o in origins])
        stats = score_block(actual, forecasts[horizon][:, :, :horizon], quantiles)
        row = {
            "horizon_days": horizon,
            "windows": stats["windows"],
            "mae_kwh": stats["mae_kwh"],
            "rmse_kwh": stats["rmse_kwh"],
            "crps_kwh": stats["crps_kwh"],
            "bias_kwh": stats["bias_kwh"],
            "mape_pct": stats["mape_pct"],
            "coverage_80_pct": stats["coverage_80_pct"],
            "width_80_kwh": stats["width_80_kwh"],
            "active_covariates": float(np.mean(active[:, h_index])) if active is not None else float(panel.n_covariates),
            "active_covariates_pct": (float(np.mean(active[:, h_index])) if active is not None
                                      else float(panel.n_covariates)) / panel.n_covariates * 100.0,
        }
        horizon_rows.append(row)
        for w, origin in enumerate(origins):
            window_rows.append(
                {
                    "horizon_days": horizon,
                    "origin_date": str(panel.dates[origin]),
                    "window_mae_kwh": float(stats["window_mae_kwh"][w]),
                    "window_crps_kwh": float(stats["window_crps_kwh"][w]),
                    "active_covariates": int(active[w, h_index]) if active is not None else panel.n_covariates,
                }
            )
    longest = max(horizons)
    actual_long = np.stack([panel.actuals(o, longest) for o in origins])
    intervals = interval_table(actual_long, forecasts[longest][:, :, :longest], quantiles)
    return horizon_rows, window_rows, intervals


def write_outputs(reports: Path, title: str, panel, origins, forecasts, quantiles, horizons,
                  horizon_rows, window_rows, intervals, seconds, efficiency: dict, report: dict) -> None:
    write_rows(reports / "metrics_by_horizon.csv", horizon_rows)
    write_rows(reports / "window_metrics.csv", window_rows)
    write_rows(reports / "interval_calibration.csv", intervals)
    write_rows(
        reports / "window_timing.csv",
        [{"origin_date": str(panel.dates[o]), "seconds": float(s)} for o, s in zip(origins, seconds)],
    )
    np.savez(
        reports / "forecasts.npz",
        **{f"h{h}": forecasts[h][:, :, :h] for h in horizons},
        **{f"actual_h{h}": np.stack([panel.actuals(o, h) for o in origins]) for h in horizons},
        origins=np.asarray(origins),
        origin_dates=panel.dates[origins].astype(str),
        quantiles=np.asarray(quantiles),
    )
    write_json(reports / "efficiency.json", efficiency)
    write_json(reports / "run_report.json", {**report, "efficiency": efficiency, "metrics": horizon_rows})
    write_charts(reports, horizon_rows, window_rows, intervals, title)


def efficiency_summary(seconds, flops, memory, panel, active, horizons, extra: dict) -> dict:
    seconds = np.asarray(seconds)
    summary = {
        "test_windows": int(len(seconds)),
        "seconds_per_window_mean": float(seconds.mean()),
        "seconds_per_window_median": float(np.median(seconds)),
        "seconds_total": float(seconds.sum()),
        "gflops_per_window": float(np.mean(flops) / 1e9),
        "peak_added_memory_mb": float(memory.added_mb),
        "baseline_memory_mb": float(memory.baseline_mb),
        "peak_gpu_memory_mb": float(getattr(memory, "gpu_peak_mb", 0.0)),
        "covariates_available": panel.n_covariates,
    }
    for h_index, horizon in enumerate(horizons):
        mean_active = float(np.mean(active[:, h_index])) if active is not None else float(panel.n_covariates)
        summary[f"active_covariates_h{horizon}"] = mean_active
        summary[f"active_covariates_pct_h{horizon}"] = mean_active / panel.n_covariates * 100.0
    summary.update(extra)
    return summary
