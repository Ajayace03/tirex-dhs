"""Point and probabilistic scores for quantile forecasts."""
from __future__ import annotations

import numpy as np


def _median_index(quantiles: np.ndarray) -> int:
    return int(np.argmin(np.abs(quantiles - 0.5)))


def crps(actual: np.ndarray, forecast: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Quantile-based CRPS per (window, step): 2 x mean pinball loss over the quantile levels.

    actual is (windows, steps); forecast is (windows, quantiles, steps).
    """
    spread = actual[..., None] - np.transpose(forecast, (0, 2, 1))
    taus = quantiles.reshape(1, 1, -1)
    pinball = np.maximum(taus * spread, (taus - 1.0) * spread)
    return 2.0 * pinball.mean(axis=-1)


def score_block(actual: np.ndarray, forecast: np.ndarray, quantiles: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if forecast.shape[0] != actual.shape[0] or forecast.shape[2] != actual.shape[1]:
        raise ValueError(f"Shape mismatch actual {actual.shape} forecast {forecast.shape}")
    if forecast.shape[1] != quantiles.shape[0]:
        raise ValueError("Forecast quantile axis does not match the model quantiles")

    point = forecast[:, _median_index(quantiles), :]
    error = point - actual
    absolute = np.abs(error)
    step_crps = crps(actual, forecast, quantiles)
    lower = forecast[:, 0, :]
    upper = forecast[:, -1, :]
    covered = (actual >= lower) & (actual <= upper)
    return {
        "windows": int(actual.shape[0]),
        "steps": int(actual.shape[1]),
        "mae_kwh": float(absolute.mean()),
        "rmse_kwh": float(np.sqrt(np.mean(error ** 2))),
        "crps_kwh": float(step_crps.mean()),
        "bias_kwh": float(error.mean()),
        "mape_pct": float(np.mean(absolute / np.maximum(np.abs(actual), 1.0)) * 100.0),
        "coverage_80_pct": float(covered.mean() * 100.0),
        "width_80_kwh": float((upper - lower).mean()),
        "window_mae_kwh": absolute.mean(axis=1),
        "window_crps_kwh": step_crps.mean(axis=1),
    }


def interval_table(actual: np.ndarray, forecast: np.ndarray, quantiles: np.ndarray) -> list[dict]:
    """Observed coverage of every central interval the quantiles can form."""
    rows = []
    for low_i in range(len(quantiles) // 2):
        high_i = len(quantiles) - 1 - low_i
        lower = forecast[:, low_i, :]
        upper = forecast[:, high_i, :]
        covered = (actual >= lower) & (actual <= upper)
        rows.append(
            {
                "nominal_coverage_pct": float(quantiles[high_i] - quantiles[low_i]) * 100.0,
                "observed_coverage_pct": float(covered.mean() * 100.0),
                "mean_width_kwh": float((upper - lower).mean()),
            }
        )
    return rows
