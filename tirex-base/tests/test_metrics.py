"""Check the score definitions without loading the model."""
from __future__ import annotations

import numpy as np

from tirex_base.metrics import interval_table, score_block

QUANTILES = np.arange(0.1, 1.0, 0.1)


def test_perfect_forecast_scores_zero():
    actual = np.array([[10.0, 12.0], [8.0, 9.0]])
    forecast = np.repeat(actual[:, None, :], len(QUANTILES), axis=1)
    stats = score_block(actual, forecast, QUANTILES)
    assert stats["mae_kwh"] == 0.0
    assert stats["rmse_kwh"] == 0.0
    assert stats["bias_kwh"] == 0.0
    assert stats["crps_kwh"] == 0.0


def test_crps_grows_with_error():
    actual = np.zeros((1, 3))
    near = np.full((1, len(QUANTILES), 3), 1.0)
    far = np.full((1, len(QUANTILES), 3), 5.0)
    assert score_block(actual, far, QUANTILES)["crps_kwh"] > score_block(actual, near, QUANTILES)["crps_kwh"]


def test_interval_coverage():
    actual = np.zeros((4, 5))
    forecast = np.linspace(-1, 1, len(QUANTILES))[None, :, None].repeat(4, 0).repeat(5, 2)
    rows = interval_table(actual, forecast, QUANTILES)
    assert rows[0]["nominal_coverage_pct"] == 80.0
    assert all(row["observed_coverage_pct"] == 100.0 for row in rows)
