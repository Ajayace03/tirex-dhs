"""Diagnostic: how much can TiRex-2 gain from past covariates at all?

Plain TiRex-2 is scored on the test windows (every 4th origin) with four covariate sets:
  none        the target alone (univariate)
  clients     the 100 client series used in this project
  oracle      one synthetic leading indicator: the target's own value 60 days
              ahead, with 2% multiplicative noise. No real covariate carries more
              information about the next 60 days.
  oracle+100  the oracle plus the 100 clients
If even the oracle barely helps, the model cannot use past-only covariates
much, and a selector that drops them loses nothing while saving their compute.

Run: python -m tirex_gather.covariate_probe   (writes outputs/covariate_probe.csv)
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .config import load_settings
from .data import load_panel
from .metrics import score_block
from .reporting import write_rows
from .runtime import load_backbone

LEAD_DAYS = 60
NOISE = 0.02


def main() -> None:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    from tirex2 import TimeseriesType

    settings = load_settings()
    torch.set_num_threads(settings.threads)
    panel = load_panel(settings)
    backbone, quantiles = load_backbone(settings.model_source, settings.device)
    target = panel.load_kwh.astype(np.float64)
    rng = np.random.default_rng(settings.seed)
    lead = np.concatenate((target[LEAD_DAYS:], target[-LEAD_DAYS:])) * (1 + NOISE * rng.standard_normal(len(target)))

    origins = panel.test_origins(settings)[::4]
    sets = {"none": [], "clients": [], "oracle": [], "oracle+100": []}
    for origin in origins:
        y, clients = panel.context(origin, settings.context_days)
        oracle = lead[origin - settings.context_days:origin].astype(np.float32)[None]
        for name, covariates in (("none", None), ("clients", clients), ("oracle", oracle),
                                 ("oracle+100", np.vstack((oracle, clients)))):
            series = TimeseriesType(target=torch.tensor(y).view(1, -1),
                                    past_covariates=None if covariates is None else torch.tensor(covariates),
                                    future_covariates=None)
            with torch.no_grad():
                out = backbone._predict_once([series], settings.max_horizon, tta_diff=False, pad_context=False)
            sets[name].append(out[0][0].numpy())

    rows = []
    for horizon in settings.horizons:
        actual = np.stack([panel.actuals(o, horizon) for o in origins])
        reference = None
        for name, forecasts in sets.items():
            stats = score_block(actual, np.stack(forecasts)[:, :, :horizon], quantiles)
            reference = reference or stats["crps_kwh"]
            rows.append({"horizon_days": horizon, "covariates": name, "windows": len(origins),
                         "crps_kwh": stats["crps_kwh"], "mae_kwh": stats["mae_kwh"],
                         "crps_change_vs_none_pct": (stats["crps_kwh"] / reference - 1) * 100})
            print(f"h={horizon:>2} {name:<11} CRPS={stats['crps_kwh']:>9.0f} ({rows[-1]['crps_change_vs_none_pct']:+.2f}%)")
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    write_rows(settings.reports_dir / "covariate_probe.csv", rows)


if __name__ == "__main__":
    main()
