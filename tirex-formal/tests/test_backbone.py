"""The staged, gated forward pass agrees with TiRex-2. Loads the model (a few seconds)."""
from __future__ import annotations

import os

import torch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from tirex_formal import backbone as staged  # noqa: E402
from tirex_formal.config import load_settings  # noqa: E402
from tirex_formal.data import load_panel  # noqa: E402
from tirex_formal.runtime import load_backbone  # noqa: E402


def _setup():
    settings = load_settings()
    panel = load_panel(settings)
    model, _ = load_backbone(settings.model_source, "cpu")
    target, covariates = panel.context(900, settings.context_days)
    return model, target, covariates


def test_no_gate_matches_tirex2_and_zero_gate_matches_removal():
    from tirex2 import TimeseriesType

    model, target, covariates = _setup()
    with torch.no_grad():
        series = TimeseriesType(target=torch.tensor(target).view(1, -1), past_covariates=torch.tensor(covariates),
                                future_covariates=None)
        reference = model._predict_once([series], 60, tta_diff=False, pad_context=False)[0][0]
        prepared = staged.prepare(model, target, covariates, 60)
        embedded = staged.embed(model, prepared)
        n = prepared.context.shape[0]
        rows = torch.arange(n)
        mine = staged.decode_target(model, prepared, staged.run_stack(
            model, embedded, rows, torch.ones(n), prepared.target_mask, None, 60)[0], n)
        assert torch.allclose(mine, reference, rtol=1e-5, atol=1.0)

        bias = torch.zeros(n)
        bias[51:] = -1e9
        silenced = staged.decode_target(model, prepared, staged.run_stack(
            model, embedded, rows, torch.ones(n), prepared.target_mask, bias, 60)[0], n)
        keep = torch.arange(51)
        removed = staged.decode_target(model, prepared, staged.run_stack(
            model, embedded, keep, torch.ones(51), prepared.target_mask[keep], None, 60)[0], n)
        assert torch.allclose(silenced, removed, rtol=1e-5, atol=1.0)
