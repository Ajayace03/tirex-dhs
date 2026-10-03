"""The shipped panel loads and the three splits never share a forecast day."""
from __future__ import annotations

from tirex_formal.config import load_settings
from tirex_formal.data import load_panel


def test_splits_are_chronological_and_disjoint():
    settings = load_settings()
    panel = load_panel(settings)
    assert panel.n_days == 1096 and panel.n_covariates == 100
    train, val, test = panel.train_origins(settings), panel.val_origins(settings), panel.test_origins(settings)
    longest = settings.max_horizon
    assert train[-1] + longest <= settings.train_end_index
    assert val[0] >= settings.train_end_index and val[-1] + longest <= settings.val_end_index
    assert test[0] >= settings.val_end_index and test[-1] + longest <= panel.n_days
