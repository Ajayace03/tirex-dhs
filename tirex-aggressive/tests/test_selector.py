"""Selector shapes and starting point, without the model."""
from __future__ import annotations

import torch

from tirex_gather.selector import DHSSelector, context_weight


def test_gates_start_near_initial_value():
    selector = DHSSelector(embedding_dim=32, n_horizons=3, initial_gate=0.9)
    logits = selector.logits(torch.randn(100, 32), torch.randn(32))
    assert logits.shape == (3, 100)
    assert torch.allclose(torch.sigmoid(logits), torch.full((3, 100), 0.9), atol=0.05)


def test_horizon_can_change_the_ranking():
    torch.manual_seed(0)
    selector = DHSSelector(embedding_dim=8, n_horizons=2, initial_gate=0.5)
    torch.nn.init.normal_(selector.v.weight, std=1.0)
    logits = selector.logits(torch.randn(20, 8), torch.randn(8))
    assert not torch.equal(logits[0].argsort(), logits[1].argsort())


def test_context_weight_is_one_for_identical_states():
    state = torch.randn(16)
    assert torch.isclose(context_weight(state, state), torch.tensor(1.0))
    assert context_weight(state, -state) < 0.2
