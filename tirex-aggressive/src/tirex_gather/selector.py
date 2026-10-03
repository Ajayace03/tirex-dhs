"""Dynamic horizon-aware covariate selector (TiRex-DHS, steps 4-6).

For covariate i, forecast horizon h, and the current window:

    g_ih = sigmoid( v . tanh(W_x x_i + W_h h_t + W_e e_h + b) + c )

x_i is covariate i's representation after the first TiRex-2 xLSTM, averaged
over the context patches. h_t is the target's state at the last context patch.
e_h is a learned horizon embedding. The tanh layer lets context and horizon
change the ranking of covariates. In the purely linear form
sigmoid(W_x x_i + W_h h_t + W_e e_h + b), the h_t and e_h terms are the same
for every covariate, so they could only move all gates together.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DHSSelector(nn.Module):
    def __init__(self, embedding_dim: int, n_horizons: int, initial_gate: float, hidden: int = 64, horizon_dim: int = 16):
        super().__init__()
        self.horizon = nn.Embedding(n_horizons, horizon_dim)
        self.w_x = nn.Linear(embedding_dim, hidden, bias=False)
        self.w_h = nn.Linear(embedding_dim, hidden, bias=False)
        self.w_e = nn.Linear(horizon_dim, hidden, bias=True)
        self.v = nn.Linear(hidden, 1)
        nn.init.normal_(self.v.weight, std=0.01)
        # Gates start near initial_gate so training begins close to plain TiRex-2.
        nn.init.constant_(self.v.bias, math.log(initial_gate / (1.0 - initial_gate)))

    @staticmethod
    def features(first_time: torch.Tensor, context_patches: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Covariate representations [N, D] and the target state [D] from the first xLSTM output."""
        context = first_time[:, :context_patches]
        covariates = F.layer_norm(context[1:].mean(dim=1), (context.shape[-1],))
        state = F.layer_norm(context[0, -1], (context.shape[-1],))
        return covariates, state

    def logits(self, covariates: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Gate logits [H, N]."""
        base = self.w_x(covariates) + self.w_h(state)
        hidden = torch.tanh(base.unsqueeze(0) + self.w_e(self.horizon.weight).unsqueeze(1))
        return self.v(hidden).squeeze(-1)


def context_weight(state: torch.Tensor, previous_state: torch.Tensor) -> torch.Tensor:
    """w_t = exp(-||h_t - h_{t-1}||) on unit-length states, so w_t is in [exp(-2), 1]."""
    a = F.normalize(state, dim=-1)
    b = F.normalize(previous_state, dim=-1)
    return torch.exp(-torch.linalg.vector_norm(a - b))
