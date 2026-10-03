"""TiRex-DHS forecasts: soft gating for training, hard selection (gather) for inference.

Soft: every covariate stays in every block, and covariate i adds log g_ih to
its attention logit. This is the differentiable model used to train the
selector.

Hard (this project's inference path): covariate i is kept for horizon h only
when g_ih > tau, and a kept covariate still carries log g_ih (g~ = g * 1[g > tau]).
Dropped series are gathered out before the first variate attention, so none of
the 12 blocks spends compute on them. If no covariate passes, the target runs
alone, which is plain univariate TiRex-2.

The three horizons run in one batched pass. Each horizon is its own attention
group, so they never see each other's rows.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .backbone import Embedded, Prepared, decode_target, embed, prepare, run_stack
from .selector import DHSSelector


class DHSModel:
    def __init__(self, backbone, selector: DHSSelector, horizons: tuple[int, ...]):
        self.backbone = backbone
        self.selector = selector
        self.horizons = horizons
        self.max_horizon = max(horizons)

    def window(self, target, covariates) -> tuple[Prepared, Embedded]:
        """Frozen part: scaling, patching, and the first xLSTM over every series."""
        with torch.no_grad():
            prepared = prepare(self.backbone, target, covariates, self.max_horizon)
            embedded = embed(self.backbone, prepared)
        return prepared, embedded

    def logits(self, embedded: Embedded) -> torch.Tensor:
        covariates, state = DHSSelector.features(embedded.first_time, embedded.context_patches)
        return self.selector.logits(covariates, state)

    def _run(self, prepared, embedded, kept: list[torch.Tensor], logits: torch.Tensor) -> list[torch.Tensor]:
        """kept[h] holds covariate indices (0-based) for horizon h. Returns one [Q, T] forecast per horizon."""
        rows, groups, bias = [], [], []
        zero = logits.new_zeros(1)
        for h, chosen in enumerate(kept):
            rows.append(torch.cat((chosen.new_zeros(1), chosen + 1)))
            groups.append(torch.full((len(chosen) + 1,), float(h + 1), device=logits.device))
            bias.append(torch.cat((zero, F.logsigmoid(logits[h, chosen]))))
        rows_t = torch.cat(rows)
        out = run_stack(
            self.backbone, embedded, rows_t, torch.cat(groups), rows_t == 0, torch.cat(bias), self.max_horizon
        )
        n_series = prepared.context.shape[0]
        starts = torch.cumsum(torch.tensor([0] + [len(r) for r in rows[:-1]]), 0)
        return [decode_target(self.backbone, prepared, out[int(start)], n_series) for start in starts]

    def soft(self, prepared, embedded, logits, subset: torch.Tensor | None = None) -> list[torch.Tensor]:
        n = logits.shape[1]
        chosen = subset if subset is not None else torch.arange(n, device=logits.device)
        return self._run(prepared, embedded, [chosen] * len(self.horizons), logits)

    def hard(self, prepared, embedded, logits, tau: float) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        gates = torch.sigmoid(logits)
        kept = [torch.nonzero(gates[h] > tau, as_tuple=False).flatten() for h in range(len(self.horizons))]
        return self._run(prepared, embedded, kept, logits), kept
