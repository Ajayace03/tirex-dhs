"""TiRex-DHS forecasts with soft gating (the PDF's formulation, used for training and inference).

Every covariate stays in every block. Covariate i adds log g_ih to its
attention logit for horizon h, so a gate near 1 leaves TiRex-2 unchanged and
a gate near 0 silences the series without removing it. Nothing is skipped:
the 12 blocks still process all series, once per horizon. That is the price
of an exact, fully differentiable soft selection, and it is what this project
measures.

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

    def soft(self, prepared, embedded, logits, subset: torch.Tensor | None = None) -> list[torch.Tensor]:
        """One [Q, T] forecast per horizon. subset limits the covariates (used only in training)."""
        n = logits.shape[1]
        chosen = subset if subset is not None else torch.arange(n, device=logits.device)
        rows, groups, bias = [], [], []
        zero = logits.new_zeros(1)
        for h in range(len(self.horizons)):
            rows.append(torch.cat((chosen.new_zeros(1), chosen + 1)))
            groups.append(torch.full((len(chosen) + 1,), float(h + 1), device=logits.device))
            bias.append(torch.cat((zero, F.logsigmoid(logits[h, chosen]))))
        rows_t = torch.cat(rows)
        out = run_stack(
            self.backbone, embedded, rows_t, torch.cat(groups), rows_t == 0, torch.cat(bias), self.max_horizon
        )
        n_series = prepared.context.shape[0]
        width = len(chosen) + 1
        return [decode_target(self.backbone, prepared, out[h * width], n_series) for h in range(len(self.horizons))]
