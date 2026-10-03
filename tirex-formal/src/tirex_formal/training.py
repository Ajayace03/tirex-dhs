"""Train the selector with L = L_forecast + lambda_1 L_sparse + lambda_2 L_stable. TiRex-2 stays frozen.

L_forecast  pinball loss over all quantiles and every step 1..h of each
            horizon, divided by the window's mean target level so windows
            weigh alike.
L_sparse    mean gate value over all covariates and horizons (L1, since g >= 0).
L_stable    w_t * mean |g_t - g_{t-1}| against the previous day's gates, with
            w_t = exp(-||h_t - h_{t-1}||). Gates may move when the context moves.

lambda_1 ramps up over the first warmup_fraction of the steps so the gates do
not all collapse before the forecast loss has shaped them. Each step trains on
a random subset of covariates to keep backpropagation through the 12-block
stack affordable on a CPU. The selector is shared across covariates, so it
learns to score any of them.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .config import Settings
from .forecast import DHSModel
from .selector import DHSSelector, context_weight


def pinball(actual: torch.Tensor, forecast: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    diff = actual.unsqueeze(0) - forecast
    taus = quantiles.view(-1, 1)
    return torch.maximum(taus * diff, (taus - 1.0) * diff).mean()


def train_selector(backbone, panel, settings: Settings, quantiles, lambda_sparse: float, logger) -> tuple[DHSSelector, dict]:
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    selector = DHSSelector(backbone.embedding_dim, len(settings.horizons), settings.initial_gate).to(settings.device)
    model = DHSModel(backbone, selector, settings.horizons)
    optimizer = torch.optim.Adam(selector.parameters(), lr=settings.learning_rate)
    q = torch.as_tensor(quantiles, dtype=torch.float32, device=settings.device)

    origins = panel.train_origins(settings)
    total_steps = settings.epochs * len(origins)
    warmup = max(1, int(settings.warmup_fraction * total_steps))
    subset_size = min(settings.covariate_subset, panel.n_covariates)
    history = []
    step = 0
    started = time.perf_counter()
    selector.train()
    for epoch in range(settings.epochs):
        sums = np.zeros(4)
        for origin in rng.permutation(origins):
            origin = int(origin)
            target, covariates = panel.context(origin, settings.context_days)
            prepared, embedded = model.window(target, covariates)
            _, previous = model.window(*panel.context(origin - 1, settings.context_days))
            logits = model.logits(embedded)
            previous_logits = model.logits(previous)

            subset = torch.as_tensor(np.sort(rng.choice(panel.n_covariates, subset_size, replace=False)),
                                     device=settings.device)
            forecasts = model.soft(prepared, embedded, logits, subset)
            scale = float(np.abs(target).mean())
            forecast_loss = sum(
                pinball(torch.as_tensor(panel.actuals(origin, h), device=settings.device), f[:, :h], q) / scale
                for h, f in zip(settings.horizons, forecasts)
            ) / len(settings.horizons)

            gates = torch.sigmoid(logits)
            previous_gates = torch.sigmoid(previous_logits)
            sparse_loss = gates.mean()
            _, state = DHSSelector.features(embedded.first_time, embedded.context_patches)
            _, previous_state = DHSSelector.features(previous.first_time, previous.context_patches)
            stable_loss = context_weight(state, previous_state) * (gates - previous_gates).abs().mean()

            ramp = min(1.0, (step + 1) / warmup)
            loss = forecast_loss + lambda_sparse * ramp * sparse_loss + settings.lambda_stable * stable_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums += [loss.item(), forecast_loss.item(), sparse_loss.item(), stable_loss.item()]
            step += 1
        n = len(origins)
        row = {
            "lambda_sparse": lambda_sparse,
            "epoch": epoch + 1,
            "loss": sums[0] / n,
            "forecast_loss": sums[1] / n,
            "mean_gate": sums[2] / n,
            "stability_loss": sums[3] / n,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        logger.info(
            "lambda_sparse=%s epoch %s/%s loss=%.4f forecast=%.4f mean_gate=%.3f stable=%.4f (%.0fs)",
            lambda_sparse, epoch + 1, settings.epochs, row["loss"], row["forecast_loss"],
            row["mean_gate"], row["stability_loss"], row["elapsed_seconds"],
        )
    selector.eval()
    return selector, {"history": history, "train_windows": len(origins), "steps": step,
                      "seconds": time.perf_counter() - started}
