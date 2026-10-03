"""Plain TiRex-2: the target and all covariates go through every block in one pass."""
from __future__ import annotations

import torch


class BaseForecaster:
    def __init__(self, backbone, prediction_length: int):
        self.backbone = backbone
        self.prediction_length = prediction_length

    def forecast(self, target, covariates) -> torch.Tensor:
        """One [Q, prediction_length] forecast. Shorter horizons are its leading steps."""
        from tirex2 import TimeseriesType

        series = TimeseriesType(
            target=torch.as_tensor(target, dtype=torch.float32).view(1, -1),
            past_covariates=torch.as_tensor(covariates, dtype=torch.float32),
            future_covariates=None,
        )
        prediction = self.backbone._predict_once(
            [series], self.prediction_length, preserve_grad=False, tta_diff=False, pad_context=False
        )[0]
        return prediction[0]
