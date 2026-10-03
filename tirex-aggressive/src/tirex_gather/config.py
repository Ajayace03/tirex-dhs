"""Run settings for TiRex-DHS with hard selection (aggressive gather)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path
    model_id: str = "NX-AI/TiRex-2"
    device: str = "cpu"
    threads: int = 8
    horizons: tuple[int, ...] = (7, 30, 60)
    context_days: int = 192
    # Days [0, 767) train the selector, [767, 876) tune it, [876, end) test it.
    train_end_index: int = 767
    val_end_index: int = 876
    train_stride_days: int = 7
    val_stride_days: int = 1
    test_stride_days: int = 1
    # Selector training (TiRex-2 stays frozen).
    epochs: int = 2
    covariate_subset: int = 32
    learning_rate: float = 1e-2
    initial_gate: float = 0.9
    lambda_sparse_grid: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1)
    lambda_stable: float = 0.01
    warmup_fraction: float = 0.3
    # Hard selection: keep covariate i for horizon h only if g_ih > tau.
    tau_grid: tuple[float, ...] = (0.3, 0.5, 0.7)
    accuracy_tolerance: float = 0.02
    seed: int = 0
    flop_windows: int = 3
    quick: bool = False
    output_dir: Path | None = None

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def daily_path(self) -> Path:
        return self.data_dir / "daily_load.csv"

    @property
    def clients_path(self) -> Path:
        return self.data_dir / "client_daily.npz"

    @property
    def model_source(self) -> str:
        local = self.root / "model"
        return str(local) if (local / "model.ckpt").exists() else self.model_id

    @property
    def reports_dir(self) -> Path:
        return self.output_dir or (self.root / "outputs")

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)


def load_settings(output_dir: Path | None = None, quick: bool = False, device: str = "cpu") -> Settings:
    settings = Settings(root=project_root(), output_dir=output_dir, quick=quick, device=device)
    if quick:
        settings = replace(
            settings,
            epochs=1,
            train_stride_days=120,
            val_stride_days=25,
            test_stride_days=40,
            lambda_sparse_grid=(0.01,),
            tau_grid=(0.5,),
            flop_windows=1,
        )
    return settings
