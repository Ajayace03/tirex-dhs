"""Run settings for the TiRex-2 baseline."""
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
    # Same chronological split as the selector projects. The baseline has nothing
    # to train or tune, so it only reads the test days [876, end).
    train_end_index: int = 767
    val_end_index: int = 876
    train_stride_days: int = 7
    val_stride_days: int = 1
    test_stride_days: int = 1
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
        settings = replace(settings, test_stride_days=40, flop_windows=1)
    return settings
