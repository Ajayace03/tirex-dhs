"""Daily electricity panel and the chronological train, validation, and test windows."""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from .config import Settings


@dataclass(frozen=True)
class Panel:
    dates: np.ndarray
    load_kwh: np.ndarray
    clients_kwh: np.ndarray
    client_names: list[str]

    @property
    def n_days(self) -> int:
        return int(self.load_kwh.shape[0])

    @property
    def n_covariates(self) -> int:
        return len(self.client_names)

    @property
    def n_series(self) -> int:
        return 1 + self.n_covariates

    def context(self, origin: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        if origin < length:
            raise ValueError(f"Origin {origin} is shorter than the {length}-day context")
        return self.load_kwh[origin - length:origin], self.clients_kwh[:, origin - length:origin]

    def actuals(self, origin: int, horizon: int) -> np.ndarray:
        return self.load_kwh[origin:origin + horizon]

    def origins(self, settings: Settings, first: int, end: int, stride: int) -> list[int]:
        """Origins whose context starts at day 0 or later and whose longest horizon ends before end."""
        start = max(first, settings.context_days)
        stop = min(end, self.n_days) - settings.max_horizon + 1
        origins = list(range(start, stop, stride))
        if not origins:
            raise ValueError(f"No windows fit between day {first} and day {end}")
        return origins

    def train_origins(self, settings: Settings) -> list[int]:
        # One day later than the first context so the stability loss has a previous day.
        return self.origins(settings, settings.context_days + 1, settings.train_end_index, settings.train_stride_days)

    def val_origins(self, settings: Settings) -> list[int]:
        return self.origins(settings, settings.train_end_index, settings.val_end_index, settings.val_stride_days)

    def test_origins(self, settings: Settings) -> list[int]:
        return self.origins(settings, settings.val_end_index, self.n_days, settings.test_stride_days)


def load_panel(settings: Settings) -> Panel:
    for path in (settings.daily_path, settings.clients_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Build it with: python -m <package>.prepare_data <LD2011_2014.txt>")

    dates: list[np.datetime64] = []
    load: list[float] = []
    with settings.daily_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "date" not in reader.fieldnames or "kwh" not in reader.fieldnames:
            raise ValueError("daily_load.csv must have date and kwh columns")
        for row in reader:
            dates.append(np.datetime64(row["date"]))
            load.append(float(row["kwh"]))
    if not dates:
        raise ValueError("daily_load.csv has no rows")

    saved = np.load(settings.clients_path, allow_pickle=False)
    client_dates = saved["dates"].astype("datetime64[D]")
    clients = np.asarray(saved["clients"], dtype=np.float32)
    names = [str(name) for name in saved["names"].tolist()]
    if clients.ndim != 2 or clients.shape[1] != len(names):
        raise ValueError(f"Unexpected client array shape {clients.shape}")
    if not np.array_equal(client_dates, np.asarray(dates, dtype="datetime64[D]")):
        raise ValueError("Client dates do not match daily_load.csv")

    return Panel(
        dates=np.asarray(dates, dtype="datetime64[D]"),
        load_kwh=np.asarray(load, dtype=np.float32),
        clients_kwh=clients.T.copy(),
        client_names=names,
    )
