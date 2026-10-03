"""Tables, archives, and charts written under outputs/."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_charts(directory: Path, horizon_rows, window_rows, intervals, title: str) -> None:
    plt = _pyplot()
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = [f"{int(row['horizon_days'])} days" for row in horizon_rows]

    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    for axis, key, name in ((axes[0], "mae_kwh", "MAE"), (axes[1], "rmse_kwh", "RMSE"), (axes[2], "crps_kwh", "CRPS")):
        axis.bar(range(len(labels)), [row[key] / 1_000 for row in horizon_rows], color="#1f4e79")
        axis.set_xticks(range(len(labels)), labels)
        axis.set(title=name, ylabel="MWh per day")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(figures / "metrics_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    grouped = [[row["window_mae_kwh"] / 1_000 for row in window_rows if row["horizon_days"] == h]
               for h in [row["horizon_days"] for row in horizon_rows]]
    axis.boxplot(grouped, tick_labels=labels, showfliers=False)
    axis.set(title=f"{title}: MAE per test window", ylabel="MWh per day")
    figure.tight_layout()
    figure.savefig(figures / "window_mae.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.4, 5))
    axis.plot([0, 100], [0, 100], color="#929aa2", ls="--", label="Ideal")
    axis.plot([r["nominal_coverage_pct"] for r in intervals], [r["observed_coverage_pct"] for r in intervals],
              marker="o", color="#087e8b", label="Observed")
    axis.set(xlim=(0, 100), ylim=(0, 100), xlabel="Nominal coverage (%)", ylabel="Observed coverage (%)",
             title=f"{title}: interval calibration (60 days)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figures / "interval_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_gate_charts(directory: Path, gates: np.ndarray, kept: np.ndarray, names: list[str],
                      horizons: tuple[int, ...], dates: np.ndarray, title: str) -> None:
    """gates is [windows, horizons, covariates]; kept is the matching 0/1 selection."""
    plt = _pyplot()
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    mean_gate = gates.mean(axis=0)
    top = np.argsort(-mean_gate.max(axis=0))[:30]
    figure, axis = plt.subplots(figsize=(6.5, 8))
    image = axis.imshow(mean_gate[:, top].T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(range(len(horizons)), [f"{h} d" for h in horizons])
    axis.set_yticks(range(len(top)), [names[i] for i in top], fontsize=7)
    axis.set(title=f"{title}: mean gate, top 30 covariates", xlabel="Forecast horizon")
    figure.colorbar(image, ax=axis, label="g")
    figure.tight_layout()
    figure.savefig(figures / "gate_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 3.8))
    for h_index, horizon in enumerate(horizons):
        axis.plot(dates, kept[:, h_index].sum(axis=1), label=f"{horizon} days", lw=1.2)
    axis.set(title=f"{title}: covariates in use per test window", ylabel="Covariates kept",
             ylim=(0, max(1, kept.shape[2])))
    axis.legend(frameon=False)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(figures / "active_covariates.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
