"""Compare the three TiRex projects on the same test windows: accuracy, cost, and significance.

Reads each project's outputs/ folder (run the projects first) and writes
results/ with tables, charts, and results.md.

Significance: for each method against the baseline, horizon, and metric, the
per-window loss difference d_t = loss_method - loss_base is tested with the
Diebold-Mariano test. It uses a Newey-West variance with h-1 lags, because
windows one day apart share h-1 forecast days, plus the Harvey-Leybourne-
Newbold small-sample correction. Holm's method controls the family-wise error
over all tests (alpha = 0.05). A Wilcoxon signed-rank p-value is reported as a
cross-check; it ignores the overlap, so it overstates significance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
METHODS = [
    ("base", "TiRex-2 baseline (all covariates)", "tirex-base"),
    ("gather", "TiRex-DHS hard selection (aggressive gather)", "tirex-aggressive-gather"),
    ("formal", "TiRex-DHS soft gating (formal selector)", "tirex-formal-selector"),
]
COLORS = {"base": "#1f4e79", "gather": "#2e8b57", "formal": "#c8742c"}
ALPHA = 0.05


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def _num(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load(projects_dir: Path) -> dict:
    loaded = {}
    for key, label, folder in METHODS:
        out = projects_dir / folder / "outputs"
        if not (out / "efficiency.json").exists():
            raise FileNotFoundError(f"{out} has no results yet. Run {folder} first (.\\run.ps1 in that folder).")
        loaded[key] = {
            "label": label,
            "folder": folder,
            "horizons": read_csv(out / "metrics_by_horizon.csv"),
            "windows": read_csv(out / "window_metrics.csv"),
            "timing": read_csv(out / "window_timing.csv"),
            "efficiency": json.loads((out / "efficiency.json").read_text(encoding="utf-8")),
            "report": json.loads((out / "run_report.json").read_text(encoding="utf-8")),
            "forecasts": np.load(out / "forecasts.npz"),
            "tuning": read_csv(out / "tuning.csv") if (out / "tuning.csv").exists() else None,
            "probe": read_csv(out / "covariate_probe.csv") if (out / "covariate_probe.csv").exists() else None,
        }
    dates = {key: [r["origin_date"] for r in m["timing"]] for key, m in loaded.items()}
    if len({tuple(v) for v in dates.values()}) != 1:
        raise ValueError("The projects were run on different test windows; rerun them with the same settings.")
    return loaded


def diebold_mariano(d: np.ndarray, horizon: int) -> tuple[float, float]:
    """Two-sided DM test of mean(d) = 0 with a Bartlett (Newey-West) long-run variance."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    lags = min(horizon - 1, n - 1)
    centered = d - d.mean()
    variance = centered @ centered / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        variance += 2.0 * weight * (centered[lag:] @ centered[:-lag]) / n
    if variance <= 0:
        return float("nan"), float("nan")
    statistic = d.mean() / np.sqrt(variance / n)
    # The correction only shrinks the statistic; its formula is meaningless once horizon approaches n.
    correction = np.sqrt(np.clip((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n, 1e-3, 1.0))
    statistic *= correction
    p = 2.0 * stats.t.sf(abs(statistic), df=n - 1)
    return float(statistic), float(p)


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def window_losses(method: dict, horizon: int, metric: str) -> np.ndarray:
    return np.array([r[metric] for r in method["windows"] if int(r["horizon_days"]) == horizon])


def accuracy_table(data: dict) -> list[dict]:
    rows = []
    base = {int(r["horizon_days"]): r for r in data["base"]["horizons"]}
    for key, method in data.items():
        for r in method["horizons"]:
            h = int(r["horizon_days"])
            rows.append({
                "method": key,
                "horizon_days": h,
                "mae_mwh": r["mae_kwh"] / 1e3,
                "rmse_mwh": r["rmse_kwh"] / 1e3,
                "crps_mwh": r["crps_kwh"] / 1e3,
                "mape_pct": r["mape_pct"],
                "bias_mwh": r["bias_kwh"] / 1e3,
                "coverage_80_pct": r["coverage_80_pct"],
                "mae_change_vs_base_pct": (r["mae_kwh"] / base[h]["mae_kwh"] - 1) * 100,
                "crps_change_vs_base_pct": (r["crps_kwh"] / base[h]["crps_kwh"] - 1) * 100,
                "active_covariates": r["active_covariates"],
            })
    return rows


def efficiency_table(data: dict) -> list[dict]:
    base = data["base"]["efficiency"]
    rows = []
    for key, method in data.items():
        e = method["efficiency"]
        horizons = method["report"]["settings"]["horizons_days"]
        rows.append({
            "method": key,
            "seconds_per_window": e["seconds_per_window_mean"],
            "seconds_per_window_median": e["seconds_per_window_median"],
            "speedup_vs_base": base["seconds_per_window_mean"] / e["seconds_per_window_mean"],
            "time_saved_pct": (1 - e["seconds_per_window_mean"] / base["seconds_per_window_mean"]) * 100,
            "gflops_per_window": e["gflops_per_window"],
            "flops_saved_pct": (1 - e["gflops_per_window"] / base["gflops_per_window"]) * 100,
            "peak_added_memory_mb": e["peak_added_memory_mb"],
            "series_through_stack_per_window": e["series_through_stack_per_window"],
            **{f"active_covariates_pct_h{h}": e[f"active_covariates_pct_h{h}"] for h in horizons},
            "selector_tuning_minutes": e.get("selector_tuning_seconds", 0.0) / 60,
        })
    return rows


def significance_table(data: dict) -> list[dict]:
    rows = []
    horizons = data["base"]["report"]["settings"]["horizons_days"]
    for key in ("gather", "formal"):
        for h in horizons:
            for metric, name in (("window_crps_kwh", "CRPS"), ("window_mae_kwh", "MAE")):
                d = window_losses(data[key], h, metric) - window_losses(data["base"], h, metric)
                statistic, p = diebold_mariano(d, h)
                wilcoxon_p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                rows.append({
                    "method": key, "versus": "base", "horizon_days": h, "metric": name,
                    "mean_difference_mwh": d.mean() / 1e3,
                    "relative_difference_pct": d.mean() / window_losses(data["base"], h, metric).mean() * 100,
                    "dm_statistic": statistic, "dm_p": p, "wilcoxon_p": wilcoxon_p,
                })
        t = np.array([r["seconds"] for r in data[key]["timing"]]) - np.array([r["seconds"] for r in data["base"]["timing"]])
        statistic, p = diebold_mariano(t, 1)
        rows.append({
            "method": key, "versus": "base", "horizon_days": "all", "metric": "seconds per window",
            "mean_difference_mwh": float("nan"),
            "relative_difference_pct": t.mean() / np.mean([r["seconds"] for r in data["base"]["timing"]]) * 100,
            "dm_statistic": statistic, "dm_p": p, "wilcoxon_p": float(stats.wilcoxon(t).pvalue),
        })
    adjusted = holm([r["dm_p"] if np.isfinite(r["dm_p"]) else 1.0 for r in rows])
    for row, p in zip(rows, adjusted):
        row["dm_p_holm"] = p
        if p >= ALPHA:
            row["verdict"] = "no significant difference"
        else:
            row["verdict"] = "significantly " + ("lower (better)" if row["relative_difference_pct"] < 0 else "higher (worse)")
    return rows


def charts(data: dict, accuracy: list[dict], efficiency: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    horizons = data["base"]["report"]["settings"]["horizons_days"]
    keys = list(data)

    figure, axis = plt.subplots(figsize=(7, 4.6))
    for row in efficiency:
        crps = np.mean([a["crps_mwh"] for a in accuracy if a["method"] == row["method"]])
        axis.scatter(row["seconds_per_window"], crps, s=140, color=COLORS[row["method"]], zorder=3,
                     label=data[row["method"]]["label"])
        axis.annotate(f"{row['seconds_per_window']:.2f}s, {row['gflops_per_window']:.0f} GFLOPs",
                      (row["seconds_per_window"], crps), textcoords="offset points", xytext=(8, 6), fontsize=8)
    from matplotlib.ticker import FormatStrFormatter, NullFormatter

    axis.set(xscale="log", xlabel="Seconds per forecast window (all 3 horizons, log scale)",
             ylabel="Mean CRPS over 7/30/60 days (MWh)",
             title="Accuracy against compute (lower-left is better)\nThe CRPS axis is zoomed: the whole spread is about 1%")
    axis.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.margins(x=0.35, y=0.35)
    axis.grid(alpha=0.3)
    axis.legend(frameon=False, fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(figures / "accuracy_vs_time.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.8 / len(keys)
    for i, key in enumerate(keys):
        rows = [a for a in accuracy if a["method"] == key]
        for axis, metric, name in ((axes[0], "crps_mwh", "CRPS"), (axes[1], "mae_mwh", "MAE")):
            axis.bar(np.arange(len(horizons)) + i * width, [r[metric] for r in rows], width,
                     color=COLORS[key], label=data[key]["label"])
            axis.set(title=f"{name} by horizon", ylabel="MWh per day")
            axis.set_xticks(np.arange(len(horizons)) + width, [f"{h} days" for h in horizons])
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "accuracy_by_horizon.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for axis, metric, name in (
        (axes[0], "seconds_per_window", "Seconds per window"),
        (axes[1], "gflops_per_window", "GFLOPs per window"),
        (axes[2], "peak_added_memory_mb", "Peak added memory (MB)"),
        (axes[3], "series_through_stack_per_window", "Series through the 12 blocks"),
    ):
        axis.bar(range(len(efficiency)), [r[metric] for r in efficiency], color=[COLORS[r["method"]] for r in efficiency])
        axis.set_xticks(range(len(efficiency)), [r["method"] for r in efficiency])
        axis.set_title(name)
    figure.suptitle("Compute cost per test window")
    figure.tight_layout()
    figure.savefig(figures / "compute_cost.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(len(horizons), 1, figsize=(10, 2.8 * len(horizons)), sharex=True)
    for axis, h in zip(np.atleast_1d(axes), horizons):
        for key in keys:
            rows = [r for r in data[key]["windows"] if int(r["horizon_days"]) == h]
            axis.plot(np.array([r["origin_date"] for r in rows], dtype="datetime64[D]"),
                      [r["window_crps_kwh"] / 1e3 for r in rows], color=COLORS[key], lw=1.1, label=key)
        axis.set(ylabel="CRPS (MWh)", title=f"{h}-day forecasts: CRPS per test window")
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=3)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(figures / "crps_per_window.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4))
    f = data["base"]["forecasts"]
    long = max(horizons)
    dates = np.array(f["origin_dates"], dtype="datetime64[D]")
    axis.plot(dates, f[f"actual_h{long}"][:, 0] / 1e3, color="black", lw=1.2, label="Actual (next day)")
    for key in keys:
        fk = data[key]["forecasts"]
        axis.plot(dates, fk[f"h{long}"][:, 4, 0] / 1e3, color=COLORS[key], lw=1, label=f"{key} median, 1 day ahead")
    axis.set(ylabel="MWh per day", title="One-day-ahead median forecasts on the test windows")
    axis.legend(frameon=False, fontsize=8)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(figures / "forecasts_one_day_ahead.png", dpi=150)
    plt.close(figure)


def markdown(data, accuracy, efficiency, significance) -> str:
    lines = ["# Results", "", f"Test windows: {data['base']['efficiency']['test_windows']} daily origins "
             f"({data['base']['timing'][0]['origin_date']} to {data['base']['timing'][-1]['origin_date']}), "
             f"{data['base']['efficiency']['covariates_available']} covariates available.", "",
             "## Compute", "",
             "| Method | s / window | Speed-up | GFLOPs / window | FLOPs saved | Peak added memory (MB) | Series through stack |",
             "|---|---|---|---|---|---|---|"]
    for r in efficiency:
        lines.append(f"| {data[r['method']]['label']} | {r['seconds_per_window']:.3f} | {r['speedup_vs_base']:.2f}x | "
                     f"{r['gflops_per_window']:.1f} | {r['flops_saved_pct']:.1f}% | {r['peak_added_memory_mb']:.0f} | "
                     f"{r['series_through_stack_per_window']:.1f} |")
    lines += ["", "## Accuracy", "",
              "| Method | Horizon | MAE (MWh) | CRPS (MWh) | MAPE % | 80% coverage | CRPS vs base | Active covariates |",
              "|---|---|---|---|---|---|---|---|"]
    for r in accuracy:
        lines.append(f"| {r['method']} | {r['horizon_days']} d | {r['mae_mwh']:.1f} | {r['crps_mwh']:.1f} | "
                     f"{r['mape_pct']:.2f} | {r['coverage_80_pct']:.1f}% | {r['crps_change_vs_base_pct']:+.2f}% | "
                     f"{r['active_covariates']:.1f} |")
    lines += ["", "## Significance against the baseline (Diebold-Mariano, Holm-adjusted)", "",
              "| Method | Horizon | Metric | Difference | DM p | Holm p | Verdict |", "|---|---|---|---|---|---|---|"]
    for r in significance:
        lines.append(f"| {r['method']} | {r['horizon_days']} | {r['metric']} | {r['relative_difference_pct']:+.2f}% | "
                     f"{r['dm_p']:.3g} | {r['dm_p_holm']:.3g} | {r['verdict']} |")
    probe = data["gather"]["probe"]
    if probe:
        lines += ["", "## How much can TiRex-2 use past covariates at all? (plain TiRex-2, every 4th test window)", "",
                  "| Horizon | Covariates given | CRPS (MWh) | vs. none |", "|---|---|---|---|"]
        for r in probe:
            lines.append(f"| {int(r['horizon_days'])} d | {r['covariates']} | {r['crps_kwh'] / 1e3:.1f} | "
                         f"{r['crps_change_vs_none_pct']:+.2f}% |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the three TiRex projects.")
    parser.add_argument("--projects", type=Path, default=ROOT.parent, help="Folder that holds the three projects.")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    data = load(args.projects)
    args.output.mkdir(parents=True, exist_ok=True)
    accuracy = accuracy_table(data)
    efficiency = efficiency_table(data)
    significance = significance_table(data)
    write_csv(args.output / "accuracy_by_horizon.csv", accuracy)
    write_csv(args.output / "efficiency.csv", efficiency)
    write_csv(args.output / "significance.csv", significance)
    if data["gather"]["probe"]:
        write_csv(args.output / "covariate_probe.csv", data["gather"]["probe"])
    for key, method in data.items():
        if method["tuning"]:
            write_csv(args.output / f"tuning_{key}.csv", method["tuning"])
    charts(data, accuracy, efficiency, args.output)
    (args.output / "results.md").write_text(markdown(data, accuracy, efficiency, significance), encoding="utf-8")
    print((args.output / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
