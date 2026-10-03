"""Build the daily panel from the raw UCI ElectricityLoadDiagrams20112014 file.

The raw file holds 15-minute average kW for 370 clients. The panel keeps the
320 clients that already record on 2012-01-01, so the total does not jump when
a new meter joins. A seeded draw of 100 of them becomes the covariate pool;
the target is the daily energy of the other 220 clients. No covariate is part
of the sum it helps forecast.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import project_root

INTERVALS_PER_DAY = 96
KW_TO_KWH = 0.25
PANEL_START = np.datetime64("2012-01-01")
N_COVARIATES = 100
SEED = 2012


def read_raw(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().rstrip("\r\n")
        names = [part.strip().strip('"') for part in header.split(";")][1:]
        stamps: list[str] = []
        rows: list[np.ndarray] = []
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue
            stamp, rest = line.replace(",", ".").split(";", 1)
            stamps.append(stamp.strip('"').replace(" ", "T"))
            rows.append(np.array(rest.split(";"), dtype=np.float32))
    values = np.stack(rows)
    if values.shape[1] != len(names):
        raise ValueError(f"Raw file has {values.shape[1]} values per row and {len(names)} names")
    return np.array(stamps, dtype="datetime64[s]"), names, values


def build(raw_path: Path, out_dir: Path) -> dict:
    stamps, names, values = read_raw(raw_path)
    # A reading stamped 00:00 closes the previous day's last interval.
    days = (stamps - np.timedelta64(15, "m")).astype("datetime64[D]")
    unique_days, codes = np.unique(days, return_inverse=True)
    counts = np.bincount(codes, minlength=len(unique_days))

    positive = values > 0
    first_day = np.where(positive.any(axis=0), codes[positive.argmax(axis=0)], len(unique_days))
    start_code = int(np.searchsorted(unique_days, PANEL_START))
    panel = np.flatnonzero(first_day <= start_code)

    rng = np.random.default_rng(SEED)
    covariate_idx = np.sort(rng.choice(panel, size=N_COVARIATES, replace=False))
    target_idx = np.setdiff1d(panel, covariate_idx)

    daily = np.zeros((len(unique_days), values.shape[1]), dtype=np.float64)
    np.add.at(daily, codes, values.astype(np.float64) * KW_TO_KWH)
    keep = (np.arange(len(unique_days)) >= start_code) & (counts == INTERVALS_PER_DAY)
    dates = unique_days[keep]
    target = daily[keep][:, target_idx].sum(axis=1)
    clients = daily[keep][:, covariate_idx].astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "daily_load.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "kwh"])
        for date, kwh in zip(dates, target):
            writer.writerow([str(date), repr(float(kwh))])
    np.savez(
        out_dir / "client_daily.npz",
        dates=dates.astype("datetime64[D]"),
        clients=clients,
        names=np.array([names[i] for i in covariate_idx]),
    )
    card = {
        "source": "UCI ElectricityLoadDiagrams20112014 (LD2011_2014.txt)",
        "raw_clients": len(names),
        "panel_rule": "clients with a positive reading on or before 2012-01-01",
        "panel_clients": int(len(panel)),
        "covariate_clients": int(len(covariate_idx)),
        "target_clients": int(len(target_idx)),
        "covariate_draw_seed": SEED,
        "days": int(len(dates)),
        "start": str(dates[0]),
        "end": str(dates[-1]),
        "unit": "kWh per day (15-minute kW x 0.25 h, summed per day)",
        "target": "daily energy of the panel clients that are not covariates",
        "covariates": [names[i] for i in covariate_idx],
    }
    (out_dir / "data_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/ from the raw UCI electricity file.")
    parser.add_argument("raw", type=Path, help="Path to LD2011_2014.txt")
    parser.add_argument("--output", type=Path, default=project_root() / "data")
    args = parser.parse_args()
    card = build(args.raw, args.output)
    print(f"wrote {args.output}: {card['days']} days, {card['covariate_clients']} covariates, "
          f"target = {card['target_clients']} clients")


if __name__ == "__main__":
    main()
