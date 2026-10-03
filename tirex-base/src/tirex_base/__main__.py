"""Command line entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run plain TiRex-2 on the test windows and write performance metrics.")
    parser.add_argument("--output", type=Path, default=None, help="Directory for tables, charts, and forecasts.")
    parser.add_argument("--quick", action="store_true", help="A few test windows, to check the setup.")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda for an NVIDIA GPU.")
    args = parser.parse_args()
    run(args.output, args.quick, device=args.device)


if __name__ == "__main__":
    main()
