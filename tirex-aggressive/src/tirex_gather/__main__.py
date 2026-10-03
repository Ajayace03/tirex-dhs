"""Command line entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TiRex-DHS, run hard selection on the test windows, and report.")
    parser.add_argument("--output", type=Path, default=None, help="Directory for tables, charts, and forecasts.")
    parser.add_argument("--quick", action="store_true", help="Few windows and one configuration, to check the setup.")
    parser.add_argument("--test-only", action="store_true",
                        help="Reuse outputs/selector.pt from an earlier run and redo only the timed test pass.")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda for an NVIDIA GPU.")
    args = parser.parse_args()
    run(args.output, args.quick, args.test_only, device=args.device)


if __name__ == "__main__":
    main()
