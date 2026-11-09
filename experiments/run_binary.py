"""Binary (Normal vs Attack) experiments across seeds and protocols."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_all import build_parser, main as run_main, normalise  # noqa: E402


def main(argv=None) -> None:
    args = build_parser().parse_args(argv or [])
    run_main(["--phases", "main", "--tasks", "binary",
              "--representations", args.representations,
              "--protocols", args.protocols, "--seeds", args.seeds,
              "--models", args.models, "--epochs", str(args.epochs),
              "--device", args.device])


if __name__ == "__main__":
    main(sys.argv[1:])
