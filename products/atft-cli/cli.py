"""
cli.py — ATFT command-line interface.

Parses subcommands and dispatches to the appropriate pipeline stage.

Commands:
    atft crystal    -m <model.pt>
    atft persistence -m <model.pt>
    atft sheaf      -m <model.pt>
    atft full       -m <model.pt>
    atft list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the atft-cli directory is on sys.path so direct imports work.
_CLI_DIR = str(Path(__file__).resolve().parent)
if _CLI_DIR not in sys.path:
    sys.path.insert(0, _CLI_DIR)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atft",
        description="ATFT — Universal Structural Measurement Tool",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── crystal ────────────────────────────────────────────────────────────
    p_crystal = sub.add_parser("crystal", help="Crystal weight distribution measurement")
    p_crystal.add_argument(
        "-m", "--model", dest="model_path", metavar="MODEL",
        help="Path to model checkpoint (.pt)",
    )

    # ── persistence ────────────────────────────────────────────────────────
    p_persist = sub.add_parser("persistence", help="H₀ persistence + Gini measurement")
    p_persist.add_argument(
        "-m", "--model", dest="model_path", metavar="MODEL",
        help="Path to model checkpoint (.pt)",
    )

    # ── sheaf ──────────────────────────────────────────────────────────────
    p_sheaf = sub.add_parser("sheaf", help="Sheaf Laplacian spectral analysis")
    p_sheaf.add_argument(
        "-m", "--model", dest="model_path", metavar="MODEL",
        help="Path to model checkpoint (.pt)",
    )
    p_sheaf.add_argument(
        "--top-n", type=int, default=None, metavar="N",
        help="Only analyze first N layers (for speed)",
    )

    # ── full ───────────────────────────────────────────────────────────────
    p_full = sub.add_parser("full", help="Full pipeline: crystal + persistence + sheaf")
    p_full.add_argument(
        "-m", "--model", dest="model_path", metavar="MODEL",
        help="Path to model checkpoint (.pt)",
    )

    # ── list ───────────────────────────────────────────────────────────────
    sub.add_parser("list", help="List available transducers")

    return parser


def main() -> int:
    """Entry point. Returns 0 on success, 1 on error, 2 on invalid args."""
    parser = _build_parser()

    # argparse itself exits on invalid args (exit code 2), but we guard for
    # edge cases by catching SystemExit and re-emitting with code 2.
    try:
        args = parser.parse_args()
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2

    try:
        if args.command == "list":
            import transducers as _transducers
            names = _transducers.list_transducers()
            if names:
                print("\n".join(names))
            else:
                print("(no transducers registered)")
            return 0

        elif args.command == "crystal":
            from pipeline.crystal import run
            run(model_path=args.model_path)
            return 0

        elif args.command == "persistence":
            from pipeline.persistence import run
            run(model_path=args.model_path)
            return 0

        elif args.command == "sheaf":
            from pipeline.sheaf import run
            top_n = getattr(args, "top_n", None)
            run(model_path=args.model_path, top_n=top_n)
            return 0

        elif args.command == "full":
            from pipeline.full import run
            run(model_path=args.model_path)
            return 0

        else:
            sys.stderr.write(f"atft: unknown command '{args.command}'\n")
            return 2

    except KeyboardInterrupt:
        sys.stderr.write("\natft: interrupted\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"atft: error — {exc}\n")
        return 1
