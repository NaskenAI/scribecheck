"""Command-line interface.

Subcommands are added milestone by milestone (see docs/build-plan.md).
Exit codes follow the spec: 0 = pass or warn, 1 = fail under the decision rule,
2 = tool error (argparse also uses 2 for usage errors).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from scribecheck import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scribecheck",
        description="Check AI-written clinical notes against the consultation transcript.",
    )
    parser.add_argument("--version", action="version", version=f"scribecheck {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0
