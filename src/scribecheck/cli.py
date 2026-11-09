"""Command-line interface.

Exit codes follow the spec: 0 = pass or warn, 1 = fail under the decision rule (full mode
with failing opted into, so never in rules mode), 2 = tool error (argparse also uses 2
for usage errors). Settings come from `--config`, else ./scribecheck.toml if present,
else the defaults; command-line options override the file.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from scribecheck import __version__
from scribecheck.checker import DEMO_PREAMBLE, check_files, check_folder, demo_cases
from scribecheck.config import CONFIG_FILENAME, STARTER_CONFIG, Config, load_config
from scribecheck.errors import InputError
from scribecheck.flags import Severity
from scribecheck.report import Report, batch_json, render_batch_markdown, render_markdown

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2

_FAIL_ON: dict[str, tuple[Severity, ...]] = {
    "critical": ("critical",),
    "supporting": ("critical", "supporting"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scribecheck",
        description="Check AI-written clinical notes against the consultation transcript.",
    )
    parser.add_argument("--version", action="version", version=f"scribecheck {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    output = argparse.ArgumentParser(add_help=False)
    output.add_argument("--format", choices=("md", "json"), default="md", help="default: md")
    output.add_argument("--output", type=Path, help="write the report here instead of stdout")

    settings = argparse.ArgumentParser(add_help=False)
    settings.add_argument("--config", type=Path, help=f"config file (default: ./{CONFIG_FILENAME})")
    settings.add_argument("--mode", choices=("rules", "full"))
    settings.add_argument("--standard", choices=("clinical", "strict"))
    settings.add_argument(
        "--fail-on",
        choices=tuple(_FAIL_ON),
        help="full mode only: fail on flags of this severity or higher",
    )

    check = commands.add_parser("check", parents=[settings, output], help="check one consultation")
    check.add_argument("-t", "--transcript", type=Path, required=True)
    check.add_argument("-n", "--note", type=Path, required=True)

    batch = commands.add_parser(
        "batch",
        parents=[settings, output],
        help="check every transcript-note pair in a folder, matched by file name",
    )
    batch.add_argument("folder", type=Path)
    batch.add_argument(
        "--live",
        action="store_true",
        help="full mode: make model calls one by one instead of through Message Batches "
        "(for small development runs)",
    )

    commands.add_parser(
        "demo", parents=[output], help="run the bundled synthetic examples in rules mode"
    )

    init = commands.add_parser("init", help=f"write a starter {CONFIG_FILENAME}")
    init.add_argument("--path", type=Path, default=Path(CONFIG_FILENAME))
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    try:
        if args.command == "init":
            return _init(args.path, force=args.force)
        if args.command == "demo":
            cases = demo_cases()
            text = (
                batch_json(cases, [])
                if args.format == "json"
                else render_batch_markdown(cases, [], preamble=DEMO_PREAMBLE)
            )
            _emit(text, args.output)
            return _exit_code([c.report for c in cases])
        config = _config(args)
        if args.command == "check":
            report = check_files(args.transcript, args.note, config)
            text = report.to_json() if args.format == "json" else render_markdown(report)
            _emit(text, args.output)
            _usage_summary([report])
            return _exit_code([report])
        result = check_folder(args.folder, config, live=args.live)
        if not result.cases and not result.problems:
            raise InputError(f"{args.folder}: no transcript-note pairs found")
        text = (
            batch_json(result.cases, result.problems)
            if args.format == "json"
            else render_batch_markdown(result.cases, result.problems)
        )
        _emit(text, args.output)
        _usage_summary(result.reports)
        for problem in result.problems:
            print(f"scribecheck: {problem}", file=sys.stderr)
        # Any problem (an unmatched, duplicate or unreadable file, or a pair that could not
        # be checked) is a tool error, so a missing case never passes silently.
        return EXIT_ERROR if result.problems else _exit_code(result.reports)
    except InputError as exc:
        print(f"scribecheck: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # an uncaught exception would exit 1, which means "fail"
        print(f"scribecheck: internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _config(args: argparse.Namespace) -> Config:
    if args.config is not None:
        config = load_config(args.config)
    elif Path(CONFIG_FILENAME).is_file():
        config = load_config(CONFIG_FILENAME)
    else:
        config = Config()
    updates: dict[str, object] = {}
    if args.mode is not None:
        updates["mode"] = args.mode
    if args.standard is not None:
        updates["standard"] = args.standard
    if args.fail_on is not None:
        updates["fail_on"] = _FAIL_ON[args.fail_on]
    config = config.model_copy(update=updates)
    if config.mode == "rules" and config.fail_on:
        print(
            "scribecheck: note: fail-on is ignored in rules mode; findings are warnings",
            file=sys.stderr,
        )
    return config


def _init(path: Path, *, force: bool) -> int:
    if path.exists() and not force:
        raise InputError(f"{path} already exists; use --force to overwrite it")
    try:
        path.write_text(STARTER_CONFIG, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise InputError(f"{path}: cannot write ({exc.strerror or exc})") from exc
    print(f"scribecheck: wrote {path}", file=sys.stderr)
    return EXIT_OK


def _emit(text: str, output: Path | None) -> None:
    if output is not None:
        try:
            output.write_text(text, encoding="utf-8", newline="\n")
        except OSError as exc:
            raise InputError(f"{output}: cannot write ({exc.strerror or exc})") from exc
        return
    # Always UTF-8 with LF line endings, whatever the console's encoding, so piped output
    # matches --output byte for byte.
    sys.stdout.flush()
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        sys.stdout.write(text)
    else:
        buffer.write(text.encode("utf-8"))
        buffer.flush()
    sys.stdout.flush()


def _usage_summary(reports: Sequence[Report]) -> None:
    """One stderr line with full mode's model calls and cost."""
    used = [r.usage for r in reports if r.usage is not None]
    if not used:
        return
    calls = sum(u.calls for u in used)
    shared = sum(u.shared_calls for u in used)
    costs = [u.cost_usd for u in used]
    cost = (
        "cost unknown (set [model.prices])"
        if any(c is None for c in costs)
        else f"estimated cost ${sum(c for c in costs if c is not None):.4f}"
    )
    print(
        f"scribecheck: {calls} model calls for single notes, {shared} shared between notes "
        f"of one transcript; {cost}",
        file=sys.stderr,
    )


def _exit_code(reports: Sequence[Report]) -> int:
    return EXIT_FAIL if any(r.decision == "fail" for r in reports) else EXIT_OK
