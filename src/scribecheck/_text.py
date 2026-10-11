"""Reading and normalising input text.

Every offset scribecheck stores refers to text after `normalise_text`, and the
models keep that normalised text, so slicing with stored offsets is exact.
"""

from __future__ import annotations

from pathlib import Path

from scribecheck.errors import InputError


def normalise_text(text: str) -> str:
    """Drop a leading byte-order mark and turn CRLF and CR line endings into LF."""
    return text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def read_text_file(path: str | Path) -> str:
    """Read a UTF-8 file (a leading BOM is dropped) with normalised line endings."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise InputError(f"{path}: cannot read file ({exc.strerror or exc})") from exc
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError(f"{path}: not valid UTF-8 (bad byte at position {exc.start})") from exc
    return normalise_text(text)


def strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink text[start:end] to exclude leading and trailing whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
