"""Exceptions raised by scribecheck."""

from __future__ import annotations


class InputError(ValueError):
    """Malformed transcript, note or other user input.

    The CLI turns this into exit code 2 (tool error).
    """
