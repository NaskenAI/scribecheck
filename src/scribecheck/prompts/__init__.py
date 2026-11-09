"""Versioned prompt files, shipped as package data.

`route_one/` holds Fox et al.'s route-one prompts, unmodified, from the MIT-licensed
OmissionBench harness (see its README and LICENSE); `scribecheck_v4/` holds scribecheck's
own (the earlier sets, which measured runs used, are kept unchanged).
Prompts are filled by literal slot replacement, never `str.format`, because the route-one
prompts end with literal JSON that `str.format` would misread.
"""

from __future__ import annotations

import hashlib
from functools import cache
from importlib import resources
from typing import Literal

PromptSet = Literal[
    "route_one", "scribecheck_v1", "scribecheck_v2", "scribecheck_v3", "scribecheck_v4"
]


@cache
def load(version: PromptSet, name: str) -> str:
    """The text of prompt `name` (without `.txt`) from folder `version`."""
    return (resources.files(__name__) / version / f"{name}.txt").read_text(encoding="utf-8")


def sha256(version: PromptSet, name: str) -> str:
    return hashlib.sha256(load(version, name).encode("utf-8")).hexdigest()


def render(template: str, **slots: str) -> str:
    """Fill `{slot}` placeholders literally. Raises KeyError for a slot the prompt lacks."""
    out = template
    for name, value in slots.items():
        token = "{" + name + "}"
        if token not in out:
            raise KeyError(f"prompt has no {token} slot")
        out = out.replace(token, value)
    return out
