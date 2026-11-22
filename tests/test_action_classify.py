"""action_classify.py on synthetic omission pairs (no network, synthetic text)."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "action_classify.py"
FACTS = {
    "syn|C1|omit|p1": "Referred to the falls clinic",
    "syn|C1|omit|p2": "Blood test for thyroid function next week",
    "syn|C1|omit|p3": "Review in two weeks",
    "syn|C1|omit|p4": "Start ramipril 2.5 mg once daily",
    "syn|C1|omit|p5": "Smokes ten a day",
    "syn|C1|omit|p6": "Stop ibuprofen",
}
ANSWERS = {
    "falls": "referral",
    "thyroid": "test",
    "Review": "follow_up",
    "ramipril": "medication_change",
    "Smokes": "not_action",
    "ibuprofen": "medication_change",
}


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("action_classify", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["action_classify"] = module
    spec.loader.exec_module(module)
    return module


def _pairs(tmp_path: Path) -> Path:
    pairs: list[dict[str, Any]] = [
        {"pair_id": pid, "type": "omit", "fact": fact, "target": {"fact": fact}}
        for pid, fact in FACTS.items()
    ]
    pairs.append({"pair_id": "syn|C1|change|p7", "type": "change", "fact": "Dose changed"})
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
    return path


class FakeClient:
    """Answers each request with the label for the first ANSWERS word in its fact."""

    provider = "anthropic"
    model = "m"

    def __init__(self, label: str | None = None) -> None:
        self.label = label
        self.calls = 0
        self.tasks: list[str] = []

    @property
    def endpoint(self) -> Any:
        from scribecheck.model.endpoint import Endpoint

        return Endpoint.from_url("https://api.anthropic.com")

    def send(self, request: Any, *, correction: str | None = None) -> Any:
        from scribecheck.model.base import RawReply, Usage

        self.calls += 1
        task = request.messages[0].blocks[0].text
        self.tasks.append(task)
        fact = task.split("REMOVED FACT:\n", 1)[1].split("\n", 1)[0]
        label = self.label or next(v for k, v in ANSWERS.items() if k in fact)
        return RawReply(json.dumps({"label": label}), "end", Usage(50, 5), "m-1")


def test_prompt_hash_is_stable_and_covers_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load()
    first = module.prompt_hash()
    assert len(first) == 64 and first == module.prompt_hash()
    monkeypatch.setattr(module, "TASK", module.TASK + " ")
    assert module.prompt_hash() != first


def test_hash_command_prints_the_hash(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load()
    assert module.main(["hash"]) == 0
    assert capsys.readouterr().out.strip() == module.prompt_hash()


def test_classify_labels_omission_pairs_and_caches(tmp_path: Path) -> None:
    from scribecheck.model.budget import Budget

    module = _load()
    pairs = json.loads(_pairs(tmp_path).read_text())["pairs"]
    items = module.omission_pairs(pairs)
    assert set(items) == set(FACTS)  # the change pair is not an omission
    assert items["syn|C1|omit|p1"] == FACTS["syn|C1|omit|p1"]  # same target fact: once
    path = tmp_path / "cache.json"
    client = FakeClient()
    counts = module.classify(items, client, Budget(), module.load_cache(path), "m", path)
    assert (counts["classified"], client.calls) == (6, 6)
    # Only the fact text is sent: the task holds the fact and the fixed instructions.
    assert all(t.startswith("REMOVED FACT:\n") for t in client.tasks)
    labels = module.labels(path, pairs)
    assert labels["syn|C1|omit|p4"] == "medication_change"
    assert labels["syn|C1|omit|p5"] == "not_action"
    cache = module.load_cache(path)
    assert cache["prompt_sha256"] == module.prompt_hash()
    # Again: nothing to call.
    counts = module.classify(items, client, Budget(), cache, "m", path)
    assert (counts["cached"], client.calls) == (6, 6)
    # A changed fact is classified afresh; another model's labels are not reused.
    assert module.lookup(cache, "syn|C1|omit|p1", "Referred to cardiology") is None
    assert module.lookup(cache, "syn|C1|omit|p1", FACTS["syn|C1|omit|p1"], "other") is None


def test_an_invalid_label_is_a_failure_not_a_label(tmp_path: Path) -> None:
    from scribecheck.model.budget import Budget

    module = _load()
    path = tmp_path / "cache.json"
    items = {"syn|C1|omit|p1": FACTS["syn|C1|omit|p1"]}
    counts = module.classify(items, FakeClient("surgery"), Budget(), {}, "m", path)
    assert counts["classified"] == 0 and sum(counts.values()) == 1
    assert module.lookup(module.load_cache(path), "syn|C1|omit|p1", items["syn|C1|omit|p1"]) is None


def test_classify_stops_without_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = tmp_path / "c.toml"
    config.write_text(
        'mode = "full"\n[model]\nprovider = "anthropic"\nname = "claude-sonnet-5-5"\n',
        encoding="utf-8",
    )
    cache = tmp_path / "cache.json"
    args = ["classify", "--pairs", str(_pairs(tmp_path)), "--config", str(config)]
    assert _load().main([*args, "--cache", str(cache)]) == 2
    assert not cache.exists()
    assert "key" in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    [
        ["a", "b"],
        {"records": ["a", "b"]},
        {"records": [{"pair_id": "a"}, {"pair_id": "b"}]},
    ],
)
def test_load_ids_accepts_lists_and_records(tmp_path: Path, content: Any) -> None:
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    assert _load().load_ids(path) == {"a", "b"}


def test_load_ids_rejects_entries_without_an_id(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    path.write_text(json.dumps({"records": [{"id": "a"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        _load().load_ids(path)


def test_sample_spreads_over_labels_and_is_fixed_by_its_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scribecheck.model.budget import Budget

    module = _load()
    pairs_path = _pairs(tmp_path)
    cache = tmp_path / "cache.json"
    items = module.omission_pairs(json.loads(pairs_path.read_text())["pairs"])
    module.classify(items, FakeClient(), Budget(), {}, "m", cache)
    out = tmp_path / "sheet.csv"
    args = ["sample", "--pairs", str(pairs_path), "--cache", str(cache), "--out", str(out)]
    assert module.main([*args, "--n", "5", "--seed", "1"]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 5
    # One of each label first: all five labels appear before any is repeated.
    assert {r["label"] for r in rows} == set(module.LABELS)
    assert all(r["reviewer_label"] == "" for r in rows)
    first = out.read_text(encoding="utf-8")
    assert module.main([*args, "--n", "5", "--seed", "1"]) == 0
    assert out.read_text(encoding="utf-8") == first
    # Only counts are printed, never fact text.
    printed = capsys.readouterr().out
    assert not any(fact in printed for fact in FACTS.values())


def test_labels_match_the_output_schema() -> None:
    module = _load()
    schema = module.Classification.model_json_schema()
    assert tuple(schema["properties"]["label"]["enum"]) == module.LABELS


def test_the_classifier_config_is_sonnet_at_effort_low() -> None:
    from scribecheck.config import load_config

    config = load_config(SCRIPT.parent.parent / "docs" / "runbooks" / "action_classify.toml")
    assert (config.model, config.effort) == ("claude-sonnet-5-5", "low")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_spot_check_sheet_is_git_ignored() -> None:
    root = SCRIPT.parent.parent
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")
    done = subprocess.run(
        ["git", "check-ignore", "-q", "docs/review/action_spot_check.csv"], cwd=root, check=False
    )
    assert done.returncode == 0
