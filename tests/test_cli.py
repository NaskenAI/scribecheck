from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scribecheck import Config, __version__, cli
from scribecheck.cli import main
from scribecheck.config import STARTER_CONFIG, load_config


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: scribecheck" in capsys.readouterr().out


def test_unknown_command_is_a_tool_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["no-such-command"])
    assert exc.value.code == 2


def test_module_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scribecheck", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


# Commands

KNEE = (
    "Doctor: What seems to be the problem?\n"
    "Patient: My left knee has been swollen for two weeks.\n"
    "Doctor: Are you taking anything for it?\n"
    "Patient: I take ibuprofen 400 mg three times a day.\n"
)
KNEE_NOTE = "History:\nKnee swollen for 2 weeks.\n\nMedications:\nIbuprofen 400 mg twice daily.\n"
CLEAN_NOTE = "History:\nLeft knee swollen for 2 weeks.\n\nMedications:\nIbuprofen 400 mg TDS.\n"


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A folder with one consultation, also the working directory (no config file)."""
    (tmp_path / "knee.transcript.txt").write_text(KNEE, encoding="utf-8")
    (tmp_path / "knee.note.md").write_text(KNEE_NOTE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _check(*extra: str) -> int:
    return main(["check", "-t", "knee.transcript.txt", "-n", "knee.note.md", *extra])


def test_check_warns_with_exit_0(case: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _check() == 0
    out = capsys.readouterr().out
    assert out.startswith("# scribecheck report: knee\n")
    assert "**Decision: warn**" in out


def test_check_clean_note_passes(case: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (case / "knee.note.md").write_text(CLEAN_NOTE, encoding="utf-8")
    assert _check("--format", "json") == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "pass"


def test_check_json_and_output_file(case: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _check("--format", "json", "--output", "report.json") == 0
    assert capsys.readouterr().out == ""
    data = json.loads((case / "report.json").read_text(encoding="utf-8"))
    assert data["consultation_id"] == "knee"
    assert data["counts"]["changed_fact"] == 1
    assert b"\r\n" not in (case / "report.json").read_bytes()


def test_rules_mode_never_exits_1_even_with_fail_on(
    case: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _check("--fail-on", "supporting") == 0
    assert "fail-on is ignored in rules mode" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [
        ["check", "-t", "absent.txt", "-n", "knee.note.md"],
        ["check", "-t", "knee.transcript.txt", "-n", "absent.md"],
        ["check", "-t", "knee.transcript.txt", "-n", "knee.note.md", "--mode", "full"],
        ["check", "-t", "knee.transcript.txt", "-n", "knee.note.md", "--config", "absent.toml"],
        ["batch", "absent-folder"],
    ],
)
def test_tool_errors_exit_2(
    case: Path, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("scribecheck: error: ")
    assert "Traceback" not in captured.err


def test_bad_config_exits_2(case: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (case / "scribecheck.toml").write_text("standard = 'loose'\n", encoding="utf-8")
    assert _check() == 2
    assert "scribecheck.toml" in capsys.readouterr().err


def test_check_reads_config_from_working_directory(
    case: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (case / "scribecheck.toml").write_text("standard = 'strict'\n", encoding="utf-8")
    assert _check("--format", "json") == 0
    assert json.loads(capsys.readouterr().out)["standard"] == "strict"
    assert _check("--format", "json", "--standard", "clinical") == 0  # the option wins
    assert json.loads(capsys.readouterr().out)["standard"] == "clinical"


def test_internal_errors_exit_2(
    case: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(cli, "check_files", broken)
    assert _check() == 2
    assert "internal error: ValueError: boom" in capsys.readouterr().err


def test_batch_reports_every_pair(case: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (case / "other.transcript.txt").write_text(KNEE, encoding="utf-8")
    (case / "other.note.md").write_text(CLEAN_NOTE, encoding="utf-8")
    assert main(["batch", ".", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [(c["id"], c["transcript"], c["report"]["decision"]) for c in data["cases"]] == [
        ("knee", "knee.transcript.txt", "warn"),
        ("other", "other.transcript.txt", "pass"),
    ]
    assert data["problems"] == []


def test_batch_rows_name_their_case_when_ids_clash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for case in ("c1", "c2"):
        data = {"consultation_id": "same", "turns": [{"speaker": "Patient", "text": "Hello."}]}
        (tmp_path / f"{case}.transcript.json").write_text(json.dumps(data), encoding="utf-8")
        (tmp_path / f"{case}.note.md").write_text(KNEE_NOTE, encoding="utf-8")
    assert main(["batch", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "| c1 | same | pass |" in out and "| c2 | same | pass |" in out
    assert "Case c1: `c1.transcript.json` and `c1.note.md`" in out


def test_batch_subfolders(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "transcripts" / "k1.txt").write_text(KNEE, encoding="utf-8")
    (tmp_path / "notes" / "k1.md").write_text(KNEE_NOTE, encoding="utf-8")
    assert main(["batch", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# scribecheck batch report\n")
    assert "| k1 | warn |" in out


def test_batch_unmatched_files_are_reported_and_exit_2(
    case: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (case / "lonely.note.md").write_text(KNEE_NOTE, encoding="utf-8")
    assert main(["batch", "."]) == 2
    captured = capsys.readouterr()
    assert "| knee | knee | warn |" in captured.out  # the matched pair is still reported
    assert "lonely.note.md: no matching transcript" in captured.out
    assert "lonely.note.md: no matching transcript" in captured.err


def test_batch_broken_pair_exits_2_after_reporting_the_rest(
    case: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (case / "bad.transcript.json").write_text("{not json", encoding="utf-8")
    (case / "bad.note.md").write_text(KNEE_NOTE, encoding="utf-8")
    assert main(["batch", ".", "--format", "json"]) == 2
    data = json.loads(capsys.readouterr().out)
    assert [c["id"] for c in data["cases"]] == ["knee"]
    assert data["problems"][0].startswith("bad: ")


def test_batch_empty_folder_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["batch", str(tmp_path)]) == 2
    assert "no transcript-note pairs found" in capsys.readouterr().err


def test_batch_with_only_unmatched_files_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a.note.md").write_text(KNEE_NOTE, encoding="utf-8")
    assert main(["batch", str(tmp_path)]) == 2


def test_init_writes_starter_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    assert (tmp_path / "scribecheck.toml").read_text(encoding="utf-8") == STARTER_CONFIG
    assert load_config(tmp_path / "scribecheck.toml") == Config()


def test_init_refuses_to_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "custom.toml"
    path.write_text("mode = 'rules'\n", encoding="utf-8")
    assert main(["init", "--path", str(path)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == "mode = 'rules'\n"
    assert main(["init", "--path", str(path), "--force"]) == 0
    assert path.read_text(encoding="utf-8") == STARTER_CONFIG


def test_stdout_is_utf8_whatever_the_console_encoding(tmp_path: Path) -> None:
    transcript = tmp_path / "caf\u00e9.transcript.txt"
    transcript.write_text(KNEE, encoding="utf-8")
    note = tmp_path / "caf\u00e9.note.md"
    note.write_text(KNEE_NOTE, encoding="utf-8")
    args = ["check", "-t", str(transcript), "-n", str(note), "--format", "json"]
    piped = subprocess.run(
        [sys.executable, "-m", "scribecheck", *args],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        check=False,
    )
    assert piped.returncode == 0, piped.stderr
    assert json.loads(piped.stdout.decode("utf-8"))["consultation_id"] == "caf\u00e9"
    assert b"\r\n" not in piped.stdout
    out = tmp_path / "out.json"
    assert main([*args, "--output", str(out)]) == 0
    assert out.read_bytes() == piped.stdout


def test_batch_json_records_elapsed_seconds(case: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["batch", ".", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data["elapsed_seconds"], float) and data["elapsed_seconds"] >= 0
    assert main(["demo", "--format", "json"]) == 0
    assert "elapsed_seconds" not in json.loads(capsys.readouterr().out)


def test_output_folder_is_created_before_the_run(case: Path) -> None:
    out = case / "new" / "deeper" / "batch.json"
    assert main(["batch", ".", "--format", "json", "--output", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["cases"]
