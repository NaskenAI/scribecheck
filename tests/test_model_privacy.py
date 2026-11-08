"""Keys come from the environment only and never show up in logs, reprs, errors or reports;
endpoint locality and its warning; rules mode never loads the client (mocked HTTP only)."""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from _mock_http import Recorder, anthropic_message, json_response, openai_completion
from pydantic import BaseModel

from scribecheck.config import Config, config_from_mapping
from scribecheck.errors import InputError
from scribecheck.model import create_client
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.base import Block, Message, ModelConfigError, ModelRequest
from scribecheck.model.budget import Budget
from scribecheck.model.endpoint import Endpoint, Secret, api_key, privacy_warning
from scribecheck.model.openai_compat import OpenAICompatClient
from scribecheck.model.structured import complete
from scribecheck.report import Counts, ModelInfo, Report, UsageInfo

SENTINEL = "sk-test-SENTINEL-7f3a9c"
SCRIPT = Path(__file__).parent.parent / "scripts" / "smoke_model.py"


class Presence(BaseModel):
    present: bool


REQUEST = ModelRequest(
    system="s", messages=(Message("user", (Block("synthetic"),)),), output=Presence
)


def test_secret_never_prints_itself() -> None:
    secret = Secret(SENTINEL)
    for text in (repr(secret), str(secret), f"{secret}", f"{secret!r}"):
        assert SENTINEL not in text
    assert secret.reveal() == SENTINEL


def test_key_absent_from_logs_reprs_errors_and_reports(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    error = {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}
    recorder = Recorder(
        json_response({}, 529),
        json_response(anthropic_message('{"present": true}')),
        json_response(error, 401),
    )
    client = AnthropicClient(
        "claude-sonnet-5-5",
        api_key=Secret(SENTINEL),
        transport=recorder.transport,
        sleep=lambda _: None,
    )
    result = complete(client, REQUEST, Budget())
    with pytest.raises(ModelConfigError) as raised:
        complete(client, REQUEST, Budget())
    assert recorder.requests[0].headers["x-api-key"] == SENTINEL  # it was sent, only there
    report = Report(
        consultation_id="c1",
        mode="full",
        standard="clinical",
        model=ModelInfo(
            name="claude-sonnet-5-5",
            version=getattr(result, "model_id", None),
            endpoint_host=client.endpoint.host,
            provider="anthropic",
        ),
        decision="pass",
        counts=Counts(),
        usage=UsageInfo(calls=2),
    )
    for text in (
        caplog.text,
        repr(client),
        str(raised.value),
        repr(raised.value),
        report.to_json(),
    ):
        assert SENTINEL not in text
    assert "api.anthropic.com" in caplog.text  # logs say where, not what


def test_openai_key_absent_from_logs_and_errors(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    recorder = Recorder(json_response({"error": "invalid key"}, 401))
    client = OpenAICompatClient(
        "m",
        base_url="https://openrouter.ai/api/v1",
        api_key=Secret(SENTINEL),
        transport=recorder.transport,
    )
    with pytest.raises(ModelConfigError) as raised:
        complete(client, REQUEST, Budget())
    for text in (caplog.text, repr(client), str(raised.value)):
        assert SENTINEL not in text


def test_logs_never_contain_prompt_or_reply_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    request = ModelRequest(
        system="SYSTEM-SECRET-TEXT",
        messages=(Message("user", (Block("PATIENT-TRANSCRIPT-TEXT"),)),),
        output=Presence,
    )
    recorder = Recorder(
        json_response({}, 500), json_response(openai_completion('{"present": true}'))
    )
    client = OpenAICompatClient(
        "m", base_url="http://localhost:8000/v1", transport=recorder.transport, sleep=lambda _: None
    )
    complete(client, request, Budget())
    assert caplog.text
    for text in ("SYSTEM-SECRET-TEXT", "PATIENT-TRANSCRIPT-TEXT", "present"):
        assert text not in caplog.text


def test_keys_come_only_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "MY_KEY"):
        monkeypatch.delenv(name, raising=False)
    remote = Endpoint.from_url("https://api.anthropic.com")
    with pytest.raises(InputError, match="ANTHROPIC_API_KEY"):
        api_key("anthropic", remote)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    key = api_key("anthropic", remote)
    assert key is not None and key.reveal() == SENTINEL
    router = Endpoint.from_url("https://openrouter.ai/api/v1")
    with pytest.raises(InputError, match="OPENROUTER_API_KEY"):
        api_key("openai", router)
    with pytest.raises(InputError, match="OPENAI_API_KEY"):
        api_key("openai", Endpoint.from_url("https://api.openai.com/v1"))
    monkeypatch.setenv("MY_KEY", "k2")
    named = api_key("openai", router, "MY_KEY")
    assert named is not None and named.reveal() == "k2"
    assert api_key("openai", Endpoint.from_url("http://localhost:11434/v1")) is None


@pytest.mark.parametrize("name", ["api_key", "API_KEY", "key", "token"])
def test_a_key_in_the_config_file_is_rejected(name: str) -> None:
    with pytest.raises(InputError, match="never put an API key in the config file") as raised:
        config_from_mapping({"model": {"provider": "openai", name: SENTINEL}})
    assert SENTINEL not in str(raised.value)


def test_config_has_no_key_field() -> None:
    assert not {"api_key", "key", "token"} & set(Config.model_fields)


def test_create_client_reads_the_environment_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    caplog.set_level(logging.WARNING)
    client = create_client(Config(provider="anthropic", model="claude-sonnet-5-5", effort="low"))
    assert isinstance(client, AnthropicClient)
    assert client.endpoint.host == "api.anthropic.com"
    assert "not on this machine or the local network" in caplog.text
    assert SENTINEL not in caplog.text and SENTINEL not in repr(client)
    caplog.clear()
    local = create_client(Config(provider="openai", model="m", base_url="http://127.0.0.1:8080/v1"))
    assert isinstance(local, OpenAICompatClient) and caplog.text == ""


def test_create_client_needs_a_provider_and_model() -> None:
    with pytest.raises(InputError, match="provider and name"):
        create_client(Config())
    with pytest.raises(InputError, match="base_url"):
        create_client(Config(provider="openai", model="m"))


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:8000", True),
        ("http://[::1]:8000", True),
        ("http://10.1.2.3/v1", True),
        ("http://172.16.0.5/v1", True),
        ("http://192.168.1.20:1234/v1", True),
        ("http://100.64.0.7/v1", True),
        ("http://gpu-box.local:8000/v1", True),
        ("http://vllm.localhost/v1", True),
        ("https://api.anthropic.com", False),
        ("https://openrouter.ai/api/v1", False),
        ("http://8.8.8.8/v1", False),
        ("https://my-gpu.example.com/v1", False),
    ],
)
def test_endpoint_locality(url: str, local: bool) -> None:
    endpoint = Endpoint.from_url(url)
    assert endpoint.is_local is local
    assert (privacy_warning(endpoint) is None) is local


def test_bad_base_url() -> None:
    with pytest.raises(InputError, match="http"):
        Endpoint.from_url("localhost:11434")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_model", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_script_against_a_mock(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    recorder = Recorder(
        json_response(anthropic_message('{"answer": true}', model="claude-sonnet-5-5-test")),
        json_response(openai_completion('{"answer": true}', model="local-test")),
    )
    script = _load_script()
    code = script.main(
        ["--anthropic", "--base-url", "http://localhost:11434/v1", "--model", "m"],
        transport=recorder.transport,
    )
    out = capsys.readouterr()
    assert code == 0
    assert "[anthropic] model id: claude-sonnet-5-5-test" in out.out
    assert "[anthropic] parsed: yes" in out.out
    assert "[anthropic] warning:" in out.out
    assert "[openai-compatible] model id: local-test" in out.out
    assert "[openai-compatible] usage: input 12, output 4" in out.out
    assert SENTINEL not in out.out + out.err
    body = json.loads(recorder.requests[0].content)
    assert body["output_config"]["effort"] == "low" and "temperature" not in body


def test_smoke_script_without_a_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    recorder = Recorder()
    assert _load_script().main(["--anthropic"], transport=recorder.transport) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert recorder.requests == []


def test_rules_mode_never_loads_the_model_package(tmp_path: Path) -> None:
    case = tmp_path / "k.transcript.txt"
    case.write_text("Patient: I take metformin 500 mg twice a day.\n", encoding="utf-8")
    (tmp_path / "k.note.md").write_text("Plan:\nReview.\n", encoding="utf-8")
    code = (
        "import sys\n"
        "from scribecheck.cli import main\n"
        f"main(['demo', '--format', 'json', '--output', {str(tmp_path / 'o1.json')!r}])\n"
        f"main(['check', '-t', {str(case)!r}, '-n', {str(tmp_path / 'k.note.md')!r}, "
        f"'--output', {str(tmp_path / 'o2.md')!r}])\n"
        f"main(['batch', {str(tmp_path)!r}, '--output', {str(tmp_path / 'o3.md')!r}])\n"
        "import scribecheck\n"
        "loaded = sorted(m for m in sys.modules if m.startswith(('scribecheck.model', 'httpx')))\n"
        "print(loaded)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_no_test_reaches_the_network_through_the_client() -> None:
    client = AnthropicClient("claude-sonnet-5-5", api_key=Secret("sk-test"), sleep=lambda _: None)
    with pytest.raises((RuntimeError, httpx.HTTPError)):
        client.send(REQUEST)  # the network guard blocks a real call
