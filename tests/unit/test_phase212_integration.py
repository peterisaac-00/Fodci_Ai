from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from backend_ai.application import Application, DEFAULT_CHECKPOINT_RELATIVE_PATH
from backend_ai.config import Settings
from backend_ai.core.contracts import LLMRequest, LLMResponse
from backend_ai.llm import LLMProviderError
from backend_ai.terminal import InteractiveSession


class SequenceInputProvider:
    def __init__(self, values: list[str | None]) -> None:
        self.values = iter(values)

    def read(self) -> str | None:
        return next(self.values)


class RecordingProvider:
    system_prompt = "test system prompt"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.responses = iter(("first answer", "second answer", "third answer"))

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=next(self.responses))


def test_session_delegates_multiple_messages_and_preserves_bounded_history() -> None:
    output = StringIO()
    provider = RecordingProvider()
    session = InteractiveSession(
        output=output,
        input_provider=SequenceInputProvider(
            ["first question\n", "second question\n", "third question\n", "/exit\n"]
        ),
        provider=provider,
        max_history_messages=4,
    )

    session.run()

    assert len(provider.requests) == 3
    assert provider.requests[0].messages == (
        provider.requests[0].messages[0],
        provider.requests[0].messages[1],
    )
    assert provider.requests[0].messages[0].role == "system"
    assert provider.requests[0].messages[1].content == "first question"
    assert [message.role for message in provider.requests[1].messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [message.content for message in session.conversation_history] == [
        "second question",
        "second answer",
        "third question",
        "third answer",
    ]
    text = output.getvalue()
    assert "Fodci > first answer" in text
    assert "Fodci > second answer" in text
    assert "Fodci > third answer" in text
    assert "Goodbye." in text


def test_session_empty_input_does_not_call_provider_or_add_history() -> None:
    provider = RecordingProvider()
    session = InteractiveSession(
        input_provider=SequenceInputProvider(["\n", "/exit\n"]),
        provider=provider,
    )

    session.run()

    assert provider.requests == []
    assert session.conversation_history == []
    assert session.received_inputs == ["", "/exit"]


def test_session_provider_failure_is_user_facing_and_does_not_add_failed_turn() -> None:
    class FailingProvider:
        system_prompt = "test system prompt"

        def generate(self, request: LLMRequest) -> LLMResponse:
            raise LLMProviderError("context length exceeded")

    output = StringIO()
    session = InteractiveSession(
        output=output,
        input_provider=SequenceInputProvider(["long prompt\n", "/exit\n"]),
        provider=FailingProvider(),
    )

    session.run()

    assert "Fodci error: context length exceeded" in output.getvalue()
    assert session.conversation_history == []


def test_application_provider_factory_is_called_once_with_project_checkpoint(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, log_level="INFO")
    session = InteractiveSession()
    calls: list[Path] = []
    provider = RecordingProvider()

    def factory(checkpoint: Path) -> RecordingProvider:
        calls.append(checkpoint)
        return provider

    application = Application(settings, session=session, provider_factory=factory)

    application.start()
    application.start()

    assert calls == [tmp_path / DEFAULT_CHECKPOINT_RELATIVE_PATH]
    assert application.provider is provider
    assert session.provider is provider


def test_application_model_boundary_never_exposes_fodci_model_to_cli() -> None:
    from importlib import import_module

    cli_module = import_module("backend_ai.cli.main")
    source = Path(cli_module.__file__).read_text(encoding="utf-8")

    assert "FodciModel" not in source
    assert "InferenceEngine" not in source
    assert "torch" not in source
    assert "checkpoint" not in source.lower()


def test_context_limit_error_from_local_provider_is_explicit() -> None:
    class ContextLimitedEngine:
        def generate(self, prompt: str) -> SimpleNamespace:
            raise ValueError("prompt exceeds context length 256; the prompt will not be truncated")

    from backend_ai.llm import FodciLocalProvider

    provider = FodciLocalProvider(ContextLimitedEngine())
    output = StringIO()
    session = InteractiveSession(
        output=output,
        input_provider=SequenceInputProvider(["prompt\n", "/exit\n"]),
        provider=provider,
    )

    session.run()

    assert "context length 256" in output.getvalue()


def test_local_integration_has_no_external_execution_or_network_calls() -> None:
    provider_source = (
        Path(__file__).resolve().parents[2] / "src" / "backend_ai" / "llm" / "fodci_provider.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "requests.",
        "urllib.",
        "socket.",
        "subprocess.",
        "os.system",
        "http://",
        "https://",
        "tool_call",
        "invoke_tool",
    )

    assert all(pattern not in provider_source for pattern in forbidden)
