"""Real Phase 6.3 read-only smoke: context -> plan -> select -> execute -> observe -> final."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend_ai.agent import AutonomousLoopRequest, AutonomousToolLoop, LoopStatus, ToolRegistry


class SmokeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class SmokeEngine:
    tokenizer = SmokeTokenizer()

    def __init__(self) -> None:
        self._outputs = iter((
            'ACTION: TOOL\nARGS: {"tool":"search_code","arguments":{"query":"AutonomousToolLoop"}}',
            'ACTION: FINAL\nARGS: {"message":"read-only smoke complete"}',
        ))
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(generated_text=next(self._outputs))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    engine = SmokeEngine()
    loop = AutonomousToolLoop(engine, registry=ToolRegistry.default())
    result = loop.run(AutonomousLoopRequest("Investigate this project", root))
    assert result.status is LoopStatus.COMPLETED
    assert result.final_answer == "read-only smoke complete"
    assert [call.name for call in result.tool_calls] == ["project_context", "search_code"]
    assert all(tool_result.success for tool_result in result.tool_results)
    assert result.steps[0].selected_tool == "project_context"
    assert result.steps[-1].selected_tool is None
    assert len(engine.prompts) == 2
    print("Phase 6.3 autonomous tool-loop smoke passed")


if __name__ == "__main__":
    main()
