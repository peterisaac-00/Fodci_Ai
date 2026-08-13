from __future__ import annotations

from backend_ai.agent import Agent
from backend_ai.evaluation import Evaluator
from backend_ai.llm import LLMProvider
from backend_ai.memory import Memory
from backend_ai.tools import Tool


def test_core_contracts_are_importable_from_their_boundaries() -> None:
    for contract in (Agent, LLMProvider, Tool, Memory, Evaluator):
        assert getattr(contract, "__protocol_attrs__", None) is not None
