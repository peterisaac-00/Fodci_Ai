"""Shared contracts and startup primitives."""

from backend_ai.core.bootstrap import bootstrap
from backend_ai.core.project import (
    InvalidProjectRootError,
    ProjectContext,
    resolve_project_context,
)
from backend_ai.core.contracts import (
    Agent,
    Evaluator,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    Memory,
    Message,
    Tool,
)

__all__ = [
    "Agent",
    "Evaluator",
    "InvalidProjectRootError",
    "LLMProviderError",
    "LLMRequest",
    "LLMResponse",
    "LLMProvider",
    "Memory",
    "Message",
    "ProjectContext",
    "Tool",
    "bootstrap",
    "resolve_project_context",
]
