from __future__ import annotations

import pytest
from backend_ai.agent.recovery import (
    ErrorCategory,
    ErrorClassification,
    ErrorClassifier,
    RecoverabilityPolicy,
    RecoveryAction,
    RecoveryConfidence,
    RecoveryContext,
    RecoveryStatus,
    normalize_error,
    compute_error_signature,
    NormalizedError,
)
from backend_ai.agent.models import ToolResult


def test_normalize_tool_error() -> None:
    result = ToolResult(call_id="call-1", tool_name="read_file", success=False, error_code="FILE_NOT_FOUND", message="No such file: foo.py")
    normalized = normalize_error(result, tool_name="read_file")
    assert normalized.category == ErrorCategory.FILE_NOT_FOUND
    assert normalized.tool_name == "read_file"
    assert "foo.py" in normalized.message


def test_normalize_command_error() -> None:
    result = ToolResult(call_id="call-1", tool_name="run_tests", success=False, error_code="TEST_FAILURE", message="pytest failed")
    normalized = normalize_error(result, tool_name="run_tests", command="pytest")
    assert normalized.category == ErrorCategory.TEST_FAILURE
    assert normalized.command == "pytest"


def test_classify_file_error() -> None:
    classifier = ErrorClassifier()
    result = ToolResult(call_id="call-1", tool_name="read_file", success=False, error_code="FILE_NOT_FOUND", message="ENOENT: no such file")
    classification = classifier.classify(result)
    assert classification.category == ErrorCategory.FILE_NOT_FOUND
    assert classification.recoverable is True


def test_classify_dependency_error() -> None:
    classifier = ErrorClassifier()
    result = ToolResult(call_id="call-1", tool_name="pip", success=False, error_code="DEPENDENCY_ERROR", message="ModuleNotFoundError: No module named 'numpy'")
    classification = classifier.classify(result)
    assert classification.category == ErrorCategory.DEPENDENCY_ERROR
    assert classification.recoverable is True


def test_signature_stability() -> None:
    result1 = ToolResult(call_id="call-1", tool_name="run_tests", success=False, error_code="TEST_FAILURE", message="FAILED tests/unit/test_foo.py - AssertionError")
    result2 = ToolResult(call_id="call-1", tool_name="run_tests", success=False, error_code="TEST_FAILURE", message="FAILED tests/unit/test_foo.py - AssertionError (timestamp 12345)")
    sig1 = compute_error_signature(result1, tool_name="run_tests", command="pytest")
    sig2 = compute_error_signature(result2, tool_name="run_tests", command="pytest")
    assert sig1 == sig2


def test_recovery_policy_decision() -> None:
    result = ToolResult(call_id="call-1", tool_name="read_file", success=False, error_code="FILE_NOT_FOUND", message="Missing file")
    context = RecoveryContext(tool_result=result, operation="read_file", has_next_plan_step=True)
    policy = RecoverabilityPolicy()
    rec_result = policy.decide(context)
    assert rec_result.decision.status == RecoveryStatus.CONTINUE
    assert rec_result.decision.action == RecoveryAction.INSPECT
