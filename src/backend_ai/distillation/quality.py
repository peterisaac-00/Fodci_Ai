"""Phase 15.3 quality filtering and verification gates."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from backend_ai.distillation.contract import (
    QualityStatus,
    RecordSplit,
    RedactionStatus,
    TeacherStudentExample,
    VerificationStatus,
)


_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]+-----"),
    re.compile(r"\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*[^\s]+", re.IGNORECASE),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
)


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    accepted: bool
    reasons: tuple[str, ...]
    secret_detected: bool
    word_count: int
    repeated_token_rate: float
    backend_signal: bool
    code_present: bool


class QualityFilter:
    """Conservative deterministic filter; it never claims semantic correctness."""

    def __init__(self, *, min_words: int = 3, max_words: int = 512, max_repeated_token_rate: float = 0.55) -> None:
        if not 1 <= min_words <= max_words:
            raise ValueError("word bounds are invalid")
        if not 0.0 <= max_repeated_token_rate < 1.0:
            raise ValueError("repetition bound is invalid")
        self.min_words = min_words
        self.max_words = max_words
        self.max_repeated_token_rate = max_repeated_token_rate

    def assess(self, prompt: str, response: str) -> QualityAssessment:
        reasons: list[str] = []
        combined = f"{prompt}\n{response}"
        secret_detected = any(pattern.search(combined) for pattern in _SECRET_PATTERNS)
        if secret_detected:
            reasons.append("secret-like content detected")
        words = re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]*", response or "")
        word_count = len(words)
        if word_count < self.min_words:
            reasons.append("response too short")
        if word_count > self.max_words:
            reasons.append("response exceeds length bound")
        normalized = [word.lower() for word in words]
        repeated = sum(max(0, normalized.count(word) - 1) for word in set(normalized))
        repeated_rate = repeated / word_count if word_count else 1.0
        if repeated_rate > self.max_repeated_token_rate:
            reasons.append("response excessively repetitive")
        backend_terms = ("backend", "api", "http", "fastapi", "sql", "database", "jwt", "pytest", "python", "endpoint", "request", "response", "test", "debug")
        backend_signal = any(term in " ".join(normalized) for term in backend_terms)
        if not backend_signal:
            reasons.append("no backend signal")
        code_present = "```" in (response or "") or bool(re.search(r"\b(?:def|class|SELECT|async def|from [A-Za-z_]+ import)\b", response or ""))
        return QualityAssessment(
            accepted=not reasons,
            reasons=tuple(reasons),
            secret_detected=secret_detected,
            word_count=word_count,
            repeated_token_rate=round(repeated_rate, 6),
            backend_signal=backend_signal,
            code_present=code_present,
        )

    def filter_example(self, example: TeacherStudentExample) -> tuple[TeacherStudentExample, QualityAssessment]:
        assessment = self.assess(example.prompt, example.response)
        if assessment.accepted:
            return example, assessment
        return example.mark_verified(
            verification_status=VerificationStatus.REJECTED,
            quality_status=QualityStatus.REJECTED,
            split=RecordSplit.BUFFER,
            redaction_status=RedactionStatus.BLOCKED if assessment.secret_detected else RedactionStatus.CLEAN,
            execution_evidence={"quality_reasons": assessment.reasons},
        ), assessment


class VerificationGate:
    """Promote only filtered examples with explicit evidence or human approval."""

    def __init__(self, quality_filter: QualityFilter | None = None) -> None:
        self.quality_filter = quality_filter or QualityFilter()

    def verify(
        self,
        example: TeacherStudentExample,
        *,
        split: RecordSplit,
        execution_passed: bool = False,
        human_approved: bool = False,
        redaction_status: RedactionStatus = RedactionStatus.CLEAN,
        execution_evidence: dict | None = None,
    ) -> tuple[TeacherStudentExample, QualityAssessment]:
        assessment = self.quality_filter.assess(example.prompt, example.response)
        if not assessment.accepted:
            rejected = example.mark_verified(
                verification_status=VerificationStatus.REJECTED,
                quality_status=QualityStatus.REJECTED,
                split=RecordSplit.BUFFER,
                redaction_status=RedactionStatus.BLOCKED if assessment.secret_detected else redaction_status,
                execution_evidence={"quality_reasons": assessment.reasons},
            )
            return rejected, assessment
        if redaction_status not in {RedactionStatus.CLEAN, RedactionStatus.REDACTED}:
            rejected = example.mark_verified(
                verification_status=VerificationStatus.REJECTED,
                quality_status=QualityStatus.REJECTED,
                split=RecordSplit.BUFFER,
                redaction_status=RedactionStatus.BLOCKED,
                execution_evidence={"quality_reasons": ("redaction not complete",)},
            )
            return rejected, QualityAssessment(False, ("redaction not complete",), False, assessment.word_count, assessment.repeated_token_rate, assessment.backend_signal, assessment.code_present)
        if not execution_passed and not human_approved:
            pending = example.mark_verified(
                verification_status=VerificationStatus.UNVERIFIED,
                quality_status=QualityStatus.PENDING,
                split=RecordSplit.BUFFER,
                redaction_status=redaction_status,
                execution_evidence={"quality_reasons": ("positive evidence required",)},
            )
            return pending, QualityAssessment(True, ("positive evidence required",), False, assessment.word_count, assessment.repeated_token_rate, assessment.backend_signal, assessment.code_present)
        status = VerificationStatus.EXECUTION_PASS if execution_passed else VerificationStatus.HUMAN_APPROVED
        accepted = example.mark_verified(
            verification_status=status,
            quality_status=QualityStatus.ACCEPTED,
            split=split,
            redaction_status=redaction_status,
            user_approved=human_approved,
            execution_evidence=execution_evidence or {"verified": True},
        )
        return accepted, assessment


__all__ = ["QualityAssessment", "QualityFilter", "VerificationGate"]
