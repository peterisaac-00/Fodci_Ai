"""Offline Teacher–Student distillation contracts and workflows."""

from backend_ai.distillation.buffer import BufferError, BufferStats, InteractionBuffer
from backend_ai.distillation.quality import QualityAssessment, QualityFilter, VerificationGate
from backend_ai.distillation.shadow import PromotionDecision, PromotionPolicy, ShadowMode, ShadowResult
from backend_ai.distillation.contract import (
    EXAMPLE_FORMAT,
    SCHEMA_VERSION,
    ContractError,
    QualityStatus,
    RecordSource,
    RecordSplit,
    RedactionStatus,
    TeacherStudentExample,
    VerificationStatus,
)

__all__ = [
    "BufferError",
    "BufferStats",
    "InteractionBuffer",
    "QualityAssessment",
    "QualityFilter",
    "VerificationGate",
    "PromotionDecision",
    "PromotionPolicy",
    "ShadowMode",
    "ShadowResult",
    "EXAMPLE_FORMAT",
    "SCHEMA_VERSION",
    "ContractError",
    "QualityStatus",
    "RecordSource",
    "RecordSplit",
    "RedactionStatus",
    "TeacherStudentExample",
    "VerificationStatus",
]
