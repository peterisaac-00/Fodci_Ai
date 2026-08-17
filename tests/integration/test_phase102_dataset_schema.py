from __future__ import annotations

from pathlib import Path

from backend_ai.agent.dataset_schema import DatasetRecord
from backend_ai.agent.experience_dataset import ExperienceDatasetExtractor
from backend_ai.agent.experience_records import ExperienceEvaluation, ExperienceProjectIdentity, ExperienceRecords, ExperienceVerification


STAMP = "2026-08-17T00:00:00Z"


def _candidate():
    records = ExperienceRecords(clock=lambda: STAMP)
    session = records.start_experience("Repair service validation", project_identity=ExperienceProjectIdentity("project-schema", "/tmp/project-schema"))
    attempt_id = session.start_attempt()
    session.record_action("inspect", "Inspect service", attempt_id=attempt_id)
    session.record_observation("Validation failure reproduced", source="test", attempt_id=attempt_id)
    error = session.record_error("test_failure", "Expected status differed", source="test", attempt_id=attempt_id)
    session.record_correction("Update validation branch", "tests passed", error_id=error.error_id, attempt_id=attempt_id)
    session.record_attempt_result("Service validation repaired", attempt_id=attempt_id)
    session.record_verification(ExperienceVerification(3, 3, 0, "PASS", "integration checks passed", STAMP))
    session.record_evaluation(ExperienceEvaluation(0.95, "accepted", "historical result recorded", ({"criterion": "integration", "passed": True},), {"source": "local"}))
    source = session.finalize(status="completed", outcome="success", final_solution="Repair validation branch", final_summary="Service validation repaired")
    return records, source, ExperienceDatasetExtractor().extract(source)


def test_phase101_candidate_round_trips_through_phase102_schema(tmp_path: Path) -> None:
    records, source, candidate = _candidate()
    source_before = source.to_dict()
    record = DatasetRecord.from_candidate(candidate)
    serialized = record.to_json()
    restored = DatasetRecord.from_json(serialized)

    assert restored.to_dict() == record.to_dict()
    assert restored.experience_id == source.experience_id
    assert restored.task == source.task
    assert restored.trajectory.to_dict() == record.trajectory.to_dict()
    assert restored.verification.to_dict() == record.verification.to_dict()
    assert restored.evaluation.to_dict() == record.evaluation.to_dict()
    assert restored.outcome.value == source.outcome.value
    assert restored.provenance.experience_id == source.experience_id
    assert records.get(source.experience_id).to_dict() == source_before
    assert str(tmp_path) not in serialized
