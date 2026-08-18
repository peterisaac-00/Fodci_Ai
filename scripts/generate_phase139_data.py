from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "training_data" / "testing_qa" / "train"
VALIDATION = ROOT / "training_data" / "testing_qa" / "validation"
BENCHMARK = ROOT / "training_data" / "testing_qa" / "evaluation" / "phase_139.jsonl"

TRAIN_RECORDS = [
    ("unit", "What is the purpose of a unit test in a backend project?", "A unit test isolates one small behavior, controls its collaborators, and makes a precise assertion about inputs, outputs, or state. It should be fast, deterministic, and explain the contract that failed.", "None"),
    ("unit", "How should a Pytest test name communicate intent?", "Use a stable name that describes the behavior and relevant condition, such as `test_rejects_expired_token`. A useful name identifies the contract without encoding incidental implementation details.", "None"),
    ("unit", "What makes a unit test deterministic?", "Control time, randomness, I/O, environment, and external collaborators through explicit seams or fakes. Avoid dependence on test order, wall-clock timing, network services, or shared mutable state.", "None"),
    ("unit", "How should a test assert an exception with Pytest?", "Use `pytest.raises` around the smallest operation expected to fail and assert the exception type and stable message or attributes when those are part of the contract. Do not wrap the entire test and hide unrelated failures.", "None"),
    ("unit", "Why should unit tests avoid testing private implementation details?", "Private-detail assertions make harmless refactors look like regressions. Prefer observable behavior, public contracts, state transitions, and externally meaningful side effects.", "None"),
    ("unit", "How should parameterization improve a Pytest suite?", "Use `pytest.mark.parametrize` for a finite table of related inputs and expected outcomes. Give cases readable IDs and keep each case focused so failures identify the boundary condition.", "None"),
    ("unit", "What should a unit test do when a dependency raises an error?", "Use a controlled fake or mock to produce the expected dependency failure, then assert the unit maps it to the documented result and does not swallow unexpected exceptions or retry without a bound.", "None"),
    ("unit", "How should a unit test protect a security-sensitive behavior?", "Assert both the allowed behavior and the rejection paths, including no secret leakage, correct authorization boundaries, and that the protected side effect is not invoked after a failed check.", "None"),
    ("integration", "What does an integration test verify?", "An integration test exercises a meaningful boundary between components such as an API, database, queue, or authentication middleware. It verifies wiring, serialization, configuration, and real collaboration rather than one isolated function.", "None"),
    ("integration", "How should an API integration test be structured?", "Arrange a controlled application and dependency environment, send a realistic request through the public boundary, assert status and response schema, and verify important side effects or persistence without relying on production services.", "None"),
    ("integration", "Why should integration tests isolate external services?", "Use disposable local services, test containers where approved, or contract-compatible fakes so the suite is repeatable and safe. Never make ordinary tests depend on an uncontrolled network or shared production data.", "None"),
    ("integration", "What should a database integration test verify beyond a repository mock?", "Verify migrations, queries, transactions, constraints, serialization, and rollback behavior against a controlled database implementation. A repository mock cannot reveal SQL or schema integration defects.", "None"),
    ("integration", "How should integration test cleanup be handled?", "Use transaction rollback, isolated schemas, disposable databases, or explicit fixture teardown. Cleanup must run after failures and must not depend on test order or a best-effort global delete.", "None"),
    ("integration", "What is a contract test useful for between services?", "It verifies that a provider and consumer agree on request shape, response schema, status semantics, and error behavior without requiring every end-to-end deployment combination.", "None"),
    ("integration", "How should an integration test handle asynchronous work?", "Control the queue or clock where possible, await observable completion with a bounded timeout, and assert the resulting state. Do not use unbounded sleeps that create flaky or slow tests.", "None"),
    ("integration", "When is an end-to-end test justified?", "Use it for a small number of critical user journeys that cross the deployed stack. Keep lower-level behavior covered by unit and integration tests so end-to-end failures remain diagnosable.", "None"),
    ("fixtures", "What is a Pytest fixture?", "A fixture is a named, reusable setup and cleanup provider injected into tests. It should expose the smallest useful resource, have an explicit scope, and guarantee cleanup when setup succeeds.", "None"),
    ("fixtures", "How should fixture scope be selected?", "Choose the narrowest scope that is fast enough: function scope for isolation, module or session scope only for immutable expensive resources with safe reset semantics. Broad scope must not leak mutable state.", "None"),
    ("fixtures", "Why are fakes often preferable to unrestricted mocks?", "A small fake models the collaborator's relevant behavior and keeps tests readable, while unrestricted mocks can validate an imagined call sequence without proving a useful contract.", "None"),
    ("fixtures", "How should a Pytest fixture manage temporary files?", "Use `tmp_path` or an equivalent managed temporary directory, create only the needed files, and let the fixture lifecycle remove them. Never write tests into the source tree or a shared fixed path.", "None"),
    ("fixtures", "What is a safe use of monkeypatch in Pytest?", "Patch an explicit module boundary for one test, record the intended replacement, and let Pytest restore it automatically. Do not patch broad global behavior or conceal missing dependency injection.", "None"),
    ("fixtures", "How should tests control time?", "Inject a clock or patch a narrow time provider and make the test's timestamps explicit. Avoid sleeping to wait for time-based behavior; advance the controlled clock or use a bounded polling helper.", "None"),
    ("fixtures", "How should flaky tests be diagnosed before adding retries?", "Collect the failure pattern, isolate shared state and timing assumptions, run the test repeatedly or in a controlled order, and fix the cause. A retry can hide a real race and must not replace diagnosis.", "None"),
    ("fixtures", "What should a test double verify about calls?", "Assert calls only when the interaction is part of the contract, such as an idempotency or audit requirement. Avoid asserting incidental call order or exact internal calls that do not affect observable behavior.", "None"),
    ("coverage", "What does line coverage measure?", "Line coverage measures which executable lines ran during a selected test run. It indicates exercised surface, not whether assertions were meaningful, branches were correct, or behavior is secure.", "None"),
    ("coverage", "Why is branch coverage useful?", "Branch coverage shows whether alternative control-flow outcomes ran, such as success, validation failure, exception, and authorization paths. It helps expose untested decisions but is not proof of correctness.", "None"),
    ("coverage", "How should a team use a coverage threshold?", "Treat the threshold as a guardrail that prevents regression, not as a target achieved by trivial assertions. Review changed code and important risk paths even when the aggregate percentage is high.", "None"),
    ("coverage", "What does a coverage report fail to prove?", "Coverage does not prove assertion quality, realistic integration, concurrency safety, performance, security, or correctness of unexecuted data combinations. It must be combined with review and targeted tests.", "None"),
    ("coverage", "How should uncovered lines be triaged?", "Classify them as a missing test, unreachable or obsolete code, generated code, or an intentional exclusion with documented justification. Do not add meaningless tests solely to inflate the number.", "None"),
    ("coverage", "How can coverage be measured for changed code?", "Compare the changed-file or diff coverage with the baseline, ensure important new branches are exercised, and keep the comparison reproducible through a fixed command and configuration.", "None"),
    ("coverage", "Why should coverage combine unit and integration runs?", "Unit tests provide fast detailed coverage while integration tests cover wiring and real boundaries. Either alone can leave important paths unmeasured or falsely reassuring.", "None"),
    ("coverage", "How should a project report a coverage regression?", "Report the baseline and current percentage, changed files, uncovered risk-relevant lines or branches, and the corresponding test plan. A small percentage change can still be serious if it removes a critical security path.", "None"),
]

VALIDATION_RECORDS = [
    ("unit", "How should a Pytest unit test verify a validator rejects malformed input?", "Call the validator with one malformed case, assert the documented exception or result and stable field information, and keep the assertion scoped so unrelated failures cannot be mistaken for validation behavior.", "Input: {email: invalid}"),
    ("unit", "What should a unit test prove when a permission check fails?", "Assert the stable denial result and prove that the protected side effect was not called. Include the relevant principal and resource boundary without asserting private implementation details.", "principal.role=viewer; operation=delete"),
    ("integration", "What should an API integration test verify after creating a resource?", "Send the request through the application boundary, assert status and response schema, then read the controlled persistence boundary or published event and verify the created resource and its important invariants.", "POST /items with a valid JSON body"),
    ("integration", "How should an integration test prove transaction rollback?", "Trigger a controlled failure after a write is attempted, assert the documented error, and query the isolated database to prove the partial state was rolled back. Cleanup must still run.", "Insert item succeeds; second constraint fails"),
    ("fixtures", "How should a fixture prevent test-order dependence?", "Use function-scoped isolated state or reset shared state deterministically before each test. Do not rely on another test to create data or on a global mutable fixture that is never cleaned.", "A test passes alone but fails after the full module."),
    ("fixtures", "When should a test use a fake instead of a mock?", "Use a small fake when the collaborator has meaningful behavior that should be exercised across cases; use a narrow mock only when verifying a contractually important interaction such as a required audit call.", "The service depends on a notification gateway."),
    ("coverage", "How should a team interpret 95 percent line coverage with an untested authorization branch?", "Do not accept the percentage as sufficient. Add focused tests for allow and deny outcomes because a small untested security branch can be higher risk than many covered low-risk lines.", "Coverage report: 95% lines; authorization deny branch uncovered."),
    ("coverage", "What evidence belongs in a trustworthy coverage report?", "Record the exact test command, configuration, source revision, measured line or branch totals, changed-file coverage, exclusions, and uncovered risk-relevant areas so the result can be reproduced and reviewed.", "CI produced a coverage.xml artifact."),
]


def write_records(directory: Path, records: list[tuple[str, str, str, str]], prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, (category, instruction, response, input_text) in enumerate(records, start=1):
        text = f"### Instruction\n{instruction}\n\n### Input\n{input_text}\n\n### Response\n{response}\n"
        (directory / f"{prefix}_{index:03d}_{category}.txt").write_text(text, encoding="utf-8")


def write_benchmark() -> None:
    BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    keyword_map = {
        "unit": ["assert", "unit", "deterministic"],
        "integration": ["integration", "boundary", "cleanup"],
        "fixtures": ["fixture", "isolated", "cleanup"],
        "coverage": ["coverage", "branch", "tests"],
    }
    with BENCHMARK.open("w", encoding="utf-8", newline="\n") as stream:
        for index, (category, question, answer, input_text) in enumerate(VALIDATION_RECORDS, start=1):
            record = {
                "benchmark_id": f"phase139_{index:03d}",
                "version": "1.0",
                "split": "benchmark",
                "category": category,
                "question": question,
                "expected_answer": answer,
                "required_keywords": keyword_map[category],
                "minimum_keyword_coverage": 0.66,
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    write_records(TRAIN, TRAIN_RECORDS, "train")
    write_records(VALIDATION, VALIDATION_RECORDS, "validation")
    write_benchmark()
    print(f"generated train={len(TRAIN_RECORDS)} validation={len(VALIDATION_RECORDS)} benchmark={len(VALIDATION_RECORDS)}")


if __name__ == "__main__":
    main()
