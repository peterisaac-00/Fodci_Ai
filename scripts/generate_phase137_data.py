from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "training_data" / "debugging" / "train"
VALIDATION = ROOT / "training_data" / "debugging" / "validation"
BENCHMARK = ROOT / "training_data" / "debugging" / "evaluation" / "phase_137.jsonl"

TRAIN_RECORDS = [
    ("traceback_reading", "Read this traceback and identify the immediate failure: `NameError: name 'user_idd' is not defined` at `return users[user_idd]`.", "The immediate failure is a NameError caused by the misspelled variable `user_idd`. Inspect the local scope and correct it to the intended `user_id` rather than changing unrelated lookup logic.", "Traceback (most recent call last):\n  File \"service.py\", line 18, in find_user\n    return users[user_idd]\nNameError: name 'user_idd' is not defined"),
    ("traceback_reading", "How should the final exception line and the deepest application frame be used when reading a traceback?", "The final exception line names the failure type and message, while the deepest relevant application frame identifies where the bad state was first observed. Read upward for the call path and inspect surrounding values before patching.", "None"),
    ("traceback_reading", "What is the likely cause of this error at an API boundary?", "`KeyError: 'email'` means code indexed a dictionary assuming `email` existed. The boundary either received an incomplete payload or validation was bypassed; use explicit validation and safe access according to the contract.", "Traceback (most recent call last):\n  File \"api.py\", line 42, in create_user\n    email = payload['email']\nKeyError: 'email'"),
    ("traceback_reading", "How should an AttributeError be analyzed when a value can be None?", "Check the value's producer and the contract that permits None before the failing attribute access. The fix should validate or handle the absent value at the correct boundary, not blindly add a broad catch.", "Traceback (most recent call last):\n  File \"orders.py\", line 73, in total\n    return order.customer.name\nAttributeError: 'NoneType' object has no attribute 'name'"),
    ("traceback_reading", "What does `JSONDecodeError` usually indicate in a backend client?", "The client attempted to parse a response that was not valid JSON, possibly an empty body, HTML error page, or malformed payload. Inspect status and content type before decoding and preserve safe diagnostic context.", "Traceback (most recent call last):\n  File \"client.py\", line 31, in fetch\n    return response.json()\njson.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)"),
    ("traceback_reading", "How should a ModuleNotFoundError be triaged before changing application code?", "Confirm the import name, active environment, package installation, and project packaging configuration. Do not immediately rewrite imports or install an unapproved dependency; the cause may be environment drift.", "Traceback (most recent call last):\n  File \"worker.py\", line 4, in <module>\n    from payments.client import Gateway\nModuleNotFoundError: No module named 'payments'"),
    ("traceback_reading", "What is the root cause of a ZeroDivisionError in a metric endpoint?", "The metric divides by a zero denominator, so the code needs a defined empty-data policy such as returning zero, null, or a documented client error. The repair must match the metric contract.", "Traceback (most recent call last):\n  File \"metrics.py\", line 27, in conversion_rate\n    return completed / total\nZeroDivisionError: division by zero"),
    ("traceback_reading", "How should a SyntaxError be fixed safely?", "Use the reported file and line as the starting point, inspect the nearby delimiter or indentation, and make the smallest syntax correction. Then compile or run the focused test before broader changes.", "Traceback (most recent call last):\n  File \"routes.py\", line 22\n    return response(\n                    ^\nSyntaxError: '(' was never closed"),
    ("root_cause", "How can an engineer distinguish a root cause from a downstream symptom?", "Trace the first invalid state backward through the call chain and data transformations. A later 500 response or assertion failure may be a symptom of an earlier validation, configuration, or dependency error.", "None"),
    ("root_cause", "Why should the first failing test be investigated before later failures?", "Later failures may cascade from shared corrupted state or one broken dependency. The first independent failure usually provides the strongest evidence for the root cause.", "None"),
    ("root_cause", "How should a missing `await` be diagnosed in an async service?", "If a coroutine object reaches code expecting its result, inspect the call chain for a missing await and verify the function's async contract. Add the await at the boundary where the result is needed.", "Traceback (most recent call last):\n  File \"service.py\", line 55, in get_name\n    return user.name\nAttributeError: 'coroutine' object has no attribute 'name'"),
    ("root_cause", "What evidence should be collected before changing a failing production path?", "Collect the exact error, request or correlation ID, relevant input shape with secrets removed, versions, recent changes, and reproducible steps. Evidence narrows the hypothesis without guessing.", "None"),
    ("root_cause", "How can configuration drift cause an apparently correct code path to fail?", "Different environment variables, dependency versions, migrations, or feature flags can invalidate assumptions. Compare effective configuration and versions safely before editing business logic.", "None"),
    ("root_cause", "What is a common root cause of an intermittent database timeout?", "Possible causes include pool exhaustion, slow plans, lock contention, or an unavailable dependency. Measure timing and pool or database evidence instead of treating every timeout as a code defect.", "None"),
    ("root_cause", "How should an engineer handle a failure that cannot yet be reproduced?", "Preserve the evidence, add safe observability, formulate competing hypotheses, and create the smallest diagnostic experiment. Do not claim a fix based only on correlation.", "None"),
    ("root_cause", "Why is changing multiple unrelated files dangerous during root-cause analysis?", "It destroys causal clarity and increases regression risk. Keep the hypothesis and patch narrow until evidence requires a broader change.", "None"),
    ("repair", "What makes a debugging patch safe and minimal?", "It addresses the demonstrated cause, preserves existing contracts, changes the smallest necessary surface, avoids unrelated refactors, and includes a regression test for the failure.", "None"),
    ("repair", "How should code repair a missing optional value without hiding bugs?", "Use an explicit domain policy such as a default, a typed optional branch, or a controlled validation error. Do not catch every exception or silently convert unexpected states.", "None"),
    ("repair", "How should a backend repair a failed downstream request?", "Classify the failure, apply a bounded timeout and retry only for known transient errors when the operation is safe, then return a stable error contract and preserve diagnostic context.", "None"),
    ("repair", "Why should a traceback repair avoid broad exception handling?", "A broad handler can turn programming defects into misleading successes and erase the original signal. Catch the narrow expected exception and let unexpected failures remain visible.", "None"),
    ("repair", "How should a repair preserve an existing API response contract?", "Keep status semantics and schema stable unless a versioned change is intentional; update the implementation and tests rather than surprising existing clients.", "None"),
    ("repair", "What should be done when a fix requires a database schema change?", "Plan a backward-compatible migration, deploy in safe order, backfill or validate data, and verify both old and new application paths during the rollout.", "None"),
    ("repair", "How should a repair handle a security-sensitive error?", "Remove secret exposure, use a safe public error, retain redacted structured logs, and add a regression test that asserts sensitive values do not appear.", "None"),
    ("repair", "Why should an automated repair prefer a patch over a full rewrite?", "A bounded patch preserves unrelated behavior and makes review, rollback, and verification tractable. A rewrite introduces unmeasured changes beyond the diagnosed failure.", "None"),
    ("verification", "What is the correct order for verifying a debugging fix?", "Reproduce the original failure, run a focused regression test, inspect the changed behavior and diff, then run the relevant broader suite and verify no new failures.", "None"),
    ("verification", "Why must a fix be tested against the original failing input?", "A green unrelated test does not prove the reported failure is fixed. The original input is the direct acceptance evidence for the repair.", "None"),
    ("verification", "How should a debugging result report uncertainty?", "State what was reproduced, the evidence for the root cause, what was changed, which checks passed, and any remaining hypotheses instead of claiming certainty without evidence.", "None"),
    ("verification", "What should a regression test assert for an error-handling repair?", "Assert the triggering input, stable error category or response, absence of secret leakage, and that valid neighboring inputs still succeed.", "None"),
    ("verification", "Why should a repair inspect the final diff before completion?", "Diff inspection catches accidental edits, debug prints, weakened validation, unrelated formatting churn, and secrets that tests may not detect.", "None"),
    ("verification", "How can a test distinguish a timeout retry fix from an infinite retry loop?", "Use a bounded fake dependency, assert the exact attempt count and backoff policy, and verify the final error or success outcome.", "None"),
    ("verification", "Why should debugging workflows record a stable error signature?", "A stable signature groups repeated occurrences of the same failure, prevents blind repeated retries, and helps measure whether the repair removes the original class of error.", "None"),
    ("verification", "What evidence is needed before declaring an autonomous repair complete?", "The original failure is reproduced and resolved, focused and relevant full tests pass, the diff is within scope, and the final state is independently verified.", "None"),
]

VALIDATION_RECORDS = [
    ("traceback_reading", "What does this traceback suggest and what should be checked first?", "The failure is a TypeError caused by adding an integer to None. Check why the producer allowed a missing value and apply the domain's explicit missing-value policy.", "Traceback (most recent call last):\n  File \"billing.py\", line 14, in total\n    return subtotal + tax\nTypeError: unsupported operand type(s) for +: 'int' and 'NoneType'"),
    ("root_cause", "How should an engineer investigate a test that fails only after another test runs?", "Check shared mutable state, fixture cleanup, database transactions, environment variables, and ordering dependence; reproduce the test in isolation and in the failing order.", "None"),
    ("repair", "How should a service repair a missing request field without masking malformed clients?", "Validate the request at the boundary and return a stable field-level client error, while reserving defaults for fields whose absence is explicitly allowed by the contract.", "None"),
    ("repair", "What is a safe repair for a transient upstream timeout?", "Use a bounded timeout and bounded retry only when the operation is safe to repeat, then map exhaustion to a stable dependency error and preserve redacted evidence.", "None"),
    ("verification", "What must be proven before accepting a traceback fix?", "Reproduce the original error, pass a focused regression test, verify valid neighboring behavior, inspect the diff, and run the relevant broader test suite.", "None"),
    ("verification", "How should an automated agent avoid repeating the same failed repair?", "Record a stable error signature and patch attempt, compare new evidence with prior attempts, and replan or escalate when the same failure persists.", "None"),
    ("root_cause", "Why can a 500 response be only a symptom?", "The 500 is a transport-level symptom; the root cause may be an earlier invalid input, missing configuration, dependency failure, or programming exception identified from the traceback and evidence.", "None"),
    ("traceback_reading", "How should secrets be handled when preserving a traceback for analysis?", "Redact tokens, passwords, authorization headers, and private payload values while preserving exception type, safe locations, correlation ID, and the call path needed for diagnosis.", "None"),
]


def write_records(directory: Path, records: list[tuple[str, str, str, str]], prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, (category, instruction, response, input_text) in enumerate(records, start=1):
        text = f"### Instruction\n{instruction}\n\n### Input\n{input_text}\n\n### Response\n{response}\n"
        (directory / f"{prefix}_{index:03d}_{category}.txt").write_text(text, encoding="utf-8")


def write_benchmark() -> None:
    BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    keyword_map = {
        "traceback_reading": ["traceback", "error", "inspect"],
        "root_cause": ["root", "cause", "evidence"],
        "repair": ["repair", "test", "bounded"],
        "verification": ["verify", "regression", "diff"],
    }
    with BENCHMARK.open("w", encoding="utf-8", newline="\n") as stream:
        for index, (category, question, answer, input_text) in enumerate(VALIDATION_RECORDS, start=1):
            record = {
                "benchmark_id": f"phase137_{index:03d}",
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
