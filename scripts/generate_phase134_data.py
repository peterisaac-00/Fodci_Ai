from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "training_data" / "python_backend" / "train"
VALIDATION = ROOT / "training_data" / "python_backend" / "validation"

TRAIN_RECORDS = [
    ("type_hints", "Why should a backend function annotate both its input and output types?", "Type annotations make the contract explicit, improve editor and static-checker feedback, and expose accidental None or wrong return values before runtime.", "None"),
    ("type_hints", "Write a typed Python function signature for creating a user and returning its integer ID.", "Use a signature such as `def create_user(name: str, email: str) -> int:` and validate the values inside the function before persistence.", "None"),
    ("type_hints", "When is `str | None` appropriate in backend Python?", "Use `str | None` when a value is legitimately either a string or None; do not use it to hide an unhandled missing-value bug.", "None"),
    ("type_hints", "What is the purpose of `TypedDict` in an API response contract?", "TypedDict describes the expected keys and value types of dictionary-shaped data while keeping the runtime value a normal dict.", "None"),
    ("type_hints", "How can a repository protocol decouple service code from a database implementation?", "Define a `Protocol` with the methods the service needs, type the service against that protocol, and inject a concrete repository at the boundary.", "None"),
    ("type_hints", "Why should a backend avoid using `Any` for every external payload?", "Any disables useful static checks and lets malformed data spread; use a precise model, TypedDict, Protocol, or validated boundary type instead.", "None"),
    ("type_hints", "What does `TypeVar` enable in a reusable backend helper?", "TypeVar lets a generic helper preserve the relationship between an input type and its corresponding output type instead of collapsing both to object.", "None"),
    ("type_hints", "How should a function accepting only specific operation names be annotated?", "Use `Literal`, for example `Literal['create', 'update']`, when the accepted values are a small fixed set.", "None"),
    ("async", "Why must blocking file or database work not run directly in an async request handler?", "Blocking work stops the event loop and delays unrelated requests; use an async driver or explicitly move unavoidable blocking work to a bounded worker.", "None"),
    ("async", "What does `await` do in an asynchronous Python function?", "Await suspends the current coroutine until the awaitable completes, allowing the event loop to run other ready tasks.", "None"),
    ("async", "When is `asyncio.gather` appropriate in a backend service?", "Use gather for independent awaitables that can run concurrently, and handle cancellation and exceptions according to the required all-or-nothing policy.", "None"),
    ("async", "How should a service limit concurrent calls to an upstream API?", "Use an `asyncio.Semaphore` around the upstream call so concurrency is bounded and the service cannot create an unbounded request storm.", "None"),
    ("async", "Why should async network calls have timeouts?", "Timeouts bound resource consumption and turn an indefinitely stalled dependency into a controlled failure that recovery logic can classify.", "None"),
    ("async", "How should an async handler treat task cancellation?", "Allow `asyncio.CancelledError` to propagate after required cleanup; swallowing cancellation can prevent graceful shutdown and request cancellation.", "None"),
    ("async", "What is the role of an async context manager for a database connection?", "It acquires the connection on entry and reliably releases or rolls it back on exit, including when the request raises an exception.", "None"),
    ("async", "What is an async generator useful for in a backend API?", "It yields records incrementally with `async for`, reducing memory pressure while allowing each fetch to await an asynchronous source.", "None"),
    ("pydantic", "What problem does a Pydantic BaseModel solve at an API boundary?", "It defines a typed schema and validates incoming data before business logic consumes it, producing structured validation errors.", "None"),
    ("pydantic", "How should an API validate a request body with Pydantic v2?", "Define a BaseModel and call `Model.model_validate(payload)` at the boundary, then pass the validated model rather than the raw dict inward.", "None"),
    ("pydantic", "Why should a Pydantic model use Field constraints for an email or page size?", "Field constraints make boundary rules executable and consistent, such as minimum length, maximum page size, or a required format.", "None"),
    ("pydantic", "What is the difference between `model_dump` and `model_validate` in Pydantic v2?", "model_validate constructs a model from input data; model_dump serializes a validated model into a dictionary for transport or persistence.", "None"),
    ("pydantic", "How should nested API data be represented with Pydantic?", "Define a nested BaseModel for the child object and use it as the parent field type so nested validation is applied recursively.", "None"),
    ("pydantic", "Why should validation errors be returned as client errors rather than internal errors?", "Invalid client input is expected boundary failure and should produce a safe structured 4xx response without exposing an internal traceback.", "None"),
    ("pydantic", "Why should an API response schema be separate from an internal database model?", "Separate schemas prevent persistence details from leaking into the public contract and let each boundary evolve independently.", "None"),
    ("pydantic", "How can Pydantic prevent unexpected fields in a strict request model?", "Configure the model to forbid or explicitly handle extra fields so clients cannot silently send data the endpoint does not understand.", "None"),
    ("error_handling", "Why should backend code catch specific exceptions instead of a bare `except`?", "Specific handlers preserve the distinction between expected domain failures, retryable dependency failures, and programming bugs.", "None"),
    ("error_handling", "How should a custom backend exception carry an HTTP-safe error code?", "Define a domain exception with a stable code and safe public message, then map it at the API boundary without exposing internal details.", "None"),
    ("error_handling", "What is the purpose of `raise NewError(...) from exc`?", "Exception chaining preserves the original cause for diagnostics while exposing a domain-specific error to the caller.", "None"),
    ("error_handling", "What belongs in a `finally` block in a backend operation?", "Put mandatory cleanup such as releasing a lock or closing a resource in finally so it runs on both success and failure.", "None"),
    ("error_handling", "How should a backend log an exception without leaking credentials?", "Log the exception type, correlation ID, operation, and safe context, but redact tokens, passwords, authorization headers, and private payloads.", "None"),
    ("error_handling", "Which failures are usually retryable?", "Transient timeouts, temporary unavailability, and rate limits may be retryable with bounded backoff; validation and authorization failures are not.", "None"),
    ("error_handling", "Why should an API map internal exceptions at one boundary?", "Central mapping keeps handlers consistent, avoids duplicated try/except code, and prevents tracebacks or implementation details from reaching clients.", "None"),
    ("error_handling", "How should error handling preserve the original failure while adding context?", "Catch the narrow exception, add operation context with exception chaining, and retain the original traceback for diagnostics.", "None"),
]

BENCHMARK = ROOT / "training_data" / "python_backend" / "evaluation" / "phase_134.jsonl"

VALIDATION_RECORDS = [
    ("type_hints", "How should a typed service represent a function that may return a user or no match?", "Annotate the return as `User | None` and make callers handle the no-match case explicitly instead of assuming a User always exists.", "None"),
    ("type_hints", "What is dependency injection through a typed Protocol?", "The service receives an object implementing the Protocol, which makes the dependency explicit and permits alternate database or test implementations.", "None"),
    ("async", "How should an async endpoint protect itself from a slow upstream dependency?", "Use an awaited timeout around the call, classify timeout as a controlled dependency failure, and release resources in cleanup.", "None"),
    ("async", "When should independent async operations be run sequentially instead of together?", "Run sequentially when order, shared mutation, rate limits, or failure semantics make concurrency unsafe; concurrency is not automatically better.", "None"),
    ("pydantic", "How should a validated Pydantic model be converted into a response payload?", "Return `model_dump()` or the framework-supported serialization of the validated response model, not an unvalidated internal object.", "None"),
    ("pydantic", "How should an endpoint handle a request with an invalid page size?", "Let the boundary schema reject it with a structured validation error and a safe client-facing 4xx response.", "None"),
    ("error_handling", "How should a backend distinguish an authorization failure from a database outage?", "Map authorization failure to a safe client response without retrying, while classify the database outage separately as a dependency failure that may be retried within bounds.", "None"),
    ("error_handling", "Why is a broad exception handler dangerous at the top of a request handler?", "It can hide programming defects, mislabel failures, and return misleading success or client errors; a boundary handler must log safely and preserve failure status.", "None"),
]


def write_records(directory: Path, records: list[tuple[str, str, str, str]], prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, (category, instruction, response, input_text) in enumerate(records, start=1):
        text = f"### Instruction\n{instruction}\n\n### Input\n{input_text}\n\n### Response\n{response}\n"
        (directory / f"{prefix}_{index:03d}_{category}.txt").write_text(text, encoding="utf-8")


def write_benchmark(records: list[tuple[str, str, str, str]]) -> None:
    BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK.open("w", encoding="utf-8", newline="\n") as stream:
        for index, (category, instruction, response, _input_text) in enumerate(records, start=1):
            keywords = {
                "type_hints": ["type", "Protocol", "None"],
                "async": ["async", "await", "timeout"],
                "pydantic": ["Pydantic", "model", "validation"],
                "error_handling": ["exception", "error", "boundary"],
            }[category]
            record = {
                "benchmark_id": f"phase134_{index:03d}",
                "version": "1.0",
                "split": "benchmark",
                "category": category,
                "question": instruction,
                "expected_answer": response,
                "required_keywords": keywords,
                "minimum_keyword_coverage": 0.66,
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    write_records(TRAIN, TRAIN_RECORDS, "train")
    write_records(VALIDATION, VALIDATION_RECORDS, "validation")
    write_benchmark(VALIDATION_RECORDS)
    print(f"generated train={len(TRAIN_RECORDS)} validation={len(VALIDATION_RECORDS)} benchmark={len(VALIDATION_RECORDS)}")


if __name__ == "__main__":
    main()
