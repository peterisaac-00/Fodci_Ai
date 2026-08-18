from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "training_data" / "rest_api" / "train"
VALIDATION = ROOT / "training_data" / "rest_api" / "validation"
BENCHMARK = ROOT / "training_data" / "rest_api" / "evaluation" / "phase_136.jsonl"

TRAIN_RECORDS = [
    ("resources", "How should REST resources be named in a backend API?", "Use stable plural nouns such as `/users` and `/orders`, and represent a single resource with its identifier such as `/users/{id}`.", "None"),
    ("resources", "What does idempotence mean for an HTTP method?", "An idempotent method has the same intended server state after one or repeated identical requests; PUT and DELETE are designed to be idempotent when implemented correctly.", "None"),
    ("resources", "When should an API use POST instead of PUT?", "Use POST to create under a collection or trigger a non-idempotent action; use PUT when the client addresses the complete target resource and repetition has the same intended result.", "None"),
    ("resources", "Why should nested REST paths be kept shallow?", "Shallow paths reduce coupling and ambiguity; use nesting only when the child relationship is essential to the resource identity or authorization boundary.", "None"),
    ("http_semantics", "What status code should a successful resource creation return?", "Return 201 Created and preferably include a Location header for the new resource; return a representation when the API contract defines one.", "None"),
    ("http_semantics", "When should an API return 204 No Content?", "Use 204 when the operation succeeds and there is intentionally no response representation, such as a successful deletion without a response body.", "None"),
    ("http_semantics", "How should an API distinguish 401 from 403?", "Use 401 when authentication is missing or invalid, and 403 when the caller is authenticated but is not allowed to perform the action.", "None"),
    ("http_semantics", "When is 409 Conflict appropriate?", "Use 409 when the request conflicts with the current resource state, such as a duplicate business identifier or a detected version conflict.", "None"),
    ("pagination", "How should an API design a stable offset pagination response?", "Accept bounded page and page_size values, apply a deterministic ORDER BY, and return items plus explicit metadata such as page, page_size, and total when total is available.", "None"),
    ("pagination", "Why can cursor pagination be better for a changing large collection?", "A cursor based on an indexed stable ordering avoids scanning large offsets and reduces duplicates or omissions while earlier rows change.", "None"),
    ("pagination", "What should an API enforce on page size?", "Enforce a positive default and a maximum page size to bound database work, response size, and memory consumption.", "None"),
    ("pagination", "How should API filtering values be validated?", "Parse and validate each filter against an allowlist of fields and operators, then pass values as parameters rather than constructing arbitrary query text.", "None"),
    ("versioning", "Why should an API version its public contract deliberately?", "Versioning provides a controlled compatibility boundary so breaking representation or behavior changes can be introduced without silently breaking existing clients.", "None"),
    ("versioning", "What is a safe way to deprecate an API field?", "Announce the field as deprecated, document a replacement, measure usage, provide a migration window, and remove it only according to the published compatibility policy.", "None"),
    ("versioning", "Why should an API avoid exposing database table names as its public contract?", "Database names couple clients to storage details; resource names should express domain behavior so persistence can evolve independently.", "None"),
    ("versioning", "What should an API documentation version identify?", "It should identify the public contract version, supported endpoints, request and response schemas, authentication requirements, and compatibility notes.", "None"),
    ("openapi", "What is the purpose of an OpenAPI document?", "OpenAPI describes endpoints, parameters, request bodies, responses, authentication, and schemas in a machine-readable contract that supports documentation and tooling.", "None"),
    ("openapi", "Why should every documented response include an explicit schema?", "An explicit schema makes client expectations testable and prevents documentation from promising an ambiguous or unstable response shape.", "None"),
    ("openapi", "How should an API document authentication requirements?", "Declare the security scheme and the operations or scopes that require it, while documenting safe error responses without publishing secrets.", "None"),
    ("openapi", "Why should examples in API documentation be valid against the schema?", "Valid examples reduce integration ambiguity and can be used as contract-test fixtures rather than teaching clients an impossible payload.", "None"),
    ("errors", "What should a consistent API error body contain?", "Use a stable machine-readable code, a safe human message, optional field details, and a correlation or request ID without exposing stack traces or secrets.", "None"),
    ("errors", "How should validation failures be represented?", "Return a client-error status with structured field-level details that identify invalid input while keeping internal implementation information private.", "None"),
    ("errors", "Why should an API not return raw exception text to clients?", "Raw exception text can leak SQL, file paths, credentials, or internal architecture; map failures to safe public error contracts and log details privately.", "None"),
    ("errors", "How should a REST API map a missing resource?", "Return 404 with the documented safe error shape, and avoid revealing whether a protected resource exists when authorization policy requires concealment.", "None"),
    ("implementation", "What is the role of a service layer behind a REST endpoint?", "The service layer coordinates domain rules and dependencies so the HTTP adapter remains focused on parsing, authorization, response mapping, and transport concerns.", "None"),
    ("implementation", "Why should an API validate request data before calling the service layer?", "Early validation rejects malformed input consistently and ensures the service receives a typed, trusted boundary object rather than an arbitrary dictionary.", "None"),
    ("implementation", "How should an API handle a slow downstream dependency?", "Set an explicit timeout, classify the failure, return a safe bounded error, and avoid holding unnecessary resources while waiting indefinitely.", "None"),
    ("implementation", "Why should mutating endpoints support an idempotency key when clients may retry?", "An idempotency key lets the server recognize a repeated logical request and return the original result instead of creating duplicate side effects.", "None"),
    ("implementation", "What should an API test at the contract boundary?", "Test method and path routing, request validation, authentication behavior, status codes, response schemas, error shapes, and compatibility guarantees.", "None"),
    ("implementation", "Why should API handlers avoid embedding complex SQL and business rules?", "Keeping transport, persistence, and domain rules separate improves testability, reuse, error mapping, and the ability to evolve each layer independently.", "None"),
    ("implementation", "How should an API expose a long-running operation?", "Return an accepted operation representation with a status URL or job identifier, then let clients poll or receive a documented completion notification.", "None"),
    ("implementation", "Why should a REST API define a correlation ID policy?", "A correlation ID connects client-visible failures with structured server logs and traces without placing sensitive diagnostic data in the response.", "None"),
]

VALIDATION_RECORDS = [
    ("resources", "How should an API update a complete user representation when the client sends a stable identifier?", "Use PUT at `/users/{id}` for a complete replacement contract, validate the representation, and document whether missing fields are reset or rejected.", "None"),
    ("http_semantics", "What response should a successful asynchronous creation request use?", "Return 202 Accepted with an operation or status resource when work is not complete, and document how the client observes completion.", "None"),
    ("pagination", "How should a list endpoint prevent an unbounded query?", "Validate page_size or cursor limits, use a stable indexed ordering, and reject or cap values beyond the documented maximum.", "None"),
    ("versioning", "How should a breaking response change be introduced safely?", "Publish a new contract version, document the differences and migration path, keep the old version during the compatibility window, and measure usage.", "None"),
    ("openapi", "What should an OpenAPI operation document for a protected endpoint?", "Document the security requirement, parameters, request and response schemas, successful statuses, and safe authentication or authorization errors.", "None"),
    ("errors", "How should an endpoint report invalid JSON and field validation errors?", "Return the documented client-error status with a stable error code and structured field details, without returning a traceback.", "None"),
    ("implementation", "How can an API avoid duplicate orders when a client retries after a timeout?", "Accept and persist an idempotency key for the logical operation, then return the stored original result for a repeated key.", "None"),
    ("implementation", "Why should an API handler pass a validated DTO to the service layer?", "A validated DTO separates transport parsing from domain logic and gives the service a predictable typed contract.", "None"),
]


def write_records(directory: Path, records: list[tuple[str, str, str, str]], prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, (category, instruction, response, input_text) in enumerate(records, start=1):
        text = f"### Instruction\n{instruction}\n\n### Input\n{input_text}\n\n### Response\n{response}\n"
        (directory / f"{prefix}_{index:03d}_{category}.txt").write_text(text, encoding="utf-8")


def write_benchmark() -> None:
    BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    keyword_map = {
        "resources": ["resource", "PUT", "POST"],
        "http_semantics": ["status", "201", "response"],
        "pagination": ["page", "cursor", "limit"],
        "versioning": ["version", "breaking", "client"],
        "openapi": ["OpenAPI", "schema", "security"],
        "errors": ["error", "status", "safe"],
        "implementation": ["service", "validation", "API"],
    }
    with BENCHMARK.open("w", encoding="utf-8", newline="\n") as stream:
        for index, (category, question, answer, _input_text) in enumerate(VALIDATION_RECORDS, start=1):
            record = {
                "benchmark_id": f"phase136_{index:03d}",
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
