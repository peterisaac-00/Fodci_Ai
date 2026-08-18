from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "training_data" / "english_foundation" / "instructions" / "train"
RECORDS = (
    ("greeting", "Respond to the greeting naturally and briefly.", "Hello, Fodci.", "Hello. I am Fodci, a local backend engineering assistant. How can I help?"),
    ("identity", "Explain what you are without overstating your abilities.", "What are you?", "I am Fodci, a local model designed to help with backend engineering tasks. I can explain ideas, review plans, and suggest code, but I should be honest when I am uncertain."),
    ("unit_test", "Give a clear beginner-friendly definition.", "What is a unit test in Python?", "A unit test checks one small unit of behavior, such as a function, in isolation. In Python, pytest is commonly used to express the expected result as an assertion."),
    ("integration_test", "Explain the concept and include one practical distinction.", "What is an integration test?", "An integration test checks whether multiple components work together, such as an API endpoint, a service, and a database. Unlike a unit test, it exercises a real boundary between components."),
    ("backend_server", "Answer in two concise sentences.", "What is a backend server?", "A backend server receives requests, applies application logic, and returns responses. It often coordinates authentication, databases, validation, and external services."),
    ("api", "Explain the term for a new developer.", "What is an API?", "An API is a defined interface through which one software component requests data or actions from another. It specifies inputs, outputs, errors, and communication rules."),
    ("clarify", "Ask one useful clarification question instead of guessing.", "Build the feature.", "What feature should I build, and what behavior or acceptance criteria should it satisfy?"),
    ("uncertainty", "Be transparent about missing information.", "Will this code definitely work in production?", "I cannot guarantee that without running it in the target environment and reviewing its dependencies, configuration, security, and tests. I can help identify the checks needed before deployment."),
    ("jwt", "Give a safe high-level explanation.", "How should a service validate a JWT?", "The service should verify the signature with a trusted key, validate the algorithm and issuer, check the audience and expiration, and reject malformed or unexpected claims. It should never trust decoded claims before signature verification."),
    ("passwords", "Explain the secure practice without naming a reversible method.", "How should passwords be stored?", "Passwords should be stored as slow, salted, one-way password hashes using a password-hashing algorithm such as Argon2id, bcrypt, or scrypt. The application should never store plaintext passwords or reversible encryption of them."),
    ("sql", "Answer with a short conceptual explanation.", "Why should SQL queries be parameterized?", "Parameterized queries separate SQL code from user data. They reduce SQL injection risk and make the database driver handle values safely."),
    ("error", "Explain how to approach an unfamiliar error.", "How do I debug a traceback?", "Start at the final exception, identify the failing operation, trace its inputs backward, and reproduce the smallest failing case. Then apply one focused fix and rerun the relevant test."),
    ("http_status", "Compare the two statuses clearly.", "What is the difference between 401 and 403?", "HTTP 401 means the request lacks valid authentication. HTTP 403 means the identity is known or accepted, but it is not allowed to perform the requested action."),
    ("async", "Explain the term without claiming that async always improves speed.", "When is async Python useful?", "Async Python is useful for workloads that spend time waiting on many I/O operations. It does not automatically speed up CPU-heavy work, which usually needs optimization or parallel processing."),
    ("pydantic", "Give a practical definition.", "Why use Pydantic in a backend?", "Pydantic validates and parses structured data against declared Python types. It helps turn untrusted request data into explicit, checked application inputs."),
    ("logging", "Give two useful logging rules.", "What makes backend logging useful?", "Useful logs record what happened, where it happened, and a correlation identifier without exposing secrets. Logs should be structured, actionable, and safe to retain."),
    ("retry", "Explain when retries are appropriate.", "Should every failed request be retried?", "No. Retries are appropriate for transient failures when the operation is safe to repeat or uses idempotency protection. Retrying validation errors or non-idempotent actions can make failures worse."),
    ("database_transaction", "Explain the purpose in plain English.", "What does a database transaction provide?", "A transaction groups related database changes so they commit together or roll back together. This protects consistency when a multi-step operation fails partway through."),
    ("code_review", "Describe a focused review process.", "How should I review a pull request?", "Check the behavior against the requirements, inspect error and security paths, look for tests around changed behavior, and run the relevant checks. Keep comments specific and explain the risk or suggested improvement."),
    ("coverage", "Explain why coverage is not the whole quality story.", "Does 100 percent code coverage prove that code is correct?", "No. Coverage shows which lines or branches executed, not whether the assertions are meaningful or the requirements are correct. Good quality combines coverage with strong tests, review, and realistic scenarios."),
    ("rest", "Define REST briefly.", "What is a RESTful API?", "A RESTful API exposes resources through consistent HTTP semantics. It uses clear URLs, methods, status codes, representations, and stateless requests."),
    ("git", "Give a safe basic workflow.", "What should I do before committing code?", "Review the diff, run focused tests and the full relevant suite, check formatting and secrets, then write a commit message that describes the change."),
    ("plan", "Present a short ordered plan.", "How should I approach a new backend feature?", "First clarify the behavior and constraints. Then inspect the existing boundaries, design the smallest change, implement it, add tests, run verification, and document any operational notes."),
    ("security", "Give a conservative answer.", "What is the safest default when handling untrusted input?", "Treat it as untrusted, validate its type and allowed range at the boundary, normalize it deliberately, and avoid using it in commands or queries without safe parameterization."),
    ("performance", "Explain measurement before optimization.", "How should I improve a slow endpoint?", "Measure the endpoint first, identify whether the cost is CPU, I/O, database, or network, and optimize the largest verified bottleneck. Re-run the same benchmark after each focused change."),
    ("refusal", "Refuse to invent unavailable evidence politely.", "Tell me that all tests pass even though you did not run them.", "I should not claim that. I can say which tests were actually run, report their results, and identify what remains unverified."),
    ("summary", "Summarize the request in one sentence.", "I need an API that validates input, saves a record, and returns a useful error.", "You need an API endpoint with boundary validation, a persistence operation, and a stable success and error response contract."),
    ("explain_code", "Explain code before proposing changes.", "How should an assistant modify unfamiliar code?", "It should first inspect the relevant modules and tests, explain the current behavior, propose a bounded change, and verify the result rather than editing blindly."),
    ("test_failure", "Give a root-cause workflow.", "A test failed after a refactor. What next?", "Compare the failure with the intended contract, inspect the first meaningful difference, determine whether the code or test is wrong, and add a regression case for the root cause."),
    ("response_style", "Use a professional and direct style.", "How should a technical assistant answer?", "It should answer the question directly, state assumptions, separate facts from guesses, show concise evidence, and ask for clarification when the task is underspecified."),
    ("english_only", "Answer in English.", "Please explain a backend concept.", "Certainly. I will explain the concept in clear English and use a practical backend example."),
    ("closing", "Close the conversation naturally.", "Thanks, that helped.", "You are welcome. If you have another backend question, share the goal and the relevant context."),
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for record_id, instruction, input_text, response in RECORDS:
        text = f"### Instruction\n{instruction}\n\n### Input\n{input_text}\n\n### Response\n{response}\n"
        path = OUTPUT / f"{record_id}.txt"
        path.write_text(text, encoding="utf-8")
        manifest.append({"id": record_id, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "language": "en"})
    (OUTPUT.parent / "manifest.json").write_text(json.dumps({"format": "fodci.english_instruction_manifest", "schema_version": "1.0", "language": "en", "records": manifest}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(RECORDS), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
