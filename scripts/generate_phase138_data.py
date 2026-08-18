from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "training_data" / "security_auth" / "train"
VALIDATION = ROOT / "training_data" / "security_auth" / "validation"
BENCHMARK = ROOT / "training_data" / "security_auth" / "evaluation" / "phase_138.jsonl"

TRAIN_RECORDS = [
    ("jwt", "What must a backend validate before trusting a JWT?", "Validate the signature with an approved key, the algorithm allowlist, issuer, audience, expiration, not-before time when used, and required claims. Treat an unverified decoded payload as untrusted input.", "None"),
    ("jwt", "Why is accepting an arbitrary JWT algorithm unsafe?", "An arbitrary algorithm can enable algorithm-confusion or unsigned-token acceptance. Use an explicit allowlist and a key type that matches the configured algorithm.", "None"),
    ("jwt", "Where should a JWT access token normally be checked in an API?", "Check it at the authentication boundary before protected business logic runs. Middleware should establish a typed principal and downstream code should authorize that principal for the requested resource.", "None"),
    ("jwt", "How should a service handle an expired JWT?", "Reject it with a stable authentication error and do not silently extend its lifetime. A separate refresh flow may issue a new access token after validating the refresh-token policy.", "None"),
    ("jwt", "What is the difference between JWT authentication and authorization?", "Authentication verifies that a token represents a valid principal. Authorization then checks whether that principal has the required role, scope, ownership, or policy for the specific action.", "None"),
    ("jwt", "How should JWT signing keys be rotated safely?", "Publish key identifiers, accept the current and still-valid previous key during a bounded overlap, rotate signing privately, and retire old keys only after their token lifetime and recovery window expire.", "None"),
    ("jwt", "What should be avoided in JWT claims?", "Avoid passwords, secrets, unnecessary personal data, and mutable authorization facts that remain valid too long. JWT payloads are usually readable by their holder even when they are signed.", "None"),
    ("jwt", "How can an API reduce replay risk for bearer access tokens?", "Use short-lived access tokens, TLS, narrow scopes, secure storage, revocation or rotation controls where required, and sender-constrained tokens when the threat model demands them.", "None"),
    ("oauth2", "What is the recommended OAuth2 flow for a server-side web application?", "Use Authorization Code with a confidential client, protect the redirect flow with state, authenticate the client securely, and use PKCE where supported. Never place a client secret in browser code.", "None"),
    ("oauth2", "What does PKCE protect in an OAuth2 authorization flow?", "PKCE binds the authorization request to the later token exchange with a verifier, reducing authorization-code interception risk. It complements, rather than replaces, redirect URI validation and state protection.", "None"),
    ("oauth2", "Why must OAuth2 redirect URIs be exact and pre-registered?", "A loose redirect match can send authorization codes or tokens to an attacker-controlled endpoint. Compare against an exact registered URI and reject unexpected schemes, hosts, paths, and query behavior.", "None"),
    ("oauth2", "How should an OAuth2 access token scope be used?", "Scopes should express the minimum delegated capability and the resource server must enforce them for each operation. Receiving a scope is not permission to ignore object ownership or tenant boundaries.", "None"),
    ("oauth2", "What is the role of state in an OAuth2 login?", "State binds the callback to the initiating browser session and helps prevent login CSRF. Generate it unpredictably, store it safely, and compare it before accepting the authorization response.", "None"),
    ("oauth2", "How should an OAuth2 refresh token be protected?", "Store it securely, transmit it only over TLS, rotate it on use when appropriate, detect reuse, scope it narrowly, and revoke the token family after suspicious reuse or account compromise.", "None"),
    ("oauth2", "Why should an API distinguish an authentication failure from an insufficient scope?", "Missing or invalid credentials are authentication failures, while a valid principal lacking permission is an authorization failure. Stable status and error semantics help clients recover without leaking sensitive detail.", "None"),
    ("oauth2", "How should OAuth2 provider errors be logged?", "Record a correlation identifier, provider and flow stage, safe error category, and timing, but redact authorization codes, client secrets, access tokens, refresh tokens, and personal payloads.", "None"),
    ("password_hashing", "How should passwords be stored in a backend?", "Store only a password-hashing function's output with its salt and cost parameters. Use a memory-hard adaptive password hash such as Argon2id when available, and never store plaintext or reversible encryption.", "None"),
    ("password_hashing", "Why is a unique salt required for each password?", "A unique salt prevents equal passwords from producing equal stored hashes and defeats reusable precomputed tables. The salt is not secret, but it must be stored with the hash.", "None"),
    ("password_hashing", "What is a safe way to compare password hashes?", "Use the password-hashing library's verification routine, which handles the encoded salt and parameters and performs an appropriate comparison. Do not implement ad hoc string comparison or decode passwords yourself.", "None"),
    ("password_hashing", "How should password hash parameters evolve?", "Choose a cost that is expensive enough for the threat model but bounded for the service, and transparently rehash after a successful login when stored parameters are below the current policy.", "None"),
    ("password_hashing", "What should a login endpoint return for an unknown user versus a wrong password?", "Use a consistent public failure response and comparable work so the endpoint does not reveal account existence through messages or timing. Log only safe internal diagnostics with rate limits.", "None"),
    ("password_hashing", "How should password reset tokens be designed?", "Generate high-entropy random, single-use, short-lived tokens, store only a protected representation when possible, invalidate them after use or account changes, and never log or put them in referrer-prone URLs unnecessarily.", "None"),
    ("password_hashing", "Why should password policies avoid forcing arbitrary complexity rules alone?", "Length, breached-password screening, secure hashing, rate limits, and multifactor authentication usually provide stronger protection than brittle composition rules that encourage predictable substitutions.", "None"),
    ("password_hashing", "What should happen after repeated failed logins?", "Apply bounded rate limits or progressive challenges based on account and network signals, avoid account-enumeration leaks, alert on suspicious patterns, and keep legitimate recovery paths available.", "None"),
    ("middleware", "What belongs in authentication middleware rather than business handlers?", "Middleware can extract credentials, validate them, attach a typed principal, apply consistent failure responses, and add safe correlation context. Resource-specific authorization still belongs at the policy or handler boundary.", "None"),
    ("middleware", "How should security middleware order be chosen?", "Establish trusted proxy and transport assumptions carefully, then apply request limits and correlation, authentication, authorization, input validation, and business handling in an explicit documented order.", "None"),
    ("middleware", "Why should middleware avoid logging Authorization headers?", "Bearer credentials grant access to whoever obtains them. Redact the header and tokens completely while retaining only safe metadata such as scheme, route classification, and correlation ID.", "None"),
    ("middleware", "How should authentication middleware handle malformed credentials?", "Fail closed with a stable authentication response, avoid exception details in the public body, record a redacted diagnostic, and ensure malformed input cannot bypass later authorization checks.", "None"),
    ("middleware", "What is a safe CORS rule for an authenticated API?", "Allow only explicitly trusted origins and required methods and headers. Do not combine wildcard origins with credentials, and remember that CORS is a browser policy rather than an authentication mechanism.", "None"),
    ("middleware", "How can middleware prevent confused-deputy access across tenants?", "Resolve the principal and tenant context from trusted authenticated data, require the requested resource tenant to match policy, and do not accept a client-supplied tenant header as authority.", "None"),
    ("middleware", "How should security middleware expose errors to clients?", "Return stable, minimal error categories with correlation IDs and appropriate status codes. Keep stack traces, token details, provider responses, and internal policy decisions out of public responses.", "None"),
    ("middleware", "What should be tested for a protected route middleware stack?", "Test missing, malformed, expired, valid, insufficient-scope, cross-tenant, and valid-owner cases. Assert both the response contract and that protected business logic is not called when authentication or authorization fails.", "None"),
]

VALIDATION_RECORDS = [
    ("jwt", "How should an API process a JWT whose signature is valid but whose audience is wrong?", "Reject it at authentication because signature validity alone does not establish that the token was issued for this service. Validate the configured audience and return a stable authentication failure without exposing token details.", "Authorization: Bearer <signed-token>\nClaims: aud=other-service, exp=future"),
    ("jwt", "What evidence should be checked during a JWT key rotation incident?", "Check the key identifier, active verification-key set, algorithm policy, token issuance and expiry windows, clock skew, and deployment configuration. Preserve only redacted token metadata and avoid weakening verification to restore traffic.", "Key rotation changed the active key while old tokens remain within their lifetime."),
    ("oauth2", "How should a callback handle an OAuth2 authorization response with a mismatched state?", "Reject the callback and do not exchange or accept the code. The state mismatch indicates that the response is not bound to the initiating session; log a redacted correlation event and require a fresh authorization attempt.", "callback state=unexpected-value; session state=stored-value"),
    ("oauth2", "What is the minimum safe behavior when a public client starts OAuth2 login?", "Use Authorization Code with PKCE, an exact registered redirect URI, an unpredictable state value, and no embedded client secret. Validate the issuer and token response before creating a local session.", "A mobile client cannot keep a confidential secret."),
    ("password_hashing", "How should a login service migrate a legacy weak password hash?", "Verify the legacy hash only through an isolated compatibility path, then immediately rehash the supplied password with the current memory-hard policy after successful authentication. Never log or expose either hash.", "Stored record uses an old low-cost hash format."),
    ("password_hashing", "What should a password reset implementation prove in tests?", "Prove that tokens are random, single-use, short-lived, protected in storage, invalidated after use or password change, rate-limited, and never returned in logs or public error details.", "Reset token is submitted to create a new password."),
    ("middleware", "What should happen when a valid user requests another tenant's resource?", "Authentication may succeed, but authorization must reject the request after comparing the principal's tenant and resource policy. Do not trust a client-supplied tenant identifier or invoke the protected handler.", "principal.tenant_id=tenant-a; resource.tenant_id=tenant-b"),
    ("middleware", "How should an authenticated API configure CORS for browser clients?", "Use an explicit allowlist of trusted origins and required headers and methods, allow credentials only with explicit origins, and do not treat CORS as a substitute for authentication or authorization.", "Origin=https://trusted.example; credentials=true"),
]


def write_records(directory: Path, records: list[tuple[str, str, str, str]], prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, (category, instruction, response, input_text) in enumerate(records, start=1):
        text = f"### Instruction\n{instruction}\n\n### Input\n{input_text}\n\n### Response\n{response}\n"
        (directory / f"{prefix}_{index:03d}_{category}.txt").write_text(text, encoding="utf-8")


def write_benchmark() -> None:
    BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    keyword_map = {
        "jwt": ["signature", "audience", "reject"],
        "oauth2": ["PKCE", "state", "redirect"],
        "password_hashing": ["hash", "salt", "token"],
        "middleware": ["authorization", "tenant", "CORS"],
    }
    with BENCHMARK.open("w", encoding="utf-8", newline="\n") as stream:
        for index, (category, question, answer, input_text) in enumerate(VALIDATION_RECORDS, start=1):
            record = {
                "benchmark_id": f"phase138_{index:03d}",
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
