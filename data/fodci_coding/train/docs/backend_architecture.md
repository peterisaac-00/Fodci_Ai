# Backend service boundaries

A maintainable backend keeps transport, application policy, persistence, and infrastructure concerns separate. An HTTP route validates the request shape and maps domain outcomes to status codes. The application service owns authorization and business rules. A repository owns parameterized SQL and transaction boundaries. Configuration is loaded once from environment variables and injected into the components that need it.

A successful request follows this path:

```text
HTTP request -> validation -> authorization -> service -> repository -> database
       ^                                                     |
       +------------- safe response and structured error ----+
```

A failed database operation must roll back its transaction, while an external side effect such as email delivery should be idempotent and retryable. Tests should cover both successful behavior and boundaries such as malformed input, missing resources, unauthorized access, dependency failure, and retry handling.
