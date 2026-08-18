from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "training_data" / "sql_database" / "train"
VALIDATION = ROOT / "training_data" / "sql_database" / "validation"
BENCHMARK = ROOT / "training_data" / "sql_database" / "evaluation" / "phase_135.jsonl"

TRAIN_RECORDS = [
    ("sql_queries", "Why should backend SQL queries use parameters instead of string concatenation?", "Parameters separate SQL code from values, preventing injection and allowing the database driver to encode values correctly.", "None"),
    ("sql_queries", "How should a query return the newest ten orders with deterministic pagination?", "Use an explicit `ORDER BY created_at DESC, id DESC` and then apply `LIMIT 10`; the unique ID makes ties deterministic.", "None"),
    ("sql_queries", "How can SQL count orders per customer while retaining customers with no orders?", "Use a LEFT JOIN from customers to orders, group by the customer identity, and count a nullable order column rather than counting every joined row.", "None"),
    ("sql_queries", "What is the purpose of GROUP BY and HAVING in an aggregate query?", "GROUP BY forms one result group per key, while HAVING filters groups after aggregation; WHERE filters rows before aggregation.", "None"),
    ("sql_queries", "How should a query test whether a nullable column has no value?", "Use `IS NULL` or `IS NOT NULL`; equality with NULL does not produce the intended SQL truth result.", "None"),
    ("sql_queries", "When is EXISTS useful in a backend query?", "EXISTS checks whether a related row exists without needing to return or count every matching related row, which is useful for authorization or membership checks.", "None"),
    ("sql_queries", "How should a backend update only a permitted set of fields?", "Build a fixed SQL statement for the permitted columns, pass values as parameters, and verify the affected row belongs to the authorized resource.", "None"),
    ("sql_queries", "Why should a query select only the columns required by the endpoint?", "Selecting required columns reduces transfer and memory costs, avoids leaking fields, and makes the response contract explicit.", "None"),
    ("schema_design", "What is the role of a primary key in a relational table?", "A primary key uniquely identifies each row, provides a stable reference target for foreign keys, and must not be NULL.", "None"),
    ("schema_design", "What does a foreign key guarantee?", "A foreign key constrains a child value to reference an existing parent key, preserving referential integrity according to the configured delete behavior.", "None"),
    ("schema_design", "How should a many-to-many relationship be represented relationally?", "Use a junction table with one foreign key to each entity and a composite primary key or unique constraint across the pair.", "None"),
    ("schema_design", "Why is normalization useful in transactional database design?", "Normalization reduces duplicated facts and update anomalies by storing each fact in an appropriate related table.", "None"),
    ("schema_design", "When should a column be NOT NULL?", "Use NOT NULL when every valid row must have a value for the field and the absence of a value has no valid meaning.", "None"),
    ("schema_design", "Why should a natural business identifier often have a UNIQUE constraint even with a surrogate primary key?", "The surrogate key identifies the row internally, while UNIQUE protects the business rule that the natural identifier cannot repeat.", "None"),
    ("schema_design", "What should a schema consider before using soft deletion?", "Define the deleted state, filtering rules, uniqueness behavior, retention policy, and how administrators can restore or permanently remove data.", "None"),
    ("schema_design", "Why should database migrations be backward compatible during a rolling deployment?", "Old and new application versions may run together, so an intermediate schema must support both until the rollout is complete.", "None"),
    ("indexes_transactions", "When does an index improve a backend query?", "An index helps when its leading columns narrow the search or support ordering enough to cost less than scanning the table.", "None"),
    ("indexes_transactions", "What is the leftmost-prefix rule for a composite index?", "A composite index on `(tenant_id, status, created_at)` is most directly useful for predicates or ordering that begin with tenant_id and then follow the index order.", "None"),
    ("indexes_transactions", "Why should an engineer inspect the query plan instead of guessing about an index?", "The query plan shows whether the optimizer scans, seeks, joins, or sorts, providing evidence for an index decision.", "None"),
    ("indexes_transactions", "What does a database transaction provide?", "A transaction groups operations into one atomic unit with defined commit and rollback behavior, so partial updates do not become visible as success.", "None"),
    ("indexes_transactions", "Why does transaction isolation matter?", "Isolation controls which concurrent changes a transaction can observe and helps balance consistency, locking, and throughput.", "None"),
    ("indexes_transactions", "How can optimistic concurrency prevent lost updates?", "Read a version or updated timestamp, include it in the update predicate, and reject the update when the stored version changed.", "None"),
    ("indexes_transactions", "How should a service respond to a deadlock victim error?", "Rollback the transaction, retry the complete bounded operation when safe, and use backoff; never retry only one statement outside its transaction semantics.", "None"),
    ("indexes_transactions", "Why should a database upsert have an explicit conflict rule?", "An explicit conflict rule defines whether a duplicate becomes an update, no-op, or error and prevents accidental data replacement.", "None"),
    ("schema_design", "Why should relationship cardinality be documented in a schema design?", "Cardinality states whether each side is one-to-one, one-to-many, or many-to-many and guides constraints, joins, and API behavior.", "None"),
    ("sql_queries", "Why should a backend avoid relying on implicit row order?", "Relational tables have no guaranteed order without ORDER BY, so implicit order makes pagination and tests unstable.", "None"),
    ("schema_design", "Why should timestamps have an explicit timezone policy?", "A documented timezone policy prevents ambiguous comparisons and makes auditing and cross-region behavior consistent.", "None"),
    ("indexes_transactions", "Why should indexes be added selectively?", "Indexes accelerate some reads but consume storage and slow writes, so each index should support a measured access pattern.", "None"),
    ("sql_queries", "What is the difference between WHERE and ON conditions in an outer join?", "ON controls which rows match during the join, while WHERE filters the joined result and can accidentally remove NULL-preserved rows.", "None"),
    ("schema_design", "Why should a database constraint enforce a rule that must always hold?", "A database constraint protects integrity for every writer, including scripts and future services that may bypass application validation.", "None"),
    ("indexes_transactions", "Why should transaction scope be kept small?", "Short transactions hold locks for less time, reduce contention and deadlock risk, and release resources sooner.", "None"),
    ("sql_queries", "Why should a query use a stable cursor for large pagination?", "A cursor based on an indexed, ordered key avoids large offsets and reduces duplicates or omissions when earlier rows change.", "None"),
]

VALIDATION_RECORDS = [
    ("sql_queries", "How should a backend safely filter users by an optional email value?", "Use a fixed query shape with a parameter for the email and handle the absent filter explicitly; never concatenate the value into SQL.", "None"),
    ("sql_queries", "How can an endpoint list customers with their order totals including zero-order customers?", "Start with customers, LEFT JOIN orders, aggregate with GROUP BY, and use a zero-preserving count or COALESCE as appropriate.", "None"),
    ("schema_design", "How should a product-tag many-to-many relationship be stored?", "Create a product_tag junction table with product_id and tag_id foreign keys and a unique or composite primary key on both IDs.", "None"),
    ("schema_design", "Why should an application rule also be represented by a database constraint when possible?", "The constraint protects integrity across all writers and turns an implicit assumption into an enforceable schema rule.", "None"),
    ("indexes_transactions", "How should an engineer decide whether to add an index to a slow query?", "Inspect the query plan and workload, then add an index whose leading columns support the predicate or ordering and verify the measured tradeoff.", "None"),
    ("indexes_transactions", "How should a service safely retry a deadlock?", "Rollback the entire transaction and retry the complete operation with a bounded attempt count and backoff when the operation is safe to repeat.", "None"),
    ("sql_queries", "Why can moving an outer-join condition from ON to WHERE change results?", "A WHERE condition can remove rows whose joined side is NULL, changing an intended outer join into inner-like behavior.", "None"),
    ("schema_design", "How should a rolling deployment add a required database field?", "Add it in a compatible nullable or defaulted form, deploy code that writes it, backfill safely, then enforce requiredness after old code is gone.", "None"),
]


def write_records(directory: Path, records: list[tuple[str, str, str, str]], prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, (category, instruction, response, input_text) in enumerate(records, start=1):
        text = f"### Instruction\n{instruction}\n\n### Input\n{input_text}\n\n### Response\n{response}\n"
        (directory / f"{prefix}_{index:03d}_{category}.txt").write_text(text, encoding="utf-8")


def write_benchmark() -> None:
    BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    keyword_map = {
        "sql_queries": ["SQL", "parameter", "query"],
        "schema_design": ["table", "foreign key", "constraint"],
        "indexes_transactions": ["index", "transaction", "rollback"],
    }
    with BENCHMARK.open("w", encoding="utf-8", newline="\n") as stream:
        for index, (category, question, answer, _input_text) in enumerate(VALIDATION_RECORDS, start=1):
            record = {
                "benchmark_id": f"phase135_{index:03d}",
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
