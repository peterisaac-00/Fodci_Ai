-- Validation corpus: schema choices should support the query patterns used by the API.
CREATE TABLE account_events (
    event_id UUID PRIMARY KEY,
    account_id UUID NOT NULL,
    event_type VARCHAR(80) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX account_events_account_created_idx
    ON account_events (account_id, created_at DESC);

-- Parameterized application code supplies the value for :account_id.
SELECT event_id, event_type, payload, created_at
FROM account_events
WHERE account_id = :account_id
ORDER BY created_at DESC
LIMIT 100;

-- A transaction keeps the state transition and audit record together.
BEGIN;
UPDATE accounts SET status = 'suspended' WHERE account_id = :account_id;
INSERT INTO account_events (event_id, account_id, event_type, payload)
VALUES (:event_id, :account_id, 'account_suspended', :payload);
COMMIT;
