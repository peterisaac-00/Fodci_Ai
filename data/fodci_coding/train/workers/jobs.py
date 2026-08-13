"""Small background-job boundary with retry-safe behavior."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmailJob:
    job_id: str
    recipient: str
    template: str
    idempotency_key: str


class Mailer(Protocol):
    def send(self, recipient: str, template: str, idempotency_key: str) -> None: ...


class JobQueue(Protocol):
    def acknowledge(self, job_id: str) -> None: ...

    def retry(self, job_id: str, delay_seconds: int) -> None: ...


def process_email_job(job: EmailJob, mailer: Mailer, queue: JobQueue) -> None:
    """Acknowledge only after success; transient failures remain retryable."""

    try:
        mailer.send(job.recipient, job.template, job.idempotency_key)
    except TimeoutError:
        queue.retry(job.job_id, delay_seconds=30)
        return
    except Exception:
        queue.retry(job.job_id, delay_seconds=300)
        return
    queue.acknowledge(job.job_id)
