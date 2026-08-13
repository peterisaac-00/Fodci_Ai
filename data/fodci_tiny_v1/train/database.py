"""Small database service example with explicit transaction boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Connection(Protocol):
    def execute(self, query: str, parameters: tuple[Any, ...] = ...): ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Order:
    order_id: int
    user_id: int
    total_cents: int
    status: str


class OrderRepository:
    """Keep SQL and transaction behavior in one backend boundary."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    def create_order(self, user_id: int, total_cents: int) -> Order:
        if user_id <= 0:
            raise ValueError("user_id must be positive")
        if total_cents < 0:
            raise ValueError("total_cents cannot be negative")
        query = (
            "INSERT INTO orders (user_id, total_cents, status) "
            "VALUES (%s, %s, %s) RETURNING id"
        )
        try:
            cursor = self.connection.execute(query, (user_id, total_cents, "pending"))
            row = cursor.fetchone()
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        if row is None:
            raise RuntimeError("database did not return an order id")
        return Order(order_id=int(row[0]), user_id=user_id, total_cents=total_cents, status="pending")

    def mark_paid(self, order_id: int) -> None:
        if order_id <= 0:
            raise ValueError("order_id must be positive")
        try:
            self.connection.execute(
                "UPDATE orders SET status = %s WHERE id = %s",
                ("paid", order_id),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def list_user_orders(self, user_id: int) -> list[tuple[Any, ...]]:
        if user_id <= 0:
            raise ValueError("user_id must be positive")
        cursor = self.connection.execute(
            "SELECT id, total_cents, status FROM orders WHERE user_id = %s ORDER BY id",
            (user_id,),
        )
        return list(cursor.fetchall())
