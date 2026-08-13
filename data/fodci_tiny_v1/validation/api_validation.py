"""Validation-only example for a small FastAPI service boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="Inventory API")


@dataclass(frozen=True, slots=True)
class Item:
    item_id: int
    name: str
    quantity: int


ITEMS = {
    1: Item(item_id=1, name="keyboard", quantity=8),
    2: Item(item_id=2, name="monitor", quantity=3),
}


@app.get("/items/{item_id}")
def get_item(item_id: int) -> Item:
    item = ITEMS.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    return item


@app.get("/items")
def list_items(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    available_only: bool = False,
) -> list[Item]:
    items = list(ITEMS.values())
    if available_only:
        items = [item for item in items if item.quantity > 0]
    return items[:limit]


def reserve_item(item_id: int, quantity: int) -> Item:
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    item = ITEMS.get(item_id)
    if item is None:
        raise KeyError("item not found")
    if item.quantity < quantity:
        raise RuntimeError("insufficient inventory")
    updated = Item(item.item_id, item.name, item.quantity - quantity)
    ITEMS[item_id] = updated
    return updated
