"""Write-once event store (the Gaudi 'transient event store' pattern).

Algorithms never call each other — they communicate exclusively through
named locations here.  Write-once enforces the immutability contract
(e.g. the measurement operator can never be mutated downstream) and the
provenance record gives every data product its producer.
"""
from __future__ import annotations

from typing import Any


class EventStore:
    def __init__(self):
        self._data: dict[str, Any] = {}
        self._producer: dict[str, str] = {}

    def put(self, key: str, value: Any, by: str = "?") -> None:
        if key in self._data:
            raise KeyError(
                f"store['{key}'] already written by {self._producer[key]} "
                f"(write-once); attempted rewrite by {by}")
        self._data[key] = value
        self._producer[key] = by

    def get(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(f"store['{key}'] not present; available: "
                           f"{sorted(self._data)}")
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def provenance(self) -> dict[str, str]:
        return dict(self._producer)

    def clear(self) -> None:
        self._data.clear()
        self._producer.clear()
