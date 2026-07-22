"""Component bases and the factory registry.

An Algorithm declares its store inputs/outputs (``reads``/``writes``) —
the sequence validator checks every read is satisfied by an earlier
write, which is the declared-dependency DAG without a scheduler.
Services are long-lived infrastructure shared by algorithms.
"""
from __future__ import annotations

from typing import Any

from .store import EventStore

ALGORITHMS: dict[str, type] = {}
SERVICES: dict[str, type] = {}


def algorithm(name: str):
    def deco(cls):
        cls.name = name
        ALGORITHMS[name] = cls
        return cls
    return deco


def service(name: str):
    def deco(cls):
        cls.name = name
        SERVICES[name] = cls
        return cls
    return deco


class Algorithm:
    name: str = "?"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()

    def __init__(self, **props: Any):
        self.props = props

    def initialize(self, services: dict[str, Any]) -> None:   # once per job
        self.services = services

    def execute(self, store: EventStore) -> None:             # per event
        raise NotImplementedError

    def finalize(self) -> dict:                               # job summary
        return {}

    # convenience
    def put(self, store: EventStore, key: str, value: Any) -> None:
        store.put(key, value, by=self.name)


class Service:
    name: str = "?"

    def __init__(self, **props: Any):
        self.props = props

    def initialize(self) -> None:
        pass
