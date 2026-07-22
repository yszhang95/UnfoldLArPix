"""Job runner: YAML config -> services + algorithm sequence -> event loop.

Usage:  python -m unfoldlarpix.fwk.runner <config.yaml>

The full resolved config (plus git commit and store provenance) is
attached to the outputs by the writer algorithms — every result file
carries its own recipe.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .component import ALGORITHMS, SERVICES, Algorithm
from .store import EventStore

# importing the packages registers their components
from .. import services as _services_pkg    # noqa: F401
from .. import algs as _algs_pkg            # noqa: F401


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_job(cfg: dict) -> tuple[dict[str, Any], list[Algorithm]]:
    services: dict[str, Any] = {}
    for name, props in (cfg.get("services") or {}).items():
        svc = SERVICES[name](**(props or {}))
        svc.initialize()
        services[name] = svc
    seq: list[Algorithm] = []
    for entry in cfg["sequence"]:
        (name, props), = entry.items()
        alg = ALGORITHMS[name](**(props or {}))
        alg.initialize(services)
        seq.append(alg)
    validate_sequence(seq)
    return services, seq


def validate_sequence(seq: list[Algorithm]) -> None:
    written: set[str] = set()
    for alg in seq:
        missing = [r for r in alg.reads if r not in written]
        if missing:
            raise ValueError(
                f"{alg.name}: reads {missing} not produced by any earlier "
                f"algorithm (have: {sorted(written)})")
        written |= set(alg.writes)


def run(config_path: str | Path) -> list[dict]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["_meta"] = {"config_path": str(config_path), "git": git_commit()}
    services, seq = build_job(cfg)
    # the first algorithm is the source: it reports how many events it has
    source = seq[0]
    n_events = source.n_events() if hasattr(source, "n_events") else 1
    summaries = []
    for i_evt in range(n_events):
        store = EventStore()
        store.put("job.config", cfg, by="runner")
        for alg in seq:
            alg.execute(store)
    for alg in seq:
        s = alg.finalize()
        if s:
            summaries.append({alg.name: s})
    return summaries


if __name__ == "__main__":
    for s in run(sys.argv[1]):
        print(s)
