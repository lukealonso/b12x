"""Content-addressed GPU observations shared by offline search stages."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import zlib
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from b12x.policy.problem import stable_identity
from b12x.policy.types import FrozenMapping


def _encode(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True, kw_only=True)
class ObservationIdentity:
    """Identity of a paired race, independent of the stage requesting it.

    The cohort distinguishes initial sampling from independent confirmation.
    Candidate order belongs to the timing protocol and is retained exactly.
    """

    component_id: str
    generation: FrozenMapping
    inputs: FrozenMapping
    candidates: tuple[FrozenMapping, ...]
    oracle_contract: str
    cohort: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.oracle_contract or not self.cohort:
            raise ValueError("observations require component, oracle, and cohort identities")
        required = {"source_revision", "source_sha256", "toolchain", "device",
                    "physical_device", "settings"}
        if not required <= set(self.generation) or any(not self.generation[key] for key in required):
            raise ValueError("observation provenance is incomplete")
        if not self.candidates or len({stable_identity(item) for item in self.candidates}) != len(self.candidates):
            raise ValueError("observation candidates must be nonempty and unique")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, "component_id": self.component_id,
                "generation": self.generation.to_dict(), "inputs": self.inputs.to_dict(),
                "candidates": [item.to_dict() for item in self.candidates],
                "oracle_contract": self.oracle_contract, "cohort": self.cohort}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ObservationIdentity":
        if value.get("schema_version") != 1:
            raise ValueError("unsupported observation identity schema")
        return cls(component_id=value["component_id"],
                   generation=FrozenMapping(value["generation"]),
                   inputs=FrozenMapping(value["inputs"]),
                   candidates=tuple(FrozenMapping(item) for item in value["candidates"]),
                   oracle_contract=value["oracle_contract"], cohort=value["cohort"])

    @property
    def key(self) -> str:
        return stable_identity(self.to_dict())


class ObservationStore:
    """Immutable compressed records, indexed by complete measurement identity.

    SQLite transactions make readers and independent GPU writers safe across
    processes. Checkpoints retain references; paired samples are stored once.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        deadline = time.monotonic() + 30
        try:
            while True:
                try:
                    connection.execute("PRAGMA journal_mode=WAL").fetchone()
                    break
                except sqlite3.OperationalError as error:
                    # Journal-mode lock promotion can bypass SQLite's busy timeout.
                    if (getattr(error, "sqlite_errorcode", 0) & 0xff != sqlite3.SQLITE_BUSY
                            or time.monotonic() >= deadline):
                        raise
                    time.sleep(0.01)
            connection.execute("CREATE TABLE IF NOT EXISTS observations ("
                               "identity TEXT PRIMARY KEY, payload BLOB NOT NULL, "
                               "payload_sha256 TEXT NOT NULL) WITHOUT ROWID")
            connection.execute("CREATE TABLE IF NOT EXISTS specializations ("
                               "identity TEXT PRIMARY KEY, payload BLOB NOT NULL) WITHOUT ROWID")
        except BaseException:
            connection.close()
            raise
        return connection

    def load(self, identity: ObservationIdentity) -> dict[str, object] | None:
        record = self.load_key(identity.key)
        if record is None:
            return None
        if _encode(record["identity"]) != _encode(identity.to_dict()):
            raise ValueError("observation identity differs from its requested inputs")
        return record["result"]

    def load_key(self, key: str) -> dict[str, object] | None:
        if not self.path.exists():
            return None
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT payload, payload_sha256 FROM observations "
                                     "WHERE identity = ?", (key,)).fetchone()
        if row is None:
            return None
        raw = zlib.decompress(row[0])
        if hashlib.sha256(raw).hexdigest() != row[1]:
            raise ValueError(f"observation {key} failed its content hash")
        record = json.loads(raw)
        if not isinstance(record, dict) or stable_identity(record.get("identity")) != key:
            raise ValueError(f"observation {key} failed its identity hash")
        if not isinstance(record.get("result"), dict):
            raise ValueError(f"observation {key} has no result object")
        for request in record["result"].get("compilation_requests", ()):
            self.load_specialization(request)
        return record

    def load_reference(self, key: str, expected: ObservationIdentity, *,
                       generation_matches: Callable[[object], bool]) -> dict[str, object] | None:
        record = self.load_key(key)
        if record is None:
            return None
        actual = dict(record["identity"])
        generation = actual.pop("generation", None)
        requested = expected.to_dict()
        requested.pop("generation")
        if actual != requested:
            raise ValueError("checkpoint references a different measurement identity")
        if not generation_matches(generation):
            return None
        return record

    def save(self, identity: ObservationIdentity, result: Mapping[str, object]) -> str:
        raw = _encode({"identity": identity.to_dict(), "result": dict(result)})
        digest = hashlib.sha256(raw).hexdigest()
        key = identity.key
        with closing(self._connect()) as connection, connection:
            connection.execute("INSERT OR IGNORE INTO observations VALUES (?, ?, ?)",
                               (key, zlib.compress(raw, level=6), digest))
            existing = connection.execute("SELECT payload_sha256 FROM observations "
                                          "WHERE identity = ?", (key,)).fetchone()[0]
            if existing != digest:
                raise ValueError("an observation identity already owns different samples; "
                                 "use an independent cohort for fresh measurements")
        return key

    def save_specialization(self, descriptor: Mapping[str, object]) -> str:
        raw = _encode(descriptor)
        key = hashlib.sha256(raw).hexdigest()
        with closing(self._connect()) as connection, connection:
            connection.execute("INSERT OR IGNORE INTO specializations VALUES (?, ?)",
                               (key, zlib.compress(raw, level=6)))
            stored = connection.execute("SELECT payload FROM specializations WHERE identity = ?", (key,)).fetchone()[0]
            if zlib.decompress(stored) != raw:
                raise ValueError("specialization identity owns different content")
        return key

    def load_specialization(self, key: str) -> dict[str, object]:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT payload FROM specializations WHERE identity = ?", (key,)).fetchone()
        if row is None:
            raise ValueError(f"missing specialization descriptor: {key}")
        raw = zlib.decompress(row[0])
        if hashlib.sha256(raw).hexdigest() != key:
            raise ValueError(f"specialization {key} failed its content hash")
        return json.loads(raw)


@dataclass(frozen=True, kw_only=True)
class ObservedRace:
    identity: ObservationIdentity | None
    result: dict[str, object]
    fresh: bool
    measurement_seconds: float
    storage_seconds: float


def measure_observation(*, context, component_id: str, inputs: FrozenMapping,
                        candidates: tuple[FrozenMapping, ...], oracle_contract: str,
                        store: ObservationStore, measure: Callable[[], dict[str, object]],
                        reference: str | None = None) -> ObservedRace:
    """Record one complete production race without duplicating stage samples."""
    identity = None
    stored = None
    if context.provenance:
        identity = ObservationIdentity(component_id=component_id,
                                       generation=FrozenMapping(context.checkpoint_metadata()),
                                       inputs=inputs, candidates=candidates,
                                       oracle_contract=oracle_contract,
                                       cohort=context.measurement_cohort)
        if reference is not None:
            record = store.load_reference(reference, identity,
                                           generation_matches=context.checkpoint_metadata_matches)
            if record is not None:
                identity = ObservationIdentity.from_dict(record["identity"])
                stored = record["result"]
        if stored is None:
            stored = store.load(identity)
    if stored is not None:
        return ObservedRace(identity=identity, result=stored, fresh=False,
                            measurement_seconds=0., storage_seconds=0.)
    started = time.monotonic()
    from .census import collect_compilation_requests

    with collect_compilation_requests() as requests:
        result = measure()
    measurement_seconds = time.monotonic() - started
    started = time.monotonic()
    if identity is not None:
        result = {**result, "compilation_requests": sorted(store.save_specialization(item) for item in requests.values())}
        store.save(identity, result)
    return ObservedRace(identity=identity, result=result, fresh=True,
                        measurement_seconds=measurement_seconds,
                        storage_seconds=time.monotonic() - started)


__all__ = ["ObservationIdentity", "ObservationStore", "ObservedRace", "measure_observation"]
