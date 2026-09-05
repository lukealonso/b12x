"""Compare checkpoint containers using complete existing measurement records.

Source checkpoints are read-only. SQLite variants use rollback journals and
FULL synchronous commits; their measurements include one commit per record.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time
import zlib

from b12x.policy.generation.store import CheckpointStore


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class SqliteRecords:
    def __init__(self, root, *, compressed):
        self.root = root
        self.compressed = compressed
        root.mkdir(parents=True)
        self.connection = sqlite3.connect(root / "records.sqlite3", isolation_level=None)
        self.connection.execute("PRAGMA journal_mode=TRUNCATE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE generation(id INTEGER PRIMARY KEY, payload TEXT UNIQUE NOT NULL)")
        self.connection.execute("CREATE TABLE records(component TEXT, key TEXT, generation_id INTEGER, payload BLOB NOT NULL, PRIMARY KEY(component,key)) WITHOUT ROWID")
        self.generations = {}

    def save(self, component, key, payload):
        value = dict(payload)
        generation_id = None
        if "generation" in value:
            generation = canonical(value.pop("generation"))
            generation_id = self.generations.get(generation)
            if generation_id is None:
                self.connection.execute("INSERT OR IGNORE INTO generation(payload) VALUES (?)", (generation,))
                generation_id = self.connection.execute("SELECT id FROM generation WHERE payload=?", (generation,)).fetchone()[0]
                self.generations[generation] = generation_id
        blob = canonical(value).encode()
        if self.compressed:
            blob = zlib.compress(blob, level=1)
        self.connection.execute("INSERT OR REPLACE INTO records VALUES(?,?,?,?)", (component, key, generation_id, blob))

    def decode(self, blob, generation):
        if self.compressed:
            blob = zlib.decompress(blob)
        value = json.loads(blob)
        if generation is not None:
            value["generation"] = json.loads(generation)
        return value

    def load(self, component, key):
        row = self.connection.execute("SELECT r.payload,g.payload FROM records r LEFT JOIN generation g ON r.generation_id=g.id WHERE r.component=? AND r.key=?", (component, key)).fetchone()
        return None if row is None else self.decode(*row)

    def records(self):
        for component, key, blob, generation in self.connection.execute("SELECT r.component,r.key,r.payload,g.payload FROM records r LEFT JOIN generation g ON r.generation_id=g.id ORDER BY r.component,r.key"):
            yield component, key, self.decode(blob, generation)

    def close(self):
        self.connection.close()


class CompactJsonRecords(CheckpointStore):
    def __init__(self, root, *, compressed=False):
        super().__init__(root)
        self.compressed = compressed

    def _path(self, component_id, key):
        path = super()._path(component_id, key)
        return path.with_suffix(".json.gz") if self.compressed else path

    def save(self, component_id, key, payload):
        path = self._path(component_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
        blob = (canonical(payload) + "\n").encode()
        if self.compressed:
            blob = gzip.compress(blob, compresslevel=1, mtime=0)
        temporary.write_bytes(blob)
        os.replace(temporary, path)
        return path

    def load(self, component_id, key):
        if not self.compressed:
            return super().load(component_id, key)
        path = self._path(component_id, key)
        return json.loads(gzip.decompress(path.read_bytes())) if path.is_file() else None


class PrettyJsonRecords(CheckpointStore):
    def save(self, component_id, key, payload):
        path = self._path(component_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path


def digest_rows(rows):
    digest = hashlib.sha256()
    for component, key, payload in rows:
        digest.update(canonical((component, key, payload)).encode() + b"\n")
    return digest.hexdigest()


def footprint(root):
    paths = tuple(path for path in root.rglob("*") if path.is_file())
    return {"files": len(paths), "bytes": sum(path.stat().st_size for path in paths),
            "allocated_bytes": sum(path.stat().st_blocks * 512 for path in paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-write-seconds", type=float, default=60.0)
    parser.add_argument(
        "--variants", nargs="+",
        choices=("json_files", "compact_json_files", "gzip_json_files",
                 "sqlite_json", "sqlite_deflate"),
        default=("json_files", "compact_json_files", "gzip_json_files",
                 "sqlite_json", "sqlite_deflate"),
    )
    args = parser.parse_args()
    if args.repetitions <= 0 or args.max_write_seconds <= 0:
        parser.error("repetitions and max-write-seconds must be positive")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variants must be distinct")
    if args.output_dir.exists():
        raise FileExistsError("the output directory must be unused")
    args.output_dir.mkdir(parents=True)
    paths = sorted(args.source.glob("*/*.json"))
    keys = [(path.parent.name, path.stem) for path in paths]
    assert keys
    source = CheckpointStore(args.source)

    def source_rows():
        for component, key in keys:
            payload = source.load(component, key)
            if payload is None:
                raise ValueError(f"invalid source checkpoint: {component}/{key}")
            yield component, key, payload

    source_digest = digest_rows(source_rows())
    store_source = Path(sys.modules[CheckpointStore.__module__].__file__)
    report = {"source": str(args.source), "source_digest": source_digest,
              "command": sys.argv, "worktree": str(Path.cwd()),
              "revision": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checkpoint_writer_sha256": hashlib.sha256(store_source.read_bytes()).hexdigest(),
              "source_filesystem": subprocess.check_output(
                  ["findmnt", "-J", "-T", str(args.source)], text=True),
              "output_filesystem": subprocess.check_output(
                  ["findmnt", "-J", "-T", str(args.output_dir)], text=True),
              "records": len(keys), "sqlite_version": sqlite3.sqlite_version,
              "journal": "TRUNCATE", "synchronous": "FULL", "variants": {}}
    variants = OrderedDict(
        json_files=PrettyJsonRecords,
        compact_json_files=CheckpointStore,
        gzip_json_files=lambda root: CompactJsonRecords(root, compressed=True),
        sqlite_json=lambda root: SqliteRecords(root, compressed=False),
        sqlite_deflate=lambda root: SqliteRecords(root, compressed=True),
    )
    for name in args.variants:
        store = variants[name](args.output_dir / name)
        expected = hashlib.sha256()
        started = time.perf_counter()
        for index, (component, key, payload) in enumerate(source_rows(), start=1):
            store.save(component, key, payload)
            expected.update(canonical((component, key, payload)).encode() + b"\n")
            if index % 25000 == 0:
                print(json.dumps({"variant": name, "written": index}), flush=True)
            if time.perf_counter() - started >= args.max_write_seconds:
                break
        write_seconds = time.perf_counter() - started
        lookup_seconds = []
        written_keys = keys[:index]
        shuffled = list(written_keys)
        random.Random(92451).shuffle(shuffled)
        for _ in range(args.repetitions):
            started = time.perf_counter()
            for component, key in shuffled:
                assert store.load(component, key) is not None
            lookup_seconds.append(time.perf_counter() - started)
        started = time.perf_counter()
        rows = (store.records() if isinstance(store, SqliteRecords) else
                ((component, key, store.load(component, key)) for component, key in written_keys))
        digest = digest_rows(rows)
        scan_seconds = time.perf_counter() - started
        assert digest == expected.hexdigest(), name
        if index == len(keys):
            assert digest == source_digest, name
        if isinstance(store, SqliteRecords):
            assert store.connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            store.close()
        report["variants"][name] = {
            "written_records": index, "complete": index == len(keys),
            "written_digest": digest,
            "write_seconds": write_seconds, "lookup_seconds": lookup_seconds,
            "verified_scan_seconds": scan_seconds, "digest_matches": True,
            **footprint(args.output_dir / name),
        }
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"variant": name, **report["variants"][name]}), flush=True)


if __name__ == "__main__":
    main()
