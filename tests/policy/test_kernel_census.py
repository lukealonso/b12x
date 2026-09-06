import ast
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import zlib

import pytest

from b12x._lib.runtime_control import report_compilation_request
from b12x.policy.generation.census import (
    collect_compilation_requests, inventory_sources, inventory_observations, merge_compilation_requests,
)
from b12x.policy.generation.observations import ObservationStore


def test_inventory_resolves_aliases_and_preserves_conditional_keys(tmp_path):
    package = tmp_path / "b12x"
    package.mkdir()
    (package / "example.py").write_text('''
from cutlass import cute as c
from b12x._lib.compiler import compile as build, launch as run
from triton import jit as tj
class Kernel:
    def __init__(self, rows):
        self.blocks = 1 if rows == 1 else 2
    @property
    def __cache_key__(self):
        return (self.blocks,)
    @c.kernel
    def device(self, rows: int):
        use(self.blocks, rows)
@tj(do_not_specialize=["rows"])
def metadata(rows, TILE: tl.constexpr):
    pass
def prepare(rows):
    key = (rows,)
    build(Kernel(rows), compile_spec=make_spec(key))
    run(Kernel(rows), compile_spec=make_spec(key), compile_args=(), runtime_args=())
    c.compile(Kernel(rows))
    metadata[(1,)](rows, TILE=128)
''')
    report = inventory_sources(tmp_path)
    assert report["counts"]["entry_points"] == 2
    assert report["counts"]["compile_sites"] == 3
    assert report["counts"]["triton_launch_sites"] == 1
    assert report["triton_launch_sites"][0]["target"] == "b12x.example.metadata"
    assert report["unowned_compile_sites"] == ["b12x.example.prepare#3"]
    kernel = report["entry_points"][0]
    assert kernel["constructor_bindings"]["self.blocks"] == ["1 if rows == 1 else 2"]
    assert report["cache_contracts"][0]["returns"] == ["(self.blocks,)"]
    assert report["compile_sites"][0]["scope_bindings"][0]["key"] == ["(rows,)"]
    json.dumps(report, sort_keys=True)


def test_inventory_accounts_for_every_explicit_production_compile_site():
    root = Path(__file__).resolve().parents[2]
    inventory = inventory_sources(root)
    found = {(item["path"], item["line"]) for item in inventory["compile_sites"]}
    expected = set()
    for path in (root / "b12x").rglob("*.py"):
        if path.name == "compiler.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and any(kw.arg == "compile_spec" for kw in node.keywords):
                expected.add((path.relative_to(root).as_posix(), node.lineno))
    assert expected <= found
    assert inventory["unowned_compile_sites"] == []


def test_inventory_records_memoized_functions_and_persistent_cache_keys(tmp_path):
    package = tmp_path / "b12x"
    package.mkdir()
    (package / "example.py").write_text('''
from functools import lru_cache as memo
from dataclasses import field
_COMPILED = {}
_LAST_KERNEL = None
@memo(maxsize=8)
def prepare(capacity, *, dtype):
    return build(capacity, dtype)
class Workspace:
    planned_launches: dict = field(default_factory=dict)
    def get(self, rows, capacity):
        key = (capacity,)
        if key not in self.planned_launches:
            self.planned_launches[key] = prepare(capacity, dtype="bf16")
        return self.planned_launches.get(key)
def launch(rows):
    local_config = {}
    return _COMPILED.get((rows,))
''')
    report = inventory_sources(tmp_path)
    assert report["counts"]["memoized_functions"] == 1
    assert report["memoized_functions"][0]["key_arguments"] == ["capacity", "dtype"]
    assert {item["id"] for item in report["persistent_state"]} == {
        "b12x.example._COMPILED", "b12x.example._LAST_KERNEL", "b12x.example.Workspace.planned_launches",
    }
    accesses = report["state_access_sites"]
    assert {item["operation"] for item in accesses} == {"contains", "store", "get"}
    assert any(item["key"] == "(rows,)" for item in accesses)
    planned = next(item for item in accesses if item["receiver"] == "self.planned_launches")
    assert report["state_scope_bindings"][planned["scope_ids"][0]]["key"] == ["(capacity,)"]
    json.dumps(report, sort_keys=True)


def test_compilation_requests_are_deduplicated_context_local_and_restored():
    def kernel():
        pass
    spec = SimpleNamespace(json_key='{"kernel":"example","version":1,"facts":[["columns",128]]}')
    with collect_compilation_requests() as outer:
        report_compilation_request(kernel, spec)
        with collect_compilation_requests() as inner:
            report_compilation_request(kernel, spec)
        report_compilation_request(kernel, spec)
    assert len(outer) == len(inner) == 1
    report_compilation_request(kernel, None)
    assert len(outer) == 1


def test_specialization_descriptors_are_stored_once_and_fail_on_corruption(tmp_path):
    store = ObservationStore(tmp_path / "observations.sqlite3")
    descriptor = {"target": "example", "compile_spec": {"kernel": "example", "version": 1}}
    key = store.save_specialization(descriptor)
    assert store.save_specialization(descriptor) == key
    assert store.load_specialization(key) == descriptor
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM specializations").fetchone()[0] == 1
        db.execute("UPDATE specializations SET payload = ? WHERE identity = ?",
                   (zlib.compress(b'{}'), key))
    with pytest.raises(ValueError, match="content hash"):
        store.load_specialization(key)


def test_worker_descriptors_reach_nested_collectors_without_compilation():
    descriptor = {"target": "worker.Kernel", "compile_spec": {"kernel": "worker", "version": 1}}
    with collect_compilation_requests() as outer, collect_compilation_requests() as inner:
        merge_compilation_requests((descriptor, descriptor))
    assert outer == inner and list(outer.values()) == [descriptor]
    merge_compilation_requests(({"target": "other", "compile_spec": None},))
    assert len(outer) == 1


def test_census_does_not_create_missing_database(tmp_path):
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        inventory_observations((path,))
    assert not path.exists()


@pytest.mark.parametrize("local_memory_fields", [True, False])
def test_cuda_trace_census_retains_distinct_resources_and_runtime_grids(tmp_path, local_memory_fields):
    from b12x.policy.generation.census import inventory_cuda_trace

    path = tmp_path / "trace.sqlite"
    fields = ("graphNodeId", "correlationId", "deviceId", "contextId", "demangledName",
              "start", "gridX", "gridY", "gridZ", "registersPerThread", "staticSharedMemory",
              "dynamicSharedMemory", "localMemoryPerThread", "localMemoryTotal", "blockX", "blockY", "blockZ")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (" + ",".join(name + " INTEGER" for name in fields) + ")")
        db.execute("CREATE TABLE TARGET_INFO_GPU (uuid BLOB, name TEXT)")
        db.execute("INSERT INTO TARGET_INFO_GPU VALUES (?, ?)", (b"gpu-uuid", "synthetic"))
        db.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        db.execute("INSERT INTO StringIds VALUES (1, 'same_symbol')")
        rows = [(0, 1, 0, 2, 1, 1, 1, 1, 1, 32, 0, 1024, 0, 0, 128, 1, 1),
                (7, 2, 0, 2, 1, 2, 3, 1, 1, 32, 0, 1024, 0, 0, 128, 1, 1),
                (8, 3, 0, 2, 1, 3, 3, 1, 1, 64, 0, 1024, 0, 0, 128, 1, 1)]
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (" + ",".join("?" for _ in fields) + ")", rows)
        if not local_memory_fields:
            db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL DROP COLUMN localMemoryPerThread")
            db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL DROP COLUMN localMemoryTotal")
    before = path.read_bytes()
    report = inventory_cuda_trace(path)
    assert path.read_bytes() == before
    assert report["launches"] == 3
    variants = report["symbol_resource_variants"]
    assert len(variants) == 2
    assert variants[0]["graph_nodes"] == [7]
    assert variants[0]["grids"] == [{"grid": [1, 1, 1], "launches": 1}, {"grid": [3, 1, 1], "launches": 1}]
    assert variants[1]["resources"]["registersPerThread"] == 64
    assert variants[0]["resources"]["localMemoryPerThread"] == (0 if local_memory_fields else None)
    assert report["missing_resource_fields"] == ([] if local_memory_fields else ["localMemoryPerThread", "localMemoryTotal"])
    assert "object identity is unverified" in report["qualification"]


def test_cuda_trace_census_rejects_missing_launch_evidence(tmp_path):
    from b12x.policy.generation.census import inventory_cuda_trace

    path = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        inventory_cuda_trace(path)
    assert not path.exists()
    with sqlite3.connect(path):
        pass
    with pytest.raises(ValueError, match="kernel nodes and launch resources"):
        inventory_cuda_trace(path)
