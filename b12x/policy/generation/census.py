"""Source inventory of GPU entry points, compilation sites, and cache contracts.

The inventory describes specialization expressions, including conditional and
dynamic names. It does not treat source reachability as GPU qualification.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from pathlib import Path

_REQUEST_COLLECTIONS = ContextVar("b12x_specialization_request_collections", default=())


def merge_compilation_requests(descriptors):
    """Import worker requests into every active census without resolving kernels."""
    import json

    for value in descriptors:
        if not isinstance(value, dict) or set(value) != {"target", "compile_spec"}:
            raise ValueError("invalid compilation request descriptor")
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        key = hashlib.sha256(encoded.encode()).hexdigest()
        for requests in _REQUEST_COLLECTIONS.get():
            requests[key] = value


@contextmanager
def collect_compilation_requests():
    """Collect exact requested specs; local callable-cache hits are not requests.

    A request may fail compilation or correctness. The surrounding measurement
    owns qualification; this recorder establishes the requested specialization.
    """
    import json
    from b12x._lib.runtime_control import _describe_target, observe_compilation_requests

    requests = {}
    def record(target, spec):
        value = {"target": _describe_target(target),
                 "compile_spec": None if spec is None else json.loads(spec.json_key)}
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        requests[hashlib.sha256(encoded.encode()).hexdigest()] = value
    token = _REQUEST_COLLECTIONS.set((*_REQUEST_COLLECTIONS.get(), requests))
    try:
        with observe_compilation_requests(record):
            yield requests
    finally:
        _REQUEST_COLLECTIONS.reset(token)


def _text(node):
    return None if node is None else ast.unparse(node)


def _bindings(node):
    values = defaultdict(list)
    class Assignments(ast.NodeVisitor):
        def visit_Assign(self, item):
            for target in item.targets:
                values[_text(target)].append(_text(item.value))
        def visit_AnnAssign(self, item):
            if item.value is not None:
                values[_text(item.target)].append(_text(item.value))
        def visit_FunctionDef(self, item):
            if item is node:
                self.generic_visit(item)
        def visit_ClassDef(self, item):
            pass
    Assignments().visit(node)
    return dict(sorted(values.items()))


def _host_state_inventory(tree, *, module, relative, resolve):
    declarations, memoized, accesses = [], [], []

    class StateVisitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []
            self.functions = []

        def location(self, node):
            return {"path": relative, "symbol": ".".join(self.scope), "line": node.lineno,
                    "source_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()}

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node):
            self.scope.append(node.name)
            self.functions.append(node)
            for decorator in node.decorator_list:
                callee = resolve(decorator.func if isinstance(decorator, ast.Call) else decorator)
                if callee in {"functools.cache", "functools.lru_cache"}:
                    memoized.append({**self.location(node), "id": f"{module}.{'.'.join(self.scope)}",
                        "decorator": _text(decorator),
                        "key_arguments": [arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)],
                        "vararg": None if node.args.vararg is None else node.args.vararg.arg,
                        "kwarg": None if node.args.kwarg is None else node.args.kwarg.arg,
                        "qualification": "unobserved"})
            self.generic_visit(node)
            self.functions.pop()
            self.scope.pop()

        def declaration(self, node, target, value, annotation=None):
            name = _text(target)
            instance = isinstance(target, ast.Attribute) and _text(target.value) == "self"
            if self.functions and not instance:
                return
            constructor = resolve(value.func) if isinstance(value, ast.Call) else None
            collection = isinstance(value, (ast.Dict, ast.Set, ast.DictComp, ast.SetComp))
            collection |= constructor in {"dict", "set", "collections.defaultdict", "collections.OrderedDict",
                                          "weakref.WeakKeyDictionary", "weakref.WeakValueDictionary"}
            if constructor in {"dataclasses.field", "field"}:
                collection |= any(kw.arg == "default_factory" and resolve(kw.value) in {"dict", "set"}
                                  for kw in value.keywords)
            annotation_name = resolve(annotation.value) if isinstance(annotation, ast.Subscript) else _text(annotation)
            collection |= annotation_name in {"dict", "set", "typing.Dict", "typing.Set"}
            named_cache = "cache" in name.lower() or name.lower().startswith("_last_kernel")
            if not collection and not named_cache:
                return
            owner = self.scope[:-1] if self.functions else self.scope
            identity = ".".join((module, *owner, name.removeprefix("self.")))
            declarations.append({**self.location(node), "id": identity, "name": name,
                "annotation": _text(annotation), "initializer": _text(value),
                "kind": "persistent_collection" if collection else "named_cache_state",
                "disposition": "requires ownership and cache-role review"})

        def visit_Assign(self, node):
            for target in node.targets:
                self.declaration(node, target, node.value)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            self.declaration(node, node.target, node.value, node.annotation)
            self.generic_visit(node)

        def access(self, node, receiver, key, operation):
            accesses.append({**self.location(node), "receiver": _text(receiver), "key": _text(key),
                "operation": operation, "functions": tuple(self.functions)})

        def visit_Subscript(self, node):
            self.access(node, node.value, node.slice, type(node.ctx).__name__.lower())
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"get", "setdefault", "pop", "add", "discard"} and node.args:
                self.access(node, node.func.value, node.args[0], node.func.attr)
            self.generic_visit(node)

        def visit_Compare(self, node):
            for left, op, right in zip((node.left, *node.comparators), node.ops, node.comparators, strict=False):
                if isinstance(op, (ast.In, ast.NotIn)):
                    self.access(node, right, left, "contains")
            self.generic_visit(node)

    StateVisitor().visit(tree)
    names = defaultdict(list)
    for item in declarations:
        names[item["name"].split(".")[-1]].append(item["id"])
    matched, scope_bindings = [], {}
    for item in accesses:
        possible = sorted(set(names.get(item["receiver"].split(".")[-1], ())))
        if possible:
            functions = item.pop("functions")
            scope_ids = []
            for function in functions:
                scope_id = f"{module}:{function.lineno}:{function.name}"
                if scope_id not in scope_bindings:
                    scope_bindings[scope_id] = _bindings(function)
                scope_ids.append(scope_id)
            item["scope_ids"] = scope_ids
            matched.append({**item, "possible_state_ids": possible})
    return declarations, memoized, matched, scope_bindings


def inventory_sources(root: Path):
    """Enumerate every CuTe kernel, Triton JIT function, and compiler call."""
    root = Path(root).resolve()
    entries, sites, contracts, modules, indexed_calls = [], [], [], {}, []
    persistent_state, memoized_functions, state_accesses = [], [], []
    state_scope_bindings = {}
    for path in sorted((root / "b12x").rglob("*.py")):
        source = path.read_bytes()
        tree = ast.parse(source, filename=str(path))
        relative = path.relative_to(root).as_posix()
        module = relative[:-3].replace("/", ".").removesuffix(".__init__")
        aliases = {}
        for item in ast.walk(tree):
            if isinstance(item, ast.Import):
                for name in item.names:
                    aliases[name.asname or name.name.split(".")[0]] = name.name if name.asname else name.name.split(".")[0]
            elif isinstance(item, ast.ImportFrom):
                prefix = item.module or ""
                if item.level:
                    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
                    parts = package.split(".")
                    prefix = ".".join((*parts[:len(parts) - item.level + 1], prefix)).rstrip(".")
                for name in item.names:
                    aliases[name.asname or name.name] = f"{prefix}.{name.name}"

        def resolve(node):
            if isinstance(node, ast.Subscript):
                return resolve(node.value)
            name = _text(node)
            head, *tail = name.split(".")
            return ".".join((aliases.get(head, head), *tail))

        module_bindings = _bindings(tree)
        classes = {item.name: item for item in ast.walk(tree) if isinstance(item, ast.ClassDef)}

        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.scope = []
                self.functions = []
                self.site_counts = Counter()

            def location(self, node):
                return {"path": relative, "symbol": ".".join(self.scope), "line": node.lineno,
                        "source_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()}

            def visit_ClassDef(self, node):
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            def visit_FunctionDef(self, node):
                self.scope.append(node.name)
                self.functions.append(node)
                decorators = [resolve(item.func if isinstance(item, ast.Call) else item) for item in node.decorator_list]
                if "cutlass.cute.kernel" in decorators or "triton.jit" in decorators:
                    dialect = "cutedsl" if "cutlass.cute.kernel" in decorators else "triton"
                    args = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                    attributes = sorted({_text(item) for item in ast.walk(node)
                                         if isinstance(item, ast.Attribute) and isinstance(item.ctx, ast.Load)
                                         and _text(item).startswith("self.")})
                    owner = classes.get(self.scope[-2]) if len(self.scope) > 1 else None
                    constructor = next((item for item in owner.body if isinstance(item, ast.FunctionDef)
                                        and item.name == "__init__"), None) if owner is not None else None
                    entries.append({**self.location(node), "id": f"{module}.{'.'.join(self.scope)}",
                        "dialect": dialect, "kind": "cuda_kernel" if dialect == "cutedsl" else "jit_function",
                        "arguments": [{"name": arg.arg, "annotation": _text(arg.annotation)} for arg in args],
                        "decorators": [_text(item) for item in node.decorator_list],
                        "instance_reads": attributes,
                        "constructor_bindings": {} if constructor is None else _bindings(constructor),
                        "qualification": "unobserved"})
                if node.name == "__cache_key__":
                    contracts.append({**self.location(node), "id": f"{module}.{'.'.join(self.scope)}",
                                      "returns": [_text(item.value) for item in ast.walk(node) if isinstance(item, ast.Return)]})
                self.generic_visit(node)
                self.functions.pop()
                self.scope.pop()

            def visit_Call(self, node):
                callee = resolve(node.func)
                indexed = isinstance(node.func, ast.Subscript)
                method = isinstance(node.func, ast.Attribute) and node.func.attr in {"warmup", "run"}
                if indexed or method:
                    target = resolve(node.func.value)
                    indexed_calls.append({**self.location(node),
                        "target": target if target.startswith("b12x.") else f"{module}.{target}",
                        "kind": "indexed" if indexed else node.func.attr,
                        "grid": _text(node.func.slice) if indexed else None,
                        "arguments": [_text(item) for item in node.args],
                        "keywords": {item.arg if item.arg is not None else f"**{index}": _text(item.value)
                                     for index, item in enumerate(node.keywords)},
                        "scope_bindings": [_bindings(item) for item in self.functions]})
                if callee in ("b12x._lib.compiler.compile", "b12x._lib.compiler.launch", "cutlass.cute.compile") and module != "b12x._lib.compiler":
                    self.site_counts[tuple(self.scope)] += 1
                    keywords = {item.arg if item.arg is not None else f"**{index}": _text(item.value)
                                for index, item in enumerate(node.keywords)}
                    spec = next((item.value for item in node.keywords if item.arg == "compile_spec"), None)
                    sites.append({**self.location(node),
                        "id": f"{module}.{'.'.join(self.scope)}#{self.site_counts[tuple(self.scope)]}",
                        "compiler": callee, "target": _text(node.args[0]) if node.args else None,
                        "compile_spec": _text(spec), "arguments": [_text(item) for item in node.args[1:]],
                        "keywords": keywords, "module_bindings": module_bindings,
                        "scope_bindings": [_bindings(item) for item in self.functions],
                        "qualification": "unobserved"})
                self.generic_visit(node)

        Visitor().visit(tree)
        declarations, memoized, accesses, scopes = _host_state_inventory(
            tree, module=module, relative=relative, resolve=resolve,
        )
        persistent_state.extend(declarations)
        memoized_functions.extend(memoized)
        state_accesses.extend(accesses)
        state_scope_bindings.update(scopes)
        modules[relative] = hashlib.sha256(source).hexdigest()
    triton_targets = {item["id"] for item in entries if item["dialect"] == "triton"}
    triton_launches = [item for item in indexed_calls if item["target"] in triton_targets]
    unresolved = [item for item in indexed_calls if item["kind"] == "indexed" and item["target"] not in triton_targets]
    return {"schema_version": 1, "status": "implemented", "scope": "complete package source inventory",
            "qualification": "source inventory does not establish instantiated specialization or GPU coverage",
            "counts": {"modules": len(modules), "entry_points": len(entries), "compile_sites": len(sites),
                       "cache_contracts": len(contracts), "triton_launch_sites": len(triton_launches),
                       "memoized_functions": len(memoized_functions),
                       "persistent_state_declarations": len(persistent_state), "state_access_sites": len(state_accesses),
                       "unresolved_indexed_calls": len(unresolved),
                       "dialects": dict(Counter(item["dialect"] for item in entries))},
            "source_files": modules, "entry_points": entries, "compile_sites": sites, "cache_contracts": contracts,
            "triton_launch_sites": triton_launches,
            "memoized_functions": memoized_functions, "persistent_state": persistent_state,
            "state_access_sites": state_accesses,
            "state_scope_bindings": state_scope_bindings,
            "state_discovery_limits": "Persistent collection declarations and named cache state are syntax evidence. "
                "Accesses are matched by receiver name within a module; aliases and dynamic attributes need manual review. "
                "Collections are not assumed to cache kernels, and a possible state match does not prove data flow.",
            "unresolved_indexed_calls": unresolved,
            "unowned_compile_sites": [item["id"] for item in sites if item["compile_spec"] is None]}


def verify_compile_manifest(path: Path):
    """Verify cached object identity without loading or modifying the object."""
    import json

    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "b12x._lib.compile_manifest.v3":
        raise ValueError(f"unsupported compile manifest schema: {path}")
    object_path = path.with_name(f"{manifest['cache_key']}.o")
    data = object_path.read_bytes()
    if len(data) != manifest["object_bytes"] or hashlib.sha256(data).hexdigest() != manifest["object_sha256"]:
        raise ValueError(f"cached object differs from manifest: {object_path}")
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    if digest(manifest["semantic_payload"]) != manifest["semantic_key"]:
        raise ValueError(f"semantic payload differs from its hash: {path}")
    artifact = {key: manifest[key] for key in ("cache_key", "object_sha256", "launch_metadata")}
    if digest(artifact) != manifest["artifact_evidence_sha256"]:
        raise ValueError(f"artifact evidence differs from its hash: {path}")
    if "compile_spec_json" in manifest:
        if hashlib.sha256(manifest["compile_spec_json"].encode()).hexdigest() != manifest["compile_spec_hash"]:
            raise ValueError(f"compile spec differs from its hash: {path}")
        if json.loads(manifest["compile_spec_json"]) != manifest["semantic_payload"]["compile_spec"]:
            raise ValueError(f"compile spec differs from semantic identity: {path}")
    return {"manifest": str(path.resolve()), "cache_key": manifest["cache_key"],
            "object_sha256": manifest["object_sha256"], "kernel_id": manifest.get("kernel_id"),
            "compile_spec": manifest["semantic_payload"].get("compile_spec"),
            "target": manifest["target"], "package_fingerprint": manifest["package_fingerprint"],
            "toolchain": manifest["toolchain"], "device_uuid": manifest["semantic_payload"]["device_uuid"],
            "launch_metadata": manifest["launch_metadata"], "qualification": "artifact integrity only"}


def inventory_observations(paths):
    """Join immutable race provenance to requested specializations, read-only.

    A successful race does not prove that every requested specialization ran.
    Object manifests and replay traces establish that stronger correspondence.
    """
    from contextlib import closing
    import json
    import sqlite3
    import zlib
    from b12x.policy.problem import stable_identity

    descriptors, observations, databases = {}, {}, []
    for path in paths:
        path = Path(path).resolve()
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            available = {}
            if "specializations" in tables:
                for key, payload in db.execute("SELECT identity, payload FROM specializations"):
                    raw = zlib.decompress(payload)
                    if hashlib.sha256(raw).hexdigest() != key:
                        raise ValueError(f"specialization {key} failed its content hash in {path}")
                    available[key] = json.loads(raw)
            recorded = 0
            for key, payload, digest in db.execute("SELECT identity, payload, payload_sha256 FROM observations"):
                raw = zlib.decompress(payload)
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError(f"observation {key} failed its content hash in {path}")
                record = json.loads(raw)
                identity, result = record["identity"], record["result"]
                if stable_identity(identity) != key:
                    raise ValueError(f"observation {key} failed its identity hash in {path}")
                if key in observations and observations[key]["payload_sha256"] != digest:
                    raise ValueError(f"observation {key} owns conflicting evidence across databases")
                requests = result.get("compilation_requests")
                if requests is not None:
                    recorded += 1
                    if not isinstance(requests, list) or any(not isinstance(item, str) for item in requests):
                        raise ValueError(f"observation {key} has invalid compilation requests")
                    for request in requests:
                        if request not in available:
                            raise ValueError(f"observation {key} references missing specialization {request}")
                        entry = descriptors.setdefault(request, {**available[request], "observations": set()})
                        entry["observations"].add(key)
                measurements = result.get("measurements", ())
                observations[key] = {"payload_sha256": digest, "identity": identity,
                    "compilation_requests": requests,
                    "measurement_count": len(measurements),
                    "explicit_correct_count": sum(item.get("correct") is True for item in measurements),
                    "outcomes": [{name: item.get(name) for name in ("candidate_id", "correct", "cosine", "latency_us", "error")}
                                 for item in measurements],
                    "qualification": "race evidence; compiler requests may include unused or failed specializations"}
            databases.append({"path": str(path), "census_recorded_observations": recorded,
                              "specialization_descriptors": len(available)})
    return {"databases": databases, "observations": observations,
            "specializations": {key: {**value, "observations": sorted(value["observations"])}
                                for key, value in sorted(descriptors.items())},
            "counts": {"observations": len(observations), "specializations": len(descriptors),
                       "unrecorded_observations": sum(item["compilation_requests"] is None for item in observations.values())}}


def inventory_cuda_trace(path):
    """Inventory executed CUDA entries from an Nsight Systems SQLite export.

    Symbols can collide across compiled objects. These are trace-local launch
    and resource records, not proof of a particular cached object identity.
    """
    from contextlib import closing
    import json
    import sqlite3

    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        columns = {row["name"] for row in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        resource_fields = ("registersPerThread", "staticSharedMemory", "dynamicSharedMemory",
                           "localMemoryPerThread", "localMemoryTotal", "blockX", "blockY", "blockZ")
        optional_resources = {"localMemoryPerThread", "localMemoryTotal"}
        required = {"graphNodeId", "correlationId", "deviceId", "contextId", "demangledName",
                    "start", "gridX", "gridY", "gridZ", *(set(resource_fields) - optional_resources)}
        if not required <= columns:
            raise ValueError("CUDA trace must include kernel nodes and launch resources")
        devices = [{key: value.hex() if isinstance(value, bytes) else value for key, value in dict(row).items()}
                   for row in db.execute("SELECT * FROM TARGET_INFO_GPU")]
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
        variants = {}
        for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"):
            descriptor = {"device": row["deviceId"], "context": row["contextId"],
                          "symbol": strings[row["demangledName"]],
                          "resources": {key: row[key] if key in columns else None for key in resource_fields}}
            key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
            variant = variants.setdefault(key, {**descriptor, "launches": 0, "graph_nodes": set(), "grids": Counter()})
            variant["launches"] += 1
            if row["graphNodeId"]:
                variant["graph_nodes"].add(row["graphNodeId"])
            variant["grids"][(row["gridX"], row["gridY"], row["gridZ"])] += 1
    return {"path": str(path), "sha256": digest, "devices": devices,
            "missing_resource_fields": sorted(set(resource_fields) - columns),
            "qualification": "executed launch/resource census; tracing perturbs timing; object identity is unverified",
            "launches": sum(item["launches"] for item in variants.values()),
            "symbol_resource_variants": [{**item, "graph_nodes": sorted(item["graph_nodes"]),
                "grids": [{"grid": list(grid), "launches": count} for grid, count in sorted(item["grids"].items())]}
                for item in variants.values()]}
