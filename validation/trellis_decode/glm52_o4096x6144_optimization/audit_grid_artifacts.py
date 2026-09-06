#!/usr/bin/env python3
"""Audit exact compiled K6/MCG objects and retain cubin/SASS evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any


CUDA_ELF_MAGIC = b"\x7fELF\x02\x01\x01\x41"
TEXT_KERNEL_SECTION_RE = re.compile(
    r"^//--------------------- \.text\.(?P<kernel>kernel_cutlass_kernel_\S+)",
    re.MULTILINE,
)
INSTRUCTION_RE = re.compile(r"^\s*/\*(?P<offset>[0-9a-fA-F]+)\*/.*;\s*$", re.MULTILINE)
INSTRUCTION_LINE_RE = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/.*;\s*$", re.MULTILINE)
REGISTER_PATTERNS = {
    "r": re.compile(r"(?<![A-Z0-9_])R(?P<index>[0-9]+)\b"),
    "ur": re.compile(r"(?<![A-Z0-9_])UR(?P<index>[0-9]+)\b"),
    "p": re.compile(r"(?<![A-Z0-9_])P(?P<index>[0-9]+)\b"),
    "up": re.compile(r"(?<![A-Z0-9_])UP(?P<index>[0-9]+)\b"),
}
SETMAXREG_RE = re.compile(
    r"\bUSETMAXREG\.(?P<operation>[A-Z_]+)\.CTAPOOL\s+"
    r"(?:(?:UP(?:T|[0-9]+)|P(?:T|[0-9]+)),\s*)?"
    r"(?P<target>0x[0-9a-fA-F]+|[0-9]+)\b"
)
LOCAL_LOAD_RE = re.compile(r"\bLDL(?:\.[A-Z0-9]+)*\b")
LOCAL_STORE_RE = re.compile(r"\bSTL(?:\.[A-Z0-9]+)*\b")
EXPECTED_MANIFEST_SCHEMA = "sparkinfer._lib.compile_manifest.v3"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def embedded_cuda_elf(object_bytes: bytes) -> bytes:
    count = object_bytes.count(CUDA_ELF_MAGIC)
    if count != 1:
        raise ValueError(f"expected one embedded CUDA ELF, found {count}")
    start = object_bytes.find(CUDA_ELF_MAGIC)
    available = len(object_bytes) - start
    if available < 64:
        raise ValueError("embedded CUDA ELF header is truncated")
    e_phoff = struct.unpack_from("<Q", object_bytes, start + 0x20)[0]
    e_shoff = struct.unpack_from("<Q", object_bytes, start + 0x28)[0]
    e_ehsize = struct.unpack_from("<H", object_bytes, start + 0x34)[0]
    e_phentsize = struct.unpack_from("<H", object_bytes, start + 0x36)[0]
    e_phnum = struct.unpack_from("<H", object_bytes, start + 0x38)[0]
    e_shentsize = struct.unpack_from("<H", object_bytes, start + 0x3A)[0]
    e_shnum = struct.unpack_from("<H", object_bytes, start + 0x3C)[0]
    if e_ehsize < 64 or e_phnum == 0xFFFF or e_shnum == 0:
        raise ValueError("embedded CUDA ELF has unsupported header fields")

    def checked_end(offset: int, size: int, label: str) -> int:
        end = offset + size
        if offset < 0 or size < 0 or end < offset or end > available:
            raise ValueError(f"embedded CUDA ELF {label} is truncated")
        return end

    extent = checked_end(0, e_ehsize, "header")
    if e_phnum:
        if not e_phoff or e_phentsize < 56:
            raise ValueError("invalid program-header table")
        extent = max(extent, checked_end(e_phoff, e_phentsize * e_phnum, "phdr"))
        for index in range(e_phnum):
            header = start + e_phoff + index * e_phentsize
            offset = struct.unpack_from("<Q", object_bytes, header + 0x08)[0]
            size = struct.unpack_from("<Q", object_bytes, header + 0x20)[0]
            extent = max(extent, checked_end(offset, size, f"segment {index}"))
    if not e_shoff or e_shentsize < 64:
        raise ValueError("invalid section-header table")
    extent = max(extent, checked_end(e_shoff, e_shentsize * e_shnum, "shdr"))
    for index in range(e_shnum):
        header = start + e_shoff + index * e_shentsize
        section_type = struct.unpack_from("<I", object_bytes, header + 0x04)[0]
        offset = struct.unpack_from("<Q", object_bytes, header + 0x18)[0]
        size = struct.unpack_from("<Q", object_bytes, header + 0x20)[0]
        if section_type != 8:
            extent = max(extent, checked_end(offset, size, f"section {index}"))
    return object_bytes[start : start + extent]


def section(disassembly: str, name: str) -> str:
    marker = f"//--------------------- {name} "
    start = disassembly.find(marker)
    if start < 0:
        return ""
    end = disassembly.find("//--------------------- ", start + len(marker))
    return disassembly[start:] if end < 0 else disassembly[start:end]


def attribute_blocks(value: str, attribute: str) -> list[str]:
    pattern = re.compile(
        rf"^[ \t]*//----- nvinfo : {re.escape(attribute)}[ \t]*\r?$"
        rf"(?P<body>.*?)"
        rf"(?=^[ \t]*//----- nvinfo :|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    return [match.group("body") for match in pattern.finditer(value)]


def attribute_block(value: str, attribute: str) -> str:
    blocks = attribute_blocks(value, attribute)
    if len(blocks) > 1:
        raise ValueError(f"duplicate per-kernel {attribute} blocks")
    return blocks[0] if blocks else ""


def resource_values(disassembly: str, attribute: str) -> dict[str, int]:
    global_info = section(disassembly, ".nv.info")
    blocks = attribute_blocks(global_info, attribute)
    if not blocks:
        raise ValueError(f"nvdisasm omitted global {attribute}")
    values: dict[str, int] = {}
    for block in blocks:
        pending: str | None = None
        for line in block.splitlines():
            index_match = re.search(r"\.word\s+index@\(([^)]+)\)", line)
            value_match = re.search(r"\.word\s+0x([0-9a-fA-F]+)", line)
            if index_match:
                if pending is not None:
                    raise ValueError(f"{attribute} index without value")
                pending = index_match.group(1)
            elif value_match:
                if pending is None or pending in values:
                    raise ValueError(f"malformed or duplicate {attribute} value")
                values[pending] = int(value_match.group(1), 16)
                pending = None
        if pending is not None:
            raise ValueError(f"{attribute} index without value")
    return values


def kernel_section(disassembly: str, prefix: str, kernel: str) -> str:
    marker = f"//--------------------- {prefix}.{kernel}"
    start = disassembly.find(marker)
    if start < 0:
        return ""
    end = disassembly.find("//--------------------- ", start + len(marker))
    return disassembly[start:] if end < 0 else disassembly[start:end]


def attribute_words(kernel_info: str, attribute: str) -> list[int]:
    block = attribute_block(kernel_info, attribute)
    return [int(value, 16) for value in re.findall(r"\.word\s+0x([0-9a-fA-F]+)", block)]


def attribute_short(kernel_info: str, attribute: str) -> int:
    block = attribute_block(kernel_info, attribute)
    match = re.search(r"\.short\s+0x([0-9a-fA-F]+)", block)
    return int(match.group(1), 16) if match else 0


def static_shared_bytes(disassembly: str, kernel: str) -> int:
    value = kernel_section(disassembly, ".nv.shared", kernel)
    return sum(int(size, 0) for size in re.findall(r"\.zero\s+([0-9xa-fA-F]+)", value))


def validate_manifest(path: Path, object_bytes: bytes) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    cache_key = path.stem
    if raw.get("schema") != EXPECTED_MANIFEST_SCHEMA:
        errors.append("schema")
    if raw.get("cache_key") != cache_key:
        errors.append("cache_key")
    if raw.get("object_bytes") != len(object_bytes):
        errors.append("object_bytes")
    if raw.get("object_sha256") != sha256_bytes(object_bytes):
        errors.append("object_sha256")
    if sha256_bytes(str(raw.get("cache_payload_repr", "")).encode()) != cache_key:
        errors.append("cache_payload_repr")
    compile_spec_json = str(raw.get("compile_spec_json", ""))
    if sha256_bytes(compile_spec_json.encode()) != raw.get("compile_spec_hash"):
        errors.append("compile_spec_hash")
    semantic_payload = raw.get("semantic_payload")
    if sha256_bytes(canonical_json(semantic_payload).encode()) != raw.get(
        "semantic_key"
    ):
        errors.append("semantic_key")
    artifact_evidence = {
        "cache_key": raw.get("cache_key"),
        "object_sha256": raw.get("object_sha256"),
        "launch_metadata": raw.get("launch_metadata"),
    }
    if sha256_bytes(canonical_json(artifact_evidence).encode()) != raw.get(
        "artifact_evidence_sha256"
    ):
        errors.append("artifact_evidence_sha256")
    if errors:
        raise ValueError(f"manifest integrity failed for {path}: {errors}")
    return raw


class OccupancyAudit:
    def __init__(self, ordinal: int) -> None:
        from cuda.bindings import driver as cuda

        self.cuda = cuda
        self.ordinal = ordinal
        self._check(cuda.cuInit(0), "cuInit")
        self.device = self._value(cuda.cuDeviceGet(ordinal), "cuDeviceGet")
        self.context = self._value(
            cuda.cuCtxCreate(None, 0, self.device), "cuCtxCreate"
        )
        raw_name = self._value(
            cuda.cuDeviceGetName(256, self.device), "cuDeviceGetName"
        )
        self.name = bytes(raw_name).split(b"\0", 1)[0].decode(errors="replace")
        uuid = self._value(cuda.cuDeviceGetUuid(self.device), "cuDeviceGetUuid")
        value = bytes(uuid.bytes).hex()
        self.uuid = (
            f"GPU-{value[:8]}-{value[8:12]}-{value[12:16]}-{value[16:20]}-{value[20:]}"
        )

    @staticmethod
    def _check(result, operation: str):
        if not result or int(result[0]) != 0:
            raise RuntimeError(f"{operation} failed: {result[0] if result else None}")
        return result[1:]

    @classmethod
    def _value(cls, result, operation: str):
        values = cls._check(result, operation)
        if len(values) != 1:
            raise RuntimeError(f"{operation} returned {len(values)} values")
        return values[0]

    def query(
        self, cubin: bytes, kernel: str, threads: int, dynamic_smem: int
    ) -> dict[str, int]:
        cuda = self.cuda
        module = self._value(cuda.cuModuleLoadData(cubin), "cuModuleLoadData")
        try:
            function = self._value(
                cuda.cuModuleGetFunction(module, kernel.encode()),
                "cuModuleGetFunction",
            )
            self._check(
                cuda.cuFuncSetAttribute(
                    function,
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    dynamic_smem,
                ),
                "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
            )
            attributes = {
                "registers": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
                "local_bytes": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                "static_shared_bytes": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                "max_threads_per_block": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            }
            result = {
                name: int(
                    self._value(
                        cuda.cuFuncGetAttribute(attribute, function),
                        f"cuFuncGetAttribute({name})",
                    )
                )
                for name, attribute in attributes.items()
            }
            result["active_ctas_per_sm"] = int(
                self._value(
                    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        function, threads, dynamic_smem
                    ),
                    "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                )
            )
            return result
        finally:
            self._check(cuda.cuModuleUnload(module), "cuModuleUnload")

    def close(self) -> None:
        if self.context is not None:
            context, self.context = self.context, None
            self._check(self.cuda.cuCtxDestroy(context), "cuCtxDestroy")


def find_grid_object(grid_dir: Path) -> tuple[Path, Path]:
    objects = list((grid_dir / "sparkinfer-cache").rglob("*.o"))
    if len(objects) != 1:
        raise ValueError(f"{grid_dir} has {len(objects)} compiled objects")
    manifest = objects[0].with_suffix(".json")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return objects[0], manifest


def audit_grid(
    grid_dir: Path,
    output_dir: Path,
    nvdisasm: str,
    occupancy: OccupancyAudit,
) -> dict[str, Any]:
    grid = int(grid_dir.name.removeprefix("grid_"))
    object_path, manifest_path = find_grid_object(grid_dir)
    object_bytes = object_path.read_bytes()
    manifest = validate_manifest(manifest_path, object_bytes)
    cubin = embedded_cuda_elf(object_bytes)
    cubin_path = output_dir / f"grid_{grid:03d}.cubin"
    cubin_path.write_bytes(cubin)
    completed = subprocess.run(
        [nvdisasm, str(cubin_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    sass_path = output_dir / f"grid_{grid:03d}.sass"
    sass_path.write_text(completed.stdout, encoding="utf-8")
    disassembly = completed.stdout
    kernels = sorted(
        {
            match.group("kernel")
            for match in TEXT_KERNEL_SECTION_RE.finditer(disassembly)
        }
    )
    if len(kernels) != 1:
        raise ValueError(f"grid {grid} has {len(kernels)} cubin entry points")
    kernel = kernels[0]
    code = kernel_section(disassembly, ".text", kernel)
    info = kernel_section(disassembly, ".nv.info", kernel)
    if not code or not info:
        raise ValueError(f"grid {grid} is missing code or nv.info")
    registers = resource_values(disassembly, "EIATTR_REGCOUNT")
    frames = resource_values(disassembly, "EIATTR_FRAME_SIZE")
    stacks = resource_values(disassembly, "EIATTR_MIN_STACK_SIZE")
    reqntid = attribute_words(info, "EIATTR_REQNTID")
    if len(reqntid) != 3:
        raise ValueError(f"grid {grid} invalid EIATTR_REQNTID: {reqntid}")
    threads = reqntid[0] * reqntid[1] * reqntid[2]
    launch_map = manifest["launch_metadata"]["launch_dynamic_smem_bytes"]
    dynamic_values = launch_map.get(kernel)
    if not isinstance(dynamic_values, list) or len(set(dynamic_values)) != 1:
        raise ValueError(f"grid {grid} lacks exact dynamic SMEM")
    dynamic_smem = int(dynamic_values[0])
    driver = occupancy.query(cubin, kernel, threads, dynamic_smem)
    if driver["registers"] != registers[kernel]:
        raise ValueError(f"grid {grid} driver/nvdisasm register mismatch")
    if driver["local_bytes"] != frames[kernel]:
        raise ValueError(f"grid {grid} driver/nvdisasm local-memory mismatch")
    instruction_lines = INSTRUCTION_LINE_RE.findall(code)
    instruction_offsets = [
        int(match.group("offset"), 16) for match in INSTRUCTION_RE.finditer(code)
    ]
    register_sets = {
        family: sorted({int(match.group("index")) for match in pattern.finditer(code)})
        for family, pattern in REGISTER_PATTERNS.items()
    }
    reconfiguration = [
        {
            "operation": match.group("operation"),
            "target": int(match.group("target"), 0),
        }
        for match in SETMAXREG_RE.finditer("\n".join(instruction_lines))
    ]
    target = re.search(r"^\s*\.target\s+(\S+)", disassembly, re.MULTILINE)
    ptxas_version = re.search(
        r'\.string\s+"(Cuda compilation tools,[^"]+)"', disassembly
    )
    ptxas_flags = next(
        (
            match.group(1).strip()
            for match in re.finditer(r'\.string\s+"([^"\r\n]+)"', disassembly)
            if re.search(r"(?:^|\s)-O\s+\d+(?:\s|$)", match.group(1))
            and re.search(r"(?:^|\s)-arch\s+\S+", match.group(1))
        ),
        "",
    )
    compile_spec = json.loads(manifest["compile_spec_json"])
    if compile_spec["facts"][-1] != grid:
        raise ValueError(f"grid {grid} compile spec does not bind its grid")
    return {
        "grid_x": grid,
        "object": {
            "path": str(object_path.resolve()),
            "bytes": len(object_bytes),
            "sha256": sha256_bytes(object_bytes),
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "schema": manifest["schema"],
            "cache_key": manifest["cache_key"],
            "artifact_evidence_sha256": manifest["artifact_evidence_sha256"],
            "semantic_key": manifest["semantic_key"],
            "compile_spec_hash": manifest["compile_spec_hash"],
            "package_fingerprint": manifest["package_fingerprint"],
            "toolchain": manifest["toolchain"],
            "compile_options": manifest["compile_options"],
            "compile_environment": manifest["compile_environment"],
            "integrity": "pass",
        },
        "cubin": {
            "path": str(cubin_path.resolve()),
            "bytes": len(cubin),
            "sha256": sha256_bytes(cubin),
            "architecture": target.group(1) if target else None,
            "ptxas_version": ptxas_version.group(1) if ptxas_version else None,
            "ptxas_flags": ptxas_flags,
        },
        "sass": {
            "path": str(sass_path.resolve()),
            "bytes": sass_path.stat().st_size,
            "sha256": sha256_file(sass_path),
            "instructions": len(instruction_offsets),
            "code_bytes": max(instruction_offsets) + 16,
            "register_sets": {
                family: {
                    "indices": indices,
                    "count": len(indices),
                    "span": max(indices, default=-1) + 1,
                }
                for family, indices in register_sets.items()
            },
            "register_reconfiguration": reconfiguration,
            "local_load_instructions": len(LOCAL_LOAD_RE.findall(code)),
            "local_store_instructions": len(LOCAL_STORE_RE.findall(code)),
        },
        "resources": {
            "threads": reqntid,
            "threads_per_cta": threads,
            "allocated_registers": registers[kernel],
            "max_register_count": attribute_short(info, "EIATTR_MAXREG_COUNT"),
            "parameter_bytes": attribute_short(info, "EIATTR_CBANK_PARAM_SIZE"),
            "frame_bytes": frames[kernel],
            "min_stack_bytes": stacks[kernel],
            "cubin_static_shared_bytes": static_shared_bytes(disassembly, kernel),
            "launch_dynamic_shared_bytes": dynamic_smem,
            "total_launch_shared_bytes": (
                static_shared_bytes(disassembly, kernel) + dynamic_smem
            ),
            "driver": driver,
            "occupancy_active_ctas_per_sm": driver["active_ctas_per_sm"],
        },
        "kernel": kernel,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sweep-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--occupancy-device", type=int, required=True)
    result.add_argument("--nvdisasm", default="/usr/local/cuda/bin/nvdisasm")
    return result


def main() -> int:
    args = parser().parse_args()
    args.sweep_root = args.sweep_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output = args.output.resolve()
    if args.output_dir.exists() or args.output.exists():
        raise FileExistsError("resource output paths must be fresh")
    args.output_dir.mkdir(parents=True)
    grid_dirs = sorted(args.sweep_root.glob("grid_[0-9][0-9][0-9]"))
    if len(grid_dirs) != 11:
        raise ValueError(f"expected 11 grid directories, found {len(grid_dirs)}")
    occupancy = OccupancyAudit(args.occupancy_device)
    try:
        rows = [
            audit_grid(grid_dir, args.output_dir, args.nvdisasm, occupancy)
            for grid_dir in grid_dirs
        ]
        gpu = {
            "ordinal": occupancy.ordinal,
            "name": occupancy.name,
            "uuid": occupancy.uuid,
        }
    finally:
        occupancy.close()
    required = [48, 64, 80, 96, 112, 120, 128, 144, 160, 176, 188]
    if [row["grid_x"] for row in rows] != required:
        raise ValueError("resource grid coverage does not match the required sweep")
    report = {
        "schema": "b12x.glm52_o4096x6144_grid_resources.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "sweep_root": str(args.sweep_root),
        "occupancy_gpu": gpu,
        "rows": rows,
        "pass": all(
            row["manifest"]["integrity"] == "pass"
            and row["cubin"]["architecture"] == "sm_120a"
            and row["resources"]["frame_bytes"] == 0
            and row["resources"]["min_stack_bytes"] == 0
            and row["sass"]["local_load_instructions"] == 0
            and row["sass"]["local_store_instructions"] == 0
            and row["resources"]["driver"]["registers"]
            == row["resources"]["allocated_registers"]
            for row in rows
        ),
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "pass": report["pass"]}))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
