"""Parse kernel nodes from ``cudaGraphDebugDotPrint`` DOT output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_KERNEL_START_RE = re.compile(r'^"(?P<node_name>[^"]+)"\[.*label="\{KERNEL$')
_KERNEL_ID_RE = re.compile(
    r"^\| \{ID \| (?P<node_id>\d+) \(topoId: (?P<topo_id>\d+)\) "
    r"\| (?P<launch>.+)\}$"
)
_LAUNCH_RE = re.compile(r"^(?P<symbol>.*?)\\<\\<\\<(?P<geometry>.+)\\>\\>\\>$")
_COOPERATIVE_RE = re.compile(r"^\| \{cooperative \| (?P<value>[01])\}$")


@dataclass(frozen=True)
class KernelNode:
    """One kernel node and its launch attributes from a CUDA graph."""

    node_name: str
    node_id: int
    topo_id: int
    symbol: str
    grid: tuple[int, ...]
    block: tuple[int, ...]
    dynamic_smem_bytes: int
    cooperative: bool

    @property
    def grid_x(self) -> int:
        """Return the first launch-grid dimension."""
        return self.grid[0]

    @property
    def block_x(self) -> int:
        """Return the first thread-block dimension."""
        return self.block[0]


def _split_launch_geometry(geometry: str) -> tuple[str, str, str]:
    fields: list[str] = []
    start = 0
    brace_depth = 0
    for index, character in enumerate(geometry):
        if character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth -= 1
            if brace_depth < 0:
                raise ValueError(f"unbalanced launch geometry: {geometry}")
        elif character == "," and brace_depth == 0:
            fields.append(geometry[start:index])
            start = index + 1
    fields.append(geometry[start:])
    if brace_depth or len(fields) != 3:
        raise ValueError(f"unsupported launch geometry: {geometry}")
    return fields[0], fields[1], fields[2]


def _parse_dimension(value: str) -> tuple[int, ...]:
    value = value.replace("\\{", "{").replace("\\}", "}")
    if value.startswith("{") and value.endswith("}"):
        values = value[1:-1].split(",")
    else:
        values = [value]
    if not values or any(not item.isdecimal() for item in values):
        raise ValueError(f"unsupported launch dimension: {value}")
    return tuple(int(item) for item in values)


def parse_cuda_graph_dot(path: Path) -> list[KernelNode]:
    """Extract kernel nodes from ``cudaGraphDebugDotPrint`` output."""
    nodes: list[KernelNode] = []
    current_name: str | None = None
    current_id: int | None = None
    current_topo_id: int | None = None
    current_launch: tuple[str, tuple[int, ...], tuple[int, ...], int] | None = None
    current_cooperative: bool | None = None

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.rstrip()
        if current_name is None:
            match = _KERNEL_START_RE.match(line)
            if match is not None:
                current_name = match.group("node_name")
            continue

        id_match = _KERNEL_ID_RE.match(line)
        if id_match is not None:
            current_id = int(id_match.group("node_id"))
            current_topo_id = int(id_match.group("topo_id"))
            launch_match = _LAUNCH_RE.match(id_match.group("launch"))
            if launch_match is None:
                raise ValueError(
                    f"{path}:{line_number}: unsupported CUDA launch record"
                )
            try:
                grid, block, dynamic_smem = _split_launch_geometry(
                    launch_match.group("geometry")
                )
                parsed_dynamic_smem = _parse_dimension(dynamic_smem)
                if len(parsed_dynamic_smem) != 1:
                    raise ValueError(
                        f"shared-memory launch value is not scalar: {dynamic_smem}"
                    )
                current_launch = (
                    launch_match.group("symbol"),
                    _parse_dimension(grid),
                    _parse_dimension(block),
                    parsed_dynamic_smem[0],
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            continue

        cooperative_match = _COOPERATIVE_RE.match(line)
        if cooperative_match is not None:
            current_cooperative = cooperative_match.group("value") == "1"
            continue

        if line != '}"];':
            continue

        if (
            current_id is None
            or current_topo_id is None
            or current_launch is None
            or current_cooperative is None
        ):
            raise ValueError(f"{path}:{line_number}: incomplete CUDA kernel node")
        symbol, grid, block, dynamic_smem_bytes = current_launch
        nodes.append(
            KernelNode(
                node_name=current_name,
                node_id=current_id,
                topo_id=current_topo_id,
                symbol=symbol,
                grid=grid,
                block=block,
                dynamic_smem_bytes=dynamic_smem_bytes,
                cooperative=current_cooperative,
            )
        )
        current_name = None
        current_id = None
        current_topo_id = None
        current_launch = None
        current_cooperative = None

    if current_name is not None:
        raise ValueError(f"{path}: unterminated CUDA kernel node {current_name}")
    return nodes


__all__ = ["KernelNode", "parse_cuda_graph_dot"]
