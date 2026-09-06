"""Dense FP16 Hadamard launches using the shared CuTe Trellis butterfly."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import Float16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_ptr

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import half2_mul
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.w4a8_trellis_decode import _w4a8_had128_quad


_KERNEL_CACHE = {}


class _Hadamard128:
    def __init__(self, columns, pre_scale, post_scale):
        self.columns = columns
        self.pre_scale = pre_scale
        self.post_scale = post_scale

    @cute.jit
    def __call__(self, source: cute.Pointer, destination: cute.Pointer,
                 pre_scale: cute.Pointer, post_scale: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        layout = cute.make_layout((Int64(rows) * Int64(self.columns),), stride=(1,))
        scales = cute.make_layout((self.columns,), stride=(1,))
        self.kernel(cute.make_tensor(source, layout), cute.make_tensor(destination, layout),
                    cute.make_tensor(pre_scale, scales), cute.make_tensor(post_scale, scales)).launch(
            grid=(rows, self.columns // 128, 1), block=(32, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, source: cute.Tensor, destination: cute.Tensor,
               pre_scale: cute.Tensor, post_scale: cute.Tensor):
        lane, _, _ = cute.arch.thread_idx()
        row, block, _ = cute.arch.block_idx()
        column = Int32(block) * Int32(128) + Int32(lane) * Int32(4)
        base = Int64(row) * Int64(self.columns) + Int64(column)
        values = cute.make_rmem_tensor((4,), Float32)
        for i in cutlass.range_constexpr(4):
            value = source[base + Int64(i)].to(Float32)
            if cutlass.const_expr(self.pre_scale):
                value = Float16(value * pre_scale[column + Int32(i)].to(Float32)).to(Float32)
            values[i] = value
        transformed = _w4a8_had128_quad(values[0], values[1], values[2], values[3], Int32(lane))
        # Preserve the FP16 boundary between H128 and the output scale.
        rounded = cute.make_rmem_tensor((4,), Float16)
        for i in cutlass.range_constexpr(4):
            rounded[i] = Float16(transformed[i])
        if cutlass.const_expr(self.post_scale):
            factors = cute.make_rmem_tensor((4,), Float16)
            for i in cutlass.range_constexpr(4):
                factors[i] = post_scale[column + Int32(i)]
            packed = cute.recast_tensor(rounded, Uint32)
            scales = cute.recast_tensor(factors, Uint32)
            for i in cutlass.range_constexpr(2):
                packed[i] = half2_mul(packed[i], scales[i])
        for i in cutlass.range_constexpr(4):
            destination[base + Int64(i)] = rounded[i]


def _pointer(address=16):
    return make_ptr(Float16, address, cute.AddressSpace.gmem, assumed_align=2)


def _resolve(columns, pre_scale, post_scale):
    key = (torch.cuda.current_device(), columns, pre_scale, post_scale)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        kernel = _Hadamard128(columns, pre_scale, post_scale)
        raise_if_kernel_resolution_frozen("CuTe Hadamard compilation", target=kernel, cache_key=key)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up the dense Trellis rotation before CUDA graph capture")
        compiled = b12x_compile(
            kernel, _pointer(), _pointer(), _pointer(), _pointer(), Int32(1), current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key("gemm.trellis_hadamard128", 2, key),
        )
        _KERNEL_CACHE[key] = compiled
    return compiled


def hadamard_128(source, destination, pre_scale=None, post_scale=None, scale=1.0):
    """Apply normalized H128 with FP16 scale boundaries and caller-owned output."""
    if source.ndim != 2 or source.shape[0] <= 0 or source.shape[0] > 2**31 - 1:
        raise ValueError("dense Trellis rotation requires a positive Int32 row count")
    columns = source.shape[1]
    if columns <= 0 or columns % 128:
        raise ValueError("dense Trellis rotation width must be a positive multiple of 128")
    if not source.is_cuda or source.dtype != torch.float16 or not source.is_contiguous():
        raise ValueError("dense Trellis rotation requires contiguous CUDA FP16 input")
    if (destination.shape != source.shape or destination.dtype != source.dtype
            or destination.device != source.device or not destination.is_contiguous()):
        raise ValueError("dense Trellis rotation output must match the input shape, dtype, and device")
    if scale != 1.0:
        raise ValueError("dense Trellis rotation uses a unit transform multiplier")
    for value in (pre_scale, post_scale):
        if value is not None and (value.shape != (columns,) or value.dtype != source.dtype
                                  or value.device != source.device or not value.is_contiguous()):
            raise ValueError("dense Trellis rotation scales must be contiguous FP16 vectors on the input device")
    with torch.cuda.device(source.device):
        compiled = _resolve(columns, pre_scale is not None, post_scale is not None)
        compiled(_pointer(source.data_ptr()), _pointer(destination.data_ptr()),
                 _pointer(source.data_ptr() if pre_scale is None else pre_scale.data_ptr()),
                 _pointer(source.data_ptr() if post_scale is None else post_scale.data_ptr()),
                 Int32(source.shape[0]), current_cuda_stream())
