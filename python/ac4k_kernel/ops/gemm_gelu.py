"""Fused FP32 GEMM + GELU activation operator.

Provides zero-overhead access to architecture-optimized fused kernel.
"""
import torch
import torch.nn.functional as F

# Direct import - zero runtime dispatch overhead
from .._cuda_ops import gemm_gelu_fused as _gemm_gelu_fused


def gemm_gelu_fused(A: torch.Tensor, B: torch.Tensor,
                    out: torch.Tensor = None) -> torch.Tensor:
    """
    Fused FP32 GEMM + GELU Activation

    Computes: C = GELU(A @ B)
    where GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

    Fusing GEMM and GELU avoids a separate kernel launch and an extra
    read/write of the full output matrix through global memory.

    Zero-overhead dispatch - kernel is selected at compile time.

    Args:
        A: Input tensor [M, K], float32, row-major, contiguous, on CUDA.
        B: Input tensor [K, N], float32, row-major, contiguous, on CUDA.
        out: Optional pre-allocated output tensor [M, N], float32.

    Returns:
        Output tensor [M, N], float32.
    """
    assert A.dtype == torch.float32, f"A must be float32, got {A.dtype}"
    assert B.dtype == torch.float32, f"B must be float32, got {B.dtype}"
    assert A.dim() == 2, f"A must be 2D, got {A.dim()}D"
    assert B.dim() == 2, f"B must be 2D, got {B.dim()}D"
    assert A.size(1) == B.size(0), (
        f"A and B inner dimensions must match: A.size(1)={A.size(1)}, B.size(0)={B.size(0)}"
    )

    M, K = A.shape
    _, N = B.shape

    if out is None:
        out = torch.empty(M, N, dtype=torch.float32, device=A.device)
    else:
        assert out.dtype == torch.float32, f"out must be float32, got {out.dtype}"
        assert out.shape == (M, N), (
            f"out shape must be ({M}, {N}), got {out.shape}"
        )

    _gemm_gelu_fused(out, A, B)
    return out


# Direct kernel export for maximum performance
__all__ = ["gemm_gelu_fused"]