"""FP32 GEMM (General Matrix-Matrix Multiplication) operators.

Provides zero-overhead access to architecture-optimized kernels.
"""
import torch

# Direct import - zero runtime dispatch overhead
from .._cuda_ops import gemm_fp32 as _gemm_fp32


def gemm_fp32(A: torch.Tensor, B: torch.Tensor, *,
              alpha: float = 1.0, beta: float = 0.0,
              out: torch.Tensor = None) -> torch.Tensor:
    """
    FP32 General Matrix-Matrix Multiplication

    Computes: C = alpha * A @ B + beta * C

    Zero-overhead dispatch - kernel is selected at compile time.

    Args:
        A: Input tensor [M, K], float32, row-major, contiguous, on CUDA.
        B: Input tensor [K, N], float32, row-major, contiguous, on CUDA.
        alpha: Scaling factor for A @ B (default: 1.0).
        beta: Scaling factor for C (default: 0.0). If 0.0, C is not read.
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

    _gemm_fp32(out, A, B, alpha, beta)
    return out


# Direct kernel export for maximum performance
__all__ = ["gemm_fp32"]