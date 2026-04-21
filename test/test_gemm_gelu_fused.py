"""Correctness test for the fused GEMM + GELU kernel.

Tests C = GELU(A @ B) where GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
for various matrix sizes, comparing against PyTorch reference (torch.mm + F.gelu).
"""
import torch
import torch.nn.functional as F
import ac4k_kernel


def _gelu_ref(mat: torch.Tensor) -> torch.Tensor:
    """Reference GELU using PyTorch's F.gelu (exact mode)."""
    return F.gelu(mat)


def test_gemm_gelu_fused_128x128x128():
    """Test the required 128x128x128 shape: C = GELU(A @ B)."""
    M, K, N = 128, 128, 128
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')

    C = ac4k_kernel.gemm_gelu_fused(A, B)
    C_ref = _gelu_ref(torch.mm(A, B))

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ 128x128x128 test passed: [{M}, {K}] @ [{K}, {N}] -> [{M}, {N}]")


def test_gemm_gelu_fused_basic():
    """Test basic fused GEMM+GELU: C = GELU(A @ B)."""
    M, K, N = 64, 32, 48
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')

    C = ac4k_kernel.gemm_gelu_fused(A, B)
    C_ref = _gelu_ref(A @ B)

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Basic test passed: [{M}, {K}] @ [{K}, {N}] -> [{M}, {N}]")


def test_gemm_gelu_fused_non_tile_aligned():
    """Test fused GEMM+GELU with non-tile-aligned dimensions."""
    M, K, N = 37, 53, 71
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')

    C = ac4k_kernel.gemm_gelu_fused(A, B)
    C_ref = _gelu_ref(A @ B)

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Non-tile-aligned test passed: [{M}, {K}] @ [{K}, {N}]")


def test_gemm_gelu_fused_small():
    """Test fused GEMM+GELU with very small matrices."""
    M, K, N = 1, 1, 1
    A = torch.tensor([[1.5]], dtype=torch.float32, device='cuda')
    B = torch.tensor([[2.0]], dtype=torch.float32, device='cuda')

    C = ac4k_kernel.gemm_gelu_fused(A, B)
    C_ref = _gelu_ref(A @ B)

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Small matrix test passed: [{M}, {K}] @ [{K}, {N}]")


def test_gemm_gelu_fused_large():
    """Test fused GEMM+GELU with larger matrices (multiple tiles)."""
    M, K, N = 512, 256, 384
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')

    C = ac4k_kernel.gemm_gelu_fused(A, B)
    C_ref = _gelu_ref(A @ B)

    torch.testing.assert_close(C, C_ref, rtol=1e-3, atol=1e-3)
    print(f"✓ Large matrix test passed: [{M}, {K}] @ [{K}, {N}]")


def test_gemm_gelu_fused_out_preallocated():
    """Test fused GEMM+GELU with pre-allocated output tensor."""
    M, K, N = 64, 32, 48
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')

    C = torch.empty(M, N, dtype=torch.float32, device='cuda')
    result = ac4k_kernel.gemm_gelu_fused(A, B, out=C)
    C_ref = _gelu_ref(A @ B)

    # Check that the returned tensor is the same object
    assert result is C, "Returned tensor should be the same object as out"
    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Pre-allocated output test passed")


def test_gemm_gelu_fused_negative_values():
    """Test that GELU correctly handles negative values (near-zero output)."""
    M, K, N = 16, 16, 16
    # Use large negative values so GELU output should be near zero
    A = -torch.randn(M, K, dtype=torch.float32, device='cuda') * 5.0
    B = torch.randn(K, N, dtype=torch.float32, device='cuda') * 5.0

    C = ac4k_kernel.gemm_gelu_fused(A, B)
    C_ref = _gelu_ref(A @ B)

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Negative values test passed")


def test_gemm_gelu_fused_zero_input():
    """Test fused GEMM+GELU with zero matrices (GELU(0) = 0)."""
    M, K, N = 32, 32, 32
    A = torch.zeros(M, K, dtype=torch.float32, device='cuda')
    B = torch.zeros(K, N, dtype=torch.float32, device='cuda')

    C = ac4k_kernel.gemm_gelu_fused(A, B)
    C_ref = _gelu_ref(A @ B)

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Zero input test passed")


def test_gemm_gelu_fused_identity():
    """Test fused GEMM+GELU with identity matrix: GELU(A @ I) = GELU(A)."""
    N = 64
    A = torch.randn(N, N, dtype=torch.float32, device='cuda')
    I = torch.eye(N, dtype=torch.float32, device='cuda')

    C = ac4k_kernel.gemm_gelu_fused(A, I)
    C_ref = _gelu_ref(A)

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Identity matrix test passed: GELU(A @ I) = GELU(A)")


def test_gemm_gelu_fused_rectangular():
    """Test fused GEMM+GELU with rectangular matrices."""
    shapes = [(128, 64, 256), (256, 128, 64), (64, 256, 128)]
    for M, K, N in shapes:
        A = torch.randn(M, K, dtype=torch.float32, device='cuda')
        B = torch.randn(K, N, dtype=torch.float32, device='cuda')

        C = ac4k_kernel.gemm_gelu_fused(A, B)
        C_ref = _gelu_ref(A @ B)

        torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
        print(f"✓ Rectangular test passed: [{M}, {K}] @ [{K}, {N}]")


def run_all_tests():
    """Run all fused GEMM+GELU correctness tests."""
    print("=" * 60)
    print("Fused GEMM + GELU Correctness Tests")
    print("=" * 60)

    tests = [
        test_gemm_gelu_fused_128x128x128,
        test_gemm_gelu_fused_basic,
        test_gemm_gelu_fused_non_tile_aligned,
        test_gemm_gelu_fused_small,
        test_gemm_gelu_fused_large,
        test_gemm_gelu_fused_out_preallocated,
        test_gemm_gelu_fused_negative_values,
        test_gemm_gelu_fused_zero_input,
        test_gemm_gelu_fused_identity,
        test_gemm_gelu_fused_rectangular,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)