"""Correctness test for the FP32 GEMM kernel.

Tests C = alpha * A @ B + beta * C for various matrix sizes and parameters.
"""
import torch
import ac4k_kernel


def test_gemm_fp32_basic():
    """Test basic GEMM: C = A @ B (alpha=1, beta=0)."""
    M, K, N = 128, 64, 96
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')

    C = ac4k_kernel.gemm_fp32(A, B)
    C_ref = A @ B

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Basic GEMM test passed: [{M}, {K}] @ [{K}, {N}] -> [{M}, {N}]")


def test_gemm_fp32_alpha():
    """Test GEMM with alpha scaling: C = alpha * A @ B."""
    M, K, N = 64, 32, 48
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    alpha = 2.5

    C = ac4k_kernel.gemm_fp32(A, B, alpha=alpha)
    C_ref = alpha * (A @ B)

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Alpha scaling test passed: alpha={alpha}")


def test_gemm_fp32_beta():
    """Test GEMM with beta: C = A @ B + beta * C."""
    M, K, N = 64, 32, 48
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    C_init = torch.randn(M, N, dtype=torch.float32, device='cuda')
    beta = 0.5

    C = C_init.clone()
    C = ac4k_kernel.gemm_fp32(A, B, beta=beta, out=C)
    C_ref = A @ B + beta * C_init

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Beta test passed: beta={beta}")


def test_gemm_fp32_alpha_beta():
    """Test GEMM with both alpha and beta: C = alpha * A @ B + beta * C."""
    M, K, N = 64, 32, 48
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    C_init = torch.randn(M, N, dtype=torch.float32, device='cuda')
    alpha = 3.0
    beta = 1.5

    C = C_init.clone()
    C = ac4k_kernel.gemm_fp32(A, B, alpha=alpha, beta=beta, out=C)
    C_ref = alpha * (A @ B) + beta * C_init

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Alpha+Beta test passed: alpha={alpha}, beta={beta}")


def test_gemm_fp32_non_tile_aligned():
    """Test GEMM with non-tile-aligned dimensions."""
    M, K, N = 37, 53, 71
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')

    C = ac4k_kernel.gemm_fp32(A, B)
    C_ref = A @ B

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Non-tile-aligned test passed: [{M}, {K}] @ [{K}, {N}]")


def test_gemm_fp32_small():
    """Test GEMM with very small matrices."""
    M, K, N = 1, 1, 1
    A = torch.tensor([[2.0]], dtype=torch.float32, device='cuda')
    B = torch.tensor([[3.0]], dtype=torch.float32, device='cuda')

    C = ac4k_kernel.gemm_fp32(A, B)
    C_ref = A @ B

    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Small matrix test passed: [{M}, {K}] @ [{K}, {N}]")


def test_gemm_fp32_large():
    """Test GEMM with larger matrices (multiple tiles)."""
    M, K, N = 512, 256, 384
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')

    C = ac4k_kernel.gemm_fp32(A, B)
    C_ref = A @ B

    torch.testing.assert_close(C, C_ref, rtol=1e-3, atol=1e-3)
    print(f"✓ Large matrix test passed: [{M}, {K}] @ [{K}, {N}]")


def test_gemm_fp32_out_preallocated():
    """Test GEMM with pre-allocated output tensor."""
    M, K, N = 64, 32, 48
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')

    C = torch.empty(M, N, dtype=torch.float32, device='cuda')
    result = ac4k_kernel.gemm_fp32(A, B, out=C)
    C_ref = A @ B

    # Check that the returned tensor is the same object
    assert result is C, "Returned tensor should be the same object as out"
    torch.testing.assert_close(C, C_ref, rtol=1e-4, atol=1e-4)
    print(f"✓ Pre-allocated output test passed")


def test_gemm_fp32_identity():
    """Test GEMM with identity matrix: A @ I = A."""
    N = 64
    A = torch.randn(N, N, dtype=torch.float32, device='cuda')
    I = torch.eye(N, dtype=torch.float32, device='cuda')

    C = ac4k_kernel.gemm_fp32(A, I)
    torch.testing.assert_close(C, A, rtol=1e-4, atol=1e-4)
    print(f"✓ Identity matrix test passed: [{N}, {N}] @ I")


def run_all_tests():
    """Run all GEMM FP32 correctness tests."""
    print("=" * 60)
    print("FP32 GEMM Correctness Tests")
    print("=" * 60)

    tests = [
        test_gemm_fp32_basic,
        test_gemm_fp32_alpha,
        test_gemm_fp32_beta,
        test_gemm_fp32_alpha_beta,
        test_gemm_fp32_non_tile_aligned,
        test_gemm_fp32_small,
        test_gemm_fp32_large,
        test_gemm_fp32_out_preallocated,
        test_gemm_fp32_identity,
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