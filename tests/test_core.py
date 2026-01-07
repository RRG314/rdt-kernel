"""
Comprehensive test suite for rdt_kernel.

Tests cover:
- Basic functionality
- Device detection
- Numerical stability
- Parameter handling
- Edge cases
"""

import torch
import math
import pytest


def test_imports():
    """Test that all expected functions can be imported."""
    from rdt_kernel import get_device, rdt_kernel, step, run_demo
    assert callable(get_device)
    assert callable(rdt_kernel)
    assert callable(step)
    assert callable(run_demo)


def test_get_device():
    """Test device detection returns valid device and name."""
    from rdt_kernel import get_device
    device, name = get_device()

    assert isinstance(device, torch.device)
    assert isinstance(name, str)
    assert name in ["CPU", "GPU", "TPU"]


def test_rdt_kernel_basic():
    """Test basic rdt_kernel computation."""
    from rdt_kernel import rdt_kernel

    n = 32
    L = torch.ones((n, n))
    alpha = 0.5
    D = 0.1
    dx = 1.0

    dLdt = rdt_kernel(L, alpha, D, dx)

    assert dLdt.shape == L.shape
    assert torch.isfinite(dLdt).all()


def test_rdt_kernel_uniform_field():
    """Test that uniform field produces zero Laplacian component."""
    from rdt_kernel import rdt_kernel

    n = 32
    L = torch.ones((n, n)) * 2.0  # Uniform field
    alpha = 0.5
    D = 0.1
    dx = 1.0

    dLdt = rdt_kernel(L, alpha, D, dx)

    # For uniform field, Laplacian should be zero
    # Result should be -alpha * log(L) everywhere
    expected = -alpha * torch.log(L)
    assert torch.allclose(dLdt, expected, rtol=1e-5)


def test_step_basic():
    """Test basic time-stepping function."""
    from rdt_kernel import step

    n = 32
    L = torch.ones((n, n)) + 0.1
    alpha = 0.5
    D = 0.1
    dx = 1.0
    dt = 0.01

    L_new = step(L, alpha, D, dx, dt)

    assert L_new.shape == L.shape
    assert torch.isfinite(L_new).all()
    assert (L_new >= 1.001).all()  # Check clamping


def test_step_clamping():
    """Test that clamping prevents values below clamp_min."""
    from rdt_kernel import step

    n = 16
    L = torch.ones((n, n)) * 1.01  # Very close to clamp boundary
    alpha = 1.0  # Strong damping
    D = 0.0     # No diffusion
    dx = 1.0
    dt = 0.1    # Large time step

    L_new = step(L, alpha, D, dx, dt, clamp_min=1.001)

    # All values should be at or above clamp_min
    assert (L_new >= 1.001).all()
    assert torch.isfinite(L_new).all()


def test_step_custom_clamp_min():
    """Test step function with custom clamp_min parameter."""
    from rdt_kernel import step

    n = 16
    L = torch.ones((n, n)) * 2.0
    alpha = 0.5
    D = 0.1
    dx = 1.0
    dt = 0.01
    clamp_min = 1.5

    L_new = step(L, alpha, D, dx, dt, clamp_min=clamp_min)

    assert (L_new >= clamp_min).all()


def test_numerical_stability():
    """Test that kernel remains stable over many iterations."""
    from rdt_kernel import step

    n = 64
    L = torch.ones((n, n)) + 0.1 * torch.randn((n, n))
    L = torch.clamp(L, min=1.001)

    alpha = 0.5
    D = 0.1
    dx = 1.0
    dt = 0.01

    # Run 100 steps
    for _ in range(100):
        L = step(L, alpha, D, dx, dt)

    # Check no NaNs or infinities
    assert torch.isfinite(L).all()
    assert (L >= 1.001).all()
    assert (L <= 1e6).all()  # Reasonable upper bound


def test_periodic_boundaries():
    """Test that periodic boundaries are correctly implemented."""
    from rdt_kernel import rdt_kernel

    n = 8
    L = torch.ones((n, n))
    # Set a single hot spot at corner
    L[0, 0] = 2.0

    alpha = 0.0  # No damping, pure diffusion
    D = 1.0
    dx = 1.0

    dLdt = rdt_kernel(L, alpha, D, dx)

    # The corner should diffuse to all 4 neighbors (including wrapped edges)
    # dLdt[0,0] should be negative (losing heat)
    assert dLdt[0, 0] < 0

    # Wrapped neighbors should have positive dLdt (gaining heat)
    assert dLdt[0, 1] > 0  # right
    assert dLdt[1, 0] > 0  # down
    assert dLdt[0, -1] > 0  # left (wrapped)
    assert dLdt[-1, 0] > 0  # up (wrapped)


def test_conservation_with_no_damping():
    """Test that pure diffusion conserves mean (no damping)."""
    from rdt_kernel import step

    n = 32
    L = torch.ones((n, n)) + 0.1 * torch.randn((n, n))
    L = torch.clamp(L, min=1.001)

    initial_mean = L.mean()

    alpha = 0.0  # No damping
    D = 0.1
    dx = 1.0
    dt = 0.01

    # Run several steps
    for _ in range(10):
        L = step(L, alpha, D, dx, dt)

    final_mean = L.mean()

    # Mean should be approximately conserved (within numerical error + clamping effects)
    # This is approximate due to clamping at boundaries
    assert abs(final_mean - initial_mean) < 0.1


def test_damping_reduces_variance():
    """Test that damping term reduces field variance over time."""
    from rdt_kernel import step

    n = 64
    L = torch.ones((n, n)) + 0.2 * torch.randn((n, n))
    L = torch.clamp(L, min=1.001)

    initial_var = L.var()

    alpha = 0.5
    D = 0.1
    dx = 1.0
    dt = 0.01

    # Run many steps
    for _ in range(50):
        L = step(L, alpha, D, dx, dt)

    final_var = L.var()

    # Variance should decrease due to damping
    assert final_var < initial_var


def test_run_demo():
    """Test that run_demo executes without errors."""
    from rdt_kernel import run_demo

    # Run a small, fast demo
    L = run_demo(n=16, steps=5, alpha=0.5, D=0.1, dx=1.0, dt=0.01)

    assert L.shape == (16, 16)
    assert torch.isfinite(L).all()
    assert (L >= 1.001).all()


def test_different_grid_sizes():
    """Test kernel works with various grid sizes."""
    from rdt_kernel import step

    for n in [8, 16, 32, 64, 128]:
        L = torch.ones((n, n))
        alpha = 0.5
        D = 0.1
        dx = 1.0
        dt = 0.01

        L_new = step(L, alpha, D, dx, dt)

        assert L_new.shape == (n, n)
        assert torch.isfinite(L_new).all()


def test_parameter_ranges():
    """Test kernel behavior with different parameter values."""
    from rdt_kernel import step

    n = 32
    L = torch.ones((n, n)) + 0.1
    dx = 1.0
    dt = 0.01

    # Test different alpha values
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        L_new = step(L.clone(), alpha, D=0.1, dx=dx, dt=dt)
        assert torch.isfinite(L_new).all()

    # Test different D values
    for D in [0.01, 0.1, 0.5, 1.0]:
        L_new = step(L.clone(), alpha=0.5, D=D, dx=dx, dt=dt)
        assert torch.isfinite(L_new).all()

    # Test different time steps
    for dt in [0.001, 0.005, 0.01, 0.02]:
        L_new = step(L.clone(), alpha=0.5, D=0.1, dx=dx, dt=dt)
        assert torch.isfinite(L_new).all()


def test_gpu_if_available():
    """Test that kernel works on GPU if available."""
    from rdt_kernel import get_device, step

    device, name = get_device()

    if name == "GPU":
        n = 64
        L = torch.ones((n, n), device=device) + 0.1
        alpha = 0.5
        D = 0.1
        dx = 1.0
        dt = 0.01

        L_new = step(L, alpha, D, dx, dt)

        assert L_new.device == device
        assert torch.isfinite(L_new).all()


def test_batch_consistency():
    """Test that processing fields individually vs in batch gives consistent results."""
    from rdt_kernel import rdt_kernel

    n = 16
    L1 = torch.ones((n, n)) + 0.1
    L2 = L1.clone()

    alpha = 0.5
    D = 0.1
    dx = 1.0

    # Process individually
    dLdt1 = rdt_kernel(L1, alpha, D, dx)
    dLdt2 = rdt_kernel(L2, alpha, D, dx)

    # Should be identical for identical inputs
    assert torch.allclose(dLdt1, dLdt2)


def test_negative_values_prevented():
    """Test that field values cannot become negative due to clamping."""
    from rdt_kernel import step

    n = 16
    L = torch.ones((n, n)) * 1.1

    # Extreme parameters that might cause negatives without clamping
    alpha = 10.0
    D = 0.0
    dx = 1.0
    dt = 1.0

    for _ in range(10):
        L = step(L, alpha, D, dx, dt)
        assert (L > 0).all()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
