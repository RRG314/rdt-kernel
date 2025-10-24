"""
RDT Kernel (Recursive Diffusion-Type Kernel)
============================================

Author: Steven Reid (RRG314)
License: MIT

A lightweight, entropy-regulated diffusion operator implemented in PyTorch.

Core equations:
    ∂L/∂t = -α·log(L) + D·∇²L

Features:
- Stable nonlinear diffusion with logarithmic potential
- Works on CPU, GPU, or TPU (via torch_xla)
- Minimal dependencies (only torch)
"""

import torch

__all__ = ["get_device", "rdt_kernel", "step"]

# ------------------------------------------------------------
# Device selection
# ------------------------------------------------------------
def get_device():
    """
    Detect best available compute device.
    Priority: GPU → TPU → CPU
    Returns:
        (device, name)
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), "GPU"
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device(), "TPU"
    except Exception:
        pass
    return torch.device("cpu"), "CPU"


# ------------------------------------------------------------
# RDT Core Kernel
# ------------------------------------------------------------
def rdt_kernel(L: torch.Tensor, alpha: float = 0.5, D: float = 0.1, dx: float = 1.0) -> torch.Tensor:
    """
    Compute the recursive diffusion-type kernel.

    Args:
        L: torch.Tensor — field (must be positive)
        alpha: float — logarithmic potential coefficient
        D: float — diffusion coefficient
        dx: float — spatial step size

    Returns:
        torch.Tensor — dL/dt (same shape as L)
    """
    L_up    = torch.roll(L,  1, -2)
    L_down  = torch.roll(L, -1, -2)
    L_left  = torch.roll(L,  1, -1)
    L_right = torch.roll(L, -1, -1)
    lap = (L_up + L_down + L_left + L_right - 4 * L) / (dx ** 2)
    return -alpha * torch.log(L) + D * lap


# ------------------------------------------------------------
# Integration Step
# ------------------------------------------------------------
def step(L: torch.Tensor, alpha: float = 0.5, D: float = 0.1, dx: float = 1.0, dt: float = 0.01) -> torch.Tensor:
    """
    Advance the field one time step using the RDT kernel.

    Args:
        L: torch.Tensor — input field
        alpha, D, dx, dt: floats — kernel parameters

    Returns:
        torch.Tensor — updated field (clamped to remain positive)
    """
    return torch.clamp(L + dt * rdt_kernel(L, alpha, D, dx), min=1.001)


# ------------------------------------------------------------
# Example (optional)
# ------------------------------------------------------------
if __name__ == "__main__":
    device, name = get_device()
    print(f"Running RDT kernel on {name}")
    L = (1 + 0.1 * torch.rand((1, 256, 256))).to(device)
    L_next = step(L)
    print(f"Mean after one step: {L_next.mean().item():.6f}")
