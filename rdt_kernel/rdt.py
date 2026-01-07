import torch
import time, math

__all__ = [
    "get_device",
    "rdt_kernel",
    "step",
    "run_demo",
]

def get_device():
    """
    Select CPU, GPU, or TPU automatically.

    Returns:
        tuple: (device, device_name) where device is torch.device and device_name is str
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), "GPU"
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device(), "TPU"
    except ImportError:
        pass
    return torch.device("cpu"), "CPU"


def rdt_kernel(L, alpha, D, dx):
    """
    Compute the Recursive Diffusion Transform PDE:

        dL/dt = -alpha * log(L) + D * Laplacian(L)

    Uses a nearest-neighbor discrete Laplacian with periodic boundaries.

    Args:
        L: 2D tensor representing the scalar field
        alpha: Nonlinear damping coefficient (float)
        D: Diffusion constant (float)
        dx: Spatial step size (float)

    Returns:
        2D tensor: Time derivative dL/dt
    """
    L_up    = torch.roll(L,  1, 0)
    L_down  = torch.roll(L, -1, 0)
    L_left  = torch.roll(L,  1, 1)
    L_right = torch.roll(L, -1, 1)

    lap = (L_up + L_down + L_left + L_right - 4 * L) / (dx * dx)
    return -alpha * torch.log(L) + D * lap


def step(L, alpha, D, dx, dt, clamp_min=1.001):
    """
    Perform one explicit Euler update with clamping.

    Args:
        L: 2D tensor representing the scalar field
        alpha: Nonlinear damping coefficient (float)
        D: Diffusion constant (float)
        dx: Spatial step size (float)
        dt: Time step size (float)
        clamp_min: Minimum value to clamp field to (default: 1.001)

    Returns:
        2D tensor: Updated field L at next time step
    """
    return torch.clamp(
        L + dt * rdt_kernel(L, alpha, D, dx),
        min=clamp_min
    )


def run_demo(n=256, steps=100, alpha=0.5, D=0.1, dx=1.0, dt=0.01):
    """
    Run a demonstration simulation of the RDT kernel.

    Args:
        n: Grid size (n x n)
        steps: Number of time steps to simulate
        alpha: Nonlinear damping coefficient
        D: Diffusion constant
        dx: Spatial step size
        dt: Time step size

    Returns:
        2D tensor: Final state of the field
    """
    device, name = get_device()
    L = torch.ones((n, n), device=device) + 0.01*torch.sin(torch.linspace(0,2*math.pi,n,device=device)).unsqueeze(1)
    print(f"Running {steps} steps on {name}...")
    if name == "GPU": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps): L = step(L, alpha, D, dx, dt)
    if name == "GPU": torch.cuda.synchronize()
    print(f"Done in {time.time()-t0:.3f}s, mean={L.mean().item():.6f}")
    return L
