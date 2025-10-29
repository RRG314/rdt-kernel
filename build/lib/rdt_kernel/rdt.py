import torch
import time, math

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "GPU"
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device(), "TPU"
    except ImportError:
        pass
    return torch.device("cpu"), "CPU"

def rdt_kernel(L, alpha, D, dx):
    L_up    = torch.roll(L,  1, 0)
    L_down  = torch.roll(L, -1, 0)
    L_left  = torch.roll(L,  1, 1)
    L_right = torch.roll(L, -1, 1)
    lap = (L_up + L_down + L_left + L_right - 4*L) / (dx**2)
    dLdt = -alpha * torch.log(L) + D * lap
    return dLdt

def step(L, alpha, D, dx, dt):
    return torch.clamp(L + dt * rdt_kernel(L, alpha, D, dx), min=1.001)

def run_demo(n=256, steps=100, alpha=0.5, D=0.1, dx=1.0, dt=0.01):
    device, name = get_device()
    L = torch.ones((n, n), device=device) + 0.01*torch.sin(torch.linspace(0,2*math.pi,n,device=device)).unsqueeze(1)
    print(f"Running {steps} steps on {name}...")
    if name == "GPU": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps): L = step(L, alpha, D, dx, dt)
    if name == "GPU": torch.cuda.synchronize()
    print(f"Done in {time.time()-t0:.3f}s, mean={L.mean().item():.6f}")
    return L
