# RDT Kernel

**Recursive Diffusion-Type Kernel (RDT)**  
Author: Steven Reid (RRG314 Research Group)  
License: MIT  

---

## Overview

The RDT Kernel is a nonlinear, entropy-regulated diffusion operator implemented in PyTorch.  
It provides a compact and efficient way to evolve scalar fields under a coupled logarithmic potential and Laplacian diffusion dynamic:

\
\frac{\partial L}{\partial t} = -\alpha \log(L) + D \nabla^2 L
\

Unlike conventional linear diffusion equations, the RDT Kernel introduces a logarithmic potential term that regulates field entropy, maintaining bounded and stable evolution even in recursive or chaotic systems.  
It is lightweight, numerically stable, and compatible with CPU, GPU, and TPU backends.

---

## Key Features

- Cross-hardware operation (CPU, GPU, TPU)  
- Entropy-regulated diffusion with a logarithmic potential  
- Fully parallelized PyTorch implementation  
- Differentiable and compatible with machine-learning workflows  
- Guaranteed positivity and bounded energy through clamping  
- Minimal dependencies (requires only torch; optional torch_xla for TPU)

---

## Typical Applications

| Domain | Example Uses |
|--------|---------------|
| Machine Learning / AI | Physics-informed neural networks, differentiable PDE layers, generative or diffusion models |
| Scientific Computing | Nonlinear diffusion and stability simulations, entropy-bounded PDE solvers |
| Computational Physics | Entropy-driven field relaxation, recursive geometric entropy studies |
| Signal & Image Processing | Nonlinear smoothing and denoising, texture evolution |
| Information Theory | Modeling entropy growth and information diffusion |

Because it is differentiable and hardware-agnostic, the RDT Kernel can be integrated directly into PyTorch autograd graphs for end-to-end training of PDE-based or hybrid AI-physics systems.

---

## Mathematical Foundation

The RDT equation combines two terms:

**1. Logarithmic Entropy Term**  
(-α log L) — acts as a restoring potential that resists uncontrolled growth and encodes entropy regulation.

**2. Diffusion Term**  
(D ∇² L) — implements standard Laplacian smoothing, spreading gradients spatially to stabilize field variations.

Together these form a self-stabilizing PDE suitable for equilibrium-seeking and diffusion-reaction systems.

---

## API Reference

### get_device()
Detects the best compute backend.

```python
device, name = get_device()
# returns ("cuda", "GPU"), ("cpu", "CPU"), or ("TPU")
```

### rdt_kernel(L, alpha=0.5, D=0.1, dx=1.0)
Computes the instantaneous field derivative:

| Parameter | Type | Description |
|------------|------|-------------|
| L | Tensor | Input scalar field (must be positive) |
| alpha | float | Logarithmic potential coefficient |
| D | float | Diffusion coefficient |
| dx | float | Spatial step size |

Returns: Tensor — field derivative (∂L/∂t)

### step(L, alpha=0.5, D=0.1, dx=1.0, dt=0.01)
Advances the field one time step using:

\
L_{t+1} = L_t + \Delta t \cdot \text{RDT}(L_t)
\

The output is clamped to remain positive and bounded.

---

## Quick Start

```python
import torch
from rdt_kernel import step, get_device

device, name = get_device()
print("Running on", name)

L = (1 + 0.1 * torch.rand((1, 256, 256))).to(device)

for _ in range(100):
    L = step(L, alpha=0.5, D=0.1, dx=1.0, dt=0.01)

print("Mean field value:", L.mean().item())
```

---

## Integration Examples

### As a Differentiable Layer
```python
import torch.nn as nn
from rdt_kernel import step

class RDTLayer(nn.Module):
    def __init__(self, alpha=0.5, D=0.1, dx=1.0, dt=0.01):
        super().__init__()
        self.alpha, self.D, self.dx, self.dt = alpha, D, dx, dt

    def forward(self, L):
        return step(L, self.alpha, self.D, self.dx, self.dt)
```

### In a Physics Simulation Pipeline
```python
from torchdiffeq import odeint
from rdt_kernel import rdt_kernel
import torch

t = torch.linspace(0, 1, 100)
L0 = torch.ones((1, 128, 128))
sol = odeint(lambda t, L: rdt_kernel(L), L0, t)
```

---

## Performance

- Efficient for large grids (512×512 and higher)  
- Linear memory scaling on GPU  
- TPU-compatible via torch_xla  
- Numerically stable for dt ≤ 0.05 in most scenarios  

---

## Installation

### From GitHub
```bash
git clone https://github.com/RRG314/rdt-kernel.git
cd rdt-kernel
pip install .
```

### From PyPI (when available)
```bash
pip install rdt-kernel
```

---

## Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- Optional: torch_xla for TPU support  

---

## Research Context

The RDT Kernel originated in the Recursive Geometric Entropy and Entropy Field Theory (REFT) framework.  
It provides a minimal computational mechanism for entropy-bounded field evolution—relevant to stability analysis, information diffusion, and recursive geometric systems.

---

## Roadmap

- Add 3-D and multi-dimensional variants  
- Benchmark mixed-precision (FP16/BF16)  
- Add symbolic auto-differentiation module  
- Publish official PyPI distribution  

---

## License

MIT License © 2025 Steven Reid  
Permission is granted to use, copy, modify, and distribute this software with attribution.

---

A lightweight, entropy-regulated diffusion operator bridging physics and computation.
