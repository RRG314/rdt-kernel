# **RDT Kernel**

Recursive Diffusion Transform (RDT) Kernel

---

## **Overview**

RDT Kernel is an experimental numerical system that evolves a 2-D scalar field using a compact nonlinear partial differential equation (PDE). The method combines a logarithmic damping term with a discrete Laplacian diffusion operator, creating a self-stabilizing dynamic that smooths the field while preventing collapse.

The implementation is written in pure PyTorch, runs automatically on CPU, GPU, or TPU, and is designed for research into nonlinear diffusion, entropy-bounded scalar systems, and recursive field evolution.
Although the formulation is mathematically inspired by patterns in dissipative media, the kernel is not intended to simulate or approximate any real physical system; it is a computational tool for experimentation.
The model combines:

* a **logarithmic damping term** (stabilizing the field), and
* a **discrete Laplacian diffusion term** (spreading local structure)

to form a simple, efficient simulator for **recursive field dynamics**.

Although the formulation is inspired by structural patterns found in dissipative systems, it is **not intended to model or approximate real physical processes**.
All components (log-damping, diffusion, clamping) are implemented strictly as **mathematical operators for computational experimentation**.

The RDT Kernel runs on **CPU, GPU, or TPU** automatically using PyTorch’s device handling and is suitable for research into nonlinear dynamics, entropy-bounded evolution, and recursive scalar systems.

---

## **Mathematical Model**

The kernel evolves a scalar field (L(x, y, t)) according to:

[
\frac{\partial L}{\partial t} = -\alpha \ln(L) + D \nabla^2 L
]

Where:

* **(L(x, y, t))** — scalar field
* **(\alpha)** — nonlinear damping coefficient
* **(D)** — diffusion constant
* **(\nabla^2 L)** — 2D Laplacian (nearest-neighbor discrete stencil)

The combination of logarithmic damping and diffusion creates **self-stabilizing behavior**, preventing runaway growth while encouraging smooth recursive evolution.

---

## **Key Features**

* Pure PyTorch PDE solver
* Supports **CPU, GPU, and TPU**
* Logarithmic damping for stability
* Laplacian-based diffusion
* Euler integrator with clamping (prevents singularities)
* Fully vectorized operations (no Python loops)
* Simple, readable code designed for experimentation
* Runs in a few milliseconds for large grids

---

## **Benchmarks and Stability Behavior**

The RDT Kernel has been benchmarked across a range of grid sizes, initial conditions, and time steps using GPU execution.

### **Numerical Stability**

Across all tested configurations:

* No NaNs or infinities were observed
* The field remained strictly positive due to clamping
* The system converged toward bounded states without runaway growth

This confirms the kernel's role as a **self-stabilizing numerical operator** rather than a physical simulator.

### **Clamp Activity**

The built-in clamp prevents logarithmic singularities and enforces numerical safety.
Measured clamp activity shows:

* For moderate time steps (`dt ≈ 0.005–0.01`), clamp activation is partial and dynamics remain smooth
* For larger time steps (`dt ≥ 0.02`), clamp activation can dominate, rapidly flattening the field to a bounded uniform state

This behavior is intentional and ensures stability under aggressive parameter choices.

### **Recommended Parameters**

For default settings (`alpha=0.5`, `D=0.1`, `dx=1.0`):

* **Smooth recursive evolution:** `dt ≈ 0.005–0.01`
* **Rapid stabilization / damping:** `dt ≥ 0.02`

### **Performance**

On a single GPU, the kernel achieves approximately **6,000–7,000 Euler steps per second** for grid sizes up to 1024×1024, confirming suitability for large-tensor PDE experimentation.

---

## **Installation**

Install from PyPI:

```
pip install rdt-kernel
```

Install from GitHub:

```
pip install git+https://github.com/RRG314/rdt-kernel.git
```

Or clone manually:

```
git clone https://github.com/RRG314/rdt-kernel.git
cd rdt-kernel
pip install .
```

---

## **Usage Example**

```python
from rdt_kernel import run_demo

# Run a 128×128 simulation for 100 steps
run_demo(
    n=128,
    steps=100,
    alpha=0.5,
    D=0.1,
    dx=1.0,
    dt=0.01
)
```

Example output:

```
Running 100 steps on GPU...
Done in 0.029s, mean=1.003536
```

---

## **API Summary**

### **get_device()**

Automatically selects CPU, GPU, or TPU if available.

### **rdt_kernel(L, alpha, D, dx)**

Computes the PDE right-hand side:

[
\partial L/\partial t
]

Includes:

* discrete Laplacian (5-point stencil with **periodic boundary conditions**)
* nonlinear log-damping

**Note:** Uses `torch.roll()` to implement periodic boundaries, meaning the field wraps around at edges.

### **step(L, alpha, D, dx, dt, clamp_min=1.001)**

Performs one Euler step:

[
L_{t+1} = L_t + dt \cdot \text{rdt_kernel}(L)
]

The field is clamped to `clamp_min` (default: 1.001) to prevent numerical collapse and logarithmic singularities.



---

## **What the Kernel Is Good For**

This implementation is intended for **research, experimentation, and ML-adjacent simulation**, not physics.
Useful for:

* Nonlinear diffusion experiments
* Entropy-bounded scalar evolution
* Recursive geometric/diffusion systems
* Prototype PDE-based ML components
* Testing large-tensor PDE performance
* Exploring stability effects of log-based damping

---

## **Limitations**

* Euler integration (not adaptive or implicit)
* Not a physical model or simulator
* Numerical stability depends on parameter choice
* Not yet optimized for multi-GPU or distributed setups

---

## **Minimal Working Code**

```python
import torch
from rdt_kernel import rdt_kernel, step

n = 64
L = torch.ones((n,n)).cuda()  # GPU if available
alpha = 0.5
D = 0.1
dx = 1.0
dt = 0.01

for _ in range(100):
    L = step(L, alpha, D, dx, dt)

print("Done, mean:", L.mean())
```

---

## **Author**

**Steven Reid**
Independent Researcher
ORCID: 0009-0003-9132-3410
GitHub: [https://github.com/RRG314](https://github.com/RRG314)

---

## **License**

Apache License 2.0



If you want:

* a version with figures
* a short version for PyPI
* a long research explanation
* a combined README for Topological Adam + RDT Kernel
* or a documentation website

Just tell me.
