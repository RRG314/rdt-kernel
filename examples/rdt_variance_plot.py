"""
RDT Kernel Variance Plot

Demonstrates self-stabilizing behavior by plotting variance decay
over time for different time step values.

This script visualizes how the RDT kernel dampens variance:
- dt=0.005: Smooth, gradual decay
- dt=0.01: Strong damping
- dt=0.02: Rapid clamp-dominated flattening
"""

import torch
import math
import matplotlib.pyplot as plt

from rdt_kernel import rdt_kernel, step, get_device


@torch.no_grad()
def run_variance_experiment(n=256, steps=300, dt=0.01, alpha=0.5, D=0.1, dx=1.0):
    """
    Run RDT kernel simulation and track variance over time.

    Args:
        n: Grid size (n x n)
        steps: Number of time steps
        dt: Time step size
        alpha: Nonlinear damping coefficient
        D: Diffusion constant
        dx: Spatial step size

    Returns:
        list: Variance at each time step
    """
    device, _ = get_device()

    # Initialize field with sinusoidal perturbation
    x = torch.linspace(0, 2*math.pi, n, device=device).unsqueeze(1)
    L = 1.01 + 0.01 * torch.sin(x)

    variances = []
    for _ in range(steps):
        variances.append(L.var(unbiased=False).item())
        L = step(L, alpha, D, dx, dt)

    return variances


def plot_variance_comparison():
    """
    Generate variance comparison plot for different time steps.
    """
    print("Running variance experiments...")

    plt.figure(figsize=(10, 6))

    # Test different time steps
    time_steps = [0.005, 0.01, 0.02]
    colors = ['blue', 'green', 'red']
    labels = [
        'dt=0.005 (smooth evolution)',
        'dt=0.01 (strong damping)',
        'dt=0.02 (rapid flattening)'
    ]

    for dt, color, label in zip(time_steps, colors, labels):
        print(f"  Running dt={dt}...")
        variances = run_variance_experiment(dt=dt, steps=300)
        plt.plot(variances, color=color, linewidth=2, label=label)

    plt.yscale("log")
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Variance (log scale)", fontsize=12)
    plt.title("RDT Kernel: Self-Stabilizing Variance Decay", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    output_file = "rdt_variance_decay.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")

    plt.show()


def plot_parameter_sensitivity():
    """
    Generate plot showing sensitivity to alpha parameter.
    """
    print("\nRunning parameter sensitivity experiments...")

    plt.figure(figsize=(10, 6))

    # Test different alpha values
    alphas = [0.2, 0.5, 1.0]
    colors = ['purple', 'green', 'orange']

    for alpha, color in zip(alphas, colors):
        print(f"  Running alpha={alpha}...")
        variances = run_variance_experiment(alpha=alpha, dt=0.01, steps=300)
        plt.plot(variances, color=color, linewidth=2, label=f'α={alpha}')

    plt.yscale("log")
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Variance (log scale)", fontsize=12)
    plt.title("RDT Kernel: Sensitivity to Damping Coefficient α", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    output_file = "rdt_alpha_sensitivity.png"
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")

    plt.show()


def main():
    """
    Run all variance experiments and generate plots.
    """
    print("=" * 60)
    print("RDT Kernel Variance Analysis")
    print("=" * 60)

    # Check device
    device, name = get_device()
    print(f"\nUsing device: {name}")

    # Generate plots
    plot_variance_comparison()
    plot_parameter_sensitivity()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
