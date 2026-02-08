"""Plot CR3BP Earth-Moon trajectory results.

Reads trajectory CSV (columns: phase,t,x,y,vx,vy,m) and produces:
  1. Capture energy evolution (phases 2-4)
  2. Trajectory with Jacobi potential contours (phases 1-4)

Usage:
    python plot_3brp.py trajectory.csv
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

# Constants (must match simulation).
days = 24 * 3600
rearth = 6378
rmoon = 1737
r12 = 384400
mu1 = 398600
mu2 = 4903.02
m1 = 5974e21
m2 = 7348e19
M = m1 + m2
pi_1 = m1 / M
pi_2 = m2 / M
W = np.sqrt((mu1 + mu2) / r12**3)
x1 = -pi_2 * r12
x2 = pi_1 * r12


def load_trajectory(path):
    """Load trajectory CSV. Returns dict mapping phase number to (N, 6) arrays [t, x, y, vx, vy, m]."""
    data = np.loadtxt(path, delimiter=",")
    phases = {}
    for p in np.unique(data[:, 0]).astype(int):
        mask = data[:, 0] == p
        phases[p] = data[mask, 1:]  # drop phase column
    return phases


def capture_energy(x, y, vx, vy):
    """Specific orbital energy relative to Moon [km^2/s^2]."""
    r = np.sqrt((x - x2) ** 2 + y**2)
    return 0.5 * (vx**2 + vy**2) - mu2 / r


def jacobi_potential(x, y):
    """Jacobi pseudo-potential (zero-velocity surface value)."""
    r1 = np.sqrt((x + pi_2 * r12) ** 2 + y**2)
    r2 = np.sqrt((x - pi_1 * r12) ** 2 + y**2)
    return -0.5 * W**2 * (x**2 + y**2) - mu1 / r1 - mu2 / r2


def plot_capture_energy(p2, p3, p4):
    """Plot capture energy for last quarter of phase 2, all of phases 3 and 4.

    Each argument: (N, 6) array [t, x, y, vx, vy, m].
    """
    fig, ax = plt.subplots()

    # Phase 2: last quarter only.
    t2 = p2[:, 0]
    t_cut = t2[0] + 0.75 * (t2[-1] - t2[0])
    mask = t2 >= t_cut
    E2 = capture_energy(p2[mask, 1], p2[mask, 2], p2[mask, 3], p2[mask, 4])
    ax.plot(t2[mask] / days, E2, "r-", label="Energía (último ¼ de Fase 2)")

    # Phase 3.
    E3 = capture_energy(p3[:, 1], p3[:, 2], p3[:, 3], p3[:, 4])
    ax.plot(p3[:, 0] / days, E3, "b-", label="Energía (Fase 3)")

    # Phase 4.
    E4 = capture_energy(p4[:, 1], p4[:, 2], p4[:, 3], p4[:, 4])
    ax.plot(p4[:, 0] / days, E4, "m-", label="Energía (Fase 4)")

    ax.set_xlabel("Tiempo (días)")
    ax.set_ylabel("Energía de captura [km²/s²]")
    ax.set_title("Evolución de la Energía de Captura")
    ax.legend()
    ax.grid(True)
    return fig


def plot_trajectory(p1, p2, p3, p4):
    """Plot trajectory phases with Jacobi potential contours, Earth, and Moon.

    Each argument: (N, 6) array [t, x, y, vx, vy, m].
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Trajectory phases.
    ax.plot(p1[:, 1], p1[:, 2], "g-")
    ax.plot(p2[:, 1], p2[:, 2], "r-")
    ax.plot(p3[:, 1], p3[:, 2], "b-")
    ax.plot(p4[:, 1], p4[:, 2], "m-")

    # Earth and Moon.
    theta = np.linspace(0, 2 * np.pi, 361)
    ax.fill(x1 + rearth * np.cos(theta), rearth * np.sin(theta), "b", alpha=0.9, label="Earth")
    ax.fill(x2 + rmoon * np.cos(theta), rmoon * np.sin(theta), "g", alpha=0.9, label="Moon")

    # Jacobi potential contours.
    xg = np.linspace(-100000, 500000, 500)
    yg = np.linspace(-250000, 250000, 500)
    X, Y = np.meshgrid(xg, yg)
    Z = jacobi_potential(X, Y)
    levels = np.linspace(-1.6649, -1.60, 4)
    cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=8)
    fig.colorbar(cs, ax=ax, label="Potencial de Jacobi [km²/s²]")

    fs = 20
    ax.set_xlabel("x [km]", fontsize=fs)
    ax.set_ylabel("y [km]", fontsize=fs)
    ax.set_title("Transferencia Tierra-Luna", fontsize=fs)
    ax.legend()
    ax.grid(True)
    ax.set_xlim(-400000, 450000)
    ax.set_ylim(-325000, 325000)
    return fig


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} trajectory.csv")
        sys.exit(1)

    phases = load_trajectory(sys.argv[1])
    plot_capture_energy(phases[2], phases[3], phases[4])
    plot_trajectory(phases[1], phases[2], phases[3], phases[4])
    plt.show()
