#!/usr/bin/env python3
"""Recompute the three vector panels actually embedded in submitted Figure 2.

Frozen from the user-supplied reproduce_intro_exact.py, SHA-256
77472c154c1e1ac1e8a11c230cb6d44f8623f06c0f0186e585a0ebfe561d86dc.
Every arrow, boundary, marker and label is drawn mathematically; no source PDF
or image is read by this renderer. The submission assembles these three panels
with LaTeX; its caption and surrounding typesetting are not part of the assets.

All panels show -gradient L(theta), not a preconditioned/Newton field, and have
no trajectories. The original panel-specific framing, artist order and PDF
archive timestamps are intentional. Exact reproduction requires the Figure 2
environment pinned separately from the other paper figures.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from common import prepare_output

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np


THETA_0 = np.array([0.0, 0.0])
THETA_HAT = np.array([1.02, -0.42])
Q = np.array([[2.4, 0.65], [0.65, 1.15]])
OUTER_RADIUS = 2.20
N1 = np.array([np.cos(np.deg2rad(70.0)), np.sin(np.deg2rad(70.0))])
N2 = np.array([np.cos(np.deg2rad(-30.0)), np.sin(np.deg2rad(-30.0))])
C1, C2 = 1.35, 1.45
ELLIPSE_SAMPLES = 1000
INTERSECTION_SAMPLES = 1600
GRID_N = 23
FIGSIZE = (7, 7)
FIELD_SCALE = 6.3
INITIAL_COLOR, TARGET_COLOR = "#2E8B57", "#B23A3A"

INNER_ARROWS = dict(width=0.0042, headwidth=3.8, headlength=4.1,
                    headaxislength=3.8, color="#4F8FE8", alpha=0.99, zorder=1)
OUTER_ARROWS = dict(width=0.0030, headwidth=3.1, headlength=3.4,
                    headaxislength=3.1, color="#98BDF0", alpha=0.32, zorder=0)
BOUNDARY_STYLE = dict(color="black", linewidth=3.9, zorder=3)
COMPONENT_STYLE = dict(color="0.66", linewidth=2.7, alpha=0.65, zorder=2)


@dataclass(frozen=True)
class Panel:
    shape: Literal["allspace", "ellipsoid", "intersection"]
    stem: str
    frame_factor: float
    original_pdf_timestamp: str


PANELS = (
    Panel("allspace", "figure_2a", 0.52, "20260921214455"),
    Panel("ellipsoid", "figure_2b", 0.52, "20260921214455"),
    Panel("intersection", "figure_2c", 0.54, "20260921231104"),
)


def ellipse_points() -> tuple[np.ndarray, np.ndarray]:
    """Retain the original eigendecomposition, samples and endpoint order."""
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    t = np.linspace(0, 2*np.pi, ELLIPSE_SAMPLES)
    circle = np.vstack([np.cos(t), np.sin(t)])
    transform = eigenvectors @ np.diag(OUTER_RADIUS / np.sqrt(eigenvalues))
    points = transform @ circle
    return points[0], points[1]


def gradient(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = x - THETA_HAT[0], y - THETA_HAT[1]
    q = Q[0, 0]*dx**2 + 2*Q[0, 1]*dx*dy + Q[1, 1]*dy**2
    denominator = np.sqrt(1 + q)
    gx = (Q[0, 0]*dx + Q[0, 1]*dy) / denominator
    gy = (Q[1, 0]*dx + Q[1, 1]*dy) / denominator
    return gx, gy


def in_ellipse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = Q[0, 0]*x**2 + 2*Q[0, 1]*x*y + Q[1, 1]*y**2
    return q <= OUTER_RADIUS**2


def in_strips(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((np.abs(N1[0]*x + N1[1]*y) <= C1)
            & (np.abs(N2[0]*x + N2[1]*y) <= C2))


def polyhedron_corners() -> np.ndarray:
    """Closed bounded parallelogram, with original corner arithmetic/order."""
    inverse = np.linalg.inv(np.vstack([N1, N2]))
    return np.array([inverse @ np.array([s1*C1, s2*C2])
                     for s1, s2 in [(1, 1), (-1, 1), (-1, -1), (1, -1)]])


def intersection_boundary() -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0, 2*np.pi, INTERSECTION_SAMPLES)
    u = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    q = np.einsum("ni,ij,nj->n", u, Q, u)
    r_ellipse = OUTER_RADIUS / np.sqrt(q)
    r1 = C1 / np.maximum(np.abs(u @ N1), 1e-12)
    r2 = C2 / np.maximum(np.abs(u @ N2), 1e-12)
    r = np.minimum.reduce([r_ellipse, r1, r2])
    boundary = r[:, None] * u
    return boundary[:, 0], boundary[:, 1]


def plot_limits(panel: Panel) -> tuple[tuple[float, float], tuple[float, float]]:
    xo, yo = ellipse_points()
    if panel.shape == "intersection":
        poly = polyhedron_corners()
        xmin, xmax = min(xo.min(), poly[:, 0].min()), max(xo.max(), poly[:, 0].max())
        ymin, ymax = min(yo.min(), poly[:, 1].min()), max(yo.max(), poly[:, 1].max())
    else:
        xmin, xmax = xo.min(), xo.max()
        ymin, ymax = yo.min(), yo.max()
    cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
    halfspan = panel.frame_factor * max(xmax-xmin, ymax-ymin)
    return (cx-halfspan, cx+halfspan), (cy-halfspan, cy+halfspan)


def prominent_text(ax: Axes, x: float, y: float, text: str,
                   color: str = "black", fontsize: int = 24):
    # Preserve left/baseline alignment, not center alignment.
    label = ax.text(x, y, text, fontsize=fontsize, color=color, zorder=8,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="none", alpha=0.88))
    label.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])
    return label


def make_figure(panel: Panel) -> Figure:
    """Draw one panel's unnormalized negative-gradient field and geometry."""
    xo, yo = ellipse_points()
    xlim, ylim = plot_limits(panel)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.linspace(xlim[0], xlim[1], GRID_N)
    y = np.linspace(ylim[0], ylim[1], GRID_N)
    X, Y = np.meshgrid(x, y)
    gx, gy = gradient(X, Y)
    U, V = -gx, -gy
    quiver_common = dict(angles="xy", scale_units="xy", scale=FIELD_SCALE)
    if panel.shape == "allspace":
        ax.quiver(X, Y, U, V, **quiver_common, **INNER_ARROWS)
    else:
        inside = in_ellipse(X, Y)
        if panel.shape == "intersection":
            inside = inside & in_strips(X, Y)
        outside = ~inside
        ax.quiver(X[outside], Y[outside], U[outside], V[outside],
                  **quiver_common, **OUTER_ARROWS)
        ax.quiver(X[inside], Y[inside], U[inside], V[inside],
                  **quiver_common, **INNER_ARROWS)
        if panel.shape == "intersection":
            ax.plot(xo, yo, **COMPONENT_STYLE)
            poly = polyhedron_corners()
            closed_poly = np.vstack([poly, poly[0]])
            ax.plot(closed_poly[:, 0], closed_poly[:, 1], **COMPONENT_STYLE)
            bx, by = intersection_boundary()
            ax.plot(bx, by, **BOUNDARY_STYLE)
        else:
            ax.plot(xo, yo, **BOUNDARY_STYLE)
            prominent_text(ax, xlim[0]+0.30, ylim[0]+0.40,
                           r"$\mathcal{A}$", fontsize=27)
    for point, color in [(THETA_0, INITIAL_COLOR), (THETA_HAT, TARGET_COLOR)]:
        ax.scatter([point[0]], [point[1]], s=150, color=color,
                   edgecolors="white", linewidths=1.8, zorder=6)
    prominent_text(ax, THETA_0[0]-0.06, THETA_0[1]+0.24,
                   r"$\theta_0$", color=INITIAL_COLOR, fontsize=24)
    prominent_text(ax, THETA_HAT[0]+0.14, THETA_HAT[1]+0.18,
                   r"$\hat{\theta}$", color=TARGET_COLOR, fontsize=24)
    if panel.shape == "allspace":
        prominent_text(ax, xlim[0]+0.03, ylim[0]+0.40,
                       r"$\mathcal{A}=\mathbb{R}^d$", fontsize=25)
    elif panel.shape == "intersection":
        prominent_text(ax, xlim[0]+0.30, ylim[0]+0.40,
                       r"$\mathcal{A}$", fontsize=27)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig


def render(artifact_root: Path, out_dir: Path) -> list[Path]:
    """Return three PDFs; associated PNGs are newly generated previews only."""
    frozen = (Path(artifact_root) / "figure 2").resolve()
    destination = Path(out_dir).resolve()
    if (destination == frozen or destination.is_relative_to(frozen)
            or frozen.is_relative_to(destination)):
        raise ValueError("Do not overwrite or write into the supplied Figure 2 bundle")
    out_dir = prepare_output(artifact_root, out_dir)
    outputs = []
    rc = dict(matplotlib.rcParamsDefault)
    rc["backend"] = "Agg"
    with matplotlib.rc_context(rc=rc):
        for panel in PANELS:
            fig = make_figure(panel)
            pdf = out_dir / f"{panel.stem}.pdf"
            timestamp = datetime.strptime(panel.original_pdf_timestamp,
                                          "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            try:
                fig.savefig(out_dir / f"{panel.stem}.png", dpi=240, bbox_inches="tight")
                fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": timestamp})
            finally:
                plt.close(fig)
            outputs.append(pdf)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    for output in render(args.artifact_root, args.out_dir):
        print(output)
