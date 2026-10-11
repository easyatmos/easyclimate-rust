from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def _array(result: Any, name: str) -> np.ndarray:
    """Return a result variable as a Numpy array from a dict or Dataset."""

    return np.asarray(result[name])


def _optional_array(result: Any, name: str) -> np.ndarray | None:
    if isinstance(result, dict):
        value = result.get(name)
    else:
        value = result[name] if name in result else None
    return None if value is None else np.asarray(value)


def _metadata(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return result.attrs.get(name, default)


def _default_levels(result: Any) -> np.ndarray:
    case = str(_metadata(result, "short_name", ""))
    if case == "GaussPert":
        return np.arange(-1.0, 1.0001, 0.05)
    return np.r_[np.arange(-10.0, 0.0, 0.2), np.arange(0.2, 10.0001, 0.2)]




def _quiver_like_matlab(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    nq: int = 5,
    scale: float = 1.0,
    color: str = "#0072BD",
):
    """Draw vectors with MATLAB ``quiver(..., 0)``-like geometry.

    MATLAB's final scale argument ``0`` disables autoscaling.  In Matplotlib,
    the closest equivalent is ``angles='xy', scale_units='xy', scale=1``.
    Increasing ``scale`` makes arrows shorter; decreasing it makes arrows longer.
    """

    return ax.quiver(
        x[::nq],
        y[::nq],
        u.T[::nq, ::nq],
        v.T[::nq, ::nq],
        angles="xy",
        scale_units="xy",
        scale=scale,
        pivot="tail",
        color=color,
        width=0.004,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=4.0,
        minlength=0.0,
    )


def plot_snapshot(
    result: Any,
    *,
    ax: plt.Axes | None = None,
    levels: np.ndarray | None = None,
    nq: int = 5,
    quiver_scale: float = 1.5,
    quiver_color: str = "k",
    title: str | None = None,
) -> plt.Axes:
    """Plot a Numpy-result dictionary or xarray Dataset final state."""

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    x = _array(result, "x")
    y = _array(result, "y")
    eta = _array(result, "eta")
    u = _array(result, "u")
    v = _array(result, "v")
    levels = _default_levels(result) if levels is None else levels

    ax.contour(x, y, eta.T, levels=levels)
    _quiver_like_matlab(
        ax, x, y, u, v, nq=nq, scale=quiver_scale, color=quiver_color
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(float(y[0]), float(y[-1]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is None:
        title = f"{_metadata(result, 'case_label', '')}\nt = {_metadata(result, 't'):.2f}"
    ax.set_title(title)
    return ax


def plot_cross_sections(
    result: Any, *, fig: plt.Figure | None = None) -> plt.Figure:
    """Plot cross sections from a Numpy-result dictionary or xarray Dataset."""

    if fig is None:
        fig = plt.figure(figsize=(8, 8))

    x = _array(result, "x")
    y = _array(result, "y")
    z = _array(result, "z")
    u = _array(result, "u")
    v = _array(result, "v")
    lz = float(_metadata(result, "lz"))

    fig.clear()
    axes = fig.subplots(3, 1)

    psiuw = -lz / np.pi * u.mean(axis=1)[:, None] * np.sin(np.pi / lz * z[None, :])
    axes[0].contour(x, z, psiuw.T, levels=np.arange(-0.4, 0.4001, 0.04))
    axes[0].set_title(f"{_metadata(result, 'case_label', '')}\nt = {_metadata(result, 't'):.2f}\nu-w")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")

    psivw = -lz / np.pi * v.mean(axis=0)[:, None] * np.sin(np.pi / lz * z[None, :])
    axes[1].contour(y, z, psivw.T, levels=np.arange(-0.1, 0.1001, 0.004))
    axes[1].set_title("v-w")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("z")

    uxm = u.mean(axis=0)[:, None] * np.cos(np.pi / lz * z[None, :])
    axes[2].contour(y, z, uxm.T, levels=np.arange(-0.4, 0.4001, 0.04))
    axes[2].set_title("u")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("z")

    for ax in axes:
        ax.set_yticks([0, z[-1]], ["0", "z"])
    fig.tight_layout()
    return fig


def animate(
    result: Any,
    *,
    nq: int = 5,
    levels: np.ndarray | None = None,
    quiver_scale: float = 1.0,
    quiver_color: str = "#0072BD",
    interval: int = 60,
    save: str | Path | None = None,
) -> animation.FuncAnimation:
    """Animate a Numpy-result dictionary or xarray Dataset history."""

    eta_hist = _optional_array(result, "eta_history")
    u_hist = _optional_array(result, "u_history")
    v_hist = _optional_array(result, "v_history")
    times = _optional_array(result, "time")
    if eta_hist is None or len(eta_hist) == 0:
        raise ValueError("No saved history. Re-run simulate(..., store_every=2), for example.")

    x = _array(result, "x")
    y = _array(result, "y")
    levels = _default_levels(result) if levels is None else levels

    fig, ax = plt.subplots(figsize=(8, 4.5))

    def draw(frame: int):
        ax.clear()
        ax.contour(x, y, eta_hist[frame].T, levels=levels)
        _quiver_like_matlab(
            ax,
            x,
            y,
            u_hist[frame],
            v_hist[frame],
            nq=nq,
            scale=quiver_scale,
            color=quiver_color,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(float(y[0]), float(y[-1]))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        t = float(times[frame]) if times is not None and len(times) else 0.0
        ax.set_title(f"{_metadata(result, 'case_label', '')}\nt = {t:.2f}")
        return ax.collections

    anim = animation.FuncAnimation(fig, draw, frames=len(eta_hist), interval=interval, blit=False)
    if save is not None:
        anim.save(str(save))
    return anim
