"""Rust/PyO3 + Python implementation of the MATLAB Gill model."""

from .api import CASES, simulate
from .plotting import animate, plot_cross_sections, plot_snapshot

__all__ = [
    "CASES",
    "simulate",
    "plot_snapshot",
    "plot_cross_sections",
    "animate",
]
