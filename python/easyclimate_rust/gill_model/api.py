from __future__ import annotations

from typing import Any

import numpy as np

from .. import _easyclimate_rust

CASES = {
    "gill_sym": "Gill: symmetric heating",
    "gill_asym": "Gill: asymmetric heating",
    "kelvin_wave": "Kelvin wave",
    "gauss_pert": "Gaussian-shape initial perturbation",
    "matsuno_heating": "Matsuno: forced stationary motion",
    "gill_sym_gaussian_x": "Gill: symmetric heating with Gaussian shape in zonal direction",
    "gill_sym_asym": "Gill: symmetric heating + asymmetric heating",
    "gill_delta_y1": "Gill: delta heating at y = 1",
}

_ARRAY_KEYS = {
    "x",
    "y",
    "z",
    "q",
    "eta",
    "u",
    "v",
    "v_staggered",
    "time",
    "eta_history",
    "u_history",
    "v_history",
}


def _as_numpy_dict(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(raw)
    for key in _ARRAY_KEYS:
        if key in out:
            out[key] = np.asarray(out[key], dtype=float)
    return out


def simulate(
    case: str = "gill_sym",
    *,
    t: float | None = None,
    nt: int | None = None,
    nx: int = 128,
    ny: int = 60,
    nz: int = 20,
    lx: float | None = None,
    ly: float | None = None,
    lz: float | None = None,
    damping: float | None = None,
    store_every: int = 0,
) -> dict[str, Any]:
    """Run the Gill-model solver.

    Parameters mirror the MATLAB script. By default, case-specific MATLAB
    overrides are preserved: ``kelvin_wave`` uses ``t=10, nt=100, damping=0``;
    ``gauss_pert`` uses ``damping=0``; and ``matsuno_heating`` uses
    ``damping=0.2``. Passing an explicit value overrides those defaults.

    Parameters
    ----------
    case:
        One of ``gill_sym``, ``gill_asym``, ``kelvin_wave``, ``gauss_pert``,
        ``matsuno_heating``, ``gill_sym_gaussian_x``, ``gill_sym_asym``,
        or ``gill_delta_y1``.
    store_every:
        Save eta/u/v snapshots every ``store_every`` steps. Use ``0`` for final
        state only.

    Returns
    -------
    dict
        Numpy arrays with MATLAB-like orientation: fields are stored as
        ``(nx, ny+1)`` except ``v_staggered``, which is ``(nx, ny)``.
    """

    raw = _easyclimate_rust.simulate(
        case=case,
        t=t,
        nt=nt,
        nx=nx,
        ny=ny,
        nz=nz,
        lx=lx,
        ly=ly,
        lz=lz,
        damping=damping,
        store_every=store_every,
    )
    return _as_numpy_dict(raw)
