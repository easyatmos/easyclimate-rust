from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

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


def _as_xarray_dataset(result: dict[str, Any]) -> Any:
    """Convert a solver result to a labelled xarray dataset."""
    coords = {
        "x": result["x"],
        "y": result["y"],
        "z": result["z"],
        "y_staggered": result["y"][:-1],
    }

    time = result["time"]
    if time.size:
        coords["time"] = time

    # Solver output: (x, y) -> xarray convention: (y, x)
    data_vars = {
        name: (("y", "x"), result[name].T)
        for name in ("q", "eta", "u", "v")
    }

    # (x, y_staggered) -> (y_staggered, x)
    data_vars["v_staggered"] = (
        ("y_staggered", "x"),
        result["v_staggered"].T,
    )

    if time.size:
        # (time, x, y) -> (time, y, x)
        for name in ("eta_history", "u_history", "v_history"):
            data_vars[name] = (
                ("time", "y", "x"),
                result[name].transpose(0, 2, 1),
            )

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={key: value for key, value in result.items() if key not in _ARRAY_KEYS},
    )


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
    kernel: str = "reyon",
    as_xarray: bool = True,
) -> dict[str, Any] | Any:
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
    kernel:
        Use ``"serial"`` for one CPU core or ``"reyon"``(default) to evaluate
        independent zonal modes in parallel. The ``"reyon"`` kernel uses at
        most six CPU cores.
    as_xarray:
        Return an :class:`xarray.Dataset` with labelled ``x``, ``y``, ``z``, and
        (when history is saved) ``time`` dimensions. The default ``False``
        preserves the original dictionary of Numpy arrays.

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
        kernel=kernel,
    )
    result = _as_numpy_dict(raw)
    return _as_xarray_dataset(result) if as_xarray else result
