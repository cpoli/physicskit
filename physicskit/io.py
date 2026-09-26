"""Save and load :class:`~physicskit.results.Result` objects.

Two formats are supported, chosen by file extension:

* ``.npz`` (NumPy's zipped archive), always available. Loading never
  unpickles anything (``allow_pickle=False``), so a file from an
  untrusted source cannot execute code.
* ``.h5`` / ``.hdf5``, when :mod:`h5py` is installed
  (``pip install "physicskit[hdf5]"``). Arrays become datasets that
  other HDF5 tools can read directly.

Both store the same thing: the ``times``, ``states`` and extra arrays,
plus a JSON header with the metadata, units, unit system and a format
version. The metadata must therefore be JSON-compatible. NumPy scalars are
converted to Python numbers, and tuples come back as lists.

Examples
--------
>>> import numpy as np, tempfile, pathlib
>>> from physicskit.results import Result
>>> from physicskit.io import save, load
>>> r = Result(np.linspace(0, 1, 5), np.eye(5), metadata={"method": "rk4"})
>>> with tempfile.TemporaryDirectory() as d:
...     back = load(save(r, pathlib.Path(d) / "run.npz"))
>>> back.metadata["method"], np.array_equal(back.states, r.states)
('rk4', True)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from physicskit.results import Result
from physicskit.units import UnitSystem

__all__ = ["FORMAT_VERSION", "save", "load"]

#: Version of the on-disk layout written by :func:`save`. :func:`load`
#: refuses files written by a newer version.
FORMAT_VERSION = 1

_FORMAT_NAME = "physicskit.Result"
_HEADER_KEY = "__physicskit_result__"
_ARRAY_PREFIX = "array__"
_NPZ_SUFFIXES = (".npz",)
_HDF5_SUFFIXES = (".h5", ".hdf5")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        raise TypeError("metadata may not contain arrays; put them in Result.arrays instead")
    raise TypeError(f"metadata value of type {type(obj).__name__} is not JSON-serializable")


def _header(result: Result) -> str:
    return json.dumps(
        {
            "format": _FORMAT_NAME,
            "version": FORMAT_VERSION,
            "has_times": result.times is not None,
            "arrays": list(result.arrays),
            "metadata": result.metadata,
            "units": result.units,
            "unit_system": None if result.unit_system is None else result.unit_system.to_dict(),
        },
        default=_json_default,
    )


def _parse_header(text: str, path: Path) -> dict[str, Any]:
    header: dict[str, Any] = json.loads(text)
    if header.get("format") != _FORMAT_NAME:
        raise ValueError(f"{path} is not a physicskit Result file")
    if int(header.get("version", 0)) > FORMAT_VERSION:
        raise ValueError(f"{path} was written by a newer physicskit (format version {header['version']} > {FORMAT_VERSION}); upgrade physicskit to read it")
    return header


def _check_storable(result: Result) -> None:
    for name in result.names:
        if result[name].dtype.kind not in "biufc":
            raise TypeError(f"array {name!r} has dtype {result[name].dtype}; only boolean and numeric arrays can be saved")


def _from_parts(header: dict[str, Any], arrays: dict[str, NDArray[Any]]) -> Result:
    system = header.get("unit_system")
    return Result(
        times=arrays.pop("times") if header["has_times"] else None,
        states=arrays.pop("states"),
        metadata=header.get("metadata", {}),
        units=header.get("units", {}),
        unit_system=None if system is None else UnitSystem.from_dict(system),
        arrays={name: arrays[name] for name in header["arrays"]},
    )


def _format_for(path: Path, fmt: str | None) -> str:
    if fmt is not None:
        if fmt not in ("npz", "hdf5"):
            raise ValueError(f"unknown format {fmt!r}; use 'npz' or 'hdf5'")
        return fmt
    suffix = path.suffix.lower()
    if suffix in _NPZ_SUFFIXES:
        return "npz"
    if suffix in _HDF5_SUFFIXES:
        return "hdf5"
    raise ValueError(f"cannot infer format from extension {suffix!r}; use .npz, .h5 or .hdf5, or pass format=")


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError('h5py is required for HDF5 files; install it with `pip install "physicskit[hdf5]"`') from exc
    return h5py


def save(result: Result, path: str | os.PathLike[str], *, format: str | None = None) -> Path:
    """Write ``result`` to ``path``, overwriting any existing file.

    Parameters
    ----------
    result : Result
        The result to save.
    path : str or path-like
        Destination. The extension (``.npz``, ``.h5``, ``.hdf5``) picks
        the format unless ``format`` is given.
    format : {"npz", "hdf5"}, optional
        Explicit format, overriding the extension.

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    TypeError
        If the metadata is not JSON-compatible or an array is not
        numeric or boolean.
    ImportError
        For HDF5 when :mod:`h5py` is not installed.
    """
    path = Path(path)
    fmt = _format_for(path, format)
    _check_storable(result)
    header = _header(result)
    if fmt == "npz":
        payload: dict[str, Any] = {_HEADER_KEY: np.array(header)}
        for name in result.names:
            key = name if name in ("times", "states") else _ARRAY_PREFIX + name
            payload[key] = result[name]
        # np.savez appends ".npz" to names without it; write via a handle to keep ``path`` exact.
        with open(path, "wb") as fh:
            np.savez_compressed(fh, **payload)
    else:
        h5py = _require_h5py()
        with h5py.File(path, "w") as f:
            f.attrs[_HEADER_KEY] = header
            for name in ("times", "states"):
                if name in result.names:
                    f.create_dataset(name, data=result[name])
            group = f.create_group("arrays")
            for name, arr in result.arrays.items():
                group.create_dataset(name, data=arr)
    return path


def load(path: str | os.PathLike[str], *, format: str | None = None) -> Result:
    """Read a :class:`~physicskit.results.Result` written by :func:`save`.

    Parameters
    ----------
    path : str or path-like
        File to read.
    format : {"npz", "hdf5"}, optional
        Explicit format, overriding the extension.

    Returns
    -------
    Result

    Raises
    ------
    ValueError
        If the file is not a physicskit result, or was written by a newer
        format version.
    ImportError
        For HDF5 when :mod:`h5py` is not installed.
    """
    path = Path(path)
    fmt = _format_for(path, format)
    if fmt == "npz":
        with np.load(path, allow_pickle=False) as data:
            if _HEADER_KEY not in data.files:
                raise ValueError(f"{path} is not a physicskit Result file")
            header = _parse_header(str(data[_HEADER_KEY]), path)
            arrays = {name: data[_ARRAY_PREFIX + name] for name in header["arrays"]}
            arrays["states"] = data["states"]
            if header["has_times"]:
                arrays["times"] = data["times"]
    else:
        h5py = _require_h5py()
        with h5py.File(path, "r") as f:
            raw = f.attrs.get(_HEADER_KEY)
            if raw is None:
                raise ValueError(f"{path} is not a physicskit Result file")
            header = _parse_header(raw.decode() if isinstance(raw, bytes) else str(raw), path)
            arrays = {name: np.asarray(f["arrays"][name]) for name in header["arrays"]}
            arrays["states"] = np.asarray(f["states"])
            if header["has_times"]:
                arrays["times"] = np.asarray(f["times"])
    return _from_parts(header, arrays)
