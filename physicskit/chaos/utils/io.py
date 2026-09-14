"""Save and load system configurations and array results (trajectories,
basin grids, Poincare sections, ...) to disk.

A system's configuration is recovered generically from its ``__init__``
signature (see :func:`physicskit.chaos.core.base_system.get_init_params`), the same
introspection that already powers every system's ``__repr__`` -- so
:func:`save_system_config` / :func:`load_system_config` work for any
concrete system in physicskit.chaos without a per-class serializer. Array results
are saved as plain ``.npz`` archives via :func:`save_arrays` /
:func:`load_arrays`.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from physicskit.chaos.core.base_system import get_init_params


def save_system_config(obj: object, path: str | Path) -> None:
    """Save a system's class and constructor parameters as JSON.

    Parameters
    ----------
    obj : object
        A physicskit.chaos system instance (or any object whose ``__init__``
        parameters are stored under same-named attributes).
    path : str or pathlib.Path
        Destination ``.json`` file.
    """
    params = {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in get_init_params(obj).items()}
    config = {
        "module": type(obj).__module__,
        "class": type(obj).__qualname__,
        "params": params,
    }
    Path(path).write_text(json.dumps(config, indent=2))


def load_system_config(path: str | Path) -> Any:
    """Reconstruct a system instance saved by :func:`save_system_config`.

    Parameters
    ----------
    path : str or pathlib.Path
        Source ``.json`` file.

    Returns
    -------
    object
        A new instance of the saved class, constructed from its saved
        parameters.
    """
    config = json.loads(Path(path).read_text())
    module = importlib.import_module(config["module"])
    cls = getattr(module, config["class"])
    return cls(**config["params"])


def save_arrays(path: str | Path, **arrays: NDArray[Any]) -> None:
    """Save named arrays (trajectories, grids, Poincare sections, ...) to a ``.npz`` archive.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination ``.npz`` file.
    **arrays
        Arrays to save, keyed by name (e.g. ``save_arrays("run.npz", t=t,
        states=states)``).
    """
    np.savez(path, **arrays)  # type: ignore[arg-type]  # numpy stubs mistype **kwds here


def load_arrays(path: str | Path) -> dict[str, NDArray[Any]]:
    """Load a ``.npz`` archive saved by :func:`save_arrays`.

    Parameters
    ----------
    path : str or pathlib.Path
        Source ``.npz`` file.

    Returns
    -------
    dict
        Mapping of array name to array, matching the keyword arguments
        `save_arrays` was called with.
    """
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
