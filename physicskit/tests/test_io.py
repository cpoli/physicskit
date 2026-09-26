"""Tests for physicskit.io: save/load round trips."""

import importlib.util
import json
import sys

import numpy as np
import pytest

import physicskit.constants as const
from physicskit import io
from physicskit.results import Result
from physicskit.units import SI, geometrized_units

needs_h5py = pytest.mark.skipif(importlib.util.find_spec("h5py") is None, reason="h5py not installed")
FORMATS = ["run.npz", pytest.param("run.h5", marks=needs_h5py), pytest.param("run.hdf5", marks=needs_h5py)]


def _full_result():
    rng = np.random.default_rng(0)
    return Result(
        times=np.linspace(0.0, 1.0, 7),
        states=rng.normal(size=(7, 3)),
        metadata={"method": "dopri5", "rtol": np.float64(1e-9), "n": np.int64(7), "params": (1.0, 2.0), "nested": {"ok": True, "none": None}},
        units={"times": "time", "states": "length", "psi": "dimensionless"},
        unit_system=geometrized_units(mass_kg=const.SOLAR_MASS_KG),
        arrays={
            "psi": rng.normal(size=(7, 4)) + 1j * rng.normal(size=(7, 4)),
            "mask": np.array([True, False, True]),
            "counts": np.arange(5, dtype=np.int32),
            "scalar": np.float64(3.5),
        },
    )


def _assert_same(a, b):
    assert a.names == b.names
    for name in a.names:
        np.testing.assert_array_equal(a[name], b[name])
        assert a[name].dtype == b[name].dtype
        assert a[name].shape == b[name].shape
    assert a.units == b.units
    assert a.unit_system == b.unit_system


@pytest.mark.parametrize("fname", FORMATS)
def test_round_trip_full(tmp_path, fname):
    r = _full_result()
    path = io.save(r, tmp_path / fname)
    assert path == tmp_path / fname and path.exists()
    back = io.load(path)
    _assert_same(r, back)
    # JSON normalizes numpy scalars to Python numbers and tuples to lists
    assert back.metadata == {"method": "dopri5", "rtol": 1e-9, "n": 7, "params": [1.0, 2.0], "nested": {"ok": True, "none": None}}
    # Units survive well enough to convert: 1 M of time is G M / c^3
    assert back.to_si().times[-1] == pytest.approx(const.G * const.SOLAR_MASS_KG / const.C**3, rel=1e-12)


@pytest.mark.parametrize("fname", FORMATS)
def test_round_trip_minimal_without_times(tmp_path, fname):
    r = Result(None, np.arange(4.0))
    back = io.load(io.save(r, tmp_path / fname))
    _assert_same(r, back)
    assert back.times is None and back.unit_system is None and back.metadata == {}


@pytest.mark.parametrize("fname", FORMATS)
def test_round_trip_si_result(tmp_path, fname):
    r = Result([0.0, 1.0], [[1.0], [2.0]], units={"states": "energy"}, unit_system=SI)
    back = io.load(io.save(r, str(tmp_path / fname)))
    assert back.unit_system == SI


def test_round_trip_classical_simulation(tmp_path):
    from physicskit.classical.systems.hamiltonian import HenonHeilesSystem
    from physicskit.results import from_simulation_result

    sim = HenonHeilesSystem(q0=[0.1, 0.0], p0=[0.0, 0.2]).integrate((0.0, 1.0), dt=0.01)
    r = from_simulation_result(sim)
    back = io.load(io.save(r, tmp_path / "hh.npz"))
    _assert_same(r, back)
    assert back.metadata["method"] == sim.method


def test_explicit_format_overrides_extension(tmp_path):
    path = io.save(Result(None, [1.0]), tmp_path / "data.bin", format="npz")
    assert path.name == "data.bin"  # np.savez would otherwise append ".npz"
    assert io.load(path, format="npz").states[0] == 1.0


def test_unknown_extension_or_format(tmp_path):
    with pytest.raises(ValueError, match="cannot infer format"):
        io.save(Result(None, [1.0]), tmp_path / "data.txt")
    with pytest.raises(ValueError, match="unknown format"):
        io.save(Result(None, [1.0]), tmp_path / "data.npz", format="csv")


def test_rejects_non_json_metadata(tmp_path):
    with pytest.raises(TypeError, match="Result.arrays"):
        io.save(Result(None, [1.0], metadata={"a": np.zeros(2)}), tmp_path / "x.npz")
    with pytest.raises(TypeError, match="set"):
        io.save(Result(None, [1.0], metadata={"a": {1}}), tmp_path / "x.npz")


def test_rejects_object_arrays(tmp_path):
    with pytest.raises(TypeError, match="numeric"):
        io.save(Result(None, np.array(["a", "b"])), tmp_path / "x.npz")


def test_load_rejects_foreign_npz(tmp_path):
    path = tmp_path / "plain.npz"
    np.savez(path, x=np.zeros(2))
    with pytest.raises(ValueError, match="not a physicskit Result"):
        io.load(path)


def _rewrite_npz_header(path, **changes):
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    header = json.loads(str(payload["__physicskit_result__"]))
    header.update(changes)
    payload["__physicskit_result__"] = np.array(json.dumps(header))
    with open(path, "wb") as fh:
        np.savez(fh, **payload)


def test_load_rejects_newer_version_and_wrong_format(tmp_path):
    path = io.save(Result(None, [1.0]), tmp_path / "x.npz")
    _rewrite_npz_header(path, version=io.FORMAT_VERSION + 1)
    with pytest.raises(ValueError, match="newer physicskit"):
        io.load(path)
    _rewrite_npz_header(path, version=1, format="something.else")
    with pytest.raises(ValueError, match="not a physicskit Result"):
        io.load(path)


@needs_h5py
def test_load_rejects_foreign_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "plain.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.zeros(2))
    with pytest.raises(ValueError, match="not a physicskit Result"):
        io.load(path)


@needs_h5py
def test_hdf5_datasets_are_readable_without_physicskit(tmp_path):
    h5py = pytest.importorskip("h5py")
    r = _full_result()
    path = io.save(r, tmp_path / "run.h5")
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["times"][()], r.times)
        np.testing.assert_array_equal(f["arrays/psi"][()], r["psi"])


def test_hdf5_missing_h5py_raises_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match=r"physicskit\[hdf5\]"):
        io.save(Result(None, [1.0]), tmp_path / "x.h5")
    with pytest.raises(ImportError, match=r"physicskit\[hdf5\]"):
        io.load(tmp_path / "x.h5")
