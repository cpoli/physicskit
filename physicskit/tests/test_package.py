"""Tests for the top-level physicskit namespace."""

import pkgutil

import physicskit as pk


def test_every_subpackage_is_exported():
    subpackages = {m.name for m in pkgutil.iter_modules(pk.__path__) if m.ispkg and m.name != "tests"}
    assert subpackages <= set(pk.__all__)
    for name in subpackages:
        assert getattr(pk, name).__name__ == f"physicskit.{name}"


def test_shared_modules_are_exported():
    for name in ["constants", "units", "results", "io"]:
        assert name in pk.__all__
        assert getattr(pk, name).__name__ == f"physicskit.{name}"
