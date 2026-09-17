"""Coverage for physicskit.relativity.utils.constants's simple geometrized-unit
conversion functions (previously untested)."""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.relativity.utils.constants import (
    C_SI,
    geometrized_to_meters,
    geometrized_to_seconds,
    geometrized_to_solar_masses,
    seconds_to_geometrized,
    solar_masses_to_geometrized,
)


def test_geometrized_to_meters_is_the_identity():
    assert geometrized_to_meters(5.0) == pytest.approx(5.0)


def test_geometrized_to_seconds_divides_by_c():
    assert geometrized_to_seconds(C_SI) == pytest.approx(1.0)


def test_seconds_to_geometrized_is_the_inverse_of_geometrized_to_seconds():
    t_s = 2.5
    assert geometrized_to_seconds(seconds_to_geometrized(t_s)) == pytest.approx(t_s)
    assert np.asarray(seconds_to_geometrized(t_s)).shape == ()


def test_geometrized_to_solar_masses_is_the_inverse_of_solar_masses_to_geometrized():
    assert geometrized_to_solar_masses(solar_masses_to_geometrized(30.0)) == pytest.approx(30.0, rel=1e-6)
