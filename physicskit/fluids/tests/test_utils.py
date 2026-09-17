import numpy as np
import pytest

from physicskit.fluids.exceptions import InvalidParameterError
from physicskit.fluids.utils.dimensionless import (
    froude_number,
    mach_number,
    reynolds_number,
    strouhal_number,
    weber_number,
)
from physicskit.fluids.utils.spectral_analysis import energy_spectrum, kolmogorov_reference_slope


def test_reynolds_number_formula():
    assert reynolds_number(velocity=2.0, length=0.1, nu=0.02) == pytest.approx(10.0)


def test_reynolds_number_rejects_nonpositive_viscosity():
    with pytest.raises(InvalidParameterError):
        reynolds_number(velocity=1.0, length=1.0, nu=0.0)


def test_reynolds_number_rejects_nonpositive_length():
    with pytest.raises(InvalidParameterError):
        reynolds_number(velocity=1.0, length=0.0, nu=1.0)


def test_froude_number_rejects_nonpositive_length_and_g():
    with pytest.raises(InvalidParameterError):
        froude_number(velocity=1.0, length=0.0, g=9.81)
    with pytest.raises(InvalidParameterError):
        froude_number(velocity=1.0, length=1.0, g=0.0)


def test_mach_number_rejects_nonpositive_speed_of_sound():
    with pytest.raises(InvalidParameterError):
        mach_number(velocity=100.0, speed_of_sound=0.0)


def test_strouhal_number_rejects_nonpositive_velocity():
    with pytest.raises(InvalidParameterError):
        strouhal_number(frequency=5.0, length=0.02, velocity=0.0)


def test_weber_number_rejects_nonpositive_surface_tension():
    with pytest.raises(InvalidParameterError):
        weber_number(rho=1000.0, velocity=1.0, length=0.001, surface_tension=0.0)


def test_froude_number_formula():
    U, L, g = 4.0, 2.0, 9.81
    assert froude_number(U, L, g) == pytest.approx(U / np.sqrt(g * L))


def test_mach_number_transonic_threshold():
    assert mach_number(velocity=343.0, speed_of_sound=343.0) == pytest.approx(1.0)
    assert mach_number(velocity=100.0, speed_of_sound=343.0) < 1.0


def test_strouhal_number_formula():
    assert strouhal_number(frequency=5.0, length=0.02, velocity=0.5) == pytest.approx(0.2)


def test_weber_number_formula():
    We = weber_number(rho=1000.0, velocity=2.0, length=0.002, surface_tension=0.072)
    assert We == pytest.approx(1000.0 * 4.0 * 0.002 / 0.072)


def test_energy_spectrum_is_isotropic_for_a_radially_symmetric_field():
    """A single-wavenumber flow deposits essentially all its energy in the
    shell matching |k|, regardless of the direction of the wavevector."""
    n, length = 64, 2 * np.pi
    x = np.linspace(0, length, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u, v = np.sin(3 * Y), np.sin(3 * X)  # |k| = 3
    k, E = energy_spectrum(u, v, length)
    assert np.argmax(E) == 3
    assert E[3] > 100 * np.sum(E[np.arange(len(E)) != 3])


def test_energy_spectrum_rejects_mismatched_shapes():
    with pytest.raises(InvalidParameterError):
        energy_spectrum(np.zeros((16, 16)), np.zeros((8, 8)), length=1.0)


def test_energy_spectrum_rejects_non_square_arrays():
    with pytest.raises(InvalidParameterError):
        energy_spectrum(np.zeros((8, 16)), np.zeros((8, 16)), length=1.0)


def test_kolmogorov_reference_slope_matches_power_law():
    k = np.array([1.0, 8.0])
    ref = kolmogorov_reference_slope(k, k0=1.0, E0=1.0)
    assert ref[1] == pytest.approx(8.0 ** (-5.0 / 3.0))


def test_kolmogorov_reference_slope_rejects_nonpositive_anchor():
    with pytest.raises(InvalidParameterError):
        kolmogorov_reference_slope(np.array([1.0]), k0=0.0, E0=1.0)
