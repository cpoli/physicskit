import numpy as np
import pytest

from physicskit.optics.wave import (
    angular_spectrum_propagate,
    circular_aperture,
    double_slit_aperture,
    field_grid,
    fraunhofer_diffraction,
    intensity,
    single_slit_aperture,
)


def test_field_grid_is_centered():
    X, Y = field_grid((4, 6), dx=0.5)
    assert X.shape == (4, 6)
    assert Y.shape == (4, 6)
    assert X[0, 3] == pytest.approx(0.0)
    assert Y[2, 0] == pytest.approx(0.0)


def test_circular_aperture_is_open_at_center_and_closed_at_corner():
    ap = circular_aperture((65, 65), dx=1.0, radius=10.0)
    assert ap[32, 32] == pytest.approx(1.0)
    assert ap[0, 0] == pytest.approx(0.0)
    assert set(np.unique(ap)) <= {0.0, 1.0}


def test_single_slit_aperture_width_and_full_height():
    shape = (10, 101)
    dx = 1.0
    width = 20.0
    ap = single_slit_aperture(shape, dx, width)
    # every row should be identical (open along full height)
    assert np.all(ap == ap[0])
    n_open = int(ap[0].sum())
    assert n_open == pytest.approx(width / dx, abs=1)


def test_double_slit_aperture_has_two_separated_openings():
    shape = (5, 201)
    dx = 1.0
    width, separation = 8.0, 60.0
    ap = double_slit_aperture(shape, dx, width, separation)
    row = ap[0]
    X, _ = field_grid(shape, dx)
    x = X[0]
    # sample the intensity mask around each expected slit center and the gap between them
    left_open = row[np.argmin(np.abs(x - (-separation / 2)))]
    right_open = row[np.argmin(np.abs(x - (separation / 2)))]
    middle_closed = row[np.argmin(np.abs(x - 0.0))]
    assert left_open == pytest.approx(1.0)
    assert right_open == pytest.approx(1.0)
    assert middle_closed == pytest.approx(0.0)


def test_intensity_is_squared_modulus():
    U = np.array([3.0 + 4.0j, 1.0 + 0.0j, 0.0 + 2.0j])
    assert np.allclose(intensity(U), [25.0, 1.0, 4.0])


def test_angular_spectrum_propagate_conserves_power_over_short_distance():
    # A circular aperture propagated a short distance, with fine enough
    # sampling (dx well below wavelength/2, so no aperture spatial
    # frequency content is evanescent) should conserve total power to
    # within a small grid/aliasing tolerance.
    wavelength = 0.5e-3  # e.g. millimeters, so this is 500 nm
    dx = 0.2e-3
    N = 256
    radius = 20 * dx
    z = 2.0

    aperture = circular_aperture((N, N), dx, radius)
    U_out = angular_spectrum_propagate(aperture, wavelength, z, dx)

    power_in = intensity(aperture).sum()
    power_out = intensity(U_out).sum()
    assert power_out == pytest.approx(power_in, rel=0.02)


def test_angular_spectrum_propagate_matches_input_grid_shape():
    ap = circular_aperture((32, 48), dx=1.0e-3, radius=5.0e-3)
    U = angular_spectrum_propagate(ap, wavelength=0.6e-3, z=1.0, dx=1.0e-3)
    assert U.shape == ap.shape
    assert np.iscomplexobj(U)


def test_fraunhofer_single_slit_first_minimum_matches_analytic_formula():
    # Standard single-slit diffraction: the first intensity zero occurs at
    # sin(theta) = wavelength / width. We locate the first minimum of the
    # Fraunhofer pattern's central row and convert its transverse position
    # x' at distance z to an angle via the small-angle relation
    # sin(theta) ~= x' / z (valid here since x' << z in the far field).
    wavelength = 0.5e-3  # mm
    width = 0.05  # mm
    dx = 0.002  # mm
    N = 512
    z = 500.0  # mm; far enough for the Fraunhofer approximation to hold

    aperture = single_slit_aperture((N, N), dx, width)
    U = fraunhofer_diffraction(aperture, wavelength, z, dx)
    row = intensity(U)[N // 2]

    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    x_prime = wavelength * z * fx

    # walk outward from the central peak until intensity starts rising again
    center = N // 2
    i = center
    while row[i + 1] < row[i]:
        i += 1
    x_min = x_prime[i]

    sin_theta_measured = x_min / z
    sin_theta_predicted = wavelength / width
    assert sin_theta_measured == pytest.approx(sin_theta_predicted, rel=0.05)
