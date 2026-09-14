import numpy as np
import pytest

from physicskit.optics.gaussian import (
    GaussianBeam,
    hermite_gaussian_mode,
    laguerre_gaussian_mode,
    m2_beam_waist,
    propagate_q,
    q_to_beam_params,
)


def _free_space(d):
    return np.array([[1.0, d], [0.0, 1.0]])


def _thin_lens(f):
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def test_propagate_q_free_space_shifts_real_part():
    q0 = 1j * 2.0
    M = _free_space(3.0)
    q1 = propagate_q(q0, M)
    assert q1 == pytest.approx(3.0 + 2.0j)


def test_propagate_q_thin_lens_leaves_flat_wavefront_flat():
    # A collimated (plane-wave, R=inf) beam is represented by q real & huge;
    # instead check a lens leaves an on-axis waist's q consistent with 1/q - 1/f rule.
    q0 = 1j * 5.0
    f = 10.0
    q1 = propagate_q(q0, _thin_lens(f))
    assert 1.0 / q1 == pytest.approx(1.0 / q0 - 1.0 / f)


def test_q_to_beam_params_at_waist_gives_infinite_radius_of_curvature():
    zR = 4.0
    wavelength = 1.0e-3
    w0 = np.sqrt(wavelength * zR / np.pi)
    w, R = q_to_beam_params(1j * zR, wavelength)
    assert R == np.inf
    assert w == pytest.approx(w0)


def test_symmetric_two_lens_imaging_system_reproduces_object_waist():
    # Symmetric imaging system built directly from the ABCD matrices given
    # in the house convention: free_space(f) -> lens(f) -> free_space(f) ->
    # lens(f) -> free_space(f), starting from a beam waist at the object
    # plane. This should reproduce the object-plane waist size at the image
    # plane (unit magnification, by symmetry).
    wavelength = 0.5e-3
    w0 = 0.1
    f = 50.0

    beam = GaussianBeam(wavelength=wavelength, w0=w0, z0=0.0)
    q0 = beam.q_parameter(0.0)

    M = _free_space(f) @ _thin_lens(f) @ _free_space(f) @ _thin_lens(f) @ _free_space(f)
    q_image = propagate_q(q0, M)
    w_image, _ = q_to_beam_params(q_image, wavelength)

    assert w_image == pytest.approx(w0, rel=1e-6)


def test_waist_at_z0_equals_w0_exactly():
    beam = GaussianBeam(wavelength=1.0e-3, w0=0.2, z0=1.5)
    assert beam.waist(beam.z0) == pytest.approx(0.2)
    assert beam.waist(beam.z0) == 0.2


def test_radius_of_curvature_is_infinite_at_the_waist():
    beam = GaussianBeam(wavelength=1.0e-3, w0=0.2, z0=1.5)
    assert beam.radius_of_curvature(beam.z0) == np.inf


def test_radius_of_curvature_far_from_waist_approaches_z_minus_z0():
    beam = GaussianBeam(wavelength=1.0e-3, w0=0.05, z0=0.0)
    z = 1.0e6  # z >> zR, so R(z) -> z - z0
    assert beam.radius_of_curvature(z) == pytest.approx(z, rel=1e-3)


def test_divergence_angle_matches_far_field_waist_slope():
    beam = GaussianBeam(wavelength=1.0e-3, w0=0.02, z0=0.0)
    z_far = 1000.0 * beam.rayleigh_range
    slope = beam.waist(z_far) / (z_far - beam.z0)
    assert slope == pytest.approx(beam.divergence_angle, rel=1e-3)


def test_gouy_phase_is_zero_at_waist_and_approaches_pi_over_2_far_away():
    beam = GaussianBeam(wavelength=1.0e-3, w0=0.05, z0=0.0)
    assert beam.gouy_phase(0.0) == pytest.approx(0.0)
    assert beam.gouy_phase(1.0e6) == pytest.approx(np.pi / 2.0, abs=1e-3)


def test_hermite_gaussian_mode_shape_and_fundamental_matches_gaussian_beam():
    beam = GaussianBeam(wavelength=1.0, w0=1.0, z0=0.0)
    x = np.linspace(-2, 2, 9)
    y = np.linspace(-2, 2, 9)
    X, Y = np.meshgrid(x, y)
    field = hermite_gaussian_mode(X, Y, 0.0, beam, 0, 0)
    assert field.shape == X.shape
    assert np.iscomplexobj(field)
    expected_intensity = np.exp(-2.0 * (X**2 + Y**2) / beam.waist(0.0) ** 2)
    assert np.allclose(np.abs(field) ** 2, expected_intensity)


def test_hermite_gaussian_mode_gouy_phase_scales_with_mode_order():
    # Use x=1 (rather than the axis) so H_0(sqrt2 x/w) and H_1(sqrt2 x/w)
    # are both strictly positive real numbers, isolating the Gouy phase
    # contribution (m+n+1)*zeta(z) from any sign flip of the real
    # Hermite-polynomial amplitude factor.
    beam = GaussianBeam(wavelength=1.0, w0=1.0, z0=0.0)
    z = 3.0 * beam.rayleigh_range
    u00 = hermite_gaussian_mode(1.0, 0.0, z, beam, 0, 0)
    u10 = hermite_gaussian_mode(1.0, 0.0, z, beam, 1, 0)
    phase_diff = np.angle(u10) - np.angle(u00)
    expected = beam.gouy_phase(z)  # (1+0+1)*zeta - (0+0+1)*zeta = zeta
    assert phase_diff == pytest.approx(expected, abs=1e-8)


def test_laguerre_gaussian_mode_shape_and_ring_structure():
    beam = GaussianBeam(wavelength=1.0, w0=1.0, z0=0.0)
    r = np.linspace(0, 3, 50)
    phi = np.zeros_like(r)
    field = laguerre_gaussian_mode(r, phi, 0.0, beam, l=1, p=0)
    assert field.shape == r.shape
    # LG_0^1 has a phase singularity (zero amplitude) on-axis
    assert abs(field[0]) == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(field)) > 0.0


def test_laguerre_gaussian_mode_orbital_phase_winds_by_2pi_l():
    beam = GaussianBeam(wavelength=1.0, w0=1.0, z0=0.0)
    r = np.full(9, 0.5)
    phi = np.linspace(0.0, 2 * np.pi, 9, endpoint=False)
    field = laguerre_gaussian_mode(r, phi, 0.0, beam, l=2, p=0)
    phase = np.unwrap(np.angle(field))
    total_winding = phase[-1] + (phase[-1] - phase[-2]) - phase[0]
    assert np.round(total_winding / (2 * np.pi)) == pytest.approx(2.0, abs=0.2)


def test_m2_beam_waist_reduces_to_ideal_at_M2_equals_1():
    wavelength = 1.0e-3
    w0 = 0.1
    z = np.linspace(-5, 5, 11)
    beam = GaussianBeam(wavelength=wavelength, w0=w0, z0=0.0)
    ideal = beam.waist(z)
    m2 = m2_beam_waist(z, wavelength, w0, M2=1.0)
    assert np.allclose(ideal, m2)


def test_m2_beam_waist_diverges_faster_for_larger_M2():
    wavelength = 1.0e-3
    w0 = 0.1
    z = 10.0
    w_ideal = m2_beam_waist(z, wavelength, w0, M2=1.0)
    w_poor = m2_beam_waist(z, wavelength, w0, M2=3.0)
    assert w_poor > w_ideal
    assert m2_beam_waist(0.0, wavelength, w0, M2=3.0) == pytest.approx(w0)
