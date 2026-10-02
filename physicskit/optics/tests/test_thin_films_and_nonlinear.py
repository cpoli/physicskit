import numpy as np
import pytest

from physicskit.optics.nonlinear import (
    extraordinary_index,
    shg_coupled_amplitudes,
    shg_phase_matched_efficiency,
    shg_undepleted_power,
    type_i_phase_matching_angle,
)
from physicskit.optics.thin_films import (
    airy_transmission,
    bloch_wavenumber,
    fabry_perot_free_spectral_range,
    finesse,
    multilayer_response,
    quarter_wave_band_gap,
    quarter_wave_stack,
)


def test_bare_interface_reproduces_fresnel_at_oblique_incidence():
    n1, n2, th = 1.0, 1.5, 0.6
    th2 = np.arcsin(n1 * np.sin(th) / n2)
    rs = (n1 * np.cos(th) - n2 * np.cos(th2)) / (n1 * np.cos(th) + n2 * np.cos(th2))
    rp = (n2 * np.cos(th) - n1 * np.cos(th2)) / (n2 * np.cos(th) + n1 * np.cos(th2))
    assert multilayer_response([], [], 500.0, n1, n2, th, "s")["R"] == pytest.approx(rs**2)
    assert multilayer_response([], [], 500.0, n1, n2, th, "p")["R"] == pytest.approx(rp**2)
    brewster = np.arctan(n2 / n1)
    assert multilayer_response([], [], 500.0, n1, n2, brewster, "p")["R"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("pol", ["s", "p"])
def test_lossless_stack_conserves_energy(pol):
    n, d = quarter_wave_stack(2.3, 1.38, 4, 600.0)
    lam = np.linspace(400, 900, 50)
    out = multilayer_response(n, d, lam, 1.0, 1.52, angle=0.4, polarization=pol)
    np.testing.assert_allclose(out["R"] + out["T"], 1.0, atol=1e-12)


def test_half_wave_layer_is_absentee():
    n = 2.0
    bare = multilayer_response([], [], 500.0, 1.0, 1.5)["R"]
    assert multilayer_response([n], [500.0 / (2 * n)], 500.0, 1.0, 1.5)["R"] == pytest.approx(bare)


def test_quarter_wave_mirror_peak_reflectance_closed_form():
    nH, nL, ns, N = 2.3, 1.38, 1.52, 6
    n, d = quarter_wave_stack(nH, nL, N, 600.0)
    # Macleod, Thin-Film Optical Filters, eq. (5.4) for (HL)^N on substrate ns
    Y = (nH / nL) ** (2 * N) * ns
    assert multilayer_response(n, d, 600.0, 1.0, ns)["R"] == pytest.approx(((1 - Y) / (1 + Y)) ** 2)


def test_absorbing_layer_takes_energy():
    out = multilayer_response([1.5 + 0.5j], [100.0], 500.0, 1.0, 1.5)
    assert out["R"] + out["T"] < 1.0


def test_cavity_resonance_and_finesse():
    R = 0.9
    delta = np.linspace(-np.pi, np.pi, 200001)
    T = airy_transmission(delta, R)
    fwhm = np.ptp(delta[T >= 0.5])
    assert 2 * np.pi / fwhm == pytest.approx(finesse(R), rel=1e-3)
    assert fabry_perot_free_spectral_range(0.5, 1.0, wavelength=1e-3) == pytest.approx(1e-6)


def test_dielectric_cavity_transmits_fully_on_resonance():
    n, d = quarter_wave_stack(2.3, 1.38, 5, 600.0, cavity=True)
    out = multilayer_response(n, d, np.array([600.0, 610.0]), 1.0, 1.0)
    assert out["T"][0] == pytest.approx(1.0)
    assert out["T"][1] < 0.1


def test_bloch_gap_width_matches_closed_form():
    nH, nL, lam0 = 2.3, 1.38, 600.0
    w = np.linspace(0.5, 1.5, 200001)  # omega / omega_0
    KL = bloch_wavenumber([nH, nL], [lam0 / (4 * nH), lam0 / (4 * nL)], lam0 / w)
    gap = w[KL.imag > 0]
    assert gap.max() - gap.min() == pytest.approx(quarter_wave_band_gap(nH, nL), rel=1e-3)
    assert KL.real[np.argmin(np.abs(w - 1.0))] == pytest.approx(np.pi)


def test_shg_undepleted_low_conversion_limit():
    z = np.linspace(0, 3, 31)
    out = shg_coupled_amplitudes(z, delta_k=2.0, kappa=1e-3)
    np.testing.assert_allclose(np.abs(out["A2"]) ** 2, shg_undepleted_power(z, 2.0, 1e-3), rtol=1e-4, atol=1e-14)


def test_shg_phase_matched_depletion_and_power_conservation():
    z = np.linspace(0, 3, 31)
    out = shg_coupled_amplitudes(z, kappa=1.0)
    np.testing.assert_allclose(np.abs(out["A2"]) ** 2, shg_phase_matched_efficiency(z), atol=1e-8)
    np.testing.assert_allclose(np.abs(out["A1"]) ** 2 + np.abs(out["A2"]) ** 2, 1.0, atol=1e-8)


def test_quasi_phase_matching_grows_at_two_over_pi():
    dk = 2 * np.pi
    z = np.linspace(0, 10, 11)
    qpm = shg_coupled_amplitudes(z, dk, kappa=1e-3, qpm_period=2 * np.pi / dk)
    pm = shg_coupled_amplitudes(z, 0.0, kappa=1e-3)
    assert abs(qpm["A2"][-1]) / abs(pm["A2"][-1]) == pytest.approx(2 / np.pi, rel=1e-3)


def test_type_i_angle_matches_index_condition():
    no_w, no_2w, ne_2w = 1.4942, 1.5129, 1.4709
    theta = type_i_phase_matching_angle(no_w, no_2w, ne_2w)
    assert extraordinary_index(theta, no_2w, ne_2w) == pytest.approx(no_w)
    with pytest.raises(ValueError):
        type_i_phase_matching_angle(1.6, 1.5129, 1.4709)
