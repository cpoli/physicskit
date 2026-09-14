import numpy as np
import pytest

from physicskit.relativity.chapters.gw_merger import BinaryMerger


def test_chirp_mass_equal_mass_case():
    merger = BinaryMerger(m1=30.0, m2=30.0, distance=1000.0)
    # for equal masses, chirp mass = m * 2^(-1/5)
    assert merger.chirp_mass == pytest.approx(30.0 * 2.0 ** (-1.0 / 5.0), rel=1e-10)


def test_symmetric_mass_ratio_bounds():
    equal = BinaryMerger(m1=20.0, m2=20.0, distance=1.0)
    unequal = BinaryMerger(m1=50.0, m2=5.0, distance=1.0)
    assert equal.symmetric_mass_ratio == pytest.approx(0.25)
    assert unequal.symmetric_mass_ratio < 0.25


def test_inspiral_frequency_increases_toward_merger():
    merger = BinaryMerger(m1=30.0, m2=25.0, distance=1000.0)
    t = np.linspace(-2.0, -0.01, 200)
    f = merger.inspiral_frequency(t, t_merger=0.0)
    assert np.all(np.diff(f) > 0.0)


def test_inspiral_phase_matches_frequency_derivative():
    merger = BinaryMerger(m1=30.0, m2=25.0, distance=1000.0)
    t = np.linspace(-2.0, -0.5, 1000)
    dt = t[1] - t[0]
    phase = merger.inspiral_phase(t, t_merger=0.0)
    f_from_phase = np.gradient(phase, dt) / (2.0 * np.pi)
    f_formula = merger.inspiral_frequency(t, t_merger=0.0)
    assert np.allclose(f_from_phase, f_formula, rtol=1e-3)


def test_inspiral_strain_shapes_and_finite():
    merger = BinaryMerger(m1=30.0, m2=25.0, distance=1000.0, inclination=0.5)
    t = np.linspace(-2.0, -0.1, 300)
    hp, hc = merger.inspiral_strain(t, t_merger=0.0)
    assert hp.shape == t.shape
    assert hc.shape == t.shape
    assert np.all(np.isfinite(hp))
    assert np.all(np.isfinite(hc))


def test_face_on_gives_constant_envelope_circular_polarization():
    # face-on (inclination=0): h_plus and h_cross combine into a constant
    # envelope (cos^2(phase) + sin^2(phase) = 1) equal to the raw amplitude.
    merger = BinaryMerger(m1=30.0, m2=25.0, distance=1000.0, inclination=0.0)
    t = np.array([-0.5])
    hp, hc = merger.inspiral_strain(t, t_merger=0.0)
    amp = (4.0 / merger.distance) * merger.chirp_mass ** (5.0 / 3.0) * (np.pi * merger.inspiral_frequency(t, 0.0)) ** (2.0 / 3.0)
    assert np.hypot(hp, hc)[0] == pytest.approx(amp[0], rel=1e-6)


def test_qnm_frequency_and_damping_positive():
    merger = BinaryMerger(m1=30.0, m2=25.0, distance=1000.0)
    f_qnm, tau_damp = merger.qnm_frequency_damping()
    assert f_qnm > 0.0
    assert tau_damp > 0.0


def test_ringdown_strain_decays():
    merger = BinaryMerger(m1=30.0, m2=25.0, distance=1000.0)
    t = np.linspace(0.0, 500.0, 500)
    hp, hc = merger.ringdown_strain(t, t_merger=0.0, amplitude=1.0)
    envelope = np.hypot(hp, hc)
    assert envelope[0] >= envelope[-1]
    assert np.all(np.isfinite(envelope))


def test_full_waveform_continuous_and_finite():
    merger = BinaryMerger(m1=30.0, m2=25.0, distance=1000.0)
    t = np.linspace(-1.0, 0.05, 500)
    hp, hc = merger.full_waveform(t, t_merger=0.0)
    assert np.all(np.isfinite(hp))
    assert np.all(np.isfinite(hc))
    assert hp.shape == t.shape


def test_remnant_mass_less_than_total_mass():
    merger = BinaryMerger(m1=30.0, m2=25.0, distance=1000.0)
    M_f, a_f = merger.remnant_estimate()
    assert 0.0 < M_f < merger.total_mass
    assert 0.0 <= a_f < M_f


def test_orbital_period_matches_keplers_third_law():
    merger = BinaryMerger(m1=1.4, m2=1.4, distance=1.0)
    a = 1.0e6
    P = merger.orbital_period(a)
    assert pytest.approx(2.0 * np.pi * np.sqrt(a**3 / merger.total_mass)) == P


def test_semi_major_axis_decay_rate_is_negative_and_grows_with_eccentricity():
    merger = BinaryMerger(m1=1.4, m2=1.4, distance=1.0)
    a = 1.0e6
    rate_circular = merger.semi_major_axis_decay_rate(a, eccentricity=0.0)
    rate_eccentric = merger.semi_major_axis_decay_rate(a, eccentricity=0.6)
    assert rate_circular < 0.0
    assert abs(rate_eccentric) > abs(rate_circular)


def test_hulse_taylor_pulsar_period_decay_matches_observation():
    # PSR B1913+16: the discovery that won Hulse and Taylor the 1993 Nobel
    # Prize. Predicted GR period decay should match the observed
    # -2.4056e-12 (dimensionless, s/s) to within a few percent.
    import physicskit.relativity.utils.constants as const

    m1 = const.solar_masses_to_geometrized(1.4398)
    m2 = const.solar_masses_to_geometrized(1.3886)
    psr = BinaryMerger(m1=m1, m2=m2, distance=1.0)
    period_s = 27906.98
    a = (psr.total_mass * (period_s * const.C_SI) ** 2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
    dPdt = psr.period_decay_rate(a, eccentricity=0.6171338)
    assert dPdt == pytest.approx(-2.4056e-12, rel=0.05)
