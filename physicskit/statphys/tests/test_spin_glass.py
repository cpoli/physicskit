import numpy as np
import pytest

from physicskit.statphys.chapters.spin_glass import EdwardsAndersonSpinGlass2D, SherringtonKirkpatrick


def test_bonds_are_bimodal():
    model = EdwardsAndersonSpinGlass2D(L=10, J=1.0, seed=0)
    assert set(np.unique(model.J_right)) <= {-1.0, 1.0}
    assert set(np.unique(model.J_down)) <= {-1.0, 1.0}


def test_frustration_density_between_zero_and_one():
    model = EdwardsAndersonSpinGlass2D(L=20, seed=1)
    f = model.frustration_density()
    assert 0.0 <= f <= 1.0


def test_frustration_density_near_one_half_for_random_bonds():
    model = EdwardsAndersonSpinGlass2D(L=40, seed=2)
    assert model.frustration_density() == pytest.approx(0.5, abs=0.05)


def test_ferromagnet_has_zero_frustration():
    model = EdwardsAndersonSpinGlass2D(L=10, seed=3)
    model.J_right[:] = 1.0
    model.J_down[:] = 1.0
    assert model.frustration_density() == pytest.approx(0.0)


def test_sweep_changes_configuration():
    model = EdwardsAndersonSpinGlass2D(L=16, seed=4)
    before = model.spins.copy()
    model.sweep(beta=1.0, n_sweeps=5)
    assert not np.array_equal(before, model.spins)


def test_energy_is_finite():
    model = EdwardsAndersonSpinGlass2D(L=12, seed=5)
    model.sweep(beta=1.0, n_sweeps=10)
    assert np.isfinite(model.energy())


def test_ea_order_parameter_bounds():
    model = EdwardsAndersonSpinGlass2D(L=8, seed=6)
    q2 = model.edwards_anderson_order_parameter(beta=2.0, n_equil=20, n_measure=20)
    assert 0.0 <= q2 <= 1.0 + 1e-9


def test_sk_coupling_matrix_is_symmetric_with_zero_diagonal():
    model = SherringtonKirkpatrick(N=30, seed=0)
    assert np.allclose(model.J, model.J.T)
    assert np.all(np.diag(model.J) == 0.0)


def test_sk_sweep_changes_configuration():
    model = SherringtonKirkpatrick(N=40, seed=1)
    before = model.spins.copy()
    model.sweep(beta=1.0, n_sweeps=5)
    assert not np.array_equal(before, model.spins)


def test_sk_energy_is_finite():
    model = SherringtonKirkpatrick(N=30, seed=2)
    model.sweep(beta=1.0, n_sweeps=5)
    assert np.isfinite(model.energy())


def test_sk_overlap_distribution_shape_and_bounds():
    model = SherringtonKirkpatrick(N=20, seed=3)
    samples = model.overlap_distribution(beta=1.5, n_disorder=4, n_equil=10, n_measure=5)
    assert samples.shape == (20,)
    assert np.all(samples >= -1.0 - 1e-9) and np.all(samples <= 1.0 + 1e-9)
