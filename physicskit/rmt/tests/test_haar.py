"""Tests for the Haar-random matrix generators (physicskit.rmt.utils.haar) and
the HaarOrthogonalEnsemble.

``haar_unitary`` is already indirectly validated via CUE's passing
spacing statistics (see tests/test_circular.py); ``haar_orthogonal`` and
``haar_symplectic`` are checked directly here since they are new.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt
from physicskit.rmt.utils.haar import haar_orthogonal, haar_symplectic, haar_unitary, symplectic_form


def test_haar_orthogonal_is_orthogonal():
    rng = np.random.default_rng(0)
    o = haar_orthogonal(20, rng)
    assert np.abs(o.T @ o - np.eye(20)).max() < 1e-10


def test_haar_orthogonal_first_column_is_uniform_on_sphere():
    # Hallmark of Haar invariance: any fixed unit vector rotated by a
    # Haar-random O(n) is uniform on the sphere -- mean 0, second
    # moment 1/n per component.
    rng = np.random.default_rng(1)
    n = 5
    first_columns = np.array([haar_orthogonal(n, rng)[:, 0] for _ in range(200_000)])
    np.testing.assert_allclose(first_columns.mean(axis=0), 0.0, atol=0.01)
    np.testing.assert_allclose((first_columns**2).mean(axis=0), 1.0 / n, atol=0.01)


def test_haar_orthogonal_determinant_is_balanced():
    rng = np.random.default_rng(2)
    dets = [np.linalg.det(haar_orthogonal(6, rng)) for _ in range(4000)]
    dets = np.array(dets)
    np.testing.assert_allclose(np.abs(dets), 1.0, atol=1e-8)
    frac_positive = np.mean(dets > 0)
    assert frac_positive == pytest.approx(0.5, abs=0.05)


def test_haar_symplectic_is_unitary_and_self_dual():
    rng = np.random.default_rng(3)
    n_quaternionic = 6
    u = haar_symplectic(n_quaternionic, rng)
    n = 2 * n_quaternionic
    assert u.shape == (n, n)
    assert np.abs(u.conj().T @ u - np.eye(n)).max() < 1e-10

    z = symplectic_form(n)
    u_dual = (-z) @ u.T @ z
    assert np.abs(u_dual - u).max() < 1e-10


def test_haar_symplectic_eigenvalues_are_doubly_degenerate():
    rng = np.random.default_rng(4)
    u = haar_symplectic(5, rng)
    eigs = np.linalg.eigvals(u)
    order = np.lexsort((eigs.imag, eigs.real))
    sorted_eigs = eigs[order]
    pair_diffs = np.abs(sorted_eigs[0::2] - sorted_eigs[1::2])
    assert pair_diffs.max() < 1e-8


def test_haar_unitary_still_exactly_unitary():
    # Regression check: existing helper unaffected by the new additions.
    rng = np.random.default_rng(5)
    u = haar_unitary(10, rng)
    assert np.abs(u.conj().T @ u - np.eye(10)).max() < 1e-10


# --- HaarOrthogonalEnsemble ---


def test_haar_orthogonal_ensemble_beta_is_none():
    ens = rmt.ensembles.HaarOrthogonalEnsemble(n=20, seed=0)
    assert ens.beta is None


def test_haar_orthogonal_ensemble_eigenvalues_on_unit_circle():
    ens = rmt.ensembles.HaarOrthogonalEnsemble(n=30, seed=1)
    spectrum = cached_sample(ens, n_samples=10)
    radii = np.abs(spectrum.eigenvalues)
    np.testing.assert_allclose(radii, 1.0, atol=1e-8)


def test_haar_orthogonal_ensemble_reproducibility():
    ens_a = rmt.ensembles.HaarOrthogonalEnsemble(n=20, seed=42)
    ens_b = rmt.ensembles.HaarOrthogonalEnsemble(n=20, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


def test_haar_orthogonal_ensemble_has_real_eigenvalues_unlike_coe():
    # The core structural distinction from COE (module docstring):
    # a raw Haar-O(n) matrix genuinely has real (+-1) eigenvalues with
    # non-negligible probability; COE/CUE generically do not.
    ens = rmt.ensembles.HaarOrthogonalEnsemble(n=40, seed=2)
    spectrum = cached_sample(ens, n_samples=200)
    real_fraction_of_samples = np.mean(np.any(np.abs(spectrum.eigenvalues.imag) < 1e-8, axis=1))
    assert real_fraction_of_samples > 0.5

    # COE stores eigenvalue PHASES (theta in [0, 2*pi)), not e^{i*theta}
    # directly -- convert back to compare on the same "is this real"
    # footing as HaarOrthogonalEnsemble's complex output above.
    coe = rmt.ensembles.COE(n=40, seed=3)
    coe_spectrum = cached_sample(coe, n_samples=200)
    coe_eigs = np.exp(1j * coe_spectrum.eigenvalues)
    coe_real_fraction = np.mean(np.abs(coe_eigs.imag) < 1e-8)
    assert coe_real_fraction < 0.01
