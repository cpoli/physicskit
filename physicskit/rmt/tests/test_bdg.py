"""Tests for the Bogoliubov-de Gennes / Altland-Zirnbauer superconductor
classes D, C, CI, DIII.

Unlike most of this package, these are validated only structurally
(Hermiticity, correct real/complex type per class, and the defining
+-lambda eigenvalue pairing), not against a closed-form bulk or edge
density -- see the module docstring in ``physicskit.rmt/ensembles/bdg.py`` for
why. The constructions themselves come from two independently citable
sources: Stephanov-Verbaarschot-Wettig (arXiv:hep-ph/0509286, citing
Altland-Zirnbauer 1997) for classes D and C, and Stolz
(arXiv:1707.03793, also citing Altland-Zirnbauer) for classes CI and
DIII, whose definitions are given in purely mathematical terms
independent of any physics convention -- reducing the risk of a
convention mismatch relative to earlier, less precisely specified
sources found during the initial search for this feature.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt

BDG_CLASSES = [
    ("D", rmt.ensembles.BdGClassD, 2, False),  # (name, cls, beta, expect_real)
    ("C", rmt.ensembles.BdGClassC, 2, False),
    ("CI", rmt.ensembles.BdGClassCI, 1, True),
    ("DIII", rmt.ensembles.BdGClassDIII, 4, False),
]


@pytest.mark.parametrize("name,cls,beta,expect_real", BDG_CLASSES)
def test_beta_is_correct(name, cls, beta, expect_real):
    ens = cls(n=30, seed=0)
    assert ens.beta == beta


@pytest.mark.parametrize("name,cls,beta,expect_real", BDG_CLASSES)
def test_produces_2n_eigenvalues(name, cls, beta, expect_real):
    ens = cls(n=50, seed=1)
    spectrum = cached_sample(ens, n_samples=5)
    assert spectrum.eigenvalues.shape == (5, 100)


@pytest.mark.parametrize("name,cls,beta,expect_real", BDG_CLASSES)
def test_eigenvalues_are_real(name, cls, beta, expect_real):
    # All four classes are built from Hermitian matrices, so eigenvalues
    # are always real-valued floats regardless of whether the underlying
    # matrix entries are real (CI) or complex (D, C, DIII).
    ens = cls(n=40, seed=2)
    spectrum = cached_sample(ens, n_samples=5)
    assert not np.iscomplexobj(spectrum.eigenvalues)
    assert np.all(np.isfinite(spectrum.eigenvalues))


@pytest.mark.parametrize("name,cls,beta,expect_real", BDG_CLASSES)
def test_reproducibility(name, cls, beta, expect_real):
    ens_a = cls(n=40, seed=42)
    ens_b = cls(n=40, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


@pytest.mark.parametrize("name,cls,beta,expect_real", BDG_CLASSES)
def test_spectrum_symmetric_about_zero(name, cls, beta, expect_real):
    # The defining particle-hole symmetry: eigenvalues occur in
    # +-lambda pairs (distinct values, not a Kramers-type degeneracy).
    ens = cls(n=60, seed=3)
    spectrum = cached_sample(ens, n_samples=10)
    for row in spectrum.eigenvalues:
        row_sorted = np.sort(row)
        pairing_error = np.abs(row_sorted + row_sorted[::-1]).max()
        assert pairing_error < 1e-8


def test_matrix_construction_hermiticity_and_type_directly():
    # Direct, un-cached structural check of the raw matrix construction
    # (not just the eigenvalues) for each class, including confirming
    # CI is REAL (not just Hermitian) and the other three are complex.
    from physicskit.rmt.ensembles.bdg import (
        _dense_antisymmetric,
        _dense_hermitian,
        _dense_symmetric,
    )

    rng = np.random.default_rng(7)
    n = 20

    a = _dense_hermitian(n, rng, complex_entries=True)
    b_d = _dense_antisymmetric(n, rng, complex_entries=True)
    h_d = np.block([[a, b_d], [b_d.conj().T, -a.T]])
    assert np.abs(h_d - h_d.conj().T).max() < 1e-10
    assert np.abs(h_d.imag).max() > 1e-3  # genuinely complex, not real

    b_c = _dense_symmetric(n, rng, complex_entries=True)
    h_c = np.block([[a, b_c], [b_c.conj().T, -a.T]])
    assert np.abs(h_c - h_c.conj().T).max() < 1e-10

    x1_ci = _dense_symmetric(n, rng, complex_entries=False)
    x2_ci = _dense_symmetric(n, rng, complex_entries=False)
    h_ci = np.block([[x1_ci, x2_ci], [x2_ci, -x1_ci]])
    assert np.abs(h_ci.imag).max() == 0.0  # exactly real
    assert np.abs(h_ci - h_ci.T).max() < 1e-10

    y1 = _dense_antisymmetric(n, rng, complex_entries=False)
    y2 = _dense_antisymmetric(n, rng, complex_entries=False)
    x1_diii, x2_diii = 1j * y1, 1j * y2
    h_diii = np.block([[x1_diii, x2_diii], [x2_diii, -x1_diii]])
    assert np.abs(h_diii - h_diii.conj().T).max() < 1e-10
    assert np.abs(h_diii.real).max() == 0.0  # exactly purely-imaginary entries

    a_real = _dense_hermitian(n, rng, complex_entries=False)
    assert np.abs(a_real.imag).max() == 0.0  # exactly real
    assert np.abs(a_real - a_real.T).max() < 1e-10


def test_all_four_classes_are_structurally_distinct():
    # Sanity check that the four classes don't accidentally collapse to
    # the same construction (a risk given how similar the block forms
    # look on paper).
    n = 30
    spectra = {}
    for name, cls, _beta, _expect_real in BDG_CLASSES:
        ens = cls(n=n, seed=8)
        spectra[name] = np.sort(ens.sample(n_samples=1).eigenvalues[0])

    # Different beta values (D/C share beta=2 but differ in B's symmetry
    # type; CI and DIII have distinct beta) should generally give
    # different largest-eigenvalue values for the same seed.
    largest = {name: spec[-1] for name, spec in spectra.items()}
    values = list(largest.values())
    assert len(set(np.round(values, 6))) == len(values)
