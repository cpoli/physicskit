"""Tests for the chiral ensembles (chGOE, chGUE, chGSE) -- Altland-
Zirnbauer classes BDI, AIII, CII: block off-diagonal Hamiltonians with
an anticommuting chiral symmetry, the QCD-type Dirac operator random
matrix ensembles of Verbaarschot and Zahed.

Two independent layers of verification, mirroring the bar set by
``physicskit.rmt/ensembles/wishart.py`` for the underlying Laguerre machinery:
(1) the fast production sampler's structural contract (beta, shape,
+-pairing, exact zero-mode count, reproducibility) -- trivially true by
construction, but worth asserting as documentation; and (2) an
independent, direct check on the actual dense block matrix
H = [[0, W], [W^dagger, 0]] (Hermiticity, exact chiral-symmetry
anticommutation with Gamma = diag(I, -I), and that its eigenvalues are
exactly +-singular-values-of-W plus exact zeros) -- confirming the
"reuse the Laguerre machinery" shortcut actually represents the linear
algebra it claims to, not merely trusting the module docstring's
derivation.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt

CHIRAL = [
    ("chGOE", rmt.ensembles.chGOE, 1),
    ("chGUE", rmt.ensembles.chGUE, 2),
    ("chGSE", rmt.ensembles.chGSE, 4),
]


@pytest.mark.parametrize("name,cls,beta", CHIRAL)
def test_beta_is_correct(name, cls, beta):
    ens = cls(n=30, seed=0)
    assert ens.beta == beta


@pytest.mark.parametrize("name,cls,beta", CHIRAL)
def test_gamma_property(name, cls, beta):
    ens = cls(n=30, nu=10, seed=0)
    assert ens.gamma == pytest.approx(30.0 / 40.0)


@pytest.mark.parametrize("name,cls,beta", CHIRAL)
def test_produces_2n_plus_nu_eigenvalues(name, cls, beta):
    ens = cls(n=40, nu=5, seed=1)
    spectrum = cached_sample(ens, n_samples=5)
    assert spectrum.eigenvalues.shape == (5, 85)


@pytest.mark.parametrize("name,cls,beta", CHIRAL)
def test_eigenvalues_are_real(name, cls, beta):
    ens = cls(n=35, nu=3, seed=2)
    spectrum = cached_sample(ens, n_samples=5)
    assert not np.iscomplexobj(spectrum.eigenvalues)
    assert np.all(np.isfinite(spectrum.eigenvalues))


@pytest.mark.parametrize("name,cls,beta", CHIRAL)
def test_reproducibility(name, cls, beta):
    ens_a = cls(n=40, nu=4, seed=42)
    ens_b = cls(n=40, nu=4, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


@pytest.mark.parametrize("name,cls,beta", CHIRAL)
def test_spectrum_symmetric_about_zero(name, cls, beta):
    # Chiral symmetry forces exact +-lambda pairing, like BdG -- but via
    # an anticommuting (not commuting) operator; see module docstring.
    ens = cls(n=50, nu=6, seed=3)
    spectrum = cached_sample(ens, n_samples=10)
    for row in spectrum.eigenvalues:
        row_sorted = np.sort(row)
        pairing_error = np.abs(row_sorted + row_sorted[::-1]).max()
        assert pairing_error < 1e-8


@pytest.mark.parametrize("name,cls,beta", CHIRAL)
def test_exact_zero_modes_match_nu(name, cls, beta):
    ens = cls(n=45, nu=7, seed=4)
    spectrum = cached_sample(ens, n_samples=5)
    for row in spectrum.eigenvalues:
        n_zeros = np.sum(np.abs(row) < 1e-10)
        assert n_zeros == 7


@pytest.mark.parametrize("name,cls,beta", CHIRAL)
def test_nonzero_eigenvalues_are_nonnegative_singular_values(name, cls, beta):
    # +-sigma_k by construction: exactly half (rounding for nu>0's
    # zeros) of the nonzero eigenvalues are strictly positive, the rest
    # their exact negatives -- already covered by the pairing test, but
    # additionally check the magnitudes are genuinely nonzero (not
    # spuriously landing on/near zero beyond the nu exact zero modes).
    ens = cls(n=30, nu=0, seed=5)
    spectrum = cached_sample(ens, n_samples=5)
    nonzero = spectrum.eigenvalues[np.abs(spectrum.eigenvalues) > 1e-10]
    assert len(nonzero) == 5 * 60  # nu=0 -> no exact zeros
    assert np.all(np.abs(nonzero) > 1e-6)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.ChiralBetaEnsemble(n=10, beta=0, seed=0)
    with pytest.raises(ValueError):
        rmt.ensembles.chGUE(n=10, nu=-1, seed=0)


def test_continuum_beta_also_produces_valid_spectrum():
    ens = rmt.ensembles.ChiralBetaEnsemble(n=200, beta=1.7, nu=3, seed=6)
    spectrum = cached_sample(ens, n_samples=10)
    assert spectrum.eigenvalues.shape == (10, 403)
    for row in spectrum.eigenvalues:
        assert np.sum(np.abs(row) < 1e-10) == 3


# --- Direct, un-cached structural verification of the actual block
# matrix (independent of the fast Laguerre-reuse sampler above) ---


@pytest.mark.parametrize("name,cls,beta", CHIRAL)
def test_dense_construction_hermitian_and_chiral_symmetric(name, cls, beta):
    from physicskit.rmt.ensembles.chiral import _dense_chiral_hamiltonian

    rng = np.random.default_rng(7)
    n, nu = 10, 3
    h = _dense_chiral_hamiltonian(n, nu, rng, beta)
    assert np.abs(h - h.conj().T).max() < 1e-10

    n_dim = n if beta != 4 else 2 * n
    m_dim = (n + nu) if beta != 4 else 2 * (n + nu)
    gamma5 = np.diag(np.concatenate([np.ones(n_dim), -np.ones(m_dim)]))
    anticommutator_residual = gamma5 @ h @ gamma5 + h
    assert np.abs(anticommutator_residual).max() < 1e-10


@pytest.mark.parametrize("name,cls,beta", CHIRAL)
def test_dense_construction_eigenvalues_are_singular_values_plus_zeros(name, cls, beta):
    from physicskit.rmt.ensembles.chiral import _dense_chiral_hamiltonian, _sample_block

    rng = np.random.default_rng(8)
    n, nu = 10, 3
    rng_for_w = np.random.default_rng(8)
    w = _sample_block(n, n + nu, rng_for_w, beta)
    singular_values = np.linalg.svd(w, compute_uv=False)

    h = _dense_chiral_hamiltonian(n, nu, rng, beta)
    eigs = np.sort(np.linalg.eigvalsh(h))

    n_dim, m_dim = w.shape
    n_zero_modes = m_dim - n_dim  # rank deficiency of W (>= 0 for nu >= 0)
    positive_eigs = eigs[eigs > 1e-8]
    zero_eigs = eigs[np.abs(eigs) <= 1e-8]
    negative_eigs = eigs[eigs < -1e-8]

    assert len(zero_eigs) == n_zero_modes
    np.testing.assert_allclose(
        np.sort(positive_eigs),
        np.sort(singular_values[singular_values > 1e-8]),
        atol=1e-8,
    )
    np.testing.assert_allclose(np.sort(-negative_eigs), np.sort(positive_eigs), atol=1e-8)


@pytest.mark.parametrize(
    "name,cls,beta",
    [("chGOE", rmt.ensembles.chGOE, 1), ("chGUE", rmt.ensembles.chGUE, 2)],
)
def test_fast_sampler_matches_dense_construction_statistically(name, cls, beta):
    # Two-sample KS test between the fast (Laguerre-reuse) sampler's
    # pooled nonzero |eigenvalue|s and the literal dense block matrix's
    # singular values -- the same cross-check standard
    # ``physicskit.rmt/ensembles/wishart.py`` already applies to the underlying
    # bidiagonal trick, done here explicitly for the chiral wrapper
    # rather than just trusted transitively. beta=4 is excluded because
    # the dense construction's explicit quaternion embedding doubles its
    # dimension relative to the fast sampler's generalized-beta n (see
    # module docstring), so the two are not directly comparable sample-
    # for-sample at matched n.
    #
    # Scale note: the fast sampler reuses ``sample_laguerre_beta_eigenvalues``,
    # which returns Marchenko-Pastur-*normalized* eigenvalues -- despite
    # dividing the raw bidiagonal-model product by (beta*m) internally
    # (see ``physicskit.rmt/utils/tridiagonal.py``), it matches dense X^dagger @ X / m
    # for unit-variance X at every beta (confirmed directly here, and
    # already the standard this repo verifies against -- see
    # ``physicskit.rmt/ensembles/wishart.py``'s module docstring), i.e. the
    # beta-dependence is absorbed into the bidiagonal model's chi
    # degrees of freedom, not left over as an extra factor in the
    # normalization. So the dense construction's raw singular values
    # (built from literal unit-variance entries) need only be divided by
    # sqrt(m), not sqrt(beta*m), to land on the same footing.
    from scipy.stats import ks_2samp

    from physicskit.rmt.ensembles.chiral import _dense_chiral_hamiltonian

    n, nu = 12, 2
    m = n + nu
    fast_ens = cls(n=n, nu=nu, seed=9)
    fast_spectrum = fast_ens.sample(n_samples=400)
    fast_nonzero = np.abs(fast_spectrum.eigenvalues[np.abs(fast_spectrum.eigenvalues) > 1e-10])

    rng = np.random.default_rng(10)
    dense_singular_values = []
    for _ in range(400):
        h = _dense_chiral_hamiltonian(n, nu, rng, beta)
        eigs = np.linalg.eigvalsh(h)
        dense_singular_values.append(np.abs(eigs[np.abs(eigs) > 1e-8]))
    dense_nonzero = np.concatenate(dense_singular_values) / np.sqrt(m)

    ks = ks_2samp(fast_nonzero, dense_nonzero)
    assert ks.statistic < 0.05, f"{name}: fast sampler diverges from dense construction, {ks}"


# --- QCD Dirac operator convention (D = i*H, purely imaginary spectrum) ---

QCD_DIRAC = [
    ("QCDDiracGOE", rmt.ensembles.QCDDiracGOE, rmt.ensembles.chGOE, 1),
    ("QCDDiracGUE", rmt.ensembles.QCDDiracGUE, rmt.ensembles.chGUE, 2),
    ("QCDDiracGSE", rmt.ensembles.QCDDiracGSE, rmt.ensembles.chGSE, 4),
]


@pytest.mark.parametrize("name,cls,ch_cls,beta", QCD_DIRAC)
def test_qcd_dirac_beta_is_correct(name, cls, ch_cls, beta):
    ens = cls(n=30, seed=0)
    assert ens.beta == beta


@pytest.mark.parametrize("name,cls,ch_cls,beta", QCD_DIRAC)
def test_qcd_dirac_eigenvalues_are_purely_imaginary(name, cls, ch_cls, beta):
    ens = cls(n=40, nu=3, seed=1)
    spectrum = cached_sample(ens, n_samples=5)
    assert np.iscomplexobj(spectrum.eigenvalues)
    assert np.abs(spectrum.eigenvalues.real).max() < 1e-10


@pytest.mark.parametrize("name,cls,ch_cls,beta", QCD_DIRAC)
def test_qcd_dirac_is_exactly_i_times_chiral_ensemble(name, cls, ch_cls, beta):
    # Thin-wrapper contract from the module docstring: D = i*H exactly,
    # for the SAME underlying random draw (matched seed) -- not just
    # matching in distribution.
    dirac_ens = cls(n=25, nu=2, seed=7)
    chiral_ens = ch_cls(n=25, nu=2, seed=7)
    dirac_spectrum = dirac_ens.sample(n_samples=4)
    chiral_spectrum = chiral_ens.sample(n_samples=4)
    np.testing.assert_allclose(dirac_spectrum.eigenvalues, 1j * chiral_spectrum.eigenvalues)


@pytest.mark.parametrize("name,cls,ch_cls,beta", QCD_DIRAC)
def test_qcd_dirac_exact_zero_modes_match_nu(name, cls, ch_cls, beta):
    ens = cls(n=45, nu=6, seed=2)
    spectrum = cached_sample(ens, n_samples=5)
    for row in spectrum.eigenvalues:
        assert np.sum(np.abs(row) < 1e-10) == 6


@pytest.mark.parametrize("name,cls,ch_cls,beta", QCD_DIRAC)
def test_qcd_dirac_reproducibility(name, cls, ch_cls, beta):
    ens_a = cls(n=30, nu=3, seed=42)
    ens_b = cls(n=30, nu=3, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)
