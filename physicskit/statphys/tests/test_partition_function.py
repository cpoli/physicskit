import numpy as np
import pytest

from physicskit.statphys.utils.partition_function import ising_partition_polynomial, yang_lee_zeros


@pytest.mark.parametrize("N", [4, 8, 12])
@pytest.mark.parametrize("periodic", [True, False])
def test_yang_lee_zeros_lie_on_unit_circle_for_ferromagnet(N, periodic):
    g = ising_partition_polynomial(N, beta=0.4, J=1.0, periodic=periodic)
    zeros = yang_lee_zeros(g)
    assert zeros.shape == (N,)
    assert np.max(np.abs(np.abs(zeros) - 1.0)) < 1e-8


def test_partition_polynomial_coefficients_are_positive():
    g = ising_partition_polynomial(6, beta=0.5, J=1.0, periodic=True)
    assert g.shape == (7,)
    assert np.all(g > 0.0)


def test_partition_polynomial_symmetric_for_periodic_chain():
    # A periodic ferromagnetic chain is symmetric under global spin flip,
    # so g_k (k up-spins) equals g_(N-k) (k down-spins).
    g = ising_partition_polynomial(8, beta=0.3, J=1.0, periodic=True)
    assert np.allclose(g, g[::-1])
