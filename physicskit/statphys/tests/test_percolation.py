import numpy as np
import pytest

from physicskit.statphys.chapters.percolation import Percolation2D, hoshen_kopelman


def test_hoshen_kopelman_simple_shape():
    grid = np.array(
        [
            [True, True, False],
            [False, True, False],
            [False, False, True],
        ]
    )
    labels = hoshen_kopelman(grid)
    assert labels[0, 0] == labels[0, 1] == labels[1, 1]
    assert labels[0, 0] != 0
    assert labels[2, 2] != 0
    assert labels[2, 2] != labels[0, 0]
    assert labels[0, 2] == 0 and labels[1, 0] == 0 and labels[2, 0] == 0 and labels[2, 1] == 0


def test_fully_occupied_lattice_is_one_cluster():
    grid = np.ones((10, 10), dtype=bool)
    labels = hoshen_kopelman(grid)
    assert len(np.unique(labels)) == 1


def test_empty_lattice_has_no_clusters():
    grid = np.zeros((10, 10), dtype=bool)
    labels = hoshen_kopelman(grid)
    assert np.all(labels == 0)


def test_low_probability_rarely_spans():
    perc = Percolation2D(L=40, p=0.1, mode="site", seed=0)
    assert not perc.spans()


def test_full_occupation_always_spans():
    perc = Percolation2D(L=20, p=1.0, mode="site", seed=1)
    assert perc.spans()


def test_spanning_probability_increases_with_p():
    perc = Percolation2D(L=25, mode="site", seed=2)
    _p_values, P_span = perc.spanning_probability([0.3, perc.p_c, 0.85], n_trials=30)
    assert P_span[0] <= P_span[1] <= P_span[2]
    assert P_span[2] > P_span[0]


def test_bond_percolation_threshold_is_half():
    perc = Percolation2D(L=10, mode="bond", seed=3)
    assert perc.p_c == pytest.approx(0.5)


def test_largest_cluster_size_grows_with_p():
    perc = Percolation2D(L=30, p=0.2, mode="site", seed=4)
    small = perc.largest_cluster_size()
    perc.generate(p=0.9)
    large = perc.largest_cluster_size()
    assert large > small


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        Percolation2D(L=10, mode="triangle")


def test_cluster_size_distribution_empty_for_no_occupation():
    perc = Percolation2D(L=15, p=0.0, mode="site", seed=5)
    assert perc.cluster_size_distribution().size == 0


def test_cluster_size_distribution_single_cluster_for_full_occupation():
    perc = Percolation2D(L=15, p=1.0, mode="site", seed=6)
    sizes = perc.cluster_size_distribution()
    assert sizes.size == 1
    assert sizes[0] == 15 * 15


def test_cluster_size_distribution_matches_largest_cluster_size():
    perc = Percolation2D(L=30, p=0.55, mode="site", seed=7)
    sizes = perc.cluster_size_distribution()
    assert sizes.max() == perc.largest_cluster_size()


def test_cluster_size_distribution_sums_to_occupied_sites():
    perc = Percolation2D(L=25, p=0.6, mode="site", seed=8)
    sizes = perc.cluster_size_distribution()
    n_occupied = int(np.sum(perc._real_labels() != 0))
    assert sizes.sum() == n_occupied
