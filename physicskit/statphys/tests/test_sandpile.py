import numpy as np

from physicskit.statphys.chapters.sandpile import BTWSandpile


def test_initial_heights_are_zero():
    pile = BTWSandpile(L=10, seed=0)
    assert np.all(pile.heights == 0)


def test_heights_stay_below_threshold_after_relaxation():
    pile = BTWSandpile(L=16, seed=1)
    pile.run(n_grains=500)
    assert np.all(pile.heights < pile.threshold)


def test_avalanche_sizes_are_non_negative():
    pile = BTWSandpile(L=16, seed=2)
    sizes = pile.run(n_grains=200)
    assert np.all(sizes >= 0)


def test_add_grain_at_specific_site_below_threshold():
    pile = BTWSandpile(L=10, threshold=4, seed=3)
    size = pile.add_grain(site=(5, 5))
    assert size == 0
    assert pile.heights[5, 5] == 1


def test_forced_toppling_conserves_grains_in_bulk():
    pile = BTWSandpile(L=20, threshold=4, seed=4)
    pile.heights[10, 10] = 3
    grains_before = pile.total_grains()
    size = pile.add_grain(site=(10, 10))
    assert size == 1
    # a single interior toppling conserves grains (4 out, 4 in); only the
    # one added grain changes the total.
    assert pile.total_grains() == grains_before + 1


def test_large_run_produces_nontrivial_avalanches():
    pile = BTWSandpile(L=24, seed=5)
    sizes = pile.run(n_grains=2000, warmup=1000)
    assert sizes.max() > 1
