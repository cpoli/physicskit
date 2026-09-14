import numpy as np
import pytest

from physicskit.statphys.chapters.renormalization import BlockSpinRG


def test_iterate_shrinks_grid_by_half_each_step():
    rg = BlockSpinRG(L=16, T=100.0, seed=0, n_equil_sweeps=5)
    grids = rg.iterate(n_steps=3)
    assert [g.shape[0] for g in grids] == [16, 8, 4, 2]


def test_default_n_steps_stops_at_2x2():
    rg = BlockSpinRG(L=16, T=100.0, seed=0, n_equil_sweeps=5)
    grids = rg.iterate()
    assert grids[-1].shape == (2, 2)


def test_high_temperature_flows_toward_disorder():
    rg = BlockSpinRG(L=32, T=1000.0, seed=1, n_equil_sweeps=5)
    grids = rg.iterate(n_steps=4)
    orders = [rg.order_parameter(g) for g in grids]
    # disordered starting point: order parameter should stay small throughout
    assert orders[0] < 0.3


def test_low_temperature_flows_toward_order():
    rg = BlockSpinRG(L=32, T=0.5, seed=2, n_equil_sweeps=200)
    grids = rg.iterate(n_steps=3)
    orders = [rg.order_parameter(g) for g in grids]
    assert orders[-1] > 0.8


def test_coarse_grain_step_requires_even_size():
    rg = BlockSpinRG(L=16, T=100.0, seed=0, n_equil_sweeps=1)
    with pytest.raises(ValueError):
        rg.coarse_grain_step(np.ones((5, 5), dtype=np.int64))


def test_majority_rule_unanimous_block():
    rg = BlockSpinRG(L=8, T=100.0, seed=3, n_equil_sweeps=1)
    block = np.ones((4, 4), dtype=np.int64)
    coarse = rg.coarse_grain_step(block)
    assert np.all(coarse == 1)
