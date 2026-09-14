import plotly.graph_objects as go

from physicskit.statphys.chapters.ising_lattice import Ising2D, XYModel2D
from physicskit.statphys.chapters.molecular_dynamics import LennardJonesGas
from physicskit.statphys.visualizers.interactive import (
    interactive_particle_snapshot,
    interactive_temperature_sweep,
    interactive_vortex_field,
)


def test_interactive_temperature_sweep_builds_figure():
    model = Ising2D(L=8, seed=0)
    result = model.run_temperature_sweep([2.0, model.T_C, 2.5], n_equil=10, n_measure=10)
    fig = interactive_temperature_sweep(result, T_c=model.T_C)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_interactive_vortex_field_builds_figure():
    model = XYModel2D(L=10, seed=1)
    model.sweep(beta=1.0, n_sweeps=5)
    fig = interactive_vortex_field(model.theta)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_interactive_particle_snapshot_builds_figure():
    gas = LennardJonesGas(n_particles=25, box_size=10.0, seed=2)
    gas.step(n_steps=10)
    fig = interactive_particle_snapshot(gas)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
