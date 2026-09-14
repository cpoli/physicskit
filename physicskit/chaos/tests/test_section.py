import numpy as np

from physicskit.chaos.systems.continuous import RestrictedThreeBody
from physicskit.chaos.visualizers.section import plot_poincare_map, poincare_crossings


def test_poincare_crossings_only_keeps_the_requested_direction():
    """Every returned crossing must have the section coordinate increasing
    through `value` when direction=1.0 (and decreasing for direction=-1.0)."""
    system = RestrictedThreeBody()
    state0 = system.initial_state()

    up = poincare_crossings(system, state0, coord=1, value=0.0, direction=1.0, t_max=2.0 * 17.0652165601579625588917206249, dt=1e-4)
    down = poincare_crossings(system, state0, coord=1, value=0.0, direction=-1.0, t_max=2.0 * 17.0652165601579625588917206249, dt=1e-4)

    assert up.shape[0] > 0
    assert down.shape[0] > 0
    # Interpolated crossings should sit essentially exactly on the surface.
    np.testing.assert_allclose(up[:, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(down[:, 1], 0.0, atol=1e-6)


def test_poincare_crossings_returns_empty_array_for_a_surface_never_reached():
    system = RestrictedThreeBody()
    state0 = system.initial_state()
    crossings = poincare_crossings(system, state0, coord=1, value=100.0, t_max=1.0, dt=1e-3)
    assert crossings.shape == (0, system.dim)


def test_plot_poincare_map_draws_one_scatter_series_per_initial_state():
    system = RestrictedThreeBody()
    state0 = system.initial_state()
    state0_perturbed = state0.copy()
    state0_perturbed[3] *= 1.001

    _, ax = plot_poincare_map(
        system,
        [state0, state0_perturbed],
        coord=1,
        value=0.0,
        direction=1.0,
        t_max=2.0 * 17.0652165601579625588917206249,
        dt=1e-4,
        labels=["reference", "perturbed"],
    )
    assert len(ax.collections) == 2
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["reference", "perturbed"]
