import numpy as np
import pytest

from physicskit.particle.collider import (
    charged_track_points,
    cluster_into_jets,
    flatten_shower,
    parton_shower,
    shower_leaves,
    simple_shower,
)
from physicskit.particle.kinematics import FourVector


def test_simple_shower_conserves_energy_momentum_at_every_vertex():
    rng = np.random.default_rng(0)
    root = simple_shower(100.0, E_threshold=5.0, rng=rng)
    for node in flatten_shower(root):
        if not node.is_leaf:
            c1, c2 = node.children
            total = c1.four_vector + c2.four_vector
            assert total.E == pytest.approx(node.four_vector.E, abs=1e-6)
            assert total.p_vec == pytest.approx(node.four_vector.p_vec, abs=1e-6)


def test_simple_shower_leaves_reconstruct_total_four_momentum():
    rng = np.random.default_rng(1)
    root = simple_shower(80.0, E_threshold=4.0, rng=rng)
    leaves = shower_leaves(root)
    assert len(leaves) >= 2
    total_leaf = None
    for leaf in leaves:
        total_leaf = leaf.four_vector if total_leaf is None else total_leaf + leaf.four_vector
    assert total_leaf.E == pytest.approx(root.four_vector.E, abs=1e-6)
    assert total_leaf.p_vec == pytest.approx(root.four_vector.p_vec, abs=1e-6)


def test_simple_shower_all_leaves_below_threshold_or_at_generation_cap():
    rng = np.random.default_rng(2)
    root = simple_shower(60.0, E_threshold=6.0, max_generations=6, rng=rng)
    for leaf in shower_leaves(root):
        assert leaf.four_vector.E < 6.0 or leaf.generation >= 6 or leaf.four_vector.mass < 1e-8


def test_parton_shower_conserves_four_momentum():
    rng = np.random.default_rng(3)
    root = parton_shower(100.0, E_threshold=5.0, rng=rng)
    leaves = shower_leaves(root)
    total_leaf = None
    for leaf in leaves:
        total_leaf = leaf.four_vector if total_leaf is None else total_leaf + leaf.four_vector
    assert total_leaf.E == pytest.approx(root.four_vector.E, abs=1e-6)
    assert total_leaf.p_vec == pytest.approx(root.four_vector.p_vec, abs=1e-6)


def test_parton_shower_flavor_conservation_rules():
    rng = np.random.default_rng(4)
    root = parton_shower(100.0, flavor0="q", E_threshold=5.0, rng=rng)
    for node in flatten_shower(root):
        if not node.is_leaf:
            f1, f2 = node.children[0].flavor, node.children[1].flavor
            if node.flavor in ("q", "qbar"):
                assert node.flavor in (f1, f2)
                assert "g" in (f1, f2)
            elif node.flavor == "g":
                assert {f1, f2} in ({"g"}, {"q", "qbar"})


def test_cluster_into_jets_splits_back_to_back_particles_correctly():
    rng = np.random.default_rng(5)
    root = parton_shower(100.0, E_threshold=8.0, rng=rng)
    leaves = shower_leaves(root)
    jets = cluster_into_jets(leaves, n_jets=2)
    assert len(jets) == 2
    assert sum(len(j) for j in jets) == len(leaves)


def test_charged_track_points_neutral_particle_is_a_straight_line():
    p = FourVector(10.0, 3.0, 4.0, 5.0)
    pts = charged_track_points(p, charge=0.0, B=1.0, path_length=2.5, n_points=10)
    assert pts.shape == (10, 2)
    direction = np.array([3.0, 4.0]) / 5.0
    assert pts[-1] == pytest.approx(direction * 2.5)


def test_charged_track_points_charged_particle_curves_with_correct_radius():
    p = FourVector(10.0, 3.0, 0.0, 5.0)  # pT = 3
    B = 2.0
    charge = 1.0
    r_expected = 3.0 / (charge * B)
    pts = charged_track_points(p, charge=charge, B=B, path_length=r_expected * 0.5, n_points=50)

    # reconstruct the radius of curvature from three points on the arc
    # (the circumradius of any three points on a circle equals its radius).
    def circumradius(a, b, c):
        ab, bc, ca = np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)
        s = (ab + bc + ca) / 2
        area = max(s * (s - ab) * (s - bc) * (s - ca), 0.0) ** 0.5
        return (ab * bc * ca) / (4 * area) if area > 0 else np.inf

    r_reconstructed = circumradius(pts[0], pts[len(pts) // 2], pts[-1])
    assert r_reconstructed == pytest.approx(r_expected, rel=1e-2)


def test_charged_track_points_initial_direction_matches_momentum():
    p = FourVector(10.0, 3.0, 4.0, 0.0)
    for charge in (1.0, -1.0):
        pts = charged_track_points(p, charge=charge, B=1.0, path_length=0.1, n_points=1000)
        step = pts[1] - pts[0]
        step_dir = step / np.linalg.norm(step)
        expected_dir = np.array([3.0, 4.0]) / 5.0
        assert step_dir == pytest.approx(expected_dir, abs=1e-2)
