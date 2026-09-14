"""Physics-correctness tests for physicskit.condensed."""

from __future__ import annotations

import numpy as np
import pytest

import physicskit as pk
from physicskit.condensed.anderson_localization import (
    anderson_chain_hamiltonian,
    inverse_participation_ratio,
    localization_length,
)
from physicskit.condensed.correlated import bdg_bcs_hamiltonian, hubbard_1d_exact_diagonalization, hubbard_spin_correlations
from physicskit.condensed.ginzburg_landau import (
    ginzburg_landau_parameter,
    gl_coherence_length,
    gl_equilibrium_order_parameter,
    gl_order_parameter_profile,
    gl_penetration_depth,
)
from physicskit.condensed.landau_levels import (
    filling_factor,
    landau_degeneracy,
    landau_level_energies,
)
from physicskit.condensed.laughlin import laughlin_metropolis_sweep, laughlin_pair_correlation, laughlin_radial_density
from physicskit.condensed.models import (
    bhz_hamiltonian,
    bhz_ribbon_hamiltonian,
    graphene_hamiltonian,
    graphene_lattice_hamiltonian,
    haldane_lattice_hamiltonian,
    haldane_model,
    harper_hofstadter_hamiltonian,
    kane_mele_hamiltonian,
    kitaev_chain_bdg_real_space,
    ssh_lattice_hamiltonian,
)
from physicskit.condensed.tight_binding import Lattice, build_finite_cluster, build_ribbon
from physicskit.condensed.topological_insulator_3d import (
    topological_insulator_3d_hamiltonian,
    topological_insulator_3d_slab_hamiltonian,
)
from physicskit.condensed.topology import compute_chern_number, z2_invariant, zak_phase
from physicskit.condensed.weyl import weyl_node_locations, weyl_semimetal_hamiltonian, weyl_semimetal_slab_hamiltonian

K_DIRAC = np.array([2 * np.pi / 3, 4 * np.pi / 3])


class TestGrapheneDiracCone:
    def test_gap_closes_at_dirac_point(self):
        eigs = np.linalg.eigvalsh(graphene_hamiltonian(*K_DIRAC))
        assert np.allclose(eigs, 0.0, atol=1e-10)

    @pytest.mark.parametrize("direction", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    def test_linear_dispersion_near_dirac_point(self, direction):
        direction = np.array(direction) / np.linalg.norm(direction)
        slopes = []
        for dq in (1e-3, 5e-4, 2.5e-4):
            eigs = np.linalg.eigvalsh(graphene_hamiltonian(*(K_DIRAC + dq * direction)))
            slopes.append(eigs[1] / dq)
        # |E(k)| grows linearly in |k - K|, so the extracted slope should be
        # essentially constant (not vanishing as expected for a quadratic band).
        assert np.allclose(slopes, slopes[0], rtol=1e-3)
        assert slopes[0] > 0.5


class TestHaldaneChernNumber:
    def test_topological_phase_chern_pm1(self):
        H = lambda k1, k2: haldane_model(k1, k2, t=1.0, t2=0.2, phi=np.pi / 2, M=0.0)
        c = compute_chern_number(H, grid_size=24)
        assert c == [1, -1]

    def test_trivial_phase_chern_zero(self):
        # M = 2.0 >> 3*sqrt(3)*t2 = 1.039, well inside the trivial insulator regime.
        H = lambda k1, k2: haldane_model(k1, k2, t=1.0, t2=0.2, phi=np.pi / 2, M=2.0)
        c = compute_chern_number(H, grid_size=24)
        assert c == [0, 0]

    def test_time_reversal_flips_chern_sign(self):
        H_plus = lambda k1, k2: haldane_model(k1, k2, phi=np.pi / 2)
        H_minus = lambda k1, k2: haldane_model(k1, k2, phi=-np.pi / 2)
        assert compute_chern_number(H_plus, grid_size=20) == [1, -1]
        assert compute_chern_number(H_minus, grid_size=20) == [-1, 1]


class TestSSHTopology:
    def test_open_boundary_hosts_near_zero_mode_when_topological(self):
        H = ssh_lattice_hamiltonian(v=0.5, w=1.0)
        H_wire = build_ribbon(H, open_direction=0, n_cells=30)
        spectrum = np.linalg.eigvalsh(H_wire(np.array([])))
        assert np.any(np.abs(spectrum) < 1e-6)

    def test_open_boundary_has_no_zero_mode_when_trivial(self):
        H = ssh_lattice_hamiltonian(v=1.0, w=0.5)
        H_wire = build_ribbon(H, open_direction=0, n_cells=30)
        spectrum = np.linalg.eigvalsh(H_wire(np.array([])))
        assert np.all(np.abs(spectrum) > 0.1)

    def test_zak_phase_quantization(self):
        from physicskit.condensed.models import ssh_hamiltonian

        topological = lambda k: ssh_hamiltonian(k, v=0.5, w=1.0)
        trivial = lambda k: ssh_hamiltonian(k, v=1.0, w=0.5)
        assert abs(zak_phase(topological)) == pytest.approx(np.pi, abs=1e-6)
        assert abs(zak_phase(trivial)) == pytest.approx(0.0, abs=1e-6)


class TestBuildFiniteCluster:
    def test_ssh_open_chain_site_and_bond_count(self):
        H, positions, bonds = build_finite_cluster(ssh_lattice_hamiltonian(v=0.5, w=1.0), n_cells=10)
        assert H.shape == (20, 20)
        assert positions.shape == (20, 1)
        # 10 intracell (v) + 9 intercell (w) bonds; the last cell's w-bond is cut open.
        assert len(bonds) == 19

    def test_ssh_hermitian_and_matches_ribbon_spectrum(self):
        H_cluster, _, _ = build_finite_cluster(ssh_lattice_hamiltonian(v=0.5, w=1.0), n_cells=15)
        assert np.allclose(H_cluster, H_cluster.conj().T)
        H_ribbon = build_ribbon(ssh_lattice_hamiltonian(v=0.5, w=1.0), open_direction=0, n_cells=15)(np.array([]))
        # A 1D lattice's ribbon (periodic in zero remaining directions) is exactly
        # the same finite, open chain that build_finite_cluster constructs.
        assert np.allclose(np.linalg.eigvalsh(H_cluster), np.linalg.eigvalsh(H_ribbon))

    def test_ssh_topological_open_chain_hosts_near_zero_mode(self):
        H, _, _ = build_finite_cluster(ssh_lattice_hamiltonian(v=0.5, w=1.0), n_cells=15)
        eigs = np.linalg.eigvalsh(H)
        assert np.abs(eigs[np.argmin(np.abs(eigs))]) < 1e-3

    def test_graphene_flake_hermitian_with_boundary_zero_modes(self):
        H, positions, bonds = build_finite_cluster(graphene_lattice_hamiltonian(t=1.0), n_cells=(10, 10))
        assert H.shape == (200, 200)
        assert positions.shape == (200, 2)
        assert np.allclose(H, H.conj().T)
        # Zigzag-terminated finite flakes host a macroscopically-growing band
        # of near-zero-energy edge states with no bulk counterpart.
        eigs = np.linalg.eigvalsh(H)
        assert np.sum(np.abs(eigs) < 0.2) > 5

    def test_keep_predicate_carves_a_disk(self):
        lat = Lattice.honeycomb()
        n_cells = 10
        center = np.array([n_cells / 2, n_cells / 2]) @ lat.lattice_vectors
        radius = 4.0

        def in_disk(cell, orbital, position):
            return np.linalg.norm(position - center) <= radius

        H, positions, _ = build_finite_cluster(haldane_lattice_hamiltonian(t=1.0, t2=0.2), n_cells=(n_cells, n_cells), keep=in_disk)
        assert len(positions) < n_cells * n_cells * 2
        assert np.all(np.linalg.norm(positions - center, axis=1) <= radius)

    def test_haldane_disk_flake_edge_state_hugs_boundary_only_when_topological(self):
        lat = Lattice.honeycomb()
        n_cells = 14
        center = np.array([n_cells / 2, n_cells / 2]) @ lat.lattice_vectors
        radius = 6.0

        def in_disk(cell, orbital, position):
            return np.linalg.norm(position - center) <= radius

        def weighted_edge_radius(M):
            H, positions, _ = build_finite_cluster(haldane_lattice_hamiltonian(t=1.0, t2=0.2, phi=np.pi / 2, M=M), n_cells=(n_cells, n_cells), keep=in_disk)
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            mid = len(eigenvalues) // 2
            density = np.sum(np.abs(eigenvectors[:, mid - 3 : mid + 3]) ** 2, axis=1)
            r = np.linalg.norm(positions - center, axis=1)
            return np.sum(r * density) / density.sum(), r.mean()

        topological_r, mean_r = weighted_edge_radius(M=0.0)
        trivial_r, _ = weighted_edge_radius(M=2.0)
        # In the topological phase the near-gap states hug the disk's
        # (crystallographically meaningless) boundary; in the trivial phase
        # the near-gap states are ordinary bulk states with no such preference.
        assert topological_r > mean_r
        assert trivial_r < topological_r


class TestKaneMeleZ2:
    def test_trivial_when_soc_zero(self):
        H = lambda k1, k2: kane_mele_hamiltonian(k1, k2, lambda_so=0.0)
        assert z2_invariant(H, grid_size=16) == 0

    def test_topological_when_soc_nonzero(self):
        H = lambda k1, k2: kane_mele_hamiltonian(k1, k2, lambda_so=0.06)
        assert z2_invariant(H, grid_size=16) == 1

    def test_hermitian(self):
        H = kane_mele_hamiltonian(0.4, 1.1, lambda_so=0.06, lambda_r=0.05)
        assert np.allclose(H, H.conj().T)


class TestBHZModel:
    def test_hermitian_and_time_reversal_block_structure(self):
        H = bhz_hamiltonian(0.3, -0.7, M=1.0, B=1.0)
        assert np.allclose(H, H.conj().T)
        assert np.allclose(H[:2, :2], H[2:, 2:].conj())

    def test_gap_closes_at_topological_transition(self):
        # M/B = 0 is the band-inversion transition where the gap must close at Gamma.
        eigs = np.linalg.eigvalsh(bhz_hamiltonian(0.0, 0.0, M=0.0, B=1.0, A=1.0))
        assert np.allclose(eigs, 0.0, atol=1e-10)


class TestKitaevChain:
    def test_majorana_zero_mode_in_topological_phase(self):
        H = kitaev_chain_bdg_real_space(n_sites=60, mu=0.0, t=1.0, delta=1.0)
        eigs = np.linalg.eigvalsh(H)
        assert np.abs(eigs[np.argmin(np.abs(eigs))]) < 1e-6

    def test_no_zero_mode_in_trivial_phase(self):
        # |mu| > 2t is the trivial (non-topological) regime.
        H = kitaev_chain_bdg_real_space(n_sites=60, mu=3.0, t=1.0, delta=1.0)
        eigs = np.linalg.eigvalsh(H)
        assert np.abs(eigs[np.argmin(np.abs(eigs))]) > 0.1


class TestSuperconductivity:
    def test_bdg_gap_equals_pairing_amplitude(self):
        eigs = np.linalg.eigvalsh(bdg_bcs_hamiltonian(k=np.pi / 2, mu=0.0, t=1.0, delta=0.5))
        assert np.isclose(eigs[1], 0.5)


class TestHubbardModel:
    def test_two_site_dimer_exact_energy(self):
        result = hubbard_1d_exact_diagonalization(n_sites=2, n_up=1, n_dn=1, t=1.0, U=4.0)
        expected = 0.5 * (4.0 - np.sqrt(4.0**2 + 16 * 1.0**2))
        assert result["ground_state_energy"] == pytest.approx(expected, abs=1e-8)

    def test_large_u_suppresses_double_occupancy_energy_gap(self):
        low_u = hubbard_1d_exact_diagonalization(n_sites=4, n_up=2, n_dn=2, t=1.0, U=1.0)
        high_u = hubbard_1d_exact_diagonalization(n_sites=4, n_up=2, n_dn=2, t=1.0, U=20.0)
        assert high_u["ground_state_energy"] > low_u["ground_state_energy"]


class TestLandauLevels:
    def test_equally_spaced_harmonic_oscillator_spectrum(self):
        energies = landau_level_energies(n_max=4, B=1.0, m=1.0, e=1.0, hbar=1.0)
        assert np.allclose(energies, np.arange(5) + 0.5)
        assert np.allclose(np.diff(energies), 1.0)

    def test_degeneracy_scales_linearly_with_field_and_area(self):
        assert landau_degeneracy(area=10.0, B=1.0) == pytest.approx(2 * landau_degeneracy(area=5.0, B=1.0))
        assert landau_degeneracy(area=10.0, B=2.0) == pytest.approx(2 * landau_degeneracy(area=10.0, B=1.0))

    def test_filling_factor_matches_degeneracy_ratio(self):
        # nu = n_e / n_B by definition: a density equal to the per-area degeneracy gives nu=1.
        B = 1.5
        n_B = landau_degeneracy(area=1.0, B=B)
        assert filling_factor(density=n_B, B=B) == pytest.approx(1.0)


class TestGinzburgLandau:
    def test_equilibrium_order_parameter_minimizes_free_energy(self):
        a, b = -2.0, 3.0
        psi0 = gl_equilibrium_order_parameter(a, b)
        f0 = pk.condensed.gl_free_energy_density(psi0, a, b)
        for psi in np.linspace(0, 2 * psi0, 25):
            assert pk.condensed.gl_free_energy_density(psi, a, b) >= f0 - 1e-10

    def test_larger_disorder_parameter_a_shrinks_coherence_length(self):
        assert gl_coherence_length(a=-1.0) > gl_coherence_length(a=-4.0)

    def test_denser_condensate_shrinks_penetration_depth_and_kappa(self):
        # Higher condensate density psi0 screens fields over a shorter distance.
        xi = gl_coherence_length(a=-1.0)
        lam_dilute = gl_penetration_depth(psi0=0.3)
        lam_dense = gl_penetration_depth(psi0=3.0)
        assert lam_dense < lam_dilute
        assert ginzburg_landau_parameter(xi, lam_dense) < ginzburg_landau_parameter(xi, lam_dilute)

    def test_order_parameter_profile_heals_from_boundary_to_bulk(self):
        xi = 2.0
        x = np.array([0.0, xi, 5 * xi, 50 * xi])
        profile = gl_order_parameter_profile(x, xi)
        assert profile[0] == pytest.approx(0.0)
        assert np.all(np.diff(profile) > 0)
        assert profile[-1] == pytest.approx(1.0, abs=1e-6)


class TestAndersonLocalization:
    def test_clean_chain_ground_state_is_extended(self):
        H = anderson_chain_hamiltonian(n_sites=200, disorder_strength=0.0, t=1.0)
        eigs, vecs = np.linalg.eigh(H)
        ipr_clean = inverse_participation_ratio(vecs[:, np.argmin(eigs)])
        assert ipr_clean < 5.0 / 200

    def test_strong_disorder_localizes_states(self):
        H = anderson_chain_hamiltonian(n_sites=200, disorder_strength=20.0, t=1.0, seed=0)
        eigs, vecs = np.linalg.eigh(H)
        mid_state = vecs[:, len(eigs) // 2]
        assert inverse_participation_ratio(mid_state) > 0.05

    def test_localized_state_has_finite_localization_length(self):
        H = anderson_chain_hamiltonian(n_sites=300, disorder_strength=15.0, t=1.0, seed=1)
        eigs, vecs = np.linalg.eigh(H)
        mid_state = vecs[:, len(eigs) // 2]
        xi = localization_length(mid_state)
        assert 0 < xi < 300


class TestBHZEdgeStates:
    def test_helical_edge_states_in_topological_regime(self):
        spectrum = np.linalg.eigvalsh(bhz_ribbon_hamiltonian(kx=0.0, n_cells=40, M=1.0, B=1.0))
        assert np.any(np.abs(spectrum) < 1e-6)

    def test_no_edge_states_in_trivial_regime(self):
        spectrum = np.linalg.eigvalsh(bhz_ribbon_hamiltonian(kx=0.0, n_cells=40, M=-1.0, B=1.0))
        assert np.all(np.abs(spectrum) > 0.1)


class TestTopologicalInsulator3D:
    def test_bulk_hermitian(self):
        H = topological_insulator_3d_hamiltonian(0.3, -0.5, 0.7, m=1.5)
        assert np.allclose(H, H.conj().T)

    def test_surface_dirac_cone_in_strong_ti_regime(self):
        # 1 < |m/t| < 3 is the strong topological insulator window.
        spectrum = np.linalg.eigvalsh(topological_insulator_3d_slab_hamiltonian(0.0, 0.0, n_layers=40, m=-2.0))
        assert np.min(np.abs(spectrum)) < 1e-8

    def test_no_surface_state_in_trivial_regime(self):
        spectrum = np.linalg.eigvalsh(topological_insulator_3d_slab_hamiltonian(0.0, 0.0, n_layers=40, m=0.0))
        assert np.min(np.abs(spectrum)) > 0.5

    def test_surface_state_localized_on_one_layer(self):
        H = topological_insulator_3d_slab_hamiltonian(0.0, 0.0, n_layers=40, m=-2.0)
        eigs, vecs = np.linalg.eigh(H)
        near_zero = vecs[:, np.argmin(np.abs(eigs))]
        density = (np.abs(near_zero.reshape(40, 4)) ** 2).sum(axis=1)
        assert density[0] + density[-1] > 0.99


class TestHarperHofstadter:
    def test_band_chern_numbers_sum_to_zero(self):
        # The full q-band Hilbert space is a trivial bundle: whatever the
        # individual bands do, their Chern numbers must cancel.
        H = lambda k1, k2: harper_hofstadter_hamiltonian(k1, k2, p=1, q=3)
        assert sum(compute_chern_number(H, grid_size=24)) == 0

    def test_flux_one_third_matches_known_chern_numbers(self):
        H = lambda k1, k2: harper_hofstadter_hamiltonian(k1, k2, p=1, q=3)
        assert compute_chern_number(H, grid_size=24) == [-1, 2, -1]

    def test_hermitian(self):
        H = harper_hofstadter_hamiltonian(0.3, -0.7, p=2, q=5)
        assert np.allclose(H, H.conj().T)


class TestHubbardSpinCorrelations:
    def test_nearest_neighbor_antiferromagnetic_at_strong_coupling(self):
        result = hubbard_1d_exact_diagonalization(n_sites=6, n_up=3, n_dn=3, t=1.0, U=8.0, pbc=True, return_eigenvectors=True)
        corr = hubbard_spin_correlations(6, result["ground_state_vector"], result["states_up"], result["states_dn"])
        assert corr[1] < 0

    def test_antiferromagnetic_correlation_strengthens_with_u(self):
        nn_corr = []
        for U in (0.0, 4.0, 16.0):
            result = hubbard_1d_exact_diagonalization(n_sites=6, n_up=3, n_dn=3, t=1.0, U=U, pbc=True, return_eigenvectors=True)
            corr = hubbard_spin_correlations(6, result["ground_state_vector"], result["states_up"], result["states_dn"])
            nn_corr.append(corr[1])
        assert nn_corr[0] > nn_corr[1] > nn_corr[2]

    def test_onsite_correlation_is_quarter_minus_double_occupancy(self):
        # <(Sz_i)^2> = 1/4 whenever site i is never doubly (or never)
        # occupied; strong repulsion pushes every basis weight there.
        result = hubbard_1d_exact_diagonalization(n_sites=4, n_up=2, n_dn=2, t=1.0, U=50.0, pbc=True, return_eigenvectors=True)
        corr = hubbard_spin_correlations(4, result["ground_state_vector"], result["states_up"], result["states_dn"])
        assert corr[0] == pytest.approx(0.25, abs=0.02)


class TestWeylSemimetal:
    def test_nodes_gapless_bulk_gapped_elsewhere(self):
        k0 = weyl_node_locations(m=2.0, t=1.0)[1]
        eigs_at_node = np.linalg.eigvalsh(weyl_semimetal_hamiltonian(0.0, 0.0, k0, m=2.0))
        assert np.max(np.abs(eigs_at_node)) < 1e-8
        eigs_away = np.linalg.eigvalsh(weyl_semimetal_hamiltonian(0.0, 0.0, 0.0, m=2.0))
        assert np.min(np.abs(eigs_away)) > 0.5

    def test_no_nodes_outside_window(self):
        assert weyl_node_locations(m=4.0, t=1.0) is None
        assert weyl_node_locations(m=0.5, t=1.0) is None

    def test_slab_gapless_between_nodes_gapped_outside(self):
        gap_inside = np.min(np.abs(np.linalg.eigvalsh(weyl_semimetal_slab_hamiltonian(0.0, 0.0, n_layers=40, m=2.0))))
        gap_outside = np.min(np.abs(np.linalg.eigvalsh(weyl_semimetal_slab_hamiltonian(0.0, np.pi, n_layers=40, m=2.0))))
        assert gap_inside < 1e-6
        assert gap_outside > 0.5


class TestLaughlin:
    def test_metropolis_sweep_keeps_particles_apart(self):
        z = np.array([0.1, -0.1, 0.2j, -0.2j]) + 0j
        for _ in range(100):
            laughlin_metropolis_sweep(z, m=3, step=0.5)
        dists = np.abs(z[:, None] - z[None, :]) + np.eye(4) * 10
        assert dists.min() > 1e-3

    def test_correlation_hole_vanishes_near_origin(self):
        rng = np.random.default_rng(1)
        N, m = 10, 3
        R0 = np.sqrt(2 * m * N)
        r0 = R0 * np.sqrt(rng.random(N))
        theta0 = rng.uniform(0, 2 * np.pi, N)
        z = (r0 * np.cos(theta0) + 1j * r0 * np.sin(theta0)).astype(np.complex128)
        for _ in range(500):
            laughlin_metropolis_sweep(z, m=m, step=1.0)
        samples = np.array([laughlin_metropolis_sweep(z, m=m, step=1.0).copy() for _ in range(150)])
        r, g = laughlin_pair_correlation(samples, m=m, r_max=8.0, n_bins=40)
        assert g[0] < 0.1
        assert g[-5:].mean() > g[0]

    def test_density_falls_off_at_droplet_edge(self):
        rng = np.random.default_rng(2)
        N, m = 10, 3
        R0 = np.sqrt(2 * m * N)
        r0 = R0 * np.sqrt(rng.random(N))
        theta0 = rng.uniform(0, 2 * np.pi, N)
        z = (r0 * np.cos(theta0) + 1j * r0 * np.sin(theta0)).astype(np.complex128)
        for _ in range(500):
            laughlin_metropolis_sweep(z, m=m, step=1.0)
        samples = np.array([laughlin_metropolis_sweep(z, m=m, step=1.0).copy() for _ in range(150)])
        r, density = laughlin_radial_density(samples, r_max=2 * R0, n_bins=20)
        assert density[-1] < density[2]


def test_public_api_exposed_via_pk_condensed():
    assert pk.condensed.compute_chern_number is compute_chern_number
    assert hasattr(pk.condensed, "haldane_model")
    assert hasattr(pk.condensed, "landau_level_energies")
    assert hasattr(pk.condensed, "gl_coherence_length")
    assert hasattr(pk.condensed, "anderson_chain_hamiltonian")
    assert hasattr(pk.condensed, "topological_insulator_3d_hamiltonian")
    assert hasattr(pk.condensed, "bhz_ribbon_hamiltonian")
    assert hasattr(pk.condensed, "harper_hofstadter_hamiltonian")
    assert hasattr(pk.condensed, "hubbard_spin_correlations")
    assert hasattr(pk.condensed, "weyl_semimetal_hamiltonian")
    assert hasattr(pk.condensed, "laughlin_pair_correlation")
