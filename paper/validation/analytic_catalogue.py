r"""
Closed-form and literature results that the gallery examples assert.

Two or three per subpackage. Each entry names the result, its reference,
the example that asserts it, and the tolerance of that assert.
``run_all.py`` re-runs every listed example, so the table in the paper
cannot claim a check that no longer passes.
"""

ANALYTIC = [
    {
        "result": "Lane-Emden polytropes: first zeros $\\xi_1 = \\sqrt6$, $\\pi$ and $6.89685$ for $n = 0, 1, 3$",
        "reference": "Chandrasekhar, Stellar Structure (1939)",
        "example": "astro/stellar_structure/plot_01_lane_emden_polytropes.py",
        "tolerance": "$10^{-6}$; $10^{-4}$ ($n = 3$)",
    },
    {
        "result": "Figure-eight three-body choreography: period $6.32591$, energy $-1.28714$",
        "reference": "Chenciner and Montgomery, Ann. Math. 152, 881 (2000)",
        "example": "astro/nbody/plot_02_figure_eight_choreography.py",
        "tolerance": "$2\\Delta t$; $10^{-4}$",
    },
    {
        "result": "Lyapunov spectra: Lorenz $\\lambda_1 = 0.906$, H\\'enon exponents summing to $\\ln 0.3$",
        "reference": "Benettin et al., Meccanica 15, 9 (1980)",
        "example": "chaos/chaos_metrics/plot_lyapunov_spectrum.py",
        "tolerance": "$0.05$; $10^{-9}$",
    },
    {
        "result": "Smale horseshoe: topological entropy $\\ln 2$, $2^n$ points of period $n$",
        "reference": "Smale, Bull. AMS 73, 747 (1967)",
        "example": "chaos/maps/plot_smale_horseshoe.py",
        "tolerance": "$10^{-12}$; exact",
    },
    {
        "result": "Euler's disk: $\\theta(t) = \\theta_0\\sqrt{1 - t/t_f}$, a finite-time singularity",
        "reference": "Moffatt, Nature 404, 833 (2000)",
        "example": "classical/rotations/plot_04_eulers_disk.py",
        "tolerance": "$10^{-10}$",
    },
    {
        "result": "Foucault pendulum: precession period of one sidereal day over $\\sin\\varphi$",
        "reference": "Foucault, C. R. Acad. Sci. 32, 135 (1851)",
        "example": "classical/newtonian/plot_03_foucault_pendulum.py",
        "tolerance": "$10^{-3}$ (relative)",
    },
    {
        "result": "Heisenberg chain: exact-diagonalization ground energies equal the Bethe ansatz",
        "reference": "Bethe, Z. Phys. 71, 205 (1931)",
        "example": "condensed/spin_chains/plot_bethe_ansatz_heisenberg_chain.py",
        "tolerance": "$10^{-8}$",
    },
    {
        "result": "Haldane model: Chern numbers $\\pm 1$, Berry flux $2\\pi$ over the zone",
        "reference": "Thouless et al., PRL 49, 405 (1982)",
        "example": "condensed/topology/plot_chern_number.py",
        "tolerance": "exact; $10^{-10}$",
    },
    {
        "result": "Transverse-field Ising chain: free-fermion spectrum, $e_0 = -4/\\pi$ at $h = J$",
        "reference": "Pfeuty, Ann. Phys. 57, 79 (1970)",
        "example": "condensed/spin_chains/plot_transverse_field_ising_free_fermions.py",
        "tolerance": "$10^{-10}$; $10^{-4}$",
    },
    {
        "result": "Casimir energy $-\\pi/24d$ between two plates (one dimension)",
        "reference": "Casimir, Proc. KNAW 51, 793 (1948)",
        "example": "fields/quantum_fields/plot_casimir_effect.py",
        "tolerance": "$10^{-6}$ (relative)",
    },
    {
        "result": "KdV soliton fission: amplitudes from the bound states of the scattering problem",
        "reference": "Zabusky and Kruskal, PRL 15, 240 (1965)",
        "example": "fields/solitons/plot_kdv_soliton_fission.py",
        "tolerance": "1%",
    },
    {
        "result": "Blasius boundary layer: $f''(0) = 0.33206$, skin friction $0.664/\\sqrt{Re_x}$",
        "reference": "Blasius, Z. Math. Phys. 56, 1 (1908)",
        "example": "fluids/viscous_flow/plot_blasius_boundary_layer.py",
        "tolerance": "$10^{-5}$; $10^{-3}$",
    },
    {
        "result": "Rayleigh-B\\'enard onset between rigid walls: $Ra_c = 1707.76$, $a_c = 3.117$",
        "reference": "Chandrasekhar, Hydrodynamic Stability (1961)",
        "example": "fluids/convection/plot_rayleigh_benard_onset.py",
        "tolerance": "$0.05$; $2\\times10^{-3}$",
    },
    {
        "result": "Jaynes-Cummings vacuum Rabi oscillation $P_e = \\cos^2 gt$",
        "reference": "Jaynes and Cummings, Proc. IEEE 51, 89 (1963)",
        "example": "optics/quantum_optics/plot_jaynes_cummings_rabi.py",
        "tolerance": "$10^{-12}$",
    },
    {
        "result": "Fock state: Fano factor $0$, Wigner minimum $-1/\\pi$",
        "reference": "Kimble, Dagenais and Mandel, PRL 39, 691 (1977)",
        "example": "optics/quantum_optics/plot_photon_antibunching.py",
        "tolerance": "$10^{-12}$; $10^{-9}$",
    },
    {
        "result": "Compton shift $\\Delta\\lambda = (h/mc)(1 - \\cos\\theta)$ from four-momentum conservation",
        "reference": "Compton, Phys. Rev. 21, 483 (1923)",
        "example": "particle/kinematics/plot_02_compton_scattering.py",
        "tolerance": "$10^{-8}$",
    },
    {
        "result": "2D U(1) lattice gauge theory: Monte Carlo string tension $-\\ln[I_1(\\beta)/I_0(\\beta)]$",
        "reference": "Wilson, PRD 10, 2445 (1974)",
        "example": "particle/qcd/plot_02_wilson_lattice_gauge_area_law.py",
        "tolerance": "5%",
    },
    {
        "result": "Landau damping in particle-in-cell: damping rate, frequency $1.416\\,\\omega_p$",
        "reference": "Landau, J. Phys. USSR 10, 25 (1946)",
        "example": "plasma/kinetic/plot_01_landau_damping.py",
        "tolerance": "10%; $0.05$",
    },
    {
        "result": "Grad-Shafranov equation: relaxation reaches Solov'ev's exact equilibrium",
        "reference": "Solov'ev, Sov. Phys. JETP 26, 400 (1968)",
        "example": "plasma/mhd/plot_03_grad_shafranov_equilibrium.py",
        "tolerance": "$10^{-5}$",
    },
    {
        "result": "Hydrogen: $E_n = -1/2n^2$, most probable radius $n^2$ for $l = n - 1$",
        "reference": 'Schr\\"odinger, Ann. Phys. 79, 361 (1926)',
        "example": "quantum/hydrogen/plot_hydrogen_orbitals.py",
        "tolerance": "$10^{-12}$; $0.02$",
    },
    {
        "result": "Dirac hydrogen: $2p$ fine-structure splitting $10.95$ GHz",
        "reference": "Dirac, Proc. R. Soc. A 117, 610 (1928)",
        "example": "quantum/relativistic/plot_dirac_hydrogen_fine_structure.py",
        "tolerance": "$0.01$ GHz",
    },
    {
        "result": "Grover search: success probability $\\sin^2[(2k+1)\\theta]$ after $k$ queries",
        "reference": "Grover, PRL 79, 325 (1997)",
        "example": "quantum/quantum_computing/plot_grover_search.py",
        "tolerance": "$10^{-12}$",
    },
    {
        "result": "Deflection of starlight at the solar limb: $1.751''$",
        "reference": "Dyson, Eddington and Davidson, Phil. Trans. R. Soc. A 220, 291 (1920)",
        "example": "relativity/schwarzschild/plot_eddington_eclipse_deflection.py",
        "tolerance": "$0.002''$",
    },
    {
        "result": "Kerr black hole: ergosphere, horizon and prograde and retrograde ISCO radii",
        "reference": "Bardeen, Press and Teukolsky, ApJ 178, 347 (1972)",
        "example": "relativity/kerr/plot_ergosphere_and_penrose.py",
        "tolerance": "$10^{-6}$ (relative)",
    },
    {
        "result": "Wigner semicircle for the three Gaussian ensembles",
        "reference": "Wigner, Ann. Math. 62, 548 (1955)",
        "example": "rmt/gaussian_ensembles/plot_wigner_semicircle_law.py",
        "tolerance": "KS distance $0.005$",
    },
    {
        "result": "Marchenko-Pastur law for Wishart matrices",
        "reference": "Marchenko and Pastur, Math. USSR Sb. 1, 457 (1967)",
        "example": "rmt/wishart/plot_marchenko_pastur_law.py",
        "tolerance": "KS distance $0.01$",
    },
    {
        "result": "Langer-corrected radial WKB gives the exact hydrogen levels",
        "reference": "Langer, Phys. Rev. 51, 669 (1937)",
        "example": "semiclassical/wkb/plot_langer_correction_hydrogen.py",
        "tolerance": "$10^{-6}$ (relative)",
    },
    {
        "result": "Van Vleck propagator: exact for the harmonic oscillator",
        "reference": "Van Vleck, PNAS 14, 178 (1928)",
        "example": "semiclassical/propagators/plot_semiclassical_propagators.py",
        "tolerance": "$10^{-10}$",
    },
    {
        "result": "2D Ising model: Monte Carlo magnetization follows the exact $(1 - \\sinh^{-4} 2\\beta J)^{1/8}$",
        "reference": "Yang, Phys. Rev. 85, 808 (1952)",
        "example": "statphys/ising/plot_ising_phase_transition.py",
        "tolerance": "$0.03$",
    },
    {
        "result": "Ferromagnetic Ising partition-function zeros on the unit circle",
        "reference": "Lee and Yang, Phys. Rev. 87, 410 (1952)",
        "example": "statphys/yang_lee/plot_yang_lee_zeros.py",
        "tolerance": "$10^{-9}$",
    },
]
