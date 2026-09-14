r"""Entanglement, Bell/EPR correlations, and the Aharonov-Bohm effect."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from ..core.operators import sigma_x, sigma_z

__all__ = ["bell_state", "BellCorrelations", "AharonovBohmRing", "IsingEntangler"]


# --- Bell states -------------------------------------------------------------


def bell_state(kind: str = "phi+") -> np.ndarray:
    r"""One of the four maximally entangled two-qubit Bell states.

    In the computational basis :math:`\lvert00\rangle, \lvert01\rangle,
    \lvert10\rangle, \lvert11\rangle`:

    .. math::

        \lvert\Phi^\pm\rangle = \tfrac{1}{\sqrt2}(\lvert00\rangle \pm \lvert11\rangle),
        \qquad
        \lvert\Psi^\pm\rangle = \tfrac{1}{\sqrt2}(\lvert01\rangle \pm \lvert10\rangle).

    Parameters
    ----------
    kind : {'phi+', 'phi-', 'psi+', 'psi-'}, default='phi+'
        Which Bell state to return. ``'psi-'`` is the spin singlet.

    Returns
    -------
    numpy.ndarray
        The 4-component state vector.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the four valid labels.
    """
    s = 1 / np.sqrt(2)
    states = {
        "phi+": s * np.array([1, 0, 0, 1], dtype=complex),
        "phi-": s * np.array([1, 0, 0, -1], dtype=complex),
        "psi+": s * np.array([0, 1, 1, 0], dtype=complex),
        "psi-": s * np.array([0, 1, -1, 0], dtype=complex),  # the spin singlet
    }
    if kind not in states:
        raise ValueError(f"kind must be one of {list(states)}")
    return states[kind]


def _spin_projector(theta: float) -> np.ndarray:
    """Single-qubit measurement operator for spin along an axis at angle
    theta from z in the x-z plane: cos(theta) sigma_z + sin(theta) sigma_x."""
    return np.cos(theta) * sigma_z + np.sin(theta) * sigma_x


@dataclass
class BellCorrelations:
    r"""EPR correlations for the spin-1/2 singlet state.

    .. math::

        \lvert\psi^-\rangle = \frac{\lvert01\rangle - \lvert10\rangle}{\sqrt2}.

    Reproduces the textbook result
    :math:`E(a,b) = -\cos(\theta_a - \theta_b)` and the CHSH violation of
    the classical (local hidden-variable) bound :math:`\lvert S\rvert \le 2`,
    up to the quantum (Tsirelson) bound :math:`2\sqrt2`.

    Parameters
    ----------
    state : numpy.ndarray or None, optional
        The two-qubit state to use; defaults to the singlet
        :func:`bell_state`\ ``('psi-')``.
    """

    state: np.ndarray = None

    def __post_init__(self):
        if self.state is None:
            self.state = bell_state("psi-")

    def correlation(self, theta_a: float, theta_b: float) -> float:
        r"""The spin correlation :math:`E(a,b) = \langle\psi\rvert
        \hat A(\theta_a) \otimes \hat B(\theta_b) \lvert\psi\rangle`.

        Parameters
        ----------
        theta_a : float
            Measurement angle for qubit 1 (from :math:`z`, in the x-z plane).
        theta_b : float
            Measurement angle for qubit 2.

        Returns
        -------
        float
        """
        A = _spin_projector(theta_a)
        B = _spin_projector(theta_b)
        op = np.kron(A, B)
        return float(np.real(np.vdot(self.state, op @ self.state)))

    def chsh_S(self, a: float, a_prime: float, b: float, b_prime: float) -> float:
        r"""The CHSH combination
        :math:`S = E(a,b) - E(a,b') + E(a',b) + E(a',b')`.

        Parameters
        ----------
        a, a_prime : float
            The two measurement angles for qubit 1.
        b, b_prime : float
            The two measurement angles for qubit 2.

        Returns
        -------
        float
        """
        return self.correlation(a, b) - self.correlation(a, b_prime) + self.correlation(a_prime, b) + self.correlation(a_prime, b_prime)

    def chsh_optimal(self) -> float:
        r""":math:`\lvert S\rvert` at the standard optimal angles
        :math:`(0, \pi/2, \pi/4, 3\pi/4)`.

        Gives the Tsirelson bound :math:`\lvert S\rvert = 2\sqrt2 \approx
        2.828`, violating the classical bound of 2 predicted by any local
        hidden-variable theory.

        Returns
        -------
        float
        """
        return abs(self.chsh_S(0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4))

    def measurement_probabilities(self, theta_a: float, theta_b: float) -> dict:
        r"""Joint outcome probabilities for projective spin measurements.

        Parameters
        ----------
        theta_a : float
            Measurement angle for qubit 1.
        theta_b : float
            Measurement angle for qubit 2.

        Returns
        -------
        dict
            Maps ``'++'``, ``'+-'``, ``'-+'``, ``'--'`` to their probabilities.
        """

        # eigenvectors of the spin projector at angle theta (in +1/-1 order)
        def eigvecs(theta):
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            return np.array([c, s]), np.array([-s, c])

        up_a, down_a = eigvecs(theta_a)
        up_b, down_b = eigvecs(theta_b)
        probs = {}
        for (label_a, va), (label_b, vb) in product([("+", up_a), ("-", down_a)], [("+", up_b), ("-", down_b)]):
            proj = np.kron(va, vb)
            amp = np.vdot(proj, self.state)
            probs[label_a + label_b] = float(np.abs(amp) ** 2)
        return probs

    def monte_carlo_correlation(self, theta_a: float, theta_b: float, n_trials: int = 20000, rng: np.random.Generator | None = None) -> float:
        r"""Simulate joint measurements and return the empirical correlation.

        Samples simulated joint measurement outcomes from the exact quantum
        joint distribution and returns the empirical correlation
        :math:`\langle AB\rangle`, statistically converging to
        :meth:`correlation`\ ``(theta_a, theta_b)``.

        Parameters
        ----------
        theta_a : float
            Measurement angle for qubit 1.
        theta_b : float
            Measurement angle for qubit 2.
        n_trials : int, default=20000
            Number of simulated measurement pairs.
        rng : numpy.random.Generator or None, optional
            Random number generator; a fresh default one is used if omitted.

        Returns
        -------
        float
        """
        rng = rng or np.random.default_rng()
        probs = self.measurement_probabilities(theta_a, theta_b)
        labels = list(probs.keys())
        p = np.array([probs[k] for k in labels])
        p /= p.sum()
        draws = rng.choice(len(labels), size=n_trials, p=p)
        outcomes = np.array([[1 if lbl[0] == "+" else -1, 1 if lbl[1] == "+" else -1] for lbl in labels])
        samples = outcomes[draws]
        return float(np.mean(samples[:, 0] * samples[:, 1]))


# --- Dynamical entanglement generation: Ising coupling ----------------------


@dataclass
class IsingEntangler:
    r"""Two initially unentangled qubits, entangled by an Ising coupling.

    Both qubits start in :math:`\lvert+\rangle = (\lvert0\rangle+\lvert1\rangle)/\sqrt2`
    (an eigenstate of :math:`\sigma_x`, i.e. unentangled and unbiased in the
    measurement basis), and evolve under

    .. math::

        \hat H = \hbar J\, \sigma_z^{(1)}\otimes\sigma_z^{(2)}.

    Since :math:`\lvert+\rangle\otimes\lvert+\rangle` is a superposition of
    all four :math:`\sigma_z\otimes\sigma_z` eigenstates,
    :math:`e^{-i\hat Ht/\hbar}` accumulates a different phase on each
    computational-basis term, generating genuine entanglement -- maximal
    (a Bell-equivalent state) at :math:`Jt=\pi/4`.

    Parameters
    ----------
    J : float, default=1.0
        Ising coupling strength.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    """

    J: float = 1.0
    hbar: float = 1.0

    def initial_state(self) -> np.ndarray:
        r"""The unentangled product state :math:`\lvert+\rangle\otimes\lvert+\rangle`.

        Returns
        -------
        numpy.ndarray
            The 4-component state vector.
        """
        plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        return np.kron(plus, plus)

    def state(self, t: float) -> np.ndarray:
        r"""The evolved two-qubit state :math:`e^{-i\hat Ht/\hbar}\lvert\psi_0\rangle`.

        :math:`\hat H` is diagonal in the computational basis with
        eigenvalues :math:`\hbar J` on :math:`\lvert00\rangle,\lvert11\rangle`
        and :math:`-\hbar J` on :math:`\lvert01\rangle,\lvert10\rangle`, so
        the propagator is applied as a plain phase per basis state.

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
            The 4-component state vector, in the
            :math:`\lvert00\rangle,\lvert01\rangle,\lvert10\rangle,\lvert11\rangle` basis.
        """
        zz_eigenvalues = np.array([1.0, -1.0, -1.0, 1.0])  # sigma_z(x)sigma_z on |00>,|01>,|10>,|11>
        phases = np.exp(-1j * self.J * zz_eigenvalues * t)
        return phases * self.initial_state()

    def state_trajectory(self, t_values: np.ndarray) -> np.ndarray:
        """Stack of :meth:`state` snapshots over a time grid.

        Parameters
        ----------
        t_values : numpy.ndarray
            Times to evaluate at.

        Returns
        -------
        numpy.ndarray
            Complex-valued, shape ``(len(t_values), 4)``.
        """
        return np.array([self.state(t) for t in t_values])

    def reduced_density_matrix(self, psi: np.ndarray) -> np.ndarray:
        r"""The reduced density matrix :math:`\rho_1=\mathrm{Tr}_2\lvert\psi\rangle\langle\psi\rvert`
        of the first qubit.

        Parameters
        ----------
        psi : numpy.ndarray
            The 4-component two-qubit state vector.

        Returns
        -------
        numpy.ndarray
            The :math:`2\times2` reduced density matrix.
        """
        psi_mat = np.asarray(psi, dtype=complex).reshape(2, 2)  # rows = qubit 1, cols = qubit 2
        return psi_mat @ psi_mat.conj().T

    def purity(self, psi: np.ndarray) -> float:
        r"""The purity :math:`\mathrm{Tr}(\rho_1^2)` of the reduced state.

        Equal to 1 for an unentangled (product) state and 1/2 (minimal, for
        a single qubit) at maximal entanglement.

        Parameters
        ----------
        psi : numpy.ndarray
            The 4-component two-qubit state vector.

        Returns
        -------
        float
        """
        rho = self.reduced_density_matrix(psi)
        return float(np.real(np.trace(rho @ rho)))

    def concurrence(self, psi: np.ndarray) -> float:
        r"""The Wootters concurrence, a two-qubit entanglement measure.

        For a pure state, :math:`C = 2\lvert\det\Psi\rvert` where
        :math:`\Psi` is the state reshaped into a :math:`2\times2` matrix
        (equivalently :math:`C=\sqrt{2(1-\mathrm{Tr}\,\rho_1^2)}`);
        :math:`C=0` for a product state and :math:`C=1` for a maximally
        entangled (Bell) state.

        Parameters
        ----------
        psi : numpy.ndarray
            The 4-component two-qubit state vector.

        Returns
        -------
        float
        """
        psi_mat = np.asarray(psi, dtype=complex).reshape(2, 2)
        return float(2 * np.abs(np.linalg.det(psi_mat)))

    def concurrence_trajectory(self, t_values: np.ndarray) -> np.ndarray:
        """The concurrence :math:`C(t)` over a time grid.

        Parameters
        ----------
        t_values : numpy.ndarray
            Times to evaluate at.

        Returns
        -------
        numpy.ndarray
        """
        return np.array([self.concurrence(self.state(t)) for t in t_values])


# --- Aharonov-Bohm effect ----------------------------------------------------


@dataclass
class AharonovBohmRing:
    r"""A charged particle confined to a 1D ring threading a magnetic flux.

    :math:`B=0` everywhere on the ring itself -- only the vector potential
    :math:`A = \Phi/(2\pi R)` is nonzero there. The eigenspectrum

    .. math::

        E_n(\Phi) = \frac{\hbar^2}{2mR^2}\left(n - \frac{\Phi}{\Phi_0}\right)^2

    shows the Aharonov-Bohm phase shift
    :math:`\Delta\phi = (q/\hbar)\oint \mathbf A\cdot d\mathbf l = 2\pi\Phi/\Phi_0`
    purely through boundary conditions, without the particle ever entering
    a region where :math:`B\neq0`.

    Parameters
    ----------
    R : float, default=1.0
        Ring radius.
    m : float, default=1.0
        Particle mass.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    q : float, default=1.0
        Particle charge.
    """

    R: float = 1.0
    m: float = 1.0
    hbar: float = 1.0
    q: float = 1.0

    @property
    def flux_quantum(self) -> float:
        r"""float: The flux quantum :math:`\Phi_0 = 2\pi\hbar/q` (:math:`h/q`
        in these units)."""
        return 2 * np.pi * self.hbar / self.q

    def energy(self, n, Phi: float) -> np.ndarray:
        r"""Eigenenergy :math:`E_n(\Phi)`.

        Parameters
        ----------
        n : int or array_like
            Angular-momentum quantum number(s).
        Phi : float
            Enclosed magnetic flux.

        Returns
        -------
        numpy.ndarray
        """
        n = np.asarray(n)
        flux_ratio = Phi / self.flux_quantum
        return (self.hbar**2 / (2 * self.m * self.R**2)) * (n - flux_ratio) ** 2

    def spectrum(self, Phi: float, n_range: int = 5) -> np.ndarray:
        """Sorted energies for :math:`n \\in [-n_\\text{range}, n_\\text{range}]`.

        Parameters
        ----------
        Phi : float
            Enclosed magnetic flux.
        n_range : int, default=5
            Range of angular-momentum quantum numbers to include.

        Returns
        -------
        numpy.ndarray
        """
        n = np.arange(-n_range, n_range + 1)
        return np.sort(self.energy(n, Phi))

    def aharonov_bohm_phase(self, Phi: float) -> float:
        r"""The Aharonov-Bohm phase :math:`\Delta\phi = 2\pi\Phi/\Phi_0`.

        The phase accumulated by a particle encircling the flux tube once.

        Parameters
        ----------
        Phi : float
            Enclosed magnetic flux.

        Returns
        -------
        float
        """
        return 2 * np.pi * Phi / self.flux_quantum

    def eigenstate(self, n: int, phi: np.ndarray) -> np.ndarray:
        r"""Real-space ring wavefunction :math:`\psi_n(\phi) = e^{in\phi}/\sqrt{2\pi R}`.

        Parameters
        ----------
        n : int
            Angular-momentum quantum number.
        phi : numpy.ndarray
            Angular positions around the ring.

        Returns
        -------
        numpy.ndarray
            Complex-valued.
        """
        return np.exp(1j * n * phi) / np.sqrt(2 * np.pi * self.R)

    def persistent_current(self, n: int, Phi: float, dPhi: float = 1e-6) -> float:
        r"""Equilibrium persistent current carried by level :math:`n`.

        .. math::

            I_n(\Phi) = -\frac{dE_n}{d\Phi},

        periodic in :math:`\Phi` with period :math:`\Phi_0` -- the direct
        experimental signature of the Aharonov-Bohm phase.

        Parameters
        ----------
        n : int
            Angular-momentum quantum number.
        Phi : float
            Enclosed magnetic flux.
        dPhi : float, default=1e-6
            Step size for the central-difference derivative.

        Returns
        -------
        float
        """
        return -(self.energy(n, Phi + dPhi) - self.energy(n, Phi - dPhi)) / (2 * dPhi)
