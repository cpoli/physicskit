"""Generic tight-binding lattice and Bloch Hamiltonian construction.

This module provides the building blocks that :mod:`physicskit.condensed.models`
uses to assemble specific band models (graphene, the Haldane model, ...):
a :class:`Lattice` describing the Bravais lattice and orbital basis, and a
:class:`Hamiltonian` that accumulates real-space hoppings and Bloch-sums them
into :math:`H(\\mathbf{k})`.

Convention
----------
Bloch Hamiltonians in this package use the **periodic gauge**: a hopping
term with integer cell offset :math:`\\mathbf{R} = (n_1, n_2, \\dots)`
(expressed in units of the primitive lattice vectors) contributes a phase
:math:`e^{i \\mathbf{k} \\cdot \\mathbf{R}}` where :math:`\\mathbf{k}` is the
*reduced* crystal momentum conjugate to the primitive-vector indices, each
component periodic on :math:`[0, 2\\pi)`. This intracell-position-free
convention keeps :math:`H(\\mathbf{k})` manifestly periodic on the Brillouin
zone torus, which is exactly what the Fukui-Hatsugai-Suzuki Chern-number
algorithm in :mod:`physicskit.condensed.topology` needs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Lattice", "Hamiltonian", "build_ribbon", "build_finite_cluster", "apply_peierls_phase"]


@dataclass
class Lattice:
    """A Bravais lattice with a basis of orbitals.

    Parameters
    ----------
    lattice_vectors : array_like, shape (dim, dim)
        Primitive lattice vectors as rows, in Cartesian coordinates.
    orbitals : array_like, shape (n_orbitals, dim)
        Orbital positions in fractional coordinates of the primitive cell.
    labels : list of str, optional
        Human-readable names for each orbital (defaults to ``orb0``, ``orb1``, ...).

    Examples
    --------
    >>> lat = Lattice.honeycomb()
    >>> lat.n_orbitals
    2
    >>> lat.dim
    2
    """

    lattice_vectors: np.ndarray
    orbitals: np.ndarray
    labels: list = field(default_factory=list)

    def __post_init__(self):
        self.lattice_vectors = np.atleast_2d(np.asarray(self.lattice_vectors, dtype=float))
        self.orbitals = np.atleast_2d(np.asarray(self.orbitals, dtype=float))
        if not self.labels:
            self.labels = [f"orb{i}" for i in range(len(self.orbitals))]

    @property
    def dim(self) -> int:
        """int: Spatial dimension (1, 2, or 3)."""
        return self.lattice_vectors.shape[0]

    @property
    def n_orbitals(self) -> int:
        """int: Number of orbitals per unit cell."""
        return self.orbitals.shape[0]

    def cartesian_orbitals(self) -> np.ndarray:
        """Return orbital positions in Cartesian coordinates.

        Returns
        -------
        ndarray, shape (n_orbitals, dim)

        Examples
        --------
        >>> lat = Lattice.square()
        >>> lat.cartesian_orbitals()
        array([[0., 0.]])
        """
        return self.orbitals @ self.lattice_vectors

    def reciprocal_vectors(self) -> np.ndarray:
        """Return primitive reciprocal lattice vectors :math:`\\mathbf{b}_i`.

        Satisfies :math:`\\mathbf{a}_i \\cdot \\mathbf{b}_j = 2\\pi\\delta_{ij}`.

        Returns
        -------
        ndarray, shape (dim, dim)

        Examples
        --------
        >>> lat = Lattice.square(a=1.0)
        >>> np.allclose(lat.reciprocal_vectors(), 2 * np.pi * np.eye(2))
        True
        """
        return 2 * np.pi * np.linalg.inv(self.lattice_vectors).T

    # -- Common lattice factories -----------------------------------------

    @classmethod
    def chain(cls, a: float = 1.0) -> Lattice:
        """1D monatomic chain with lattice constant ``a``."""
        return cls(lattice_vectors=[[a]], orbitals=[[0.0]])

    @classmethod
    def square(cls, a: float = 1.0) -> Lattice:
        """2D square lattice, one orbital per cell."""
        return cls(lattice_vectors=[[a, 0.0], [0.0, a]], orbitals=[[0.0, 0.0]])

    @classmethod
    def triangular(cls, a: float = 1.0) -> Lattice:
        """2D triangular lattice, one orbital per cell."""
        return cls(lattice_vectors=[[a, 0.0], [a / 2, a * np.sqrt(3) / 2]], orbitals=[[0.0, 0.0]])

    @classmethod
    def honeycomb(cls, a: float = 1.0) -> Lattice:
        """2D honeycomb lattice (graphene structure), sublattices A and B.

        ``a`` is the lattice constant (nearest-neighbor bond length is ``a/sqrt(3)``).
        """
        return cls(
            lattice_vectors=[[a, 0.0], [a / 2, a * np.sqrt(3) / 2]],
            orbitals=[[1 / 3, 1 / 3], [2 / 3, 2 / 3]],
            labels=["A", "B"],
        )

    @classmethod
    def kagome(cls, a: float = 1.0) -> Lattice:
        """2D kagome lattice, three orbitals per cell (edge midpoints of a triangular lattice)."""
        return cls(
            lattice_vectors=[[a, 0.0], [a / 2, a * np.sqrt(3) / 2]],
            orbitals=[[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]],
            labels=["A", "B", "C"],
        )

    @classmethod
    def cubic(cls, a: float = 1.0) -> Lattice:
        """3D simple cubic lattice, one orbital per cell."""
        return cls(lattice_vectors=[[a, 0, 0], [0, a, 0], [0, 0, a]], orbitals=[[0.0, 0.0, 0.0]])


class Hamiltonian:
    """Real-space tight-binding Hamiltonian, Bloch-summed into :math:`H(\\mathbf{k})`.

    Parameters
    ----------
    lattice : Lattice
        The underlying Bravais lattice and orbital basis.
    onsite : array_like, shape (n_orbitals,), optional
        Onsite energies. Defaults to zero for every orbital.

    Examples
    --------
    Nearest-neighbor graphene, reproducing the linear Dirac dispersion:

    >>> import numpy as np
    >>> lat = Lattice.honeycomb()
    >>> H = Hamiltonian(lat)
    >>> t = 1.0
    >>> H.add_hopping(0, 1, (0, 0), t)
    >>> H.add_hopping(0, 1, (-1, 0), t)
    >>> H.add_hopping(0, 1, (0, -1), t)
    >>> K = np.array([2 * np.pi / 3, 4 * np.pi / 3])
    >>> np.round(np.linalg.eigvalsh(H.bloch(K)), 8)
    array([-0.,  0.])
    """

    def __init__(self, lattice: Lattice, onsite=None):
        self.lattice = lattice
        n = lattice.n_orbitals
        self.onsite = np.zeros(n) if onsite is None else np.asarray(onsite, dtype=float)
        self._hoppings: list[tuple[int, int, tuple, complex]] = []

    def add_hopping(self, i: int, j: int, cell_offset, amplitude: complex) -> None:
        """Add a hopping term :math:`t\\, c_i^\\dagger(0) c_j(\\mathbf{R})` (+ h.c.).

        Parameters
        ----------
        i, j : int
            Orbital indices within the unit cell.
        cell_offset : tuple of int
            Integer offset :math:`\\mathbf{R}` (in primitive-vector units) of
            orbital ``j``'s cell relative to orbital ``i``'s cell.
        amplitude : complex
            Hopping amplitude. The Hermitian conjugate term is added
            automatically; do not add both a bond and its reverse.

        Raises
        ------
        ValueError
            If ``(i, j, cell_offset)`` describes a diagonal onsite term
            (``i == j`` and ``cell_offset`` all zero); use ``onsite`` instead.
        """
        cell_offset = tuple(int(x) for x in cell_offset)
        if i == j and all(x == 0 for x in cell_offset):
            raise ValueError("Use the `onsite` array for i == j, R == 0 terms.")
        self._hoppings.append((i, j, cell_offset, complex(amplitude)))

    def bloch(self, k) -> np.ndarray:
        """Evaluate the Bloch Hamiltonian at reduced crystal momentum ``k``.

        Parameters
        ----------
        k : array_like, shape (dim,)
            Reduced crystal momentum, each component periodic on :math:`[0, 2\\pi)`.

        Returns
        -------
        ndarray, shape (n_orbitals, n_orbitals)
            Hermitian Bloch Hamiltonian matrix.
        """
        k = np.asarray(k, dtype=float)
        H = np.diag(self.onsite).astype(complex)
        for i, j, R, amp in self._hoppings:
            phase = np.exp(1j * np.dot(k, R))
            H[i, j] += amp * phase
            H[j, i] += np.conj(amp * phase)
        return H

    def bands(self, k) -> np.ndarray:
        """Eigenvalues of :meth:`bloch` at ``k``, sorted ascending.

        Examples
        --------
        >>> lat = Lattice.chain()
        >>> H = Hamiltonian(lat)
        >>> H.add_hopping(0, 0, (1,), 1.0)
        >>> import numpy as np
        >>> np.round(H.bands([np.pi]), 8)
        array([-2.])
        """
        return np.linalg.eigvalsh(self.bloch(k))


def build_ribbon(hamiltonian: Hamiltonian, open_direction: int, n_cells: int):
    """Build a ribbon/slab: periodic in-plane, open (finite) along ``open_direction``.

    Truncates the periodic boundary condition along one primitive-lattice
    direction, producing a quasi-1D (2D lattice) or quasi-2D (3D lattice)
    strip Hamiltonian as a function of the remaining reduced momenta. This
    exposes edge/surface states localized at the two open boundaries.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        A Hamiltonian built on a 2D (or higher) :class:`Lattice`.
    open_direction : int
        Index of the primitive-lattice direction to truncate.
    n_cells : int
        Number of unit cells stacked along ``open_direction``.

    Returns
    -------
    callable
        A function ``H_ribbon(k_parallel)`` returning the
        ``(n_cells * n_orbitals, n_cells * n_orbitals)`` ribbon Hamiltonian,
        where ``k_parallel`` is a reduced-momentum vector with one fewer
        component than the bulk lattice (the remaining periodic directions).

    Examples
    --------
    SSH chain cut open into a finite 20-site wire; the topological phase
    (``v < w``) hosts a mid-gap zero mode:

    >>> import numpy as np
    >>> from physicskit.condensed.models import ssh_lattice_hamiltonian
    >>> H = ssh_lattice_hamiltonian(v=0.5, w=1.0)
    >>> H_wire = build_ribbon(H, open_direction=0, n_cells=30)
    >>> spectrum = np.linalg.eigvalsh(H_wire(np.array([])))
    >>> bool(np.any(np.abs(spectrum) < 1e-6))
    True
    """
    lat = hamiltonian.lattice
    norb = lat.n_orbitals
    dim = lat.dim
    parallel_dirs = [d for d in range(dim) if d != open_direction]

    def H_ribbon(k_parallel):
        k_parallel = np.atleast_1d(np.asarray(k_parallel, dtype=float))
        N = n_cells * norb
        H = np.zeros((N, N), dtype=complex)
        for c in range(n_cells):
            for orb in range(norb):
                H[c * norb + orb, c * norb + orb] = hamiltonian.onsite[orb]
        for i, j, R, amp in hamiltonian._hoppings:
            n_open = R[open_direction]
            k_perp = np.array([R[d] for d in parallel_dirs], dtype=float)
            phase = np.exp(1j * np.dot(k_parallel, k_perp)) if parallel_dirs else 1.0 + 0j
            for c in range(n_cells):
                c2 = c + n_open
                if 0 <= c2 < n_cells:
                    H[c * norb + i, c2 * norb + j] += amp * phase
                    H[c2 * norb + j, c * norb + i] += np.conj(amp * phase)
        return H

    return H_ribbon


def build_finite_cluster(hamiltonian: Hamiltonian, n_cells, keep=None):
    """Build a fully finite (open in every direction) real-space cluster.

    Unlike :func:`build_ribbon`, which stays periodic along all but one
    lattice direction, this truncates *every* direction, producing a
    genuinely finite flake with real Cartesian site positions -- what a
    boundary-sensitive real-space plot (e.g.
    :func:`~physicskit.condensed.visualizers.plot_lattice_structure`) needs.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        A Hamiltonian built on a :class:`Lattice` of any dimension.
    n_cells : int or sequence of int
        Number of unit cells spanned along each lattice direction (the
        bounding-box shape). A single int is broadcast to every direction.
    keep : callable, optional
        ``keep(cell, orbital, position) -> bool`` predicate selecting which
        sites within the bounding box survive, for carving non-rectangular
        shapes (e.g. a disk) out of it. ``cell`` is the integer cell-index
        tuple, ``orbital`` the orbital index within the cell, and
        ``position`` its Cartesian coordinate. Defaults to keeping every
        site in the bounding box.

    Returns
    -------
    H : ndarray, shape (n_sites, n_sites)
        Dense, open-boundary real-space Hamiltonian.
    positions : ndarray, shape (n_sites, dim)
        Cartesian coordinates of the surviving sites, in the same order as
        ``H``'s rows/columns.
    bonds : list of tuple(int, int, complex)
        Surviving real-space hoppings ``(i, j, amplitude)``, indexing into
        ``positions``/``H``.

    Examples
    --------
    A 6-site open SSH chain (3 unit cells of 2 orbitals each) has exactly
    5 nearest-neighbor bonds, one fewer than a periodic ring would:

    >>> import numpy as np
    >>> from physicskit.condensed.models import ssh_lattice_hamiltonian
    >>> H_bulk = ssh_lattice_hamiltonian(v=0.5, w=1.0)
    >>> H, positions, bonds = build_finite_cluster(H_bulk, n_cells=3)
    >>> H.shape
    (6, 6)
    >>> len(bonds)
    5
    >>> positions.shape
    (6, 1)
    """
    lat = hamiltonian.lattice
    norb = lat.n_orbitals
    dim = lat.dim

    n_cells = np.atleast_1d(np.asarray(n_cells, dtype=int))
    if n_cells.size == 1 and dim > 1:
        n_cells = np.full(dim, n_cells[0])
    cell_ranges = [range(int(n)) for n in n_cells]

    index_map: dict[tuple, int] = {}
    positions_list = []
    for cell in itertools.product(*cell_ranges):
        for orb in range(norb):
            pos = (np.array(cell, dtype=float) + lat.orbitals[orb]) @ lat.lattice_vectors
            if keep is not None and not keep(cell, orb, pos):
                continue
            index_map[(cell, orb)] = len(positions_list)
            positions_list.append(pos)
    positions = np.array(positions_list)

    n_sites = len(positions_list)
    H = np.zeros((n_sites, n_sites), dtype=complex)
    for (_cell, orb), idx in index_map.items():
        H[idx, idx] = hamiltonian.onsite[orb]

    bonds: list[tuple[int, int, complex]] = []
    for i, j, R, amp in hamiltonian._hoppings:
        for cell in itertools.product(*cell_ranges):
            key_i = (cell, i)
            if key_i not in index_map:
                continue
            cell2 = tuple(c + r for c, r in zip(cell, R, strict=True))
            key_j = (cell2, j)
            if key_j not in index_map:
                continue
            idx_i, idx_j = index_map[key_i], index_map[key_j]
            H[idx_i, idx_j] += amp
            H[idx_j, idx_i] += np.conj(amp)
            bonds.append((idx_i, idx_j, amp))

    return H, positions, bonds


def apply_peierls_phase(positions: np.ndarray, hoppings, flux_quanta_per_plaquette: float, area_per_plaquette: float):
    """Apply a Peierls substitution phase to real-space hoppings for a uniform magnetic field.

    In the Landau gauge :math:`\\mathbf{A} = (-By, 0)`, a hopping from site
    at ``r_i`` to site at ``r_j`` acquires the phase
    :math:`\\exp\\!\\big(i\\frac{2\\pi\\Phi}{\\Phi_0}\\,\\bar{y}\\,(x_j-x_i)/a^2\\big)`
    where :math:`\\Phi/\\Phi_0` is the flux per plaquette in units of the flux
    quantum, evaluated at the bond midpoint :math:`\\bar y`.

    Parameters
    ----------
    positions : ndarray, shape (n_sites, 2)
        Real-space Cartesian coordinates of every site in a finite lattice.
    hoppings : list of tuple(int, int, complex)
        ``(i, j, amplitude)`` real-space bonds (indices into ``positions``).
    flux_quanta_per_plaquette : float
        Magnetic flux per unit-cell plaquette, in units of the flux quantum
        :math:`\\Phi_0 = h/e`.
    area_per_plaquette : float
        Real-space area of one plaquette, used to convert flux density to
        the vector potential prefactor.

    Returns
    -------
    list of tuple(int, int, complex)
        The same bonds with Peierls phases multiplied into the amplitude.

    Examples
    --------
    >>> import numpy as np
    >>> positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    >>> bonds = [(0, 1, 1.0)]
    >>> out = apply_peierls_phase(positions, bonds, flux_quanta_per_plaquette=0.25, area_per_plaquette=1.0)
    >>> bool(abs(out[0][2] - 1.0) < 1e-12)
    True
    """
    positions = np.asarray(positions, dtype=float)
    B_eff = 2 * np.pi * flux_quanta_per_plaquette / area_per_plaquette
    out = []
    for i, j, amp in hoppings:
        y_mid = 0.5 * (positions[i, 1] + positions[j, 1])
        dx = positions[j, 0] - positions[i, 0]
        phase = np.exp(1j * B_eff * y_mid * dx)
        out.append((i, j, amp * phase))
    return out
