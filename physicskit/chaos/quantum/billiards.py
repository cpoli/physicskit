"""Quantum billiards: eigenvalues and eigenfunctions of physicskit.chaos's classical billiard shapes.

Solves the Dirichlet Helmholtz (quantum-particle-in-a-box) eigenproblem
``-laplacian(psi) = k^2 * psi`` (``psi = 0`` on the boundary) inside any
:class:`~physicskit.chaos.core.base_system.BilliardSystem` shape physicskit.chaos ships,
using a straightforward five-point finite-difference discretization -- lower
accuracy than a dedicated boundary-integral eigensolver, but simple, robust,
and shape-agnostic, since it only needs each billiard's existing
:meth:`~physicskit.chaos.core.base_system.BilliardSystem.boundary_polyline` to know
which grid points are inside.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from physicskit.chaos.core.base_system import BilliardSystem, get_init_params
from physicskit.chaos.exceptions import InvalidParameterError


def _split_loops(polyline: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    """Split a NaN-separated boundary polyline into its closed-loop components."""
    nan_rows = np.flatnonzero(np.isnan(polyline[:, 0])).tolist() + [polyline.shape[0]]
    loops = []
    start = 0
    for end in nan_rows:
        loop = polyline[start:end]
        if loop.shape[0] >= 3:
            loops.append(loop)
        start = end + 1
    return loops


def _ray_cast(loop: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Even-odd ray-casting point-in-polygon test, vectorized over query points."""
    inside = np.zeros(x.shape, dtype=bool)
    x1, y1 = loop[-1]
    for x2, y2 in loop:
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= crosses & (x < x_intersect)
        x1, y1 = x2, y2
    return inside


def points_in_billiard(billiard: BilliardSystem, points: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Test which of `points` lie inside a billiard's boundary.

    Applies the even-odd fill rule across every closed component of the
    billiard's :meth:`~physicskit.chaos.core.base_system.BilliardSystem.boundary_polyline`,
    so shapes with an excised interior scatterer (e.g.
    :class:`~physicskit.chaos.systems.billiards.SinaiBilliard`) are handled correctly
    regardless of each component's winding direction.

    Parameters
    ----------
    billiard : BilliardSystem
        Billiard whose boundary defines the domain.
    points : ndarray of float, shape (n, 2)
        Query points.

    Returns
    -------
    ndarray of bool, shape (n,)
        Whether each point lies inside the billiard.
    """
    loops = _split_loops(billiard.boundary_polyline())
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(x.shape, dtype=bool)
    for loop in loops:
        inside ^= _ray_cast(loop, x, y)
    return inside


class QuantumBilliard:
    """Dirichlet Helmholtz eigenstates of a billiard: "particle in a box" quantum chaos.

    A quantum particle confined to a chaotic billiard is one of the two
    textbook playgrounds of quantum chaos (alongside quantized maps like
    :class:`~physicskit.chaos.quantum.maps.QuantumKickedRotor`): its energy
    eigenvalues ``E_n = k_n^2`` (in units where ``hbar^2 / 2m = 1``) obey
    Weyl's law on average (see :meth:`weyl_counting_function`) but fluctuate
    around it in a way that reflects the underlying classical dynamics, and
    its eigenfunctions can "scar" -- show anomalously enhanced density -- on
    unstable classical periodic orbits.

    Built directly from any :class:`~physicskit.chaos.core.base_system.BilliardSystem`
    already in physicskit.chaos (no shape-specific code needed): a regular grid is
    laid over the shape's bounding box, points inside the boundary (via
    :func:`points_in_billiard`) become unknowns of a standard five-point
    finite-difference Laplacian, and the lowest eigenpairs of the resulting
    sparse, symmetric positive-definite matrix are found by shift-invert
    Lanczos iteration.

    Parameters
    ----------
    billiard : BilliardSystem
        The billiard shape to quantize.
    resolution : int, default 150
        Number of grid points along the longer side of the billiard's
        bounding box; the grid spacing (and hence both the accuracy and the
        cost of solving for eigenstates) scales with this.

    Attributes
    ----------
    billiard : BilliardSystem
        The billiard shape.
    resolution : int
        Grid resolution.

    Raises
    ------
    InvalidParameterError
        If `resolution` is smaller than 10.

    Notes
    -----
    Finite differences converge slowly (error ``O(h^2)``) and, being defined
    on a Cartesian grid, represent curved or slanted boundaries only
    approximately; treat eigenvalues as accurate to a few percent at the
    default `resolution`; increase `resolution` for tighter results, at
    roughly quadratic cost in memory and eigensolver runtime.
    """

    def __init__(self, billiard: BilliardSystem, resolution: int = 150):
        if resolution < 10:
            raise InvalidParameterError("resolution must be at least 10")
        self.billiard = billiard
        self.resolution = int(resolution)
        self._build_grid()

    def _build_grid(self) -> None:
        polyline = self.billiard.boundary_polyline()
        finite = polyline[~np.isnan(polyline[:, 0])]
        xmin, xmax = finite[:, 0].min(), finite[:, 0].max()
        ymin, ymax = finite[:, 1].min(), finite[:, 1].max()

        h = max(xmax - xmin, ymax - ymin) / self.resolution
        nx = int(np.ceil((xmax - xmin) / h)) + 3
        ny = int(np.ceil((ymax - ymin) / h)) + 3
        xs = (xmin - h) + h * np.arange(nx)
        ys = (ymin - h) + h * np.arange(ny)

        x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
        points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        inside = points_in_billiard(self.billiard, points).reshape(x_grid.shape)

        self._h = h
        self._xs, self._ys = xs, ys
        self._x_grid, self._y_grid = x_grid, y_grid
        self._inside = inside

    def eigenstates(self, n_states: int = 6) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve for the lowest `n_states` Dirichlet eigenpairs.

        Parameters
        ----------
        n_states : int, default 6
            Number of lowest eigenstates to compute.

        Returns
        -------
        eigenvalues : ndarray of float, shape (n_states,)
            Eigenvalues ``k_n^2``, ascending.
        eigenfunctions : ndarray of float, shape (n_states, nx, ny)
            Eigenfunctions on the grid returned by :meth:`grid`, each
            normalized to a peak absolute value of 1; grid points outside
            the billiard are ``nan`` (so ``plt.imshow``/``pcolormesh`` leave
            them blank).
        """
        inside = self._inside
        nx, ny = inside.shape
        index = -np.ones((nx, ny), dtype=np.int64)
        interior_ij = np.argwhere(inside)
        index[inside] = np.arange(interior_ij.shape[0])
        n_interior = interior_ij.shape[0]
        h2 = self._h**2

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for k, (i, j) in enumerate(interior_ij):
            rows.append(k)
            cols.append(k)
            data.append(4.0 / h2)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny and inside[ni, nj]:
                    rows.append(k)
                    cols.append(int(index[ni, nj]))
                    data.append(-1.0 / h2)

        laplacian = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_interior, n_interior))
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(laplacian, k=n_states, sigma=0.0, which="LM")
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        eigenfunctions = np.full((n_states, nx, ny), np.nan)
        for m in range(n_states):
            full = np.full((nx, ny), np.nan)
            full[inside] = eigenvectors[:, m]
            full /= np.nanmax(np.abs(full))
            eigenfunctions[m] = full
        return eigenvalues, eigenfunctions

    def wavenumbers(self, n_states: int = 6) -> NDArray[np.float64]:
        """Wavenumbers ``k_n = sqrt(eigenvalue)`` of the lowest `n_states` states.

        Parameters
        ----------
        n_states : int, default 6
            Number of lowest eigenstates to compute.

        Returns
        -------
        ndarray of float, shape (n_states,)
            Wavenumbers, ascending.
        """
        eigenvalues, _ = self.eigenstates(n_states)
        return np.sqrt(eigenvalues)

    def grid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The Cartesian grid eigenfunctions are returned on.

        Returns
        -------
        X, Y : ndarray of float, shape (nx, ny)
            Grid coordinates (as from ``np.meshgrid(..., indexing="ij")``).
        """
        return self._x_grid, self._y_grid

    def area(self) -> float:
        """Interior area, estimated by counting grid points inside the boundary.

        Returns
        -------
        float
            Approximate billiard area.
        """
        return float(np.sum(self._inside)) * self._h**2

    def weyl_counting_function(self, k: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """Weyl's law: the average number of eigenvalues below wavenumber `k`.

        ``N(k) ~ Area * k^2 / (4*pi) - Perimeter * k / (4*pi)``, the leading
        area term plus the first-order (Dirichlet) boundary correction. Real
        billiards' actual eigenvalue counts fluctuate around this smooth
        curve; how they fluctuate (level "rigidity") is a hallmark of
        whether the classical billiard is integrable or chaotic -- exactly
        the kind of spectral statistic ``physicskit.rmt`` is built to quantify, using
        :meth:`wavenumbers` as its input.

        Parameters
        ----------
        k : float or array_like of float
            Wavenumber(s) at which to evaluate the counting function.

        Returns
        -------
        float or ndarray of float
            Expected (smoothed) number of eigenvalues below `k`.
        """
        area = self.area()
        perimeter = self.billiard.perimeter()
        return area * k**2 / (4.0 * np.pi) - perimeter * k / (4.0 * np.pi)

    def __repr__(self) -> str:
        parts = [f"{name}={value!r}" for name, value in get_init_params(self).items()]
        return f"{type(self).__name__}({', '.join(parts)})"
