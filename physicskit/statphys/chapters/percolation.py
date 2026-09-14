"""Site and bond percolation on the square lattice, via Hoshen-Kopelman cluster labeling.

Percolation theory studies the sudden, geometric phase transition that
appears when a fraction ``p`` of a lattice's sites (or bonds) are randomly
occupied: below a critical threshold :math:`p_c` only small, finite clusters
exist, while above it a single cluster spans the entire system. The
transition is purely geometric -- there is no energy scale or temperature --
yet it shares the full apparatus of critical phenomena: a sharp threshold,
power-law cluster-size distributions, and universal critical exponents.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Percolation2D", "hoshen_kopelman"]


def hoshen_kopelman(occupied):
    """Label connected clusters of occupied sites with the Hoshen-Kopelman algorithm.

    A single raster pass assigns provisional labels while a union-find
    structure tracks label equivalences (created whenever a site connects two
    previously separate provisional clusters); a second pass resolves every
    label to its cluster's root.

    Parameters
    ----------
    occupied : ndarray of shape (L, L), dtype bool
        Site occupation grid (open boundary conditions -- no wraparound).

    Returns
    -------
    ndarray of shape (L, L), dtype int64
        Cluster label at each site; ``0`` marks unoccupied sites, and
        occupied sites sharing a label belong to the same connected cluster.
        Labels are not necessarily contiguous integers.

    Examples
    --------
    >>> grid = np.array([[True, True, False], [False, True, False], [False, False, True]])
    >>> labels = hoshen_kopelman(grid)
    >>> bool(labels[0, 0] == labels[0, 1] == labels[1, 1])
    True
    >>> bool(labels[2, 2] not in (0, labels[0, 0]))
    True
    """
    L0, L1 = occupied.shape
    labels = np.zeros((L0, L1), dtype=np.int64)
    parent = [0]

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    next_label = 1
    for i in range(L0):
        for j in range(L1):
            if not occupied[i, j]:
                continue
            left = labels[i, j - 1] if j > 0 and occupied[i, j - 1] else 0
            up = labels[i - 1, j] if i > 0 and occupied[i - 1, j] else 0
            if left == 0 and up == 0:
                labels[i, j] = next_label
                parent.append(next_label)
                next_label += 1
            elif left != 0 and up == 0:
                labels[i, j] = find(left)
            elif left == 0 and up != 0:
                labels[i, j] = find(up)
            else:
                union(left, up)
                labels[i, j] = find(left)

    for i in range(L0):
        for j in range(L1):
            if labels[i, j] != 0:
                labels[i, j] = find(labels[i, j])
    return labels


class Percolation2D:
    """Site or bond percolation on a periodic-free (open-boundary) square lattice.

    Parameters
    ----------
    L : int, default=64
        Linear lattice size.
    p : float, default=0.5
        Occupation probability: fraction of sites (site percolation) or
        bonds (bond percolation) that are open.
    mode : {"site", "bond"}, default="site"
        Percolation type. Known exact thresholds for the square lattice are
        :math:`p_c \\approx 0.592746` (site) and :math:`p_c = 0.5` (bond,
        exact by self-duality).
    seed : int, optional
        Seed for reproducible lattice realizations.

    Attributes
    ----------
    labels : ndarray of shape (L, L)
        Cluster labels of the current realization, from
        :func:`hoshen_kopelman`.
    """

    #: Known threshold for square-lattice site percolation (numerical estimate).
    P_C_SITE = 0.592746
    #: Exact threshold for square-lattice bond percolation (self-duality).
    P_C_BOND = 0.5

    def __init__(self, L=64, p=0.5, mode="site", seed=None):
        if mode not in ("site", "bond"):
            raise ValueError("mode must be 'site' or 'bond'")
        self.L = L
        self.p = p
        self.mode = mode
        self._rng = np.random.default_rng(seed)
        self.labels = None
        self.generate()

    @property
    def p_c(self):
        """Threshold occupation probability for the current ``mode``."""
        return self.P_C_SITE if self.mode == "site" else self.P_C_BOND

    def generate(self, p=None):
        """Draw a fresh random realization and label its clusters.

        Parameters
        ----------
        p : float, optional
            Occupation probability to use; defaults to ``self.p``. If given,
            also updates ``self.p``.

        Returns
        -------
        ndarray of shape (L, L)
            The updated cluster ``labels`` array.
        """
        if p is not None:
            self.p = p
        if self.mode == "site":
            occupied = self._rng.random((self.L, self.L)) < self.p
        else:
            occupied = self._bond_occupation_to_sites(self.p)
        self.labels = hoshen_kopelman(occupied)
        return self.labels

    def _bond_occupation_to_sites(self, p):
        """Build an effective all-occupied site grid whose HK labeling respects bond connectivity.

        Bond percolation is reduced to Hoshen-Kopelman site labeling by
        working on a doubled lattice: real sites at even indices, and
        "bond sites" at the odd indices between them, present only when the
        corresponding bond is open.
        """
        L = self.L
        expanded = np.zeros((2 * L - 1, 2 * L - 1), dtype=bool)
        expanded[0::2, 0::2] = True  # all real sites present

        h_bonds = self._rng.random((L, L - 1)) < p  # horizontal bonds
        v_bonds = self._rng.random((L - 1, L)) < p  # vertical bonds
        expanded[0::2, 1::2] = h_bonds
        expanded[1::2, 0::2] = v_bonds
        return expanded

    def spanning_labels(self):
        """Cluster labels of the current realization that span top to bottom.

        For bond percolation the doubled lattice is used internally, but
        returned labels index the real (even-index) sites.

        Returns
        -------
        ndarray
            Labels present on both the first and last row of real sites.
        """
        if self.mode == "site":
            top_labels = set(self.labels[0, :]) - {0}
            bottom_labels = set(self.labels[-1, :]) - {0}
        else:
            top_labels = set(self.labels[0, 0::2]) - {0}
            bottom_labels = set(self.labels[-1, 0::2]) - {0}
        return np.array(sorted(top_labels & bottom_labels), dtype=np.int64)

    def spans(self):
        """Whether the current realization has a top-to-bottom spanning cluster."""
        return len(self.spanning_labels()) > 0

    def _real_labels(self):
        """Cluster labels restricted to real (non-bond-auxiliary) lattice sites."""
        return self.labels if self.mode == "site" else self.labels[0::2, 0::2]

    def largest_cluster_size(self):
        """Number of (real) sites in the largest cluster of the current realization."""
        sizes = self.cluster_size_distribution()
        return int(sizes.max()) if sizes.size else 0

    def cluster_size_distribution(self):
        """Sizes of every cluster in the current realization.

        At the percolation threshold, the number of clusters of size ``s``
        follows a power law :math:`n_s \\sim s^{-\\tau}` with the Fisher
        exponent :math:`\\tau = 187/91 \\approx 2.055` for 2D percolation --
        the geometric analogue of the divergent susceptibility seen at a
        thermal critical point.

        Returns
        -------
        ndarray
            Cluster sizes, in sites, one entry per cluster (including the
            spanning cluster if one exists).
        """
        real_labels = self._real_labels()
        labels = real_labels[real_labels != 0]
        if labels.size == 0:
            return np.array([], dtype=np.int64)
        _, counts = np.unique(labels, return_counts=True)
        return counts

    def spanning_probability(self, p_values, n_trials=100):
        """Monte Carlo estimate of the spanning probability :math:`P_{\\text{span}}(p)`.

        For each probability in ``p_values``, generates ``n_trials``
        independent realizations and reports the fraction that contain a
        top-to-bottom spanning cluster. As :math:`L \\to \\infty` this curve
        sharpens into a step function at :math:`p_c`.

        Parameters
        ----------
        p_values : array_like
            Occupation probabilities to sample.
        n_trials : int, default=100
            Independent realizations per probability.

        Returns
        -------
        p_values : ndarray
        P_span : ndarray
            Estimated spanning probability at each ``p``.

        Examples
        --------
        >>> perc = Percolation2D(L=20, seed=0)
        >>> p, P = perc.spanning_probability([0.3, 0.593, 0.8], n_trials=20)
        >>> bool(P[0] < P[2])
        True
        """
        p_values = np.asarray(p_values, dtype=np.float64)
        P_span = np.empty(len(p_values))
        for k, p in enumerate(p_values):
            hits = 0
            for _ in range(n_trials):
                self.generate(p)
                hits += self.spans()
            P_span[k] = hits / n_trials
        return p_values, P_span

    def fractal_dimension(self, n_trials=20):
        """Estimate the fractal dimension of the largest cluster at :math:`p = p_c`.

        A single lattice size gives no direct mass-versus-``L`` scaling, so
        instead this estimates :math:`d_f` from the mass-radius relation of
        the largest cluster in each realization,
        :math:`d_f = \\log(\\text{cluster size}) / \\log(R_g)`, where
        :math:`R_g` is the cluster's radius of gyration, averaged over
        trials. The 2D percolation universality class predicts
        :math:`d_f = 91/48 \\approx 1.896`.

        Parameters
        ----------
        n_trials : int, default=20
            Number of critical realizations to average over.

        Returns
        -------
        float
            Estimated fractal dimension.
        """
        dims = []
        for _ in range(n_trials):
            self.generate(self.p_c)
            real_labels = self._real_labels()
            labels = real_labels[real_labels != 0]
            if labels.size == 0:
                continue
            values, counts = np.unique(labels, return_counts=True)
            biggest = values[np.argmax(counts)]
            ii, jj = np.nonzero(real_labels == biggest)
            size = len(ii)
            if size < 4:
                continue
            Rg = np.sqrt(np.mean((ii - ii.mean()) ** 2 + (jj - jj.mean()) ** 2))
            if Rg > 0:
                dims.append(np.log(size) / np.log(Rg))
        return float(np.mean(dims)) if dims else float("nan")
