"""Visualizations of black hole shadows and gravitationally lensed accretion disks.

:func:`render_black_hole_image` dispatches on the spin parameter ``a``: at
``a=0`` it uses the fast, exact 2D Schwarzschild orbit-equation raytracer in
:mod:`physicskit.relativity.core.raytracer`; at ``a>0`` it uses the more general Carter-
formalism Kerr raytracer in :mod:`physicskit.relativity.core.kerr_raytracer`, which
reproduces the same physics at ``a=0`` (and is slower, hence the dedicated
fast path). Both report each pixel's disk-frame azimuthal angular momentum
:math:`L_\\phi` alongside the usual outcome/hit-radius arrays, letting
:func:`plot_black_hole_shadow` optionally color the accretion disk by its
combined gravitational and Doppler redshift rather than radius alone.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from physicskit.relativity.core.kerr_raytracer import render_kerr_shadow_image
from physicskit.relativity.core.raytracer import (
    OUTCOME_CAPTURED,
    OUTCOME_DISK,
    OUTCOME_ESCAPED,
    OUTCOME_UNRESOLVED,
    camera_basis,
    render_shadow_image,
)

__all__ = ["animate_shadow_spin_sweep", "plot_black_hole_shadow", "render_black_hole_image"]


def _pixel_grids(ny, nx, screen_half_width, screen_half_height):
    """Per-pixel (alpha, beta) screen coordinates, each of shape (ny, nx)."""
    alpha = screen_half_width * (2.0 * np.arange(nx) / (nx - 1) - 1.0)
    beta = screen_half_height * (1.0 - 2.0 * np.arange(ny) / (ny - 1))
    return np.meshgrid(alpha, beta)


def render_black_hole_image(
    M,
    a=0.0,
    observer_distance=500.0,
    inclination=1.3,
    screen_half_width=15.0,
    ny=200,
    nx=200,
    r_disk_inner=None,
    r_disk_outer=None,
    n_steps=None,
    **kwargs,
):
    """Backward ray-trace a black hole shadow and thin equatorial accretion disk.

    Parameters
    ----------
    M : float
        Black hole mass, in geometrized units.
    a : float, default=0.0
        Kerr spin parameter, :math:`0 \\le a < M`. ``a=0`` uses the fast,
        exact Schwarzschild orbit-equation raytracer; ``a>0`` uses the more
        general (and slower) Kerr Carter-formalism raytracer.
    observer_distance : float, default=500.0
        Observer distance from the black hole, in units of ``M`` (should be
        :math:`\\gg M` for the flat, far-away camera approximation to hold).
    inclination : float, default=1.3
        Observer inclination from the disk's normal axis, in radians
        (``pi/2`` is edge-on; ``1.3`` rad :math:`\\approx 74^\\circ` gives a
        classic, strongly lensed oblique view).
    screen_half_width : float, default=15.0
        Half-width of the camera screen, in units of ``M``.
    ny, nx : int, default=200
        Image resolution, in pixels.
    r_disk_inner, r_disk_outer : float, optional
        Accretion disk radii. Default to the ISCO (prograde, for Kerr) and
        ``20M``.
    n_steps : int, optional
        Maximum steps per ray; forwarded to the underlying raytracer
        (defaults differ between the Schwarzschild and Kerr backends).
    **kwargs
        Extra keyword arguments forwarded to the underlying raytracer
        (e.g. ``dphi`` for Schwarzschild, ``dlambda_max`` for Kerr).

    Returns
    -------
    dict
        Keys ``"outcomes"``, ``"hit_radii"``, ``"L_phi"`` (disk-frame
        azimuthal angular momentum of each pixel's photon, used for
        redshift coloring), ``"extent"``, ``"r_disk_inner"``,
        ``"r_disk_outer"``, ``"M"``, ``"a"``, ready for
        :func:`plot_black_hole_shadow`.

    Examples
    --------
    >>> result = render_black_hole_image(M=1.0, ny=20, nx=20)
    >>> result["outcomes"].shape
    (20, 20)
    """
    if r_disk_inner is None:
        if a == 0.0:
            r_disk_inner = 6.0 * M
        else:
            from physicskit.relativity.chapters.kerr import KerrBlackHole

            r_disk_inner = KerrBlackHole(M=M, a=a).isco_radius(prograde=True)
    if r_disk_outer is None:
        r_disk_outer = 20.0 * M

    alpha_grid, beta_grid = _pixel_grids(ny, nx, screen_half_width, screen_half_width)

    if a == 0.0:
        view_dir, right, up = camera_basis(inclination)
        obs_pos = -view_dir * observer_distance
        outcomes, hit_radii = render_shadow_image(
            ny,
            nx,
            screen_half_width,
            screen_half_width,
            obs_pos,
            right,
            up,
            view_dir,
            M,
            r_disk_inner,
            r_disk_outer,
            n_steps=n_steps or 3000,
            **kwargs,
        )
        # L_phi = z-component of (ray_origin x view_dir); disk axis is +Z.
        # Only the x,y components of each vector enter a cross product's z-component.
        ray_origin_x = obs_pos[0] + alpha_grid * right[0] + beta_grid * up[0]
        ray_origin_y = obs_pos[1] + alpha_grid * right[1] + beta_grid * up[1]
        L_phi = ray_origin_x * view_dir[1] - ray_origin_y * view_dir[0]
    else:
        outcomes, hit_radii = render_kerr_shadow_image(
            ny,
            nx,
            screen_half_width,
            screen_half_width,
            observer_distance,
            inclination,
            a,
            M,
            r_disk_inner,
            r_disk_outer,
            n_steps=n_steps or 4000,
            **kwargs,
        )
        L_phi = -alpha_grid * np.sin(inclination)

    return {
        "outcomes": outcomes,
        "hit_radii": hit_radii,
        "L_phi": L_phi,
        "extent": (-screen_half_width, screen_half_width, -screen_half_width, screen_half_width),
        "r_disk_inner": r_disk_inner,
        "r_disk_outer": r_disk_outer,
        "M": M,
        "a": a,
        "inclination": inclination,
    }


def plot_black_hole_shadow(result, ax=None, cmap_disk="inferno", redshift=False):
    """Render a :func:`render_black_hole_image` result as a matplotlib image.

    Captured and unresolved (deep near-critical) pixels are drawn black
    (the shadow); escaped pixels are drawn as a dark background; disk-hit
    pixels are colored by a temperature-like radius gradient, optionally
    modulated by their combined gravitational and Doppler redshift.

    Parameters
    ----------
    result : dict
        Output of :func:`render_black_hole_image`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created if not given.
    cmap_disk : str, default="inferno"
        Colormap used for the accretion disk's radius gradient.
    redshift : bool, default=False
        If True, scale each disk pixel's brightness by :math:`g^3` (the
        approximate relativistic Doppler-beaming boost of bolometric
        intensity for a locally isotropic emitter), so the side of the
        disk rotating toward the observer appears brighter and the
        receding side dimmer -- the signature asymmetric brightness seen
        in real black hole images.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    image = shadow_image_array(result, cmap_disk=cmap_disk, redshift=redshift)
    ax.imshow(image, extent=result["extent"], origin="upper")
    ax.set_xlabel("impact parameter x [M]")
    ax.set_ylabel("impact parameter y [M]")
    title = "Black hole shadow and gravitationally lensed accretion disk"
    if redshift:
        title += " (Doppler-shaded)"
    ax.set_title(title)
    return ax


def shadow_image_array(result, cmap_disk="inferno", redshift=False):
    """Build the RGB image array a :func:`render_black_hole_image` result maps to.

    Factored out of :func:`plot_black_hole_shadow` so
    :func:`animate_shadow_spin_sweep` can reuse the identical shadow/disk
    coloring logic when assembling animation frames.

    Parameters
    ----------
    result : dict
        Output of :func:`render_black_hole_image`.
    cmap_disk : str, default="inferno"
        Colormap used for the accretion disk's radius gradient.
    redshift : bool, default=False
        If True, scale each disk pixel's brightness by the combined
        gravitational and Doppler redshift, exactly as in
        :func:`plot_black_hole_shadow`.

    Returns
    -------
    ndarray of shape (ny, nx, 3)
        RGB image, clipped to ``[0, 1]``.
    """
    outcomes = result["outcomes"]
    hit_radii = result["hit_radii"]
    image = np.zeros((*outcomes.shape, 3))

    escaped_mask = outcomes == OUTCOME_ESCAPED
    image[escaped_mask] = [0.02, 0.02, 0.06]

    disk_mask = outcomes == OUTCOME_DISK
    if np.any(disk_mask):
        r_in, r_out = result["r_disk_inner"], result["r_disk_outer"]
        norm_r = np.clip((hit_radii - r_in) / max(r_out - r_in, 1e-12), 0.0, 1.0)
        cmap = plt.get_cmap(cmap_disk)
        colors = cmap(1.0 - norm_r)[..., :3]

        if redshift:
            g = _redshift_factor(result, disk_mask)
            brightness = np.clip(g**3, 0.15, 4.0)
            brightness_norm = brightness / np.max(brightness[disk_mask])
            colors = colors * brightness_norm[..., None]

        image[disk_mask] = colors[disk_mask]

    captured_mask = (outcomes == OUTCOME_CAPTURED) | (outcomes == OUTCOME_UNRESOLVED)
    image[captured_mask] = [0.0, 0.0, 0.0]

    return np.clip(image, 0.0, 1.0)


def animate_shadow_spin_sweep(
    M=1.0,
    a_values=None,
    observer_distance=500.0,
    inclination=1.3,
    screen_half_width=15.0,
    ny=100,
    nx=100,
    r_disk_inner=None,
    r_disk_outer=20.0,
    redshift=True,
    cmap_disk="inferno",
    interval=200,
):
    """Animate a black hole's shadow and lensed accretion disk as the spin :math:`a` is swept.

    Backward ray-traces a full image at each spin value with
    :func:`render_black_hole_image`, then plays the resulting frames back
    with :class:`~matplotlib.animation.FuncAnimation`: the shadow visibly
    shrinks and flattens toward the equatorial photon orbit, the inner
    edge of the disk creeps inward toward the shrinking ISCO, and (with
    ``redshift=True``) the Doppler-brightened approaching side of the disk
    grows more pronounced, as :math:`a \\to M`.

    Parameters
    ----------
    M : float, default=1.0
        Black hole mass, in geometrized units.
    a_values : array_like of float, optional
        Spin values to sweep through, each satisfying ``0 <= a < M``.
        Defaults to 50 values from 0 to ``0.98 * M``.
    observer_distance, inclination, screen_half_width, ny, nx : see :func:`render_black_hole_image`.
    r_disk_inner : float, optional
        Inner disk radius; defaults to the ISCO at each frame's spin
        (:meth:`~physicskit.relativity.chapters.kerr.KerrBlackHole.isco_radius`
        at ``a=0`` reduces to the Schwarzschild ISCO, :math:`6M`), so the
        disk visibly creeps inward as the spin increases.
    r_disk_outer : float, default=20.0
        Outer disk radius, held fixed across frames.
    redshift : bool, default=True
        Doppler/gravitational-redshift-shade the disk, as in
        :func:`plot_black_hole_shadow`.
    cmap_disk : str, default="inferno"
        Colormap for the disk.
    interval : int, default=200
        Delay between frames, in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Assign it to a variable to keep it alive, and display it with
        ``plt.show()`` or save it with ``anim.save(...)``.
    """
    from physicskit.relativity.chapters.kerr import KerrBlackHole
    from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole

    if a_values is None:
        a_values = np.linspace(0.0, 0.98 * M, 50)

    frames = []
    for a in a_values:
        r_in = r_disk_inner
        if r_in is None:
            r_in = SchwarzschildBlackHole(M=M).isco_radius if a == 0.0 else KerrBlackHole(M=M, a=a).isco_radius(prograde=True)
        result = render_black_hole_image(
            M=M,
            a=a,
            observer_distance=observer_distance,
            inclination=inclination,
            screen_half_width=screen_half_width,
            ny=ny,
            nx=nx,
            r_disk_inner=r_in,
            r_disk_outer=r_disk_outer,
        )
        image = shadow_image_array(result, cmap_disk=cmap_disk, redshift=redshift)
        frames.append((image, result["extent"], r_in))

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0][0], extent=frames[0][1], origin="upper")
    ax.set_xlabel("impact parameter x [M]")
    ax.set_ylabel("impact parameter y [M]")
    title = ax.set_title(f"a = {a_values[0]:.2f} M, ISCO = {frames[0][2]:.2f} M")

    def update(frame_idx):
        image, extent, r_in = frames[frame_idx]
        im.set_data(image)
        im.set_extent(extent)
        title.set_text(f"a = {a_values[frame_idx]:.2f} M, ISCO = {r_in:.2f} M")
        return im, title

    return FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)


def _redshift_factor(result, disk_mask):
    """Compute the disk-frame redshift factor g at every disk-hit pixel."""
    M, a = result["M"], result["a"]
    hit_radii = result["hit_radii"]
    L_phi = result["L_phi"]
    g = np.ones_like(hit_radii)
    if a == 0.0:
        from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole

        bh = SchwarzschildBlackHole(M=M)
        g[disk_mask] = bh.disk_redshift_factor(hit_radii[disk_mask], L_phi[disk_mask])
    else:
        from physicskit.relativity.chapters.kerr import KerrBlackHole

        bh = KerrBlackHole(M=M, a=a)
        g[disk_mask] = bh.disk_redshift_factor(hit_radii[disk_mask], L_phi[disk_mask], prograde=True)
    return g
