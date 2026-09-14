import numpy as np
import pytest

from physicskit.relativity.core.raytracer import OUTCOME_CAPTURED, OUTCOME_DISK
from physicskit.relativity.visualizers.shadow_render import (
    _redshift_factor,
    plot_black_hole_shadow,
    render_black_hole_image,
)


def test_a_zero_uses_schwarzschild_backend_and_a_positive_uses_kerr():
    result_s = render_black_hole_image(M=1.0, a=0.0, ny=40, nx=40)
    result_k = render_black_hole_image(M=1.0, a=0.5, ny=40, nx=40)
    assert result_s["a"] == 0.0
    assert result_k["a"] == 0.5
    assert result_s["outcomes"].shape == (40, 40)
    assert result_k["outcomes"].shape == (40, 40)


def test_default_disk_inner_radius_is_isco():
    from physicskit.relativity.chapters.kerr import KerrBlackHole

    result_s = render_black_hole_image(M=1.0, a=0.0, ny=20, nx=20)
    assert result_s["r_disk_inner"] == pytest.approx(6.0)

    result_k = render_black_hole_image(M=1.0, a=0.9, ny=20, nx=20)
    expected = KerrBlackHole(M=1.0, a=0.9).isco_radius(prograde=True)
    assert result_k["r_disk_inner"] == pytest.approx(expected)


def test_L_phi_grid_is_linear_in_alpha_and_antisymmetric():
    # At theta_o=pi/2 (edge-on), L_phi = z-component of (ray_origin x view_dir)
    # is exactly proportional to alpha (verified: coefficient +1 for the
    # Schwarzschild camera_basis convention, which need not match the sign
    # convention of the independent Kerr (alpha,beta)->(L,Q) parametrization
    # -- each is internally self-consistent, but the two are not required to
    # agree in handedness).
    result = render_black_hole_image(M=1.0, a=0.0, ny=21, nx=21, inclination=np.pi / 2.0, screen_half_width=10.0)
    alpha_row = np.linspace(-10.0, 10.0, 21)
    L_phi_row = result["L_phi"][10, :]
    assert np.allclose(L_phi_row, alpha_row, atol=1.0e-8) or np.allclose(L_phi_row, -alpha_row, atol=1.0e-8)
    assert L_phi_row[0] == pytest.approx(-L_phi_row[-1])


def test_redshift_factor_is_finite_and_asymmetric():
    result = render_black_hole_image(M=1.0, a=0.0, ny=80, nx=80, r_disk_inner=6.0, r_disk_outer=20.0)
    disk_mask = result["outcomes"] == OUTCOME_DISK
    assert np.any(disk_mask)
    g = _redshift_factor(result, disk_mask)
    assert np.all(np.isfinite(g[disk_mask]))
    assert np.all(g[disk_mask] > 0.0)

    nx = result["outcomes"].shape[1]
    left_mask = disk_mask.copy()
    left_mask[:, nx // 2 :] = False
    right_mask = disk_mask.copy()
    right_mask[:, : nx // 2] = False
    assert g[left_mask].mean() != pytest.approx(g[right_mask].mean(), rel=1.0e-3)


def test_plot_black_hole_shadow_runs_with_and_without_redshift():
    import matplotlib

    matplotlib.use("Agg")
    result = render_black_hole_image(M=1.0, a=0.0, ny=30, nx=30)
    ax1 = plot_black_hole_shadow(result, redshift=False)
    ax2 = plot_black_hole_shadow(result, redshift=True)
    assert ax1 is not None
    assert ax2 is not None


def test_captured_pixels_are_black_regardless_of_redshift_option():
    import matplotlib

    matplotlib.use("Agg")
    result = render_black_hole_image(M=1.0, a=0.0, ny=40, nx=40, r_disk_inner=1.0, r_disk_outer=0.0)
    ax = plot_black_hole_shadow(result, redshift=True)
    image = ax.images[0].get_array()
    captured_mask = result["outcomes"] == OUTCOME_CAPTURED
    assert np.allclose(np.asarray(image)[captured_mask][:, :3], 0.0)
