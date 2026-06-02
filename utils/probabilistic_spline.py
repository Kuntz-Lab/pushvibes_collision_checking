#!/usr/bin/env python3
"""Artery point cloud -> Gaussian B-spline (mu_w, Sigma_w, basis matrices Phi).

This is the bridge that turns a real artery point cloud into the Gaussian
spline inputs the probabilistic ranker needs (Section 3 of the writeup). The
deterministic side (``spline_fit.py``) fits a single ``splprep`` curve and
treats the artery as sitting exactly on it; here we instead fit an explicit
**linear B-spline regression** so we get a full Gaussian over the control
points and can propagate that uncertainty forward.

Fit (per spatial coordinate, sharing one basis):

    nodes  = ordered Mean-Shift skeleton (reused from spline_fit)
    u      = normalized chord-length parameter of the nodes, in [0, 1]
    B      = BSpline.design_matrix(u, knots, k)        (n_nodes, n_ctrl)
    R      = second-difference curvature penalty       (n_ctrl, n_ctrl)
    A      = B^T B + penalty * R
    c      = A^{-1} B^T y                               (ridge control points)

The covariance follows the ridge posterior form ``Sigma_c = alpha_c * A^{-1}``.
The scale ``alpha_c`` is set from the *point cloud's* scatter about the mean
curve rather than from the skeleton nodes: the nodes are denoised Mean-Shift
cluster centers, so the spline fits them almost exactly and their residual
variance collapses to ~0 (which would make every push trivially "safe"). The
raw points carry the real, occlusion-driven positional uncertainty, so we scale
``A^{-1}`` so the average spatial variance over u in [0, 1] matches the
per-coordinate point-cloud spread (floored so an unusually tight cloud still
carries uncertainty). Because ``A^{-1}``'s shape inflates where the basis
support is sparse -- and the basis is evaluated with ``extrapolate=True`` --
uncertainty grows out along the obscured continuation past the visible ends,
exactly the behavior the writeup wants.

Stacking convention (must match ``basis_matrices``): the control parameters are
``mu_w = [c_x ; c_y ; c_z]`` of length ``3 * n_ctrl``; coordinates are modeled as
independent, so ``Sigma_w`` is block-diagonal with one ``A^{-1}``-shaped block
per coordinate.

``splprep`` / ``spline_fit.py`` are left untouched; only the node ordering
helper is reused.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

try:  # importable as a package (from the visualizers) ...
    from utils.spline_fit import _order_skeleton_nodes
except ImportError:  # ... and runnable as a standalone script from utils/
    from spline_fit import _order_skeleton_nodes


@dataclass
class GaussianSpline:
    """A Gaussian over cubic B-spline control points.

    Attributes:
        mu_w: (3K,) mean control parameters, stacked [c_x; c_y; c_z].
        Sigma_w: (3K, 3K) block-diagonal covariance of the control parameters.
        knots: (K + k + 1,) clamped knot vector on u in [0, 1].
        k: spline degree (3 = cubic).
        n_ctrl: number of control points per coordinate (K).
    """
    mu_w: np.ndarray
    Sigma_w: np.ndarray
    knots: np.ndarray
    k: int
    n_ctrl: int


def _clamped_knots(n_ctrl, k):
    """Clamped knot vector on [0, 1] for ``n_ctrl`` degree-``k`` control points."""
    n_interior = n_ctrl - k - 1
    if n_interior < 0:
        raise ValueError(f"n_ctrl={n_ctrl} too small for degree k={k} "
                         f"(need n_ctrl >= k + 1).")
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    return np.concatenate([np.zeros(k + 1), interior, np.ones(k + 1)])


def _second_difference_penalty(n_ctrl):
    """Second-difference curvature penalty R = D2^T D2  (n_ctrl, n_ctrl)."""
    D2 = np.zeros((n_ctrl - 2, n_ctrl))
    for i in range(n_ctrl - 2):
        D2[i, i:i + 3] = [1.0, -2.0, 1.0]
    return D2.T @ D2


def _design_matrix(u, knots, k):
    """Dense B-spline design matrix (len(u), n_ctrl), extrapolating outside [0,1]."""
    u = np.atleast_1d(np.asarray(u, dtype=float))
    return BSpline.design_matrix(u, knots, k, extrapolate=True).toarray()


def fit_gaussian_centerline(points, bandwidth=0.004, n_ctrl=10, penalty=1e-4,
                            k=3, sigma_floor_frac=0.25, n_resid_samples=400,
                            resid_max_points=4000):
    """Fit a Gaussian cubic B-spline to the centerline of a tube point cloud.

    Args:
        points: (N, 3) artery point cloud.
        bandwidth: Mean-Shift radius for skeleton extraction (~ tube radius, m).
        n_ctrl: number of B-spline control points per coordinate (clamped down
            if there are too few skeleton nodes to support it).
        penalty: ridge weight on the second-difference curvature penalty
            (higher = stiffer mean curve).
        k: spline degree (3 = cubic).
        sigma_floor_frac: floor on the per-coordinate position std as a fraction
            of ``bandwidth`` (keeps the covariance non-degenerate).
        n_resid_samples: samples along the mean curve used to estimate the
            point-cloud residual spread.
        resid_max_points: cap on points used for the residual estimate (random
            subsample above this, for speed).

    Returns:
        GaussianSpline with mu_w, Sigma_w, knots, k, n_ctrl.
    """
    nodes = np.asarray(_order_skeleton_nodes(points, bandwidth), dtype=float)
    n_nodes = len(nodes)
    if n_nodes < k + 2:
        raise ValueError(f"Only {n_nodes} skeleton nodes; need at least "
                         f"{k + 2} for a degree-{k} fit (lower `bandwidth`).")
    # Keep the regression over-determined: at most one control point fewer than
    # the number of data nodes (and never fewer than k + 1).
    max_ctrl = max(k + 1, n_nodes - 1)
    if n_ctrl > max_ctrl:
        print(f"  probabilistic_spline: clamping n_ctrl {n_ctrl} -> {max_ctrl} "
              f"({n_nodes} skeleton nodes available).")
        n_ctrl = max_ctrl

    # Normalized chord-length parameterization of the ordered nodes.
    seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)])
    u = u / u[-1]

    knots = _clamped_knots(n_ctrl, k)
    B = _design_matrix(u, knots, k)              # (n_nodes, n_ctrl)
    R = _second_difference_penalty(n_ctrl)       # (n_ctrl, n_ctrl)
    A = B.T @ B + penalty * R
    A_inv = np.linalg.inv(A)
    K = n_ctrl

    # Mean control points per coordinate (ridge fit to the ordered nodes).
    coef_mat = np.array([A_inv @ (B.T @ nodes[:, c]) for c in range(3)])  # (3,K)
    mu_w = coef_mat.reshape(-1)                  # [c_x; c_y; c_z]

    # Covariance scale from the point-cloud scatter about the mean curve (the
    # nodes are denoised, so their residual collapses the covariance -- see the
    # module docstring). Estimate a per-coordinate spatial variance, floored.
    u_dense = np.linspace(0.0, 1.0, n_resid_samples)
    b_dense = _design_matrix(u_dense, knots, k)  # (S, K)
    curve = b_dense @ coef_mat.T                 # (S, 3) mean curve
    pts = np.asarray(points, dtype=float)
    if len(pts) > resid_max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), resid_max_points,
                                                   replace=False)]
    nn = np.argmin(np.linalg.norm(pts[:, None, :] - curve[None], axis=-1), axis=1)
    sigma2 = ((pts - curve[nn]) ** 2).mean(axis=0)            # (3,)
    sigma2 = np.maximum(sigma2, (sigma_floor_frac * bandwidth) ** 2)

    # Map spatial variance back to control points: Sigma_w^c = alpha_c * A^{-1},
    # with alpha_c chosen so the mean spatial variance over u in [0, 1] equals
    # sigma2_c. A^{-1}'s shape grows the uncertainty toward sparse / extrapolated
    # regions.
    q = np.einsum("si,ij,sj->s", b_dense, A_inv, b_dense)     # (S,)
    mean_q = float(q.mean())
    Sigma_w = np.zeros((3 * K, 3 * K))
    for c in range(3):
        Sigma_w[c * K:(c + 1) * K, c * K:(c + 1) * K] = (sigma2[c] / mean_q) * A_inv

    return GaussianSpline(mu_w=mu_w, Sigma_w=Sigma_w, knots=knots, k=k,
                          n_ctrl=K)


def basis_matrices(u_values, gspline):
    """Per-waypoint basis matrices Phi (M, 3, 3K) at parameter values ``u``.

    ``Phi_u`` is block-diagonal: row ``i`` selects coordinate ``i`` from the
    matching control-point block, so ``Phi_u @ mu_w`` is the spline point at
    ``u`` (consistent with the [c_x; c_y; c_z] stacking). Evaluated with
    extrapolation so ``u`` outside [0, 1] gives the projected continuation.
    """
    b = _design_matrix(u_values, gspline.knots, gspline.k)  # (M, K)
    m, K = b.shape
    Phi = np.zeros((m, 3, 3 * K))
    for coord in range(3):
        Phi[:, coord, coord * K:(coord + 1) * K] = b
    return Phi


def mean_curve(u_values, gspline):
    """Spline mean points (M, 3) at parameter values ``u`` (for plotting)."""
    b = _design_matrix(u_values, gspline.knots, gspline.k)  # (M, K)
    K = gspline.n_ctrl
    c = gspline.mu_w.reshape(3, K)                          # rows: x, y, z
    return (b @ c.T)                                        # (M, 3)


def _push_waypoints(cand_starts, cand_ends, n_waypoints):
    """(N, M, 3) points linearly interpolated from each push start to its end."""
    s = np.asarray(cand_starts, dtype=float)
    e = np.asarray(cand_ends, dtype=float)
    frac = np.linspace(0.0, 1.0, n_waypoints)
    return s[:, None, :] + frac[None, :, None] * (e - s)[:, None, :]


def discretize_candidates(cand_starts, cand_ends, gspline, n_waypoints=12,
                          pairing="nearest", u_range=(0.0, 1.0), n_grid=200,
                          grid_u_range=None):
    """Discretize candidate pushes into waypoints + their spline basis matrices.

    Builds, for each push, ``n_waypoints`` points linearly interpolated from its
    start to its end, plus the basis matrices ``Phi`` that pair each waypoint
    with a point on the spline. Two pairing schemes:

    * ``"nearest"`` (default): each push waypoint is paired with the spline point
      *closest* to it (the same centerline-to-centerline geometry the
      deterministic ranker uses). ``Phi`` is then per-candidate, shape
      (N, M, 3, 3K). The nearest spline point is found over a dense grid of
      ``n_grid`` parameter values spanning ``grid_u_range`` (defaults to a span
      that includes the extrapolated continuation).
    * ``"index"`` (literal Algorithm 2): push waypoint ``t`` is paired with
      spline parameter ``u_t = linspace(*u_range, M)[t]`` *by index*, and ``Phi``
      is shared across all candidates, shape (M, 3, 3K). On data where pushes are
      far from the corresponding-index spline point this collapses to "all safe".

    Args:
        cand_starts, cand_ends: (N, 3) push endpoints (camera frame, meters).
        gspline: the fitted GaussianSpline.
        n_waypoints: M waypoints per push.
        pairing: "nearest" or "index".
        u_range: (index pairing) spline-parameter span paired with push fraction.
        n_grid: (nearest pairing) number of spline samples for the search.
        grid_u_range: (nearest pairing) (u0, u1) span to search; defaults to
            (-0.75, 1.75) so the obscured continuation is reachable.

    Returns:
        waypoints: (N, M, 3) interpolated push waypoints.
        Phi: (M, 3, 3K) for "index", or (N, M, 3, 3K) for "nearest".
    """
    waypoints = _push_waypoints(cand_starts, cand_ends, n_waypoints)

    if pairing == "index":
        u = np.linspace(u_range[0], u_range[1], n_waypoints)
        return waypoints, basis_matrices(u, gspline)

    if pairing != "nearest":
        raise ValueError(f"unknown pairing {pairing!r} (use 'nearest'/'index').")

    if grid_u_range is None:
        grid_u_range = (-0.75, 1.75)
    u_grid = np.linspace(grid_u_range[0], grid_u_range[1], n_grid)
    curve = mean_curve(u_grid, gspline)                 # (G, 3)
    b_grid = _design_matrix(u_grid, gspline.knots, gspline.k)  # (G, K)

    # Nearest grid sample for every (push, waypoint).
    dist = np.linalg.norm(waypoints[:, :, None, :] - curve[None, None, :, :],
                          axis=-1)                       # (N, M, G)
    idx = np.argmin(dist, axis=2)                        # (N, M)
    b_sel = b_grid[idx]                                  # (N, M, K)

    n, m, K = b_sel.shape
    Phi = np.zeros((n, m, 3, 3 * K))
    for coord in range(3):
        Phi[:, :, coord, coord * K:(coord + 1) * K] = b_sel
    return waypoints, Phi


# Backwards-compatible alias: index pairing (literal Algorithm 2).
def candidate_waypoints(cand_starts, cand_ends, gspline, n_waypoints=12,
                        u_range=(0.0, 1.0)):
    """Index-pairing discretization (see ``discretize_candidates``)."""
    return discretize_candidates(cand_starts, cand_ends, gspline,
                                 n_waypoints=n_waypoints, pairing="index",
                                 u_range=u_range)


if __name__ == "__main__":
    # Self-test on a synthetic noisy tube: shapes, SPD covariance, mean curve
    # tracks the tube, and uncertainty grows toward the extrapolated ends.
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 800)
    centerline = np.vstack((t, np.sin(t) * 2, np.cos(t / 2) * 2)).T
    tube = centerline + rng.normal(scale=0.3, size=centerline.shape)

    g = fit_gaussian_centerline(tube, bandwidth=0.8, n_ctrl=10, penalty=1e-3)
    K = g.n_ctrl
    print(f"mu_w shape {g.mu_w.shape}, Sigma_w shape {g.Sigma_w.shape}, "
          f"n_ctrl {K}")

    eig = np.linalg.eigvalsh(g.Sigma_w)
    print(f"Sigma_w SPD: {bool(eig.min() > 0)} (min eig {eig.min():.2e})")

    # Mean curve should track the underlying centerline.
    u_fit = np.linspace(0, 1, 200)
    curve = mean_curve(u_fit, g)
    d_to_tube = np.min(np.linalg.norm(
        curve[:, None, :] - centerline[None, :, :], axis=-1), axis=1)
    print(f"mean curve max dist to true centerline: {d_to_tube.max():.3f} "
          f"(tube noise ~0.3)")

    # Spatial uncertainty: largest eigenvalue of Sigma_t interior vs extrapolated.
    def lam_max(u):
        Phi = basis_matrices([u], g)[0]
        Sigma_t = Phi @ g.Sigma_w @ Phi.T
        return np.linalg.eigvalsh(Sigma_t).max()

    interior, extrap = lam_max(0.5), lam_max(1.3)
    print(f"lambda_max interior(u=0.5)={interior:.2e} < "
          f"extrapolated(u=1.3)={extrap:.2e}: {bool(extrap > interior)}")

    # Waypoint builder smoke test: both pairings + nearest <= index distance.
    starts = rng.uniform(0, 10, size=(5, 3))
    ends = starts + rng.uniform(-1, 1, size=(5, 3))
    wp_i, Phi_i = discretize_candidates(starts, ends, g, n_waypoints=12,
                                        pairing="index")
    wp_n, Phi_n = discretize_candidates(starts, ends, g, n_waypoints=12,
                                        pairing="nearest")
    print(f"index   : waypoints {wp_i.shape}, Phi {Phi_i.shape}")
    print(f"nearest : waypoints {wp_n.shape}, Phi {Phi_n.shape}")
    mu_i = np.einsum("tij,j->ti", Phi_i, g.mu_w)            # (M, 3)
    mu_n = np.einsum("ntij,j->nti", Phi_n, g.mu_w)          # (N, M, 3)
    d_i = np.linalg.norm(wp_i - mu_i[None], axis=-1).mean()
    d_n = np.linalg.norm(wp_n - mu_n, axis=-1).mean()
    print(f"mean paired dist: nearest {d_n:.3f} <= index {d_i:.3f}: "
          f"{bool(d_n <= d_i)}")
