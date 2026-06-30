#!/usr/bin/env python3
"""Rank candidate pushes by joint clearance-AND-success probability (Section 4).

Builds on the probabilistic clearance ranker (Section 3 / Algorithm 2): we still
fit a *Gaussian* B-spline to the obstacle cloud (``utils.probabilistic_spline``),
propagate that control-point uncertainty to each push waypoint, and score every
candidate's probability of clearing the artery. Section 4 then adds a second
factor -- the action's probability of *task success* -- modeled as a decaying
function of its squared Mahalanobis distance from the mean (nominal) action
(``utils.action_success``). The final metric re-ranks candidates by the joint
probability

    P(clearance, success | a, obstacle) = P(clearance | a, obstacle) * P(success | a)

so the chosen push is both safe and close to the behavior the network wanted.
We then draw the 5 best (highest joint probability) and 5 worst capsules,
colored on a 0..1 probability scale.

Usage:
    python visualize_clearance_success_ranked.py          # default: version 1
    python visualize_clearance_success_ranked.py 2        # version 2
    python visualize_clearance_success_ranked.py 1 2 3    # all three side by side
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import proj3d

# Import the scipy-based helpers first (visualize + the Gaussian-spline bridge),
# then the torch ranker last. On systems where torch's libstdc++ shadows scipy's
# this ordering keeps scipy's compiled extensions loading correctly.
from visualize import (
    load_version, downsample, capsule_mesh, plane_rectangle,
    ARTERY_BANDWIDTH, ROBOT_TIP_RADIUS, SPLINE_RADIUS,
)
from utils.probabilistic_spline import (
    fit_gaussian_centerline, mean_curve, discretize_candidates,
)
from utils.probabilistic_collision_ranking import collision_probabilities
from utils.action_success import rank_joint_trajectories

N_BEST = 5    # best pushes to draw (highest joint clearance+success probability)
N_WORST = 5   # worst pushes to draw (lowest joint probability)

# Gaussian B-spline fit + probabilistic scoring tuning.
N_CTRL = 6         # B-spline control points per coordinate (clamped to node count)
PENALTY = 1e-1     # ridge curvature penalty (higher = stiffer / more confident)
N_WAYPOINTS = 12   # M waypoints per push paired with spline parameters
PAIRING = "nearest"  # "nearest" (centerline-to-centerline) or "index" (Alg. 2)

# How far past the visible artery to draw the spline, split by which end faces
# the tissue point cloud (the occluded continuation projected through tissue,
# drawn longer) vs the open-space end (drawn shorter).
MEAN_EXTRAP_TISSUE = 0.75    # mean centerline curve, toward the tissue cloud
MEAN_EXTRAP_OPEN = 0.25      # mean centerline curve, out into open space
SAMPLE_EXTRAP_TISSUE = 0.5   # uncertainty sample curves, toward the tissue cloud
SAMPLE_EXTRAP_OPEN = 0.25    # uncertainty sample curves, out into open space

# Probability heat map: 0 (certain collision) -> red, 1 (certain clear) -> green.
PROB_CMAP = plt.get_cmap("RdYlGn")
PROB_NORM = Normalize(vmin=0.0, vmax=1.0)


def sample_spline_curves(gspline, u_values, n_samples=15, seed=0):
    """A few curves drawn from w ~ N(mu_w, Sigma_w), to visualize the spread."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(gspline.Sigma_w + np.eye(gspline.Sigma_w.shape[0]) * 1e-9)
    K = gspline.n_ctrl
    from utils.probabilistic_spline import _design_matrix
    b = _design_matrix(u_values, gspline.knots, gspline.k)   # (M, K)
    curves = []
    for _ in range(n_samples):
        w = gspline.mu_w + L @ rng.standard_normal(gspline.mu_w.shape[0])
        c = w.reshape(3, K)
        curves.append(b @ c.T)                               # (M, 3)
    return curves


def draw_capsule(ax, p1, p2, p_value):
    """Draw one push capsule colored by its joint probability (0..1)."""
    color = PROB_CMAP(PROB_NORM(p_value))
    cx, cy, cz = capsule_mesh(p1, p2, ROBOT_TIP_RADIUS)
    ax.plot_surface(cx, cy, cz, color=color, alpha=1.0,
                    linewidth=0, antialiased=True, shade=True)


def render_push_axis(ax3d, data, title, show_title=True, show_legend=True,
                     show_ticks=True, face_on=False):
    """Draw the ranked-push 3D scene onto ``ax3d``. Shared by both visualizers.

    ``face_on`` orients the camera straight down the table-plane normal, so the
    table appears flat-on (looking down at the table) instead of the default
    oblique view.
    """
    start = downsample(data["start_pc"])
    goal = downsample(data["goal_pc"])
    obs = data["obstacle_pc"]
    cand_s = data["cand_starts"]
    cand_e = data["cand_ends"]

    ax3d.scatter(*start.T, s=1, c="steelblue", alpha=0.15)
    ax3d.scatter(*goal.T, s=1, c="orange", alpha=0.15)
    ax3d.scatter(*obs.T, s=4, c="limegreen", alpha=0.05)

    # ---- Gaussian B-spline fit of the artery centerline ----
    gspline = fit_gaussian_centerline(obs, bandwidth=ARTERY_BANDWIDTH,
                                      n_ctrl=N_CTRL, penalty=PENALTY)

    # Decide which spline end (u=0 vs u=1) faces the tissue: extrapolating past
    # that end projects the artery through the occluding tissue, so we draw it
    # longer; the opposite end heads into open space and is drawn shorter.
    tissue_centroid = np.vstack([start, goal]).mean(axis=0)
    end0, end1 = mean_curve(np.array([0.0, 1.0]), gspline)
    u0_is_tissue = (np.linalg.norm(end0 - tissue_centroid)
                    < np.linalg.norm(end1 - tissue_centroid))

    def extrap_range(tissue_amt, open_amt):
        """(lo, hi) u-span with the tissue-facing end extended by ``tissue_amt``."""
        lo_amt, hi_amt = ((tissue_amt, open_amt) if u0_is_tissue
                          else (open_amt, tissue_amt))
        return -lo_amt, 1.0 + hi_amt

    # mean curve over the visible extent (solid) + extrapolated tips (dashed)
    u_vis = np.linspace(0.0, 1.0, 200)
    mean_vis = mean_curve(u_vis, gspline)
    ax3d.plot(*mean_vis.T, "-", color="darkgreen", linewidth=2.5)
    mean_lo, mean_hi = extrap_range(MEAN_EXTRAP_TISSUE, MEAN_EXTRAP_OPEN)
    head = mean_curve(np.linspace(mean_lo, 0.0, 60), gspline)
    tail = mean_curve(np.linspace(1.0, mean_hi, 60), gspline)
    for tip in (head, tail):
        ax3d.plot(*tip.T, "--", color="darkgreen", linewidth=2.0, alpha=0.8)

    # uncertainty: faint curves sampled from the control-point Gaussian
    samp_lo, samp_hi = extrap_range(SAMPLE_EXTRAP_TISSUE, SAMPLE_EXTRAP_OPEN)
    u_full = np.linspace(samp_lo, samp_hi, 300)
    for s_curve in sample_spline_curves(gspline, u_full, n_samples=15):
        ax3d.plot(*s_curve.T, "-", color="seagreen", linewidth=0.6, alpha=0.18)

    # Table-plane normal (only used to orient the camera); the plane itself is
    # not drawn.
    scene = np.vstack([start, goal, obs])
    _, plane_n = plane_rectangle(data["plane"], scene)

    # ---- probabilistic clearance scoring of every candidate (Section 3) ----
    waypoints, Phi = discretize_candidates(cand_s, cand_e, gspline,
                                           n_waypoints=N_WAYPOINTS,
                                           pairing=PAIRING)
    d = ROBOT_TIP_RADIUS + SPLINE_RADIUS
    p_clear = collision_probabilities(
        gspline.mu_w, gspline.Sigma_w, waypoints, Phi, d)

    # ---- joint clearance-AND-success re-ranking (Section 4) ----
    order, p_success, p_joint = rank_joint_trajectories(cand_s, cand_e, p_clear)

    best = order[:N_BEST]                  # highest joint probability (best)
    worst = order[-N_WORST:][::-1]         # lowest joint probability (worst)
    print(f"  {title}: scored {len(order)} candidates "
          f"(P_joint {p_joint.min():.3f} .. {p_joint.max():.3f})")
    fmt = lambda i: f"#{i}=J{p_joint[i]:.3f}(C{p_clear[i]:.2f},S{p_success[i]:.2f})"
    print(f"    best : " + ", ".join(fmt(i) for i in best))
    print(f"    worst: " + ", ".join(fmt(i) for i in worst))

    for idx in best:
        draw_capsule(ax3d, cand_s[idx], cand_e[idx], p_joint[idx])
    for idx in worst:
        draw_capsule(ax3d, cand_s[idx], cand_e[idx], p_joint[idx])

    best_c = PROB_CMAP(PROB_NORM(p_joint[best].mean()))
    worst_c = PROB_CMAP(PROB_NORM(p_joint[worst].mean()))

    # equal-aspect cube cropped just outside the point clouds (where the pushes
    # and artery interact); the spline extrapolation / capsule tails extend well
    # past this, so we deliberately exclude them from the framing.
    center = (scene.max(axis=0) + scene.min(axis=0)) / 2
    half = (scene.max(axis=0) - scene.min(axis=0)).max() / 2 * 1.05
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_box_aspect((1, 1, 1))
    if face_on:
        # Look straight down the table-plane normal (camera above the table).
        n = plane_n / np.linalg.norm(plane_n)
        elev = np.degrees(np.arcsin(np.clip(n[2], -1.0, 1.0)))
        azim = np.degrees(np.arctan2(n[1], n[0]))
        ax3d.view_init(elev=elev, azim=azim)
    else:
        ax3d.view_init(elev=20, azim=-60)

    if show_title:
        ax3d.set_title(title)
    if show_ticks:
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    else:
        # Strip every axis decoration for a clean paper figure: ticks, labels,
        # the grid, and the gray bounding panes / spines.
        ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
        ax3d.set_xlabel(""); ax3d.set_ylabel(""); ax3d.set_zlabel("")
        ax3d.grid(False)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.pane.set_visible(False)
            # Transparent (not hidden) so savefig(bbox_inches="tight") can still
            # compute a bounding box for the 3D axis.
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))
    if show_legend:
        # Explicit, full-opacity proxy handles: the on-plot artists are drawn
        # faded (low-alpha scatters, thin samples) so their auto-legend markers
        # are nearly invisible. These proxies keep the colors matching while
        # staying crisp and legible.
        def cloud(color):
            return Line2D([], [], marker="o", linestyle="none", color=color,
                          markersize=7, markeredgecolor="none")

        handles = [
            (cloud("steelblue"), "Start tissue"),
            (cloud("orange"), "Goal tissue"),
            (cloud("limegreen"), "Artery (obstacle)"),
            (Line2D([], [], color="darkgreen", linewidth=2.5),
             "Artery centerline (mean)"),
            (Line2D([], [], color="darkgreen", linewidth=2.0, linestyle="--"),
             "Projected continuation"),
            (Line2D([], [], color="seagreen", linewidth=1.5, alpha=0.6),
             "Spline uncertainty samples"),
            (Patch(facecolor=best_c, edgecolor="0.3", linewidth=0.5),
             f"{N_BEST} safest pushes (best P={p_joint[best[0]]:.2f})"),
            (Patch(facecolor=worst_c, edgecolor="0.3", linewidth=0.5),
             f"{N_WORST} riskiest pushes (worst P={p_joint[worst[0]]:.2f})"),
        ]
        legend = ax3d.legend(
            [h for h, _ in handles], [t for _, t in handles],
            loc="upper left", fontsize=7.5, framealpha=1.0,
            borderpad=0.8, labelspacing=0.6, handlelength=1.8)
        # Fully opaque white box so plot content never shows through it.
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("0.3")
        legend.set_zorder(1000)

    return ax3d


def plot_version(data, fig, col, ncols, title):
    """Standard two-row view: ranked 3D scene on top, raw scene image below."""
    ax3d = fig.add_subplot(2, ncols, col, projection="3d")
    render_push_axis(ax3d, data, title)

    ax_img = fig.add_subplot(2, ncols, ncols + col)
    ax_img.imshow(data["image"])
    ax_img.axis("off")
    ax_img.set_title(f"{title} — scene image")
    return ax3d


def main(versions):
    ncols = len(versions)
    np.random.seed(0)
    fig = plt.figure(figsize=(6 * ncols, 10))
    fig.suptitle("PushVIBES — candidate pushes ranked by joint clearance × "
                 "success probability (green = best, red = worst)",
                 fontsize=11)

    push_axes = []
    for col, v in enumerate(versions, start=1):
        print(f"Loading version {v}...")
        data = load_version(v)
        push_axes.append(plot_version(data, fig, col, ncols, f"Version {v}"))

    sm = cm.ScalarMappable(norm=PROB_NORM, cmap=PROB_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=push_axes, fraction=0.02, pad=0.04)
    cbar.set_label("Joint probability of artery clearance AND reaching goal shape\n"
                   "← unsafe / off-distribution   |   safe & nominal →")

    plt.tight_layout()
    plt.show()


def _points_display_bbox(ax, points):
    """Display-pixel bbox of 3D ``points`` as currently projected onto ``ax``."""
    pts = np.asarray(points, dtype=float)
    xs, ys, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2],
                                      ax.get_proj())
    disp = ax.transData.transform(np.column_stack([xs, ys]))
    x0, y0 = disp.min(axis=0)
    x1, y1 = disp.max(axis=0)
    return Bbox.from_extents(x0, y0, x1, y1)


def _crop_bbox_around_clouds(fig, axes, cloud_pts, pad_frac=0.06, keep=()):
    """Tight save-bbox (in inches) cropped around the projected point clouds.

    Reproduces the figure viewer's "zoom to the point clouds": projects each
    axis's cloud to display pixels, unions those boxes, pads a little, then folds
    in any ``keep`` artists (legend, colorbar) so they stay in frame.
    """
    fig.canvas.draw()  # finalize projections + artist extents
    renderer = fig.canvas.get_renderer()
    boxes = [_points_display_bbox(ax, c) for ax, c in zip(axes, cloud_pts)]
    crop = Bbox.union(boxes)
    pad_x, pad_y = pad_frac * crop.width, pad_frac * crop.height
    crop = Bbox.from_extents(crop.x0 - pad_x, crop.y0 - pad_y,
                             crop.x1 + pad_x, crop.y1 + pad_y)
    # get_tightbbox includes each artist's own labels/ticks (e.g. the colorbar
    # label), unlike get_window_extent which would clip them.
    extras = []
    for a in keep:
        if a is None:
            continue
        try:
            extras.append(a.get_tightbbox(renderer))
        except TypeError:
            extras.append(a.get_window_extent(renderer))
    crop = Bbox.union([crop, *extras]) if extras else crop
    return crop.transformed(fig.dpi_scale_trans.inverted())


def paper_visualizer(versions, save_path=None, dpi=300, crop_to_clouds=True):
    """Clean, publication-ready figure: just the ranked 3D scene(s), no images.

    Drops the scene-image panels and per-axis titles so each version is a single
    uncluttered 3D plot. Saves a high-DPI figure when ``save_path`` is given
    (otherwise shows it interactively). With ``crop_to_clouds`` the saved figure
    is cropped tightly around the point clouds (like the viewer's zoom tool),
    trimming whitespace and the trailing spline tails.
    """
    ncols = len(versions)
    np.random.seed(0)
    fig = plt.figure(figsize=(6.5 * ncols, 6.5))

    push_axes, cloud_pts = [], []
    for col, v in enumerate(versions, start=1):
        print(f"Loading version {v}...")
        data = load_version(v)
        ax3d = fig.add_subplot(1, ncols, col, projection="3d")
        render_push_axis(ax3d, data, f"Exp. {col}",
                         show_title=(ncols > 1), show_legend=(col == 1),
                         show_ticks=False, face_on=True)
        push_axes.append(ax3d)
        cloud_pts.append(np.vstack([data["start_pc"], data["goal_pc"],
                                    data["obstacle_pc"]]))

    sm = cm.ScalarMappable(norm=PROB_NORM, cmap=PROB_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=push_axes, fraction=0.02, pad=0.04)
    cbar.set_label("Joint probability of clearance AND success\n"
                   "← unsafe / off-distribution   |   safe & nominal →")

    if save_path:
        save_kw = {}
        if crop_to_clouds:
            # Keep the colorbar (bar + its rotated label), the legend, and any
            # per-panel titles in frame.
            keep = [cbar.ax, cbar.ax.yaxis.label]
            keep += [ax.get_legend() for ax in push_axes]
            keep += [ax.title for ax in push_axes if ax.title.get_text()]
            save_kw["bbox_inches"] = _crop_bbox_around_clouds(
                fig, push_axes, cloud_pts, keep=keep)
        else:
            save_kw["bbox_inches"] = "tight"
        fig.savefig(save_path, dpi=dpi, **save_kw)
        print(f"Saved paper figure to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # `--paper [out.png]` renders the clean 3D-only figure; otherwise the
    # standard two-row (3D + scene image) view. Trailing ints select versions.
    args = sys.argv[1:]
    if args and args[0] == "--paper":
        args = args[1:]
        save_path = args[0] if args and not args[0].isdigit() else None
        if save_path:
            args = args[1:]
        versions = [int(a) for a in args] if args else [1]
        paper_visualizer(versions, save_path=save_path)
    else:
        versions = [int(a) for a in args] if args else [1]
        main(versions)
