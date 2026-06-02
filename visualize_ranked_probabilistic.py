#!/usr/bin/env python3
"""Rank candidate pushes by collision *probability* and visualize the extremes.

The probabilistic counterpart to ``visualize_ranked.py`` (Section 3 / Algorithm
2 of the writeup). Instead of assuming the artery sits exactly on a single
fitted spline, we fit a *Gaussian* B-spline to the obstacle cloud
(``utils.probabilistic_spline``), propagate that control-point uncertainty to
each push waypoint, and score every candidate by its joint probability of
clearing the artery. We then draw the 5 safest (highest probability) and 5
riskiest (lowest probability) capsules, colored on a 0..1 probability scale.

Usage:
    python visualize_ranked_probabilistic.py          # default: version 1
    python visualize_ranked_probabilistic.py 2        # version 2
    python visualize_ranked_probabilistic.py 1 2 3    # all three side by side
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import Normalize

# Import the scipy-based helpers first (visualize + the Gaussian-spline bridge),
# then the torch ranker last. On systems where torch's libstdc++ shadows scipy's
# this ordering keeps scipy's compiled extensions loading correctly.
from visualize import (
    load_version, downsample, capsule_mesh, plane_rectangle,
    ARTERY_BANDWIDTH, ARTERY_EXTRAPOLATE, ROBOT_TIP_RADIUS, SPLINE_RADIUS,
)
from utils.probabilistic_spline import (
    fit_gaussian_centerline, mean_curve, discretize_candidates,
)
from utils.probabilistic_collision_ranking import rank_probabilistic_trajectories

N_BEST = 5    # safest pushes to draw (highest clearance probability)
N_WORST = 5   # riskiest pushes to draw (lowest clearance probability)

# Gaussian B-spline fit + probabilistic scoring tuning.
N_CTRL = 6         # B-spline control points per coordinate (clamped to node count)
PENALTY = 1e-3     # ridge curvature penalty (higher = stiffer / more confident)
N_WAYPOINTS = 12   # M waypoints per push paired with spline parameters
PAIRING = "nearest"  # "nearest" (centerline-to-centerline) or "index" (Alg. 2)

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


def draw_capsule(ax, p1, p2, p_safe, rank_label):
    """Draw one push capsule colored by its clearance probability (0..1)."""
    color = PROB_CMAP(PROB_NORM(p_safe))
    cx, cy, cz = capsule_mesh(p1, p2, ROBOT_TIP_RADIUS)
    ax.plot_surface(cx, cy, cz, color=color, alpha=1.0,
                    linewidth=0, antialiased=True, shade=True)
    mid = (np.asarray(p1) + np.asarray(p2)) / 2
    ax.text(*mid, rank_label, fontsize=6, color="black")


def plot_version(data, fig, col, ncols, title):
    ax3d = fig.add_subplot(2, ncols, col, projection="3d")

    start = downsample(data["start_pc"])
    goal = downsample(data["goal_pc"])
    obs = data["obstacle_pc"]
    cand_s = data["cand_starts"]
    cand_e = data["cand_ends"]

    ax3d.scatter(*start.T, s=1, c="steelblue", alpha=0.15, label="Start tissue")
    ax3d.scatter(*goal.T, s=1, c="orange", alpha=0.15, label="Goal tissue")
    ax3d.scatter(*obs.T, s=4, c="limegreen", alpha=0.05, label="Artery (obstacle)")

    # ---- Gaussian B-spline fit of the artery centerline ----
    gspline = fit_gaussian_centerline(obs, bandwidth=ARTERY_BANDWIDTH,
                                      n_ctrl=N_CTRL, penalty=PENALTY)

    # mean curve over the visible extent (solid) + extrapolated tips (dashed)
    u_vis = np.linspace(0.0, 1.0, 200)
    u_full = np.linspace(-ARTERY_EXTRAPOLATE, 1.0 + ARTERY_EXTRAPOLATE, 300)
    mean_vis = mean_curve(u_vis, gspline)
    ax3d.plot(*mean_vis.T, "-", color="darkgreen", linewidth=2.5,
              label="Artery centerline (mean)")
    head = mean_curve(np.linspace(-ARTERY_EXTRAPOLATE, 0.0, 60), gspline)
    tail = mean_curve(np.linspace(1.0, 1.0 + ARTERY_EXTRAPOLATE, 60), gspline)
    for tip in (head, tail):
        ax3d.plot(*tip.T, "--", color="darkgreen", linewidth=2.0, alpha=0.8)
    ax3d.plot([], [], [], "--", color="darkgreen", alpha=0.8,
              label="Projected continuation")

    # uncertainty: faint curves sampled from the control-point Gaussian
    for s_curve in sample_spline_curves(gspline, u_full, n_samples=15):
        ax3d.plot(*s_curve.T, "-", color="seagreen", linewidth=0.6, alpha=0.18)
    ax3d.plot([], [], [], "-", color="seagreen", alpha=0.5,
              label="Spline uncertainty samples")

    # table plane
    scene = np.vstack([start, goal, obs])
    corners, _ = plane_rectangle(data["plane"], scene)
    quad = Poly3DCollection([corners], alpha=0.25, facecolor="tan",
                            edgecolor="saddlebrown", linewidths=1.0)
    ax3d.add_collection3d(quad)

    # ---- probabilistic collision ranking of every candidate ----
    waypoints, Phi = discretize_candidates(cand_s, cand_e, gspline,
                                           n_waypoints=N_WAYPOINTS,
                                           pairing=PAIRING)
    d = ROBOT_TIP_RADIUS + SPLINE_RADIUS
    order, p_safe = rank_probabilistic_trajectories(
        gspline.mu_w, gspline.Sigma_w, waypoints, Phi, d)

    best = order[:N_BEST]                  # highest probability (safest)
    worst = order[-N_WORST:][::-1]         # lowest probability (riskiest)
    print(f"  {title}: scored {len(order)} candidates "
          f"(P_safe {p_safe.min():.3f} .. {p_safe.max():.3f})")
    print(f"    safest  : " + ", ".join(f"#{i}={p_safe[i]:.3f}" for i in best))
    print(f"    riskiest: " + ", ".join(f"#{i}={p_safe[i]:.3f}" for i in worst))

    drawn = []
    for rank, idx in enumerate(best, start=1):
        draw_capsule(ax3d, cand_s[idx], cand_e[idx], p_safe[idx], f"B{rank}")
        drawn.append((cand_s[idx], cand_e[idx]))
    for rank, idx in enumerate(worst, start=1):
        draw_capsule(ax3d, cand_s[idx], cand_e[idx], p_safe[idx], f"W{rank}")
        drawn.append((cand_s[idx], cand_e[idx]))

    best_c = PROB_CMAP(PROB_NORM(p_safe[best].mean()))
    worst_c = PROB_CMAP(PROB_NORM(p_safe[worst].mean()))
    ax3d.plot([], [], [], "s", color=best_c,
              label=f"{N_BEST} safest (B1..B{N_BEST}, "
                    f"P={p_safe[best[0]]:.3f} best)")
    ax3d.plot([], [], [], "s", color=worst_c,
              label=f"{N_WORST} riskiest (W1..W{N_WORST}, "
                    f"P={p_safe[worst[0]]:.3f} worst)")

    # equal-aspect cube around everything drawn
    drawn_pts = np.vstack([np.vstack(c) for c in drawn]) if drawn else cand_s
    bounds = np.vstack([scene, corners, drawn_pts, mean_vis])
    center = (bounds.max(axis=0) + bounds.min(axis=0)) / 2
    half = (bounds.max(axis=0) - bounds.min(axis=0)).max() / 2
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=20, azim=-60)

    ax3d.set_title(f"{title} — {N_BEST} safest vs {N_WORST} riskiest "
                   f"(collision probability)")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    plane_proxy = plt.Rectangle((0, 0), 1, 1, fc="tan", alpha=0.5)
    handles, labels = ax3d.get_legend_handles_labels()
    ax3d.legend(handles + [plane_proxy], labels + ["Table plane"],
                loc="upper left", markerscale=4, fontsize=7)

    # ---- image panel ----
    ax_img = fig.add_subplot(2, ncols, ncols + col)
    ax_img.imshow(data["image"])
    ax_img.axis("off")
    ax_img.set_title(f"{title} — scene image")

    return ax3d


def main(versions):
    ncols = len(versions)
    np.random.seed(0)
    fig = plt.figure(figsize=(6 * ncols, 10))
    fig.suptitle("PushVIBES — candidate pushes ranked by artery collision "
                 "probability (green = safest, red = most likely collision)",
                 fontsize=11)

    push_axes = []
    for col, v in enumerate(versions, start=1):
        print(f"Loading version {v}...")
        data = load_version(v)
        push_axes.append(plot_version(data, fig, col, ncols, f"Version {v}"))

    sm = cm.ScalarMappable(norm=PROB_NORM, cmap=PROB_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=push_axes, fraction=0.02, pad=0.04)
    cbar.set_label("Probability of clearing the artery\n"
                   "← likely collision   |   likely clear →")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    versions = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [1]
    main(versions)
