#!/usr/bin/env python3
"""Rank the candidate pushes by collision clearance and visualize the extremes.

For each version we score all ~500 predicted candidate trajectories with the
batched, GPU-friendly ranker in ``utils.collision_ranking`` (signed surface
clearance against the artery tube), then draw the 5 safest (highest score) and
5 riskiest (lowest score) push capsules, each colored on the same red->green
clearance heat scale used by ``visualize.py``.

Usage:
    python visualize_ranked.py          # default: version 1
    python visualize_ranked.py 2        # version 2
    python visualize_ranked.py 1 2 3    # all three versions side by side
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

# Reuse the geometry/loading helpers and tuning constants from visualize.py
# (its script body is guarded behind __main__, so importing has no side effects
# beyond selecting the matplotlib backend).
from visualize import (
    load_version, downsample, capsule_mesh, plane_rectangle,
    ARTERY_BANDWIDTH, ARTERY_SMOOTHING, ARTERY_EXTRAPOLATE,
    ROBOT_TIP_RADIUS, SPLINE_RADIUS,
    CLEARANCE_CMAP, CLEARANCE_NORM,
)
from utils.spline_fit import fit_centerline_spline
from utils.collision_ranking import rank_trajectories

N_BEST = 5   # safest pushes to draw (highest clearance)
N_WORST = 5  # riskiest pushes to draw (lowest clearance / deepest penetration)


def artery_spline_pieces(obs):
    """Fit the artery centerline and return its polyline pieces (visible + tips).

    Returns a list of (Ni, 3) arrays, or an empty list if the fit fails.
    """
    try:
        _, spline, projections = fit_centerline_spline(
            obs, bandwidth=ARTERY_BANDWIDTH, smoothing=ARTERY_SMOOTHING,
            extrapolate=ARTERY_EXTRAPOLATE)
        return [spline, *projections]
    except ValueError as e:
        print(f"  artery spline fit skipped: {e}")
        return []


def draw_capsule(ax, p1, p2, clearance_mm, rank_label):
    """Draw one push capsule colored by its signed clearance (mm)."""
    color = CLEARANCE_CMAP(CLEARANCE_NORM(clearance_mm))
    cx, cy, cz = capsule_mesh(p1, p2, ROBOT_TIP_RADIUS)
    ax.plot_surface(cx, cy, cz, color=color, alpha=1.0,
                    linewidth=0, antialiased=True, shade=True)
    # small text tag at the capsule midpoint so best/worst rank is readable
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

    # artery centerline (+ projected continuation) as the collision tube
    spline_pts = artery_spline_pieces(obs)
    if spline_pts:
        ax3d.plot(*spline_pts[0].T, "-", color="darkgreen", linewidth=2.5,
                  label="Artery centerline")
        for p in spline_pts[1:]:
            ax3d.plot(*p.T, "--", color="darkgreen", linewidth=2.0, alpha=0.8)
        ax3d.plot([], [], [], "--", color="darkgreen", alpha=0.8,
                  label="Projected continuation")

    # table plane
    scene = np.vstack([start, goal, obs])
    corners, _ = plane_rectangle(data["plane"], scene)
    quad = Poly3DCollection([corners], alpha=0.25, facecolor="tan",
                            edgecolor="saddlebrown", linewidths=1.0)
    ax3d.add_collection3d(quad)

    # ---- batched collision ranking of every candidate trajectory ----
    if not spline_pts:
        raise RuntimeError(f"{title}: no artery spline -> cannot score pushes.")
    order, scores, _ = rank_trajectories(
        cand_s, cand_e, ROBOT_TIP_RADIUS, spline_pts, SPLINE_RADIUS)
    clearance_mm = scores * 1000.0  # signed surface clearance per candidate

    best = order[:N_BEST]                 # highest clearance (safest)
    worst = order[-N_WORST:][::-1]        # lowest clearance (riskiest), worst first
    print(f"  {title}: scored {len(order)} candidates "
          f"(clearance {clearance_mm.min():+.1f} .. {clearance_mm.max():+.1f} mm)")
    print(f"    safest  : " + ", ".join(
        f"#{i}={clearance_mm[i]:+.1f}mm" for i in best))
    print(f"    riskiest: " + ", ".join(
        f"#{i}={clearance_mm[i]:+.1f}mm" for i in worst))

    # draw the 5 safest then the 5 riskiest capsules, colored by clearance
    drawn = []
    for rank, idx in enumerate(best, start=1):
        draw_capsule(ax3d, cand_s[idx], cand_e[idx], clearance_mm[idx],
                     f"B{rank}")
        drawn.append((cand_s[idx], cand_e[idx]))
    for rank, idx in enumerate(worst, start=1):
        draw_capsule(ax3d, cand_s[idx], cand_e[idx], clearance_mm[idx],
                     f"W{rank}")
        drawn.append((cand_s[idx], cand_e[idx]))

    # legend proxies summarizing the two groups (plot_surface has no handle)
    best_c = CLEARANCE_CMAP(CLEARANCE_NORM(clearance_mm[best].mean()))
    worst_c = CLEARANCE_CMAP(CLEARANCE_NORM(clearance_mm[worst].mean()))
    ax3d.plot([], [], [], "s", color=best_c,
              label=f"{N_BEST} safest (B1..B{N_BEST}, "
                    f"{clearance_mm[best[0]]:+.1f} mm best)")
    ax3d.plot([], [], [], "s", color=worst_c,
              label=f"{N_WORST} riskiest (W1..W{N_WORST}, "
                    f"{clearance_mm[worst[0]]:+.1f} mm worst)")

    # equal-aspect cube around everything we drew
    drawn_pts = np.vstack([np.vstack(c) for c in drawn]) if drawn else cand_s
    bounds = np.vstack([scene, corners, drawn_pts, *spline_pts])
    center = (bounds.max(axis=0) + bounds.min(axis=0)) / 2
    half = (bounds.max(axis=0) - bounds.min(axis=0)).max() / 2
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=20, azim=-60)

    ax3d.set_title(f"{title} — {N_BEST} safest vs {N_WORST} riskiest pushes")
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
    fig.suptitle("PushVIBES — candidate pushes ranked by artery clearance "
                 "(green = safest, red = deepest collision)", fontsize=11)

    push_axes = []
    for col, v in enumerate(versions, start=1):
        print(f"Loading version {v}...")
        data = load_version(v)
        push_axes.append(plot_version(data, fig, col, ncols, f"Version {v}"))

    sm = cm.ScalarMappable(norm=CLEARANCE_NORM, cmap=CLEARANCE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=push_axes, fraction=0.02, pad=0.04)
    cbar.set_label("Push surface clearance (mm)\n← penetration   |   clearance →")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    versions = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [1]
    main(versions)
