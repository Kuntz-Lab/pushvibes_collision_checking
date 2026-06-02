#!/usr/bin/env python3
"""
Visualize point clouds and scene image for a given version (1, 2, or 3).

Usage:
    python visualize.py          # default: version 1
    python visualize.py 2        # version 2
    python visualize.py 1 2 3    # all three versions side by side
"""

import sys
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

from utils.spline_fit import fit_centerline_spline
from utils.deterministic_capsule_collision import check_capsule_splines_collision
from utils.object_frame import candidate_pushes_in_camera_frame

matplotlib.use("TkAgg")  # interactive window; change to "Agg" to save instead

DATA_DIR = Path("data/PushVIB3S_procedure_level_plan_example")
MAX_PTS = 4000  # downsample large clouds for speed

# Artery centerline spline. bandwidth ~ the tube radius (meters); extrapolate
# extends the fitted curve past each visible end by this fraction of its length,
# projecting how the tube likely continues where it is obscured by the tissue.
ARTERY_BANDWIDTH = 0.004
ARTERY_SMOOTHING = 1e-4
ARTERY_EXTRAPOLATE = 0.75

# Collision model: the robot tip is a sphere of this radius (meters). Swept
# along the push it forms a capsule; the artery is a tube of SPLINE_RADIUS.
# A collision is flagged when the capsule and tube overlap.
ROBOT_TIP_RADIUS = 0.0015
SPLINE_RADIUS = ARTERY_BANDWIDTH

# Heat-map coloring of the push capsule by signed surface clearance (mm):
# deeper penetration -> red, safer clearance -> green, with the contact
# boundary (0 mm) at the yellow midpoint. The scale (mm) sets how much
# penetration/clearance saturates the colormap.
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
CLEARANCE_CMAP = plt.get_cmap("RdYlGn")
CLEARANCE_NORM = TwoSlopeNorm(vmin=-10.0, vcenter=0.0, vmax=20.0)


def plane_coeffs(sympy_expr):
    """Return (a, b, c, d) for ax + by + cz + d = 0 from a sympy Add."""
    syms = {s.name: s for s in sympy_expr.free_symbols}
    a = float(sympy_expr.coeff(syms["x"]))
    b = float(sympy_expr.coeff(syms["y"]))
    c = float(sympy_expr.coeff(syms["z"]))
    d = float(sympy_expr.subs([(syms["x"], 0), (syms["y"], 0), (syms["z"], 0)]))
    return a, b, c, d


# The table_plane_equation was fit in millimeters, but camera_frame_*_pc are
# in meters. Correcting the offset by 1/1000 puts the table where it belongs:
# tissue/goal sit ~15 mm above it, and the artery ~8 mm below it (underneath).
PLANE_UNIT_SCALE = 1000.0  # mm per meter


def plane_rectangle(coeffs, scene_pts, pad=1.15):
    """Build a rectangle (4 corners) lying on the true table plane.

    `coeffs` are (a, b, c, d) for the mm-frame plane; we rescale the offset to
    meters so it is consistent with the camera-frame (meter) point clouds, then
    span a rectangle over the scene footprint centered on the plane.
    Returns (corners 4x3, unit normal pointing toward the camera).
    """
    a, b, c, d = coeffs
    d = d / PLANE_UNIT_SCALE  # mm-fit offset -> meters
    raw = np.array([a, b, c], dtype=float)
    nmag = np.linalg.norm(raw)
    n = raw / nmag

    # foot of perpendicular from the scene centroid onto the plane = anchor
    c0 = scene_pts.mean(axis=0)
    signed = (raw @ c0 + d) / nmag
    anchor = c0 - signed * n

    # orient unit normal toward the camera (origin) for a sensible legend/use
    if n @ c0 > 0:
        n = -n

    # two in-plane orthonormal axes
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, seed); u /= np.linalg.norm(u)
    w = np.cross(n, u)

    # extent of the scene in the (u, w) plane, centered on the anchor
    rel = scene_pts - anchor
    su, sw = rel @ u, rel @ w
    umin, umax = su.min() * pad, su.max() * pad
    wmin, wmax = sw.min() * pad, sw.max() * pad

    corners = np.array([
        anchor + umin * u + wmin * w,
        anchor + umax * u + wmin * w,
        anchor + umax * u + wmax * w,
        anchor + umin * u + wmax * w,
    ])
    return corners, n


def capsule_mesh(p1, p2, radius, n_theta=20, n_cap=6):
    """Surface mesh (X, Y, Z) for a capsule: a cylinder with hemispherical caps.

    The capsule core runs p1 -> p2 and is inflated by `radius`, matching the
    volume swept by a sphere of `radius` moving linearly between the endpoints.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    axis = p2 - p1
    length = np.linalg.norm(axis)
    if length < 1e-9:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / length

    # Two in-plane axes orthonormal to the capsule axis
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, seed); u /= np.linalg.norm(u)
    w = np.cross(axis, u)

    # Meridian profile as (axial offset from p1, ring radius): start cap, the
    # straight cylinder, then the end cap. Caps bulge past the endpoints by R.
    cap = np.linspace(-np.pi / 2, 0.0, n_cap)
    t_start = radius * np.sin(cap)          # negative, behind p1
    r_start = radius * np.cos(cap)
    t_end = length - radius * np.sin(cap)   # mirror, ahead of p2
    r_end = radius * np.cos(cap)
    t_prof = np.concatenate([t_start, [0.0, length], t_end])
    r_prof = np.concatenate([r_start, [radius, radius], r_end])

    theta = np.linspace(0, 2 * np.pi, n_theta)
    ct, st = np.cos(theta), np.sin(theta)

    # Sweep each profile ring around the axis
    X = np.empty((len(t_prof), n_theta))
    Y = np.empty_like(X)
    Z = np.empty_like(X)
    for i, (t, r) in enumerate(zip(t_prof, r_prof)):
        ring = p1 + t * axis + r * (np.outer(ct, u) + np.outer(st, w))
        X[i], Y[i], Z[i] = ring[:, 0], ring[:, 1], ring[:, 2]
    return X, Y, Z


def load_version(v):
    pkl_path = DATA_DIR / f"thanks_joe_vibes_v{v}.pickle"
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    # The chosen push is stored in the CAMERA frame as pred_start/pred_disp/
    # pred_end (identical here to original_pred_*). pred_start lands on the
    # current (start) tissue surface (~4-11 mm) and pred_end on the goal tissue
    # surface (~4-6 mm) -- i.e. the tip interacting with the two clouds.
    #
    # d["preds"][...] (start_point/displacement) is the 500-sample candidate set
    # in DiffDef's OBJECT frame (PCA-aligned, origin-centered), so it does not
    # overlay the camera-frame clouds as stored. utils.object_frame reconstructs
    # the object->camera transform from the start cloud and maps the candidates
    # back into the camera frame (validated: the chosen push round-trips onto
    # its candidate to <1.5 mm). cand_starts/cand_ends are then camera-frame.
    cand_starts, cand_ends = candidate_pushes_in_camera_frame(d)
    return {
        "start_pc": d["camera_frame_current_pc"].astype(np.float32),
        "goal_pc": d["camera_frame_goal_pc"].astype(np.float32),
        "push_start": np.asarray(d["pred_start"]).ravel(),
        "push_end": np.asarray(d["pred_end"]).ravel(),
        "cand_starts": cand_starts,
        "cand_ends": cand_ends,
        "obstacle_pc": np.load(DATA_DIR / f"obstacle_pc_v{v}.npy"),
        "image": plt.imread(DATA_DIR / f"image_v{v}.png"),
        "plane": plane_coeffs(d["table_plane_equation"]),
    }


def downsample(pts, n=MAX_PTS):
    if len(pts) <= n:
        return pts
    idx = np.random.choice(len(pts), n, replace=False)
    return pts[idx]


def plot_version(data, fig, col, ncols, title):
    # ---- 3-D point cloud panel ----
    ax3d = fig.add_subplot(2, ncols, col, projection="3d")

    start = downsample(data["start_pc"])
    goal = downsample(data["goal_pc"])
    obs = data["obstacle_pc"]
    push_s = data["push_start"]
    push_e = data["push_end"]

    ax3d.scatter(*start.T, s=1, c="steelblue", alpha=0.15, label="Start tissue")
    ax3d.scatter(*goal.T, s=1, c="orange", alpha=0.15, label="Goal tissue")
    ax3d.scatter(*obs.T, s=4, c="limegreen", alpha=0.05, label="Artery (obstacle)")

    # fit a centerline spline to the artery and project it past its visible
    # ends, estimating how the tube continues where the tissue obscures it
    spline_pts = []
    try:
        _, spline, projections = fit_centerline_spline(
            obs, bandwidth=ARTERY_BANDWIDTH, smoothing=ARTERY_SMOOTHING,
            extrapolate=ARTERY_EXTRAPOLATE)
        ax3d.plot(*spline.T, "-", color="darkgreen", linewidth=2.5,
                  label="Artery centerline")
        for p in projections:
            ax3d.plot(*p.T, "--", color="darkgreen", linewidth=2.0, alpha=0.8)
        # legend proxy for the dashed projected continuation
        ax3d.plot([], [], [], "--", color="darkgreen", alpha=0.8,
                  label="Projected continuation")
        spline_pts = [spline, *projections]
    except ValueError as e:
        print(f"  artery spline fit skipped: {e}")

    # table plane: use equation's normal, re-anchored beneath the scene
    scene = np.vstack([start, goal, obs])
    corners, _ = plane_rectangle(data["plane"], scene)
    quad = Poly3DCollection([corners], alpha=0.25, facecolor="tan",
                            edgecolor="saddlebrown", linewidths=1.0)
    ax3d.add_collection3d(quad)

    # draw the 500 candidate pushes (preds), mapped from DiffDef's object frame
    # into this camera frame, as faint start->end segments: the predicted action
    # distribution the chosen push was selected from.
    cand_s = data["cand_starts"]
    cand_e = data["cand_ends"]
    for i, (cs, ce) in enumerate(zip(cand_s, cand_e)):
        ax3d.plot(*np.vstack([cs, ce]).T, "-", color="crimson", linewidth=0.5,
                  alpha=0.12, label="Candidate pushes" if i == 0 else None)
    ax3d.scatter(*cand_s.T, s=2, c="crimson", alpha=0.3)

    # draw the chosen push (camera frame) as the capsule swept by the robot
    # tip (sphere of ROBOT_TIP_RADIUS) moving from the current tissue to the
    # goal tissue. Collision-check that capsule against the artery tube.
    collision, min_dist = check_capsule_splines_collision(
        push_s, push_e, ROBOT_TIP_RADIUS, spline_pts, SPLINE_RADIUS)
    status = "COLLISION" if collision else "clear"
    # Signed surface clearance (like an SDF): negative = penetration depth,
    # positive = room to spare. min_dist is the true global closest approach.
    clearance = min_dist - (ROBOT_TIP_RADIUS + SPLINE_RADIUS)
    clearance_mm = clearance * 1000
    # color on the red->green heat scale (red = penetrating, green = clear)
    cap_color = CLEARANCE_CMAP(CLEARANCE_NORM(clearance_mm))
    if collision:
        print(f"  push vs artery: {status} (penetration {-clearance_mm:.1f} mm)")
    else:
        print(f"  push vs artery: {status} (clearance {clearance_mm:.1f} mm)")

    cx, cy, cz = capsule_mesh(push_s, push_e, ROBOT_TIP_RADIUS)
    ax3d.plot_surface(cx, cy, cz, color=cap_color, alpha=1.0,
                      linewidth=0, antialiased=True, shade=True)
    # legend proxy for the capsule (plot_surface has no legend handle)
    ax3d.plot([], [], [], "s", color=cap_color,
              label=f"Push capsule ({status}, {clearance_mm:+.1f} mm)")

    # equal aspect: center a cube on the data so nothing gets flattened
    bounds = np.vstack([scene, corners, push_s[None], push_e[None],
                        cand_s, cand_e, *spline_pts])
    center = (bounds.max(axis=0) + bounds.min(axis=0)) / 2
    half = (bounds.max(axis=0) - bounds.min(axis=0)).max() / 2
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=20, azim=-60)

    ax3d.set_title(title)
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
    fig.suptitle("PushVIBES — start (blue) / goal (orange) / artery (green) / pushes (red)",
                 fontsize=11)

    push_axes = []
    for col, v in enumerate(versions, start=1):
        print(f"Loading version {v}...")
        data = load_version(v)
        push_axes.append(plot_version(data, fig, col, ncols, f"Version {v}"))

    # shared heat-map colorbar for the push capsule's signed surface clearance
    sm = cm.ScalarMappable(norm=CLEARANCE_NORM, cmap=CLEARANCE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=push_axes, fraction=0.02, pad=0.04)
    cbar.set_label("Push surface clearance (mm)\n← penetration   |   clearance →")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    versions = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [1]
    main(versions)
