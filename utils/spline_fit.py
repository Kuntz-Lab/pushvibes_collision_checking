#!/usr/bin/env python3
"""Extract a centerline from a tube-shaped point cloud and fit a B-spline.

Pipeline: Mean Shift to find skeleton nodes -> minimum spanning tree to order
them along the tube -> regularized cubic B-spline through the ordered nodes.

The fitted spline can be extrapolated past the visible ends so an obscured
continuation of the tube (e.g. an artery running underneath tissue) can be
estimated. See `fit_centerline_spline`.
"""

import numpy as np
import networkx as nx
from sklearn.cluster import MeanShift
from scipy.spatial import distance_matrix
from scipy.interpolate import splprep, splev


def _order_skeleton_nodes(points, bandwidth):
    """Mean Shift skeleton nodes ordered along the tube via an MST longest path."""
    # bin_seeding speeds up Mean Shift significantly for large point clouds
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(points)
    centers = ms.cluster_centers_

    if len(centers) < 4:
        raise ValueError(
            f"Mean Shift found only {len(centers)} nodes; lower `bandwidth` "
            f"(currently {bandwidth})."
        )

    # Build an MST over the nodes and walk its longest path. For a single
    # unbranched tube the longest path visits every node in tube order.
    dist_mat = distance_matrix(centers, centers)
    mst = nx.minimum_spanning_tree(nx.from_numpy_array(dist_mat))

    endpoints = [n for n, deg in mst.degree() if deg == 1]
    if len(endpoints) < 2:
        longest_path = list(mst.nodes())
    else:
        longest_path, max_len = [], 0
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                path = nx.shortest_path(mst, endpoints[i], endpoints[j])
                if len(path) > max_len:
                    max_len, longest_path = len(path), path

    return centers[longest_path]


def fit_centerline_spline(
    points,
    bandwidth=0.006,
    smoothing=1e-4,
    n_samples=200,
    extrapolate=0.0,
):
    """Fit a regularized B-spline to the centerline of a tube point cloud.

    Args:
        points: (N, 3) point cloud of the tube surface/volume.
        bandwidth: Mean Shift radius, roughly the tube radius (meters).
        smoothing: splprep `s` smoothing condition. Higher = stiffer curve.
        n_samples: number of points sampled along the visible spline (u in [0, 1]).
        extrapolate: fraction of the parameter range to extend past each end
            (0.2 => project 20% of the curve length off each tip). The
            extrapolated tips estimate where an obscured tube continues.

    Returns:
        ordered_centers: (M, 3) MST-ordered Mean Shift skeleton nodes.
        spline_points: (n_samples, 3) sampled spline over the visible extent.
        projections: list of up to two (K, 3) arrays, the extrapolated tips
            (empty when `extrapolate` is 0). Each starts at a visible end so it
            connects cleanly to `spline_points`.
    """
    ordered_centers = _order_skeleton_nodes(points, bandwidth)

    # Regularized cubic B-spline. `s` trades data fidelity against stiffness.
    tck, _ = splprep(ordered_centers.T, s=smoothing, k=3)

    u_fine = np.linspace(0.0, 1.0, n_samples)
    spline_points = np.asarray(splev(u_fine, tck)).T

    projections = []
    if extrapolate > 0:
        n_ext = max(2, int(round(n_samples * extrapolate)))
        # splev extrapolates the boundary polynomial pieces for u outside [0, 1].
        head = np.linspace(-extrapolate, 0.0, n_ext)
        tail = np.linspace(1.0, 1.0 + extrapolate, n_ext)
        projections = [
            np.asarray(splev(head, tck)).T,
            np.asarray(splev(tail, tck)).T,
        ]

    return ordered_centers, spline_points, projections


if __name__ == "__main__":
    # Quick self-test against a synthetic S-curve tube.
    import matplotlib.pyplot as plt

    t = np.linspace(0, 10, 2500)
    centerline = np.vstack((t, np.sin(t) * 2, np.cos(t / 2) * 2)).T
    tube = centerline + np.random.normal(scale=0.3, size=centerline.shape)

    nodes, spline, projs = fit_centerline_spline(
        tube, bandwidth=0.8, smoothing=10.0, extrapolate=0.2
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*tube.T, c="gray", s=1, alpha=0.1, label="Raw point cloud")
    ax.plot(*nodes.T, "o-", color="blue", markersize=4, label="Ordered centers (MST)")
    ax.plot(*spline.T, "-", color="red", linewidth=3, label="Fitted spline")
    for p in projs:
        ax.plot(*p.T, "--", color="red", linewidth=2, alpha=0.7)
    ax.legend()
    ax.set_title("Tube centerline extraction & spline fitting")
    plt.show()
