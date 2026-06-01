#!/usr/bin/env python3
"""Collision checking between a swept-sphere capsule and a tube-shaped spline.

The robot tip is approximated as a sphere of radius R; as it travels linearly
along a push it sweeps out a capsule (cylinder + two hemispherical caps). The
artery centerline spline is treated as a tube of constant radius. A collision
occurs when the shortest distance between the two cores is <= the sum of radii.
"""

import numpy as np
from scipy.interpolate import splprep, splev


def closest_distance_between_segments(p1, q1, p2, q2):
    """Minimum distance between two 3D line segments (p1->q1 and p2->q2).

    Based on Christer Ericson's Real-Time Collision Detection.
    """
    epsilon = 1e-8

    d1 = q1 - p1  # Direction vector of segment 1
    d2 = q2 - p2  # Direction vector of segment 2
    r = p1 - p2

    a = np.dot(d1, d1)  # Squared length of segment 1
    e = np.dot(d2, d2)  # Squared length of segment 2
    f = np.dot(d2, r)

    # Both segments degenerate into points
    if a <= epsilon and e <= epsilon:
        return np.linalg.norm(p1 - p2)

    # Segment 1 degenerates into a point
    if a <= epsilon:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        # Segment 2 degenerates into a point
        if e <= epsilon:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            # General non-degenerate case
            b = np.dot(d1, d2)
            denom = a * e - b * b

            # If not parallel, closest point on L1 to L2, clamped to S1
            if denom != 0.0:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0  # Arbitrary point; segments are parallel

            # Point on L2 closest to S1(s)
            t = (b * s + f) / e

            # If t out of [0, 1], clamp it and recompute s
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    # Closest points on both segments
    c1 = p1 + d1 * s
    c2 = p2 + d2 * t

    return np.linalg.norm(c1 - c2)


def check_capsule_spline_collision(capsule_p1, capsule_p2, capsule_radius,
                                   spline_points, spline_radius):
    """Check for collision between a capsule and a discretized spline tube.

    Args:
        capsule_p1, capsule_p2: (3,) arrays, endpoints of the capsule core.
        capsule_radius: float, radius of the capsule (e.g. robot tip radius).
        spline_points: (N, 3) array, ordered points defining the spline core.
        spline_radius: float, radius of the spline tube.

    Returns:
        collision: bool, True if colliding.
        min_dist: float, shortest distance found between the cores. This is the
            true global minimum over all segments (no early exit), so it is
            meaningful whether or not there is a collision -- like a signed
            distance, the surface gap `min_dist - (capsule_radius +
            spline_radius)` tells you how deeply you penetrate (negative) or
            how much room you have to spare (positive).
    """
    capsule_p1 = np.asarray(capsule_p1, dtype=float)
    capsule_p2 = np.asarray(capsule_p2, dtype=float)
    spline_points = np.asarray(spline_points, dtype=float)

    collision_distance_threshold = capsule_radius + spline_radius
    min_dist = float("inf")

    # Treat each spline segment as a stationary capsule core. Scan all of them
    # so min_dist is the true closest approach, not just the first violation.
    for i in range(len(spline_points) - 1):
        dist = closest_distance_between_segments(
            capsule_p1, capsule_p2,
            spline_points[i], spline_points[i + 1],
        )
        if dist < min_dist:
            min_dist = dist

    collision = min_dist <= collision_distance_threshold
    return collision, min_dist


def check_capsule_splines_collision(capsule_p1, capsule_p2, capsule_radius,
                                    spline_segments, spline_radius):
    """Check a capsule against several spline pieces, returning the min distance.

    `spline_segments` is an iterable of (N, 3) arrays (e.g. the visible spline
    plus its extrapolated tips). Each piece is checked independently so no
    spurious segment is created where two pieces would otherwise join.

    Returns (collision, min_dist) aggregated across all pieces.
    """
    min_dist = float("inf")
    collision = False
    for seg in spline_segments:
        if len(seg) < 2:
            continue
        col, dist = check_capsule_spline_collision(
            capsule_p1, capsule_p2, capsule_radius, seg, spline_radius)
        min_dist = min(min_dist, dist)
        collision = collision or col
    return collision, min_dist


if __name__ == "__main__":
    # 1. Generate a synthetic continuous spline
    t = np.linspace(0, 10, 10)
    control_points = np.vstack((t, np.sin(t) * 2, np.cos(t / 2) * 2))

    # Fit and oversample to discretize the continuous curve
    tck, u = splprep(control_points, s=0)
    u_fine = np.linspace(0, 1, 100)  # 100 line segments
    discretized_spline = np.array(splev(u_fine, tck)).T

    spline_radius = 0.5
    capsule_radius = 0.2

    # Test Case A: capsule far from the spline (safe)
    capsule_A_p1 = np.array([5.0, 5.0, 5.0])
    capsule_A_p2 = np.array([6.0, 6.0, 6.0])

    # Test Case B: capsule intersecting the spline (collision)
    capsule_B_p1 = np.array([5.0, 0.0, 0.0])
    capsule_B_p2 = np.array([5.0, 1.0, 0.0])

    is_col_A, dist_A = check_capsule_spline_collision(
        capsule_A_p1, capsule_A_p2, capsule_radius, discretized_spline, spline_radius)
    print(f"Test A -> Collision: {is_col_A}, Min Distance: {dist_A:.3f}")

    is_col_B, dist_B = check_capsule_spline_collision(
        capsule_B_p1, capsule_B_p2, capsule_radius, discretized_spline, spline_radius)
    print(f"Test B -> Collision: {is_col_B}, Min Distance: {dist_B:.3f}")
