#!/usr/bin/env python3
"""Deterministic batched, GPU-friendly collision scoring & ranking of candidate pushes.

Each candidate push is the capsule swept by the robot tip (a sphere of
``capsule_radius``) travelling linearly from its start to its end point. The
artery is a tube of ``spline_radius`` around a discretized centerline spline.

Where ``utils.capsule_collision`` checks one capsule at a time with a Python
loop over spline segments, this module evaluates *all* candidate capsules
against *all* spline segments at once as a single batched tensor op, so the
whole 500-candidate fan is scored in one shot (on the GPU when available).

The score is the signed surface clearance (meters):

    score = min_core_distance - (capsule_radius + spline_radius)

Higher is safer: positive = clearance to spare, negative = penetration depth,
0 = grazing contact. Ranking by this score puts the safest pushes on top and
the deepest collisions at the bottom -- the same quantity the visualizer
heat-maps onto each capsule.
"""

import numpy as np
import torch


def _pick_device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _segment_pairs(spline_segments):
    """Flatten an iterable of (Ni, 3) polylines into segment endpoint pairs.

    Returns ``(P, Q)`` each (M, 3): consecutive points within a polyline become
    one segment ``P[i] -> Q[i]``. Pieces are handled independently so no bogus
    segment bridges the gap between two separate polylines (e.g. the visible
    spline and an extrapolated tip).
    """
    p_list, q_list = [], []
    for seg in spline_segments:
        seg = np.asarray(seg, dtype=np.float32)
        if len(seg) < 2:
            continue
        p_list.append(seg[:-1])
        q_list.append(seg[1:])
    if not p_list:
        raise ValueError("No spline segment has >= 2 points to form a tube.")
    return np.concatenate(p_list, 0), np.concatenate(q_list, 0)


def batch_segment_segment_distance(p1, q1, p2, q2, eps=1e-8):
    """Pairwise minimum distance between two sets of 3D segments.

    Vectorized form of Christer Ericson's clamped closest-point-on-segments
    routine (Real-Time Collision Detection), computed for every pair.

    Args:
        p1, q1: (N, 3) tensors, endpoints of the N "capsule" segments.
        p2, q2: (M, 3) tensors, endpoints of the M "spline" segments.

    Returns:
        (N, M) tensor of closest-approach distances between the cores.
    """
    d1 = q1 - p1                       # (N, 3) capsule directions
    d2 = q2 - p2                       # (M, 3) spline directions
    # r[n, m] = p1[n] - p2[m]
    r = p1[:, None, :] - p2[None, :, :]            # (N, M, 3)

    a = (d1 * d1).sum(-1)[:, None]                 # (N, 1) |d1|^2
    e = (d2 * d2).sum(-1)[None, :]                 # (1, M) |d2|^2
    b = d1 @ d2.t()                                # (N, M) d1.d2
    c = torch.einsum("nk,nmk->nm", d1, r)          # (N, M) d1.r
    f = torch.einsum("mk,nmk->nm", d2, r)          # (N, M) d2.r

    denom = a * e - b * b                          # (N, M)
    # s along capsule, t along spline. Default s from the non-parallel formula;
    # fall back to 0 where the pair is parallel (denom ~ 0).
    s = torch.where(denom > eps, (b * f - c * e) / denom.clamp_min(eps),
                    torch.zeros_like(denom))
    s = s.clamp(0.0, 1.0)

    # t for the chosen s, then re-clamp s when t falls outside [0, 1].
    t = (b * s + f) / e.clamp_min(eps)

    t_lo = t < 0.0
    t_hi = t > 1.0
    t = t.clamp(0.0, 1.0)
    s_lo = (-c / a.clamp_min(eps)).clamp(0.0, 1.0)
    s_hi = ((b - c) / a.clamp_min(eps)).clamp(0.0, 1.0)
    s = torch.where(t_lo, s_lo, torch.where(t_hi, s_hi, s))

    # Degenerate capsules (a ~ 0): force s = 0 and recompute t from p1.
    degen = (a <= eps).expand_as(s)
    s = torch.where(degen, torch.zeros_like(s), s)
    t = torch.where(degen, (f / e.clamp_min(eps)).clamp(0.0, 1.0), t)

    c1 = p1[:, None, :] + d1[:, None, :] * s[..., None]   # (N, M, 3)
    c2 = p2[None, :, :] + d2[None, :, :] * t[..., None]    # (N, M, 3)
    return torch.linalg.vector_norm(c1 - c2, dim=-1)       # (N, M)


def collision_scores(cand_starts, cand_ends, capsule_radius,
                     spline_segments, spline_radius, device=None,
                     chunk_size=None):
    """Signed surface clearance (meters) for every candidate push, batched.

    Args:
        cand_starts, cand_ends: (N, 3) array-likes, capsule core endpoints in
            the camera frame.
        capsule_radius: float, swept robot-tip radius (meters).
        spline_segments: iterable of (Ni, 3) polylines defining the artery
            centerline (e.g. the visible spline plus extrapolated tips).
        spline_radius: float, artery tube radius (meters).
        device: torch device or string; defaults to CUDA when available.
        chunk_size: optional cap on candidates processed per batch, to bound
            the (N, M) intermediate when N or M is very large. ``None`` does it
            all at once.

    Returns:
        scores: (N,) float32 numpy array of signed clearance (min_dist minus the
            summed radii). Higher = safer; negative = penetrating.
        min_dists: (N,) float32 numpy array of raw core closest-approach
            distances (meters), independent of the radii.
    """
    dev = _pick_device(device)
    starts = torch.as_tensor(np.asarray(cand_starts, np.float32), device=dev)
    ends = torch.as_tensor(np.asarray(cand_ends, np.float32), device=dev)

    p2_np, q2_np = _segment_pairs(spline_segments)
    p2 = torch.as_tensor(p2_np, device=dev)
    q2 = torch.as_tensor(q2_np, device=dev)

    n = starts.shape[0]
    step = chunk_size or n
    min_dists = torch.empty(n, device=dev)
    for i in range(0, n, step):
        d = batch_segment_segment_distance(
            starts[i:i + step], ends[i:i + step], p2, q2)   # (chunk, M)
        min_dists[i:i + step] = d.min(dim=1).values

    scores = min_dists - (capsule_radius + spline_radius)
    return (scores.cpu().numpy().astype(np.float32),
            min_dists.cpu().numpy().astype(np.float32))


def rank_trajectories(cand_starts, cand_ends, capsule_radius,
                      spline_segments, spline_radius, device=None):
    """Rank candidate pushes safest-first by signed clearance.

    Returns:
        order: (N,) int array of candidate indices sorted by descending score
            (safest push first, deepest collision last).
        scores: (N,) signed clearance (meters), in the original candidate order.
        min_dists: (N,) raw core distances (meters), original candidate order.
    """
    scores, min_dists = collision_scores(
        cand_starts, cand_ends, capsule_radius,
        spline_segments, spline_radius, device=device)
    order = np.argsort(-scores)  # descending: highest score (safest) first
    return order, scores, min_dists


if __name__ == "__main__":
    # Cross-check the batched scores against the per-capsule reference loop.
    from deterministic_capsule_collision import check_capsule_splines_collision

    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 40)
    spline = np.vstack((t, np.sin(t), np.cos(t / 2))).T
    segs = [spline]
    starts = rng.uniform(-1, 11, size=(50, 3))
    ends = starts + rng.uniform(-2, 2, size=(50, 3))
    cap_r, sp_r = 0.2, 0.5

    scores, min_dists = collision_scores(starts, ends, cap_r, segs, sp_r,
                                         device="cpu")
    max_err = 0.0
    for i in range(len(starts)):
        _, ref = check_capsule_splines_collision(
            starts[i], ends[i], cap_r, segs, sp_r)
        max_err = max(max_err, abs(ref - min_dists[i]))
    print(f"max |batched - reference| min-distance error: {max_err:.2e} m")
    print(f"safest candidate {int(np.argmax(scores))} "
          f"(clearance {scores.max() * 1000:+.1f} mm), "
          f"riskiest {int(np.argmin(scores))} "
          f"(clearance {scores.min() * 1000:+.1f} mm)")
