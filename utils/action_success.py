#!/usr/bin/env python3
"""Action success probability + joint clearance-success score (Section 4).

The clearance ranking (Section 3 / Algorithm 2) only asks "does this push avoid
the artery?". Section 4 adds a second factor: "is this push also close to the
nominal behavior the network actually wanted?". A push that swings wide of the
artery is useless if it no longer accomplishes the task, so we re-rank by the
*joint* probability of clearing the obstacle **and** succeeding functionally.

We model the success probability of a sampled action ``a`` as a decaying
function of its squared Mahalanobis distance from the mean action ``a0``,
scaled by the empirical covariance of the generated candidate distribution:

    P(success | a) = P_max * exp( -1/2 * (a - a0)^T Sigma_A^+ (a - a0) )

Mapping the low-dimensional latent space to the higher-dimensional action space
leaves ``Sigma_A`` rank-deficient, so we use the Moore-Penrose pseudo-inverse
``Sigma_A^+``. We take ``P_max = 1.0`` (the mean action is the optimal nominal
behavior). Assuming clearance and success are independent given the action, the
final scalar metric is their product:

    P(clearance, success | a, obstacle) = P(success | a) * P(clearance | a, obstacle)
"""

import numpy as np


def action_vectors(cand_starts, cand_ends):
    """Stack candidate pushes into the model's native action vectors.

    A push is a start point plus a displacement (no rotation), matching how the
    model samples actions (``start_point`` + ``displacement``). We form the 6D
    action ``a = [start, end - start]`` so the success metric scores both where
    a push begins and which way it drives.

    Args:
        cand_starts: (N, 3) push start points.
        cand_ends: (N, 3) push end points.

    Returns:
        (N, 6) array of action vectors [start_x, start_y, start_z, dx, dy, dz].
    """
    starts = np.asarray(cand_starts, dtype=float)
    ends = np.asarray(cand_ends, dtype=float)
    return np.hstack([starts, ends - starts])


def success_probabilities(actions, p_max=1.0, rcond=1e-12):
    """P(success | a) for every candidate via Mahalanobis distance from the mean.

    The empirical mean ``a0`` and covariance ``Sigma_A`` are estimated from the
    batch of candidate actions itself. ``Sigma_A`` is generally rank-deficient
    (a low-dimensional latent lifted into a higher-dimensional action space), so
    we invert it with the Moore-Penrose pseudo-inverse and read the squared
    Mahalanobis distance off the pseudo-inverse directly.

    Args:
        actions: (N, D) candidate action vectors (see ``action_vectors``).
        p_max: probability assigned to the mean action (1.0 per the writeup).
        rcond: relative singular-value cutoff for the pseudo-inverse; drops the
            near-zero directions that come from the rank deficiency.

    Returns:
        (N,) array of success probabilities in the original candidate order.
    """
    actions = np.asarray(actions, dtype=float)
    a0 = actions.mean(axis=0)
    deviations = actions - a0                         # (N, D)

    # Empirical covariance of the generated action distribution.
    Sigma_A = np.cov(actions, rowvar=False)           # (D, D)
    Sigma_A_pinv = np.linalg.pinv(Sigma_A, rcond=rcond)

    # Squared Mahalanobis distance per candidate: row_i Sigma^+ row_i^T.
    maha_sq = np.einsum("ni,ij,nj->n", deviations, Sigma_A_pinv, deviations)
    maha_sq = np.clip(maha_sq, 0.0, None)             # guard tiny negatives
    return p_max * np.exp(-0.5 * maha_sq)


def joint_clearance_success(p_clear, p_success):
    """Joint probability of clearing the artery *and* succeeding functionally.

    Assuming the two events are independent given the action, the joint metric
    is their elementwise product. This is the final scalar used to re-rank.

    Args:
        p_clear: (N,) per-push clearance probabilities (Section 3).
        p_success: (N,) per-push success probabilities (Section 4).

    Returns:
        (N,) joint probabilities, original candidate order.
    """
    return np.asarray(p_clear, dtype=float) * np.asarray(p_success, dtype=float)


def rank_joint_trajectories(cand_starts, cand_ends, p_clear,
                            p_max=1.0, rcond=1e-12):
    """Rank candidate pushes by joint clearance-and-success probability.

    Convenience wrapper tying Section 4 together: build the action vectors,
    score their success probability, multiply by the supplied clearance
    probabilities, and sort safest/best-first.

    Args:
        cand_starts, cand_ends: (N, 3) candidate push endpoints.
        p_clear: (N,) clearance probabilities from the Section 3 ranker.
        p_max, rcond: forwarded to ``success_probabilities``.

    Returns:
        order: (N,) candidate indices sorted by descending joint probability.
        p_success: (N,) success probabilities, original order.
        p_joint: (N,) joint probabilities, original order.
    """
    actions = action_vectors(cand_starts, cand_ends)
    p_success = success_probabilities(actions, p_max=p_max, rcond=rcond)
    p_joint = joint_clearance_success(p_clear, p_success)
    order = np.argsort(-p_joint)  # descending: best joint score first
    return order, p_success, p_joint


if __name__ == "__main__":
    # Smoke test: a tight cluster of actions with a few outliers. Outliers
    # should get low success probability; the mean action should get ~p_max.
    rng = np.random.default_rng(0)
    starts = rng.standard_normal((50, 3)) * 0.01
    ends = starts + np.array([0.05, 0.0, 0.0]) + rng.standard_normal((50, 3)) * 0.01
    starts[0] += 0.2                                  # an outlier push

    actions = action_vectors(starts, ends)
    p_success = success_probabilities(actions)
    print("success prob range:", round(p_success.min(), 4),
          "..", round(p_success.max(), 4))
    print("outlier #0 success prob:", round(float(p_success[0]), 4))

    p_clear = rng.uniform(0.0, 1.0, size=len(starts))
    order, p_succ, p_joint = rank_joint_trajectories(starts, ends, p_clear)
    print("best joint candidate:", int(order[0]),
          "p_joint=", round(float(p_joint[order[0]]), 4))
