#!/usr/bin/env python3
"""Probabilistic collision checking -- reference loop (Section 3, Algorithm 2).

This is the readable, per-push reference for the probabilistic ranking, the
analogue of ``deterministic_capsule_collision.py`` on the deterministic side.

Where the deterministic side assumes the artery sits exactly on its fitted
spline, here the spline control parameters are Gaussian, ``w ~ N(mu_w, Sigma_w)``.
At each waypoint ``t`` the parameter uncertainty is projected into Euclidean
space via the basis matrix ``Phi_t``:

    mu_t    = Phi_t @ mu_w                  (3,)   spline mean at t
    Sigma_t = Phi_t @ Sigma_w @ Phi_t.T     (3,3)  spatial covariance at t

The offset from the deterministic robot waypoint ``y_t`` to the spline is then
Gaussian, ``v_t ~ N(mu_t - y_t, Sigma_t)``, and the squared distance
``D^2 = v_t^T v_t`` is a quadratic form in normal variables (a generalized
non-central chi-squared). Following the whitening approximation in the paper we
scale the threshold by the largest spatial eigenvalue and evaluate the survival
function of a non-central chi-squared with 3 dof, the non-centrality being the
squared Mahalanobis offset of ``y_t`` from the spline mean:

    P(clearance_t) = P(D^2 > d^2) ~ 1 - F_{ncx2}( d^2 / lambda_max ; df=3, nc )

The per-waypoint clearance probabilities are multiplied into a joint safety
probability for the whole push, and pushes are ranked safest-first.

``utils.probabilistic_collision_ranking`` is the batched / GPU equivalent of
this loop; the two are cross-checked in that module's ``__main__``.
"""

import numpy as np
from scipy.stats import ncx2
from scipy.linalg import cholesky, solve_triangular


def push_safety_probability(mu_w, Sigma_w, waypoints, Phi_matrices, d,
                            jitter=1e-6):
    """Joint probability that a single push clears the Gaussian-spline artery.

    Args:
        mu_w: (3K,) mean of the B-spline control parameters.
        Sigma_w: (3K, 3K) covariance of the control parameters.
        waypoints: (M, 3) the push discretized into M spatial waypoints y_t.
        Phi_matrices: (M, 3, 3K) basis matrices Phi_t, one per waypoint.
        d: float, safe clearance threshold (R_robot + R_artery).
        jitter: small value added to the diagonal of Sigma_t for numerically
            stable Cholesky / eigendecomposition.

    Returns:
        p_safe: float, product over waypoints of the per-waypoint clearance
            probability (the joint probability of remaining collision-free).
    """
    mu_w = np.asarray(mu_w, dtype=float)
    Sigma_w = np.asarray(Sigma_w, dtype=float)
    waypoints = np.asarray(waypoints, dtype=float)

    p_safe = 1.0
    for t in range(len(waypoints)):
        y_t = waypoints[t]          # robot coordinate at waypoint t
        Phi_t = Phi_matrices[t]     # spline basis matrix at waypoint t

        # 1. Project parameter uncertainty into Euclidean space.
        mu_t = Phi_t @ mu_w
        Sigma_t = Phi_t @ Sigma_w @ Phi_t.T

        # Tiny jitter keeps the decompositions well-conditioned.
        Sigma_t = Sigma_t + np.eye(3) * jitter

        # 2. Largest spatial variance -> conservative threshold scaling.
        lambda_max = np.max(np.linalg.eigvalsh(Sigma_t))

        # 3. Whiten the coordinate space via the Cholesky factor.
        L = cholesky(Sigma_t, lower=True)
        y_tilde = solve_triangular(L, y_t - mu_t, lower=True)

        # Squared Mahalanobis offset acts as the non-centrality parameter.
        mahalanobis_sq = float(np.sum(y_tilde ** 2))

        # 4. Probability the squared distance exceeds the threshold d.
        scaled_threshold_sq = (d ** 2) / lambda_max
        p_clearance = 1.0 - ncx2.cdf(scaled_threshold_sq, df=3,
                                     nc=mahalanobis_sq)

        # Update the joint safety probability for this push.
        p_safe *= p_clearance

    return p_safe


def score_probabilistic_trajectories(mu_w, Sigma_w, trajectories, Phi_matrices,
                                     d, jitter=1e-6):
    """Per-push safety probabilities, in the original candidate order.

    Args:
        mu_w, Sigma_w: Gaussian B-spline parameters (see above).
        trajectories: iterable of (M, 3) arrays, one per candidate push.
        Phi_matrices: (M, 3, 3K) basis matrices shared across candidates,
            indexed by waypoint t (as in Algorithm 2).
        d: float, safe clearance threshold.

    Returns:
        (N,) float array of joint safety probabilities, candidate order.
    """
    return np.array([
        push_safety_probability(mu_w, Sigma_w, T_i, Phi_matrices, d, jitter)
        for T_i in trajectories
    ], dtype=float)


def rank_probabilistic_trajectories(mu_w, Sigma_w, trajectories, Phi_matrices,
                                    d, jitter=1e-6):
    """Rank candidate pushes safest-first by joint clearance probability.

    Returns the list of trajectories sorted from highest to lowest probability
    of safety (mirrors the user's original reference signature).
    """
    probabilities = score_probabilistic_trajectories(
        mu_w, Sigma_w, trajectories, Phi_matrices, d, jitter)
    order = np.argsort(-probabilities)  # descending: safest first
    return [trajectories[i] for i in order]


if __name__ == "__main__":
    # Tiny smoke test on synthetic Gaussian-spline inputs.
    rng = np.random.default_rng(0)
    K, M, N = 6, 8, 5
    A = rng.standard_normal((3 * K, 3 * K))
    Sigma_w = A @ A.T + np.eye(3 * K)        # SPD
    mu_w = rng.standard_normal(3 * K)
    Phi = rng.standard_normal((M, 3, 3 * K))
    trajs = [rng.standard_normal((M, 3)) for _ in range(N)]

    probs = score_probabilistic_trajectories(mu_w, Sigma_w, trajs, Phi, d=0.5)
    print("safety probabilities:", np.round(probs, 4))
    print("safest candidate:", int(np.argmax(probs)),
          "riskiest:", int(np.argmin(probs)))
