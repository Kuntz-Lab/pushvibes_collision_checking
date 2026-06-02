#!/usr/bin/env python3
"""Batched, GPU-friendly probabilistic collision scoring & ranking of pushes.

This is the vectorized / GPU equivalent of the per-push reference loop in
``utils.probabilistic_collision`` (Section 3, Algorithm 2). Where the reference
loops over candidates and waypoints in Python with ``scipy.stats.ncx2``, this
module evaluates the whole candidate fan at once as batched tensor ops and ships
its own torch implementation of the non-central chi-squared CDF, so everything
runs on the GPU when one is available.

The artery B-spline control parameters are Gaussian, ``w ~ N(mu_w, Sigma_w)``.
At each waypoint ``t`` the basis matrix ``Phi_t`` projects this into Euclidean
space (``mu_t = Phi_t mu_w``, ``Sigma_t = Phi_t Sigma_w Phi_t^T``). The squared
offset of the deterministic robot waypoint from the spline is a non-central
chi-squared (df=3); the per-waypoint clearance probability is its survival
function at the eigenvalue-scaled threshold ``d^2 / lambda_max``, and the joint
push safety probability is the product across waypoints.

The basis matrices ``Phi`` are shared across candidates (indexed by waypoint
``t``), matching the reference and Algorithm 2. See ``utils.probabilistic_spline``
for deriving ``mu_w, Sigma_w, Phi`` from a real artery point cloud.
"""

import math

import numpy as np
import torch


def _pick_device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ncx2_cdf(x, nc, df=3, max_terms=None):
    """Non-central chi-squared CDF, vectorized in torch (GPU + autograd ready).

    Uses the Poisson-mixture series

        F(x; k, lambda) = sum_j  Pois(j; lambda/2) * P((k + 2j)/2, x/2)

    where ``P(a, z) = torch.special.gammainc(a, z)`` is the regularized lower
    incomplete gamma (the central chi-squared CDF). The Poisson weights are
    accumulated in log-space for stability and the central case ``lambda = 0``
    is handled exactly (only the ``j = 0`` term survives).

    Args:
        x: tensor of thresholds (>= 0), broadcastable with ``nc``.
        nc: tensor of non-centrality parameters (>= 0).
        df: degrees of freedom (3 for 3D space).
        max_terms: number of series terms; defaults to an adaptive count that
            covers the Poisson mass for the largest ``nc``.

    Returns:
        Tensor (broadcast shape of ``x`` and ``nc``) of CDF values.
    """
    x, nc = torch.broadcast_tensors(torch.as_tensor(x), torch.as_tensor(nc))
    half_x = x / 2.0
    half_nc = nc / 2.0
    log_half_nc = torch.log(half_nc)  # -inf where nc == 0 (handled per-term)

    if max_terms is None:
        m = float(half_nc.max()) if half_nc.numel() else 0.0
        # Mean of the mixing Poisson is half_nc; go well past it. Capped because
        # when nc >> x every term's gamma factor is ~0, so the tail is harmless.
        max_terms = min(int(m + 10.0 * np.sqrt(m) + 50.0), 2000)

    cdf = torch.zeros_like(half_x)
    for j in range(max_terms):
        a_j = torch.full_like(half_x, (df + 2 * j) / 2.0)
        gamma_term = torch.special.gammainc(a_j, half_x)  # central chi2 CDF
        if j == 0:
            log_w = -half_nc
        else:
            # lgamma(j+1) = log(j!)
            log_w = -half_nc + j * log_half_nc - math.lgamma(j + 1)
        weight = torch.exp(log_w)  # -> 0 where nc == 0 and j >= 1
        cdf = cdf + weight * gamma_term
    return cdf


def _chunk_safety(mu_w, Sigma_w, traj_c, Phi_c, d, dev, dtype, jitter,
                  max_terms):
    """Joint safety probability for one chunk of pushes.

    ``Phi_c`` is either (M, 3, 3K) -- shared across the chunk (index pairing) --
    or (c, M, 3, 3K) -- per push (nearest pairing). Leading dims are handled by
    ``...`` einsum / broadcasting so both shapes flow through unchanged.
    """
    mu_t = torch.einsum("...ij,j->...i", Phi_c, mu_w)            # (M,3)/(c,M,3)
    Sigma_t = torch.einsum("...ik,kl,...jl->...ij", Phi_c, Sigma_w, Phi_c)
    Sigma_t = Sigma_t + torch.eye(3, device=dev, dtype=dtype) * jitter
    lam = torch.linalg.eigvalsh(Sigma_t)[..., -1]               # (M,)/(c,M)
    L = torch.linalg.cholesky(Sigma_t)                          # (..,3,3)

    offset = (traj_c - mu_t).unsqueeze(-1)                      # (c,M,3,1)
    if L.dim() == 3:                       # shared (M,3,3) -> broadcast over c
        L = L.unsqueeze(0)
    y_tilde = torch.linalg.solve_triangular(L, offset, upper=False)
    maha_sq = (y_tilde.squeeze(-1) ** 2).sum(-1)               # (c, M)

    scaled_thr = (float(d) ** 2) / lam                         # (M,)/(c,M)
    p_clear = 1.0 - ncx2_cdf(scaled_thr, maha_sq, df=3, max_terms=max_terms)
    return p_clear.prod(dim=1)                                 # (c,)


def collision_probabilities(mu_w, Sigma_w, trajectories, Phi, d,
                            device=None, dtype=torch.float64, jitter=1e-6,
                            chunk_size=None, max_terms=None):
    """Joint safety probability for every candidate push, batched.

    Args:
        mu_w: (3K,) mean of the B-spline control parameters.
        Sigma_w: (3K, 3K) covariance of the control parameters.
        trajectories: (N, M, 3) array-like of M waypoints per push (lists ok).
        Phi: basis matrices pairing waypoints to the spline, either
            (M, 3, 3K) shared across pushes (index pairing) or (N, M, 3, 3K)
            per push (nearest pairing). See ``utils.probabilistic_spline``.
        d: float, safe clearance threshold (R_robot + R_artery).
        device: torch device or string; defaults to CUDA when available.
        dtype: compute dtype; float64 keeps tight agreement with the scipy
            reference and stable products of many small probabilities.
        jitter: diagonal added to Sigma_t for stable decompositions.
        chunk_size: cap on candidates processed per batch (bounds the (chunk, M)
            intermediates for large N). ``None`` does them all at once.
        max_terms: override the ncx2 series length (see ``ncx2_cdf``).

    Returns:
        (N,) float numpy array of joint safety probabilities, candidate order.
    """
    dev = _pick_device(device)
    traj = torch.as_tensor(np.asarray(trajectories), device=dev, dtype=dtype)
    if traj.ndim != 3 or traj.shape[-1] != 3:
        raise ValueError(f"trajectories must be (N, M, 3); got {tuple(traj.shape)}")
    n, m = traj.shape[0], traj.shape[1]

    mu_w_t = torch.as_tensor(np.asarray(mu_w), device=dev, dtype=dtype)
    Sigma_w_t = torch.as_tensor(np.asarray(Sigma_w), device=dev, dtype=dtype)
    Phi_t = torch.as_tensor(np.asarray(Phi), device=dev, dtype=dtype)

    per_candidate = Phi_t.ndim == 4
    if per_candidate:
        if Phi_t.shape[:2] != (n, m):
            raise ValueError(f"per-candidate Phi must be (N, M, 3, 3K)=({n}, {m}, "
                             f"3, .); got {tuple(Phi_t.shape)}")
    elif Phi_t.ndim == 3:
        if Phi_t.shape[0] != m:
            raise ValueError(f"shared Phi has M={Phi_t.shape[0]} but trajectories "
                             f"have M={m}.")
    else:
        raise ValueError(f"Phi must be (M,3,3K) or (N,M,3,3K); got "
                         f"{tuple(Phi_t.shape)}")

    step = chunk_size or n
    p_safe = torch.empty(n, device=dev, dtype=dtype)
    for i in range(0, n, step):
        traj_c = traj[i:i + step]                          # (c, M, 3)
        Phi_c = Phi_t[i:i + step] if per_candidate else Phi_t
        p_safe[i:i + step] = _chunk_safety(
            mu_w_t, Sigma_w_t, traj_c, Phi_c, d, dev, dtype, jitter, max_terms)

    return p_safe.cpu().numpy().astype(np.float64)


def rank_probabilistic_trajectories(mu_w, Sigma_w, trajectories, Phi, d,
                                    device=None, dtype=torch.float64,
                                    jitter=1e-6, chunk_size=None,
                                    max_terms=None):
    """Rank candidate pushes safest-first by joint clearance probability.

    Returns:
        order: (N,) int array of candidate indices sorted by descending safety
            probability (safest push first).
        p_safe: (N,) joint safety probability, in the original candidate order.
    """
    p_safe = collision_probabilities(
        mu_w, Sigma_w, trajectories, Phi, d, device=device, dtype=dtype,
        jitter=jitter, chunk_size=chunk_size, max_terms=max_terms)
    order = np.argsort(-p_safe)  # descending: safest first
    return order, p_safe


if __name__ == "__main__":
    # Cross-check the batched scores against the scipy per-push reference, and
    # the torch ncx2 CDF against scipy.stats.ncx2.
    from scipy.stats import ncx2 as scipy_ncx2
    from probabilistic_collision import score_probabilistic_trajectories

    rng = np.random.default_rng(0)

    # --- torch ncx2 CDF vs scipy over a grid (df = 3) ---
    xs = np.linspace(0.01, 30.0, 40)
    ncs = np.array([0.0, 0.5, 2.0, 8.0, 25.0])
    xx, nn = np.meshgrid(xs, ncs, indexing="ij")
    torch_cdf = ncx2_cdf(torch.as_tensor(xx, dtype=torch.float64),
                         torch.as_tensor(nn, dtype=torch.float64),
                         df=3).numpy()
    scipy_cdf = scipy_ncx2.cdf(xx, df=3, nc=nn)
    print(f"max |torch - scipy| ncx2 CDF error: "
          f"{np.abs(torch_cdf - scipy_cdf).max():.2e}")

    # --- batched safety probabilities vs scipy reference loop ---
    K, M, N = 6, 10, 40
    A = rng.standard_normal((3 * K, 3 * K))
    Sigma_w = A @ A.T + np.eye(3 * K)                  # SPD
    mu_w = rng.standard_normal(3 * K) * 0.1
    Phi = rng.standard_normal((M, 3, 3 * K)) * 0.3
    trajs = rng.standard_normal((N, M, 3)) * 0.2
    d = 0.8

    order, p_batched = rank_probabilistic_trajectories(
        mu_w, Sigma_w, trajs, Phi, d, device="cpu")
    p_ref = score_probabilistic_trajectories(mu_w, Sigma_w, list(trajs), Phi, d)

    print(f"[shared Phi]   max |batched - reference| p_safe error: "
          f"{np.abs(p_batched - p_ref).max():.2e}")
    print(f"               rankings match: "
          f"{np.array_equal(order, np.argsort(-p_ref))}")

    # --- per-candidate (nearest-pairing) Phi vs the reference loop ---
    Phi_pc = rng.standard_normal((N, M, 3, 3 * K)) * 0.3      # (N, M, 3, 3K)
    p_pc = collision_probabilities(mu_w, Sigma_w, trajs, Phi_pc, d, device="cpu")
    p_pc_ref = np.array([
        score_probabilistic_trajectories(mu_w, Sigma_w, [trajs[i]], Phi_pc[i], d)[0]
        for i in range(N)])
    print(f"[per-cand Phi] max |batched - reference| p_safe error: "
          f"{np.abs(p_pc - p_pc_ref).max():.2e}")
    print(f"safest candidate {int(order[0])} (p={p_batched[order[0]]:.4f}), "
          f"riskiest {int(order[-1])} (p={p_batched[order[-1]]:.4f})")
