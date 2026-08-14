"""Measure the mismatch r between the density the sample was ACTUALLY drawn
from and the denominator the shipped estimator reports against.

r = qbar_star / qbar_reported, where
  qbar_star      : draw-proportional mixture of MANY reconstructed proposals,
                   with the prior component carrying its TRUE share nu_n
  qbar_reported  : what extract_posterior uses -- m = 20 snapshots, prior mass
                   floored at delta = 0.5/(m+1)

The two contributions are separated:
  r_floor : same m, floored prior weight vs true nu_n   -> the Proposition-4 tilt
  r_quad  : same prior weight, m = 20 vs m = M_REF      -> the snapshot quadrature

Everything is post hoc: one evaluated history, several denominators over it.
No re-simulation, no propagator change.
"""
import sys, os, numpy as np, random
sys.path.insert(0, "propulate")
from scipy.special import logsumexp
from propulate.propagators.abcpmc import ABCPMC
from propulate.population import Individual

LO, HI = -3.0, 3.0
LIMITS = {"x": (LO, HI)}
SIG, NOBS, MU_TRUE, SEED = 1.0, 25, 0.4, 20260730
N_SIMS, K = 12000, 100
M_REF = 400
CACHE = "/tmp/claude-1000/-home-juhe-bwSyncShare-Code-async-abc-paper/07b2399a-f7fb-4aca-ae08-9ce6bfef146c/scratchpad/hist_cache.npz"


def build_history():
    rng_np = np.random.default_rng(SEED)
    ybar = float(rng_np.normal(MU_TRUE, SIG / np.sqrt(NOBS)))
    prop = ABCPMC(LIMITS, k=K, kernel="gaussian", scheduler_type="quantile",
                  amis_snapshots=20, rng=random.Random(SEED))
    hist = []
    for i in range(N_SIMS):
        c = prop(hist)
        c.generation = i
        c.loss = abs(float(rng_np.normal(c.position[0], SIG / np.sqrt(NOBS))) - ybar)
        hist.append(c)
    return prop, hist, ybar


def denominator(prop, hist, n_proposals, prior_weight):
    """Replicates extract_posterior's denominator with (m, w_prior) exposed.

    Returns log qbar evaluated at every history point, and the number of
    snapshots actually used.
    """
    n = len(hist)
    positions = np.stack([ind.position for ind in hist])
    archive_idx = [i for i, ind in enumerate(hist) if ind.tolerance is not None]
    step = max(1, len(archive_idx) // n_proposals)
    picks = list(archive_idx[::step][:n_proposals])
    if archive_idx[-1] not in picks:
        picks.append(archive_idx[-1])
    snaps, taus = [], []
    for tau in picks:
        arch = prop._reconstruct_archive(hist[:tau], hist[tau].tolerance)
        if arch is None:
            continue
        snaps.append(prop._build_proposal(arch, hist[tau].tolerance))
        taus.append(tau)
    m = len(snaps)
    arch_a = np.asarray(archive_idx)
    seg = np.searchsorted(np.asarray(taus[1:]), arch_a, side="right")
    counts = np.bincount(seg, minlength=m).astype(float)
    wp = prior_weight if prior_weight is not None else max(
        (n - len(archive_idx)) / n, 0.5 / (m + 1))
    mix_w = np.append(counts * ((1.0 - wp) / counts.sum()), wp)
    log_mix = np.log(mix_w)
    log_prior = float(np.log(prop.prior_density))
    out = np.empty(n)
    for s0 in range(0, n, 65536):
        s1 = min(s0 + 65536, n)
        comp = np.empty((m + 1, s1 - s0))
        for s, sn in enumerate(snaps):
            comp[s] = sn.log_mixture_density(positions[s0:s1])
        comp[m] = log_prior
        out[s0:s1] = logsumexp(comp + log_mix[:, None], axis=0)
    return out, m, wp


def reported_weights(prop, hist, log_qbar):
    losses = np.array([float(i.loss) for i in hist])
    eps = min(i.tolerance for i in hist if i.tolerance is not None)
    lw = float(np.log(prop.prior_density)) + prop._kernel_fn.log_weight(losses, eps) - log_qbar
    lw -= lw.max()
    w = np.exp(lw)
    return w / w.sum()


if __name__ == "__main__":
    prop, hist, ybar = build_history()
    n = len(hist)
    n_prior = sum(1 for i in hist if i.tolerance is None)
    nu = n_prior / n
    pos = np.stack([i.position for i in hist])[:, 0]
    print(f"n={n}  bootstrap draws={n_prior}  nu_n={nu:.3e}  "
          f"analytic posterior N({ybar:.4f}, {SIG/np.sqrt(NOBS):.4f}^2)")

    lq_rep, m_rep, wp_rep = denominator(prop, hist, 20, None)          # as shipped
    lq_nof, _, wp_nof = denominator(prop, hist, 20, max(nu, 1e-12))    # floor removed
    lq_ref, m_ref, wp_ref = denominator(prop, hist, M_REF, max(nu, 1e-12))  # both fixed
    print(f"shipped: m={m_rep}, w_prior={wp_rep:.4f} (floor active: "
          f"{wp_rep > nu})   reference: m={m_ref}, w_prior={wp_ref:.3e}\n")

    w_rep = reported_weights(prop, hist, lq_rep)
    w_ref = reported_weights(prop, hist, lq_ref)

    def zeta(log_num, log_den, w):
        r = np.exp(log_num - log_den)
        return float(np.sum(w * np.abs(r - 1.0))), r

    z_tot, r_tot = zeta(lq_ref, lq_rep, w_rep)
    z_flo, r_flo = zeta(lq_nof, lq_rep, w_rep)
    z_qua, r_qua = zeta(lq_ref, lq_nof, w_rep)
    for name, z, r in (("total", z_tot, r_tot), ("floor only", z_flo, r_flo),
                       ("quadrature only", z_qua, r_qua)):
        bound = z / (1 - z) if z < 1 else float("nan")
        print(f"{name:>16}: zeta = E_pi|r-1| = {z:.4f}   TV bound = {bound:.4f}"
              f"   r in [{r.min():.3f}, {r.max():.3f}]")

    mean_rep = float(np.sum(w_rep * pos)); mean_ref = float(np.sum(w_ref * pos))
    sd_rep = float(np.sqrt(np.sum(w_rep * (pos - mean_rep) ** 2)))
    sd_ref = float(np.sqrt(np.sum(w_ref * (pos - mean_ref) ** 2)))
    tv = 0.5 * float(np.abs(w_rep - w_ref).sum())
    print(f"\nreported  (m=20, floored): mean {mean_rep:+.5f}  sd {sd_rep:.5f}  "
          f"ESS frac {1/np.sum(w_rep**2)/n:.4f}")
    print(f"reference (m={m_ref}, true nu): mean {mean_ref:+.5f}  sd {sd_ref:.5f}  "
          f"ESS frac {1/np.sum(w_ref**2)/n:.4f}")
    print(f"exact TV between the two reported posteriors: {tv:.4f}")
    print(f"|mean shift| {abs(mean_rep-mean_ref):.5f}   analytic sd {SIG/np.sqrt(NOBS):.4f}")
