"""The same denominator-mismatch diagnostic, on a REAL production history:
the Cellular Potts run behind the paper's CPM posterior (rerun_20260707).

Multi-rank (48 workers), 2-D with a correlated archive covariance, so this
exercises both things the 1-D toy could not: real scale/ordering, and the
product-of-marginals truncation normalizer.

Re-analysis only: reads the stored raw_results.csv, rebuilds the history, and
re-runs the SHIPPED extract_posterior denominator at several snapshot counts.
No simulation, no propagator change.

Caveat recorded in the output: raw_results.csv is the set of evaluations tagged
by worker, not any single rank's arrival-ordered view, so the history order here
is reconstructed from sim_end_time and is a *valid* history rather than *the*
history the run reported against.
"""
import sys, random, numpy as np, pandas as pd
sys.path.insert(0, "/home/juhe/bwSyncShare/Code/async-abc-paper/propulate")
from scipy.special import logsumexp
from propulate.propagators.abcpmc import ABCPMC
from propulate.population import Individual

CSV = ("/tmp/claude-1000/-home-juhe-bwSyncShare-Code-async-abc-paper/"
       "07b2399a-f7fb-4aca-ae08-9ce6bfef146c/scratchpad/cpm/raw_results.csv")
LIMITS = {"division_rate": (0.0, 1.0), "motility": (0.0, 1.0)}
K, PSCALE, M_SHIPPED, M_REF = 100, 0.8, 20, 400


def load(replicate):
    df = pd.read_csv(CSV, low_memory=False)
    df = df[(df.method == "async_propulate_abc") & (df.replicate == replicate)]
    df = df.sort_values("sim_end_time", kind="mergesort").reset_index(drop=True)
    hist = []
    for r in df.itertuples(index=False):
        ind = Individual({"division_rate": float(r.param_division_rate),
                          "motility": float(r.param_motility)}, LIMITS)
        ind.loss = float(r.loss)
        ind.tolerance = None if pd.isna(r.tolerance) else float(r.tolerance)
        ind.weight = None if pd.isna(r.weight) else float(r.weight)
        hist.append(ind)
    return hist


def denominator(prop, hist, n_proposals, prior_weight):
    n = len(hist)
    positions = np.stack([i.position for i in hist])
    arch = [i for i, ind in enumerate(hist) if ind.tolerance is not None]
    step = max(1, len(arch) // n_proposals)
    picks = list(np.asarray(arch)[::step][:n_proposals])
    if arch[-1] not in picks:
        picks.append(arch[-1])
    snaps, taus = [], []
    for tau in picks:
        a = prop._reconstruct_archive(hist[:tau], hist[tau].tolerance)
        if a is None:
            continue
        snaps.append(prop._build_proposal(a, hist[tau].tolerance))
        taus.append(tau)
    m = len(snaps)
    seg = np.searchsorted(np.asarray(taus[1:]), np.asarray(arch), side="right")
    counts = np.bincount(seg, minlength=m).astype(float)
    wp = prior_weight if prior_weight is not None else max((n - len(arch)) / n, 0.5 / (m + 1))
    mix_w = np.append(counts * ((1.0 - wp) / counts.sum()), wp)
    log_mix, log_prior = np.log(mix_w), float(np.log(prop.prior_density))
    out = np.empty(n)
    for s0 in range(0, n, 65536):
        s1 = min(s0 + 65536, n)
        comp = np.empty((m + 1, s1 - s0))
        for s, sn in enumerate(snaps):
            comp[s] = sn.log_mixture_density(positions[s0:s1])
        comp[m] = log_prior
        out[s0:s1] = logsumexp(comp + log_mix[:, None], axis=0)
    return out, m, wp


def weights_from(prop, hist, log_qbar):
    losses = np.array([i.loss for i in hist])
    eps = min(i.tolerance for i in hist if i.tolerance is not None)
    lw = float(np.log(prop.prior_density)) + prop._kernel_fn.log_weight(losses, eps) - log_qbar
    lw -= lw.max()
    w = np.exp(lw)
    return w / w.sum()


if __name__ == "__main__":
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hist = load(rep)
    n = len(hist)
    n_prior = sum(1 for i in hist if i.tolerance is None)
    nu = n_prior / n
    pos = np.stack([i.position for i in hist])
    print(f"CPM replicate {rep}: n={n} async evaluations, bootstrap={n_prior} "
          f"(nu_n={nu:.4f}), delta at m=20 is {0.5/21:.4f} -> floor "
          f"{'binds' if nu < 0.5/21 else 'INACTIVE'}")
    prop = ABCPMC(LIMITS, k=K, kernel="gaussian", scheduler_type="acceptance_rate",
                  perturbation_scale=PSCALE, amis_snapshots=M_SHIPPED,
                  rng=random.Random(0))
    lq_ship, m_s, wp_s = denominator(prop, hist, M_SHIPPED, None)
    # Quadrature isolated: same prior weight on both sides, only m differs.
    # Immune to the writer bug that erases the bootstrap/archive split.
    lq_nof, _, _ = denominator(prop, hist, M_SHIPPED, wp_s)
    lq_ref, m_r, wp_r = denominator(prop, hist, M_REF, wp_s)
    print(f"shipped: m={m_s} w_prior={wp_s:.4f}   reference: m={m_r} w_prior={wp_r:.4f}\n")
    w_ship, w_ref = weights_from(prop, hist, lq_ship), weights_from(prop, hist, lq_ref)

    def z(a, b, w):
        r = np.exp(a - b)
        return float(np.sum(w * np.abs(r - 1.0))), r
    for name, (zz, rr) in (("total", z(lq_ref, lq_ship, w_ship)),
                           ("floor only", z(lq_nof, lq_ship, w_ship)),
                           ("quadrature only", z(lq_ref, lq_nof, w_ship))):
        b = zz / (1 - zz) if zz < 1 else float("nan")
        print(f"{name:>16}: zeta = {zz:.4f}   TV bound = {b:.4f}   "
              f"r in [{rr.min():.3f}, {rr.max():.3f}]")
    tv = 0.5 * float(np.abs(w_ship - w_ref).sum())
    for lab, w in (("shipped  (m=21, floored)", w_ship), ("reference (m=401, true nu)", w_ref)):
        mu = w @ pos
        sd = np.sqrt(w @ (pos - mu) ** 2)
        print(f"{lab}: mean=({mu[0]:.5f}, {mu[1]:.5f})  sd=({sd[0]:.5f}, {sd[1]:.5f})  "
              f"ESS frac={1/np.sum(w**2)/n:.4f}")
    print(f"exact TV between the two reported posteriors: {tv:.4f}")
