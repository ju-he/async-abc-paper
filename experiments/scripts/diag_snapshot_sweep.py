"""Post-hoc re-analysis: how much does the reported posterior depend on the
snapshot count m (and hence on the delta = 0.5/(m+1) prior floor)?

Drives the shipped ABCPMC propagator serially (it is a pure function of the
history, so no MPI is needed), then re-runs extract_posterior at a range of
n_proposals on the SAME history.  Nothing here re-simulates: every variant
reads the identical evaluated history, exactly as a post-hoc pass over a
stored run would.

Benchmark: Gaussian mean in d dimensions.  y_j ~ N(mu, sigma^2), observed
ybar; discrepancy rho = |mean(sim) - ybar| (d=1) or the Euclidean norm.
The analytic posterior is Gaussian, so the reported posterior can be scored.
"""
import sys, numpy as np, random
from scipy.stats import norm

sys.path.insert(0, "propulate")
from propulate.propagators.abcpmc import ABCPMC
from propulate.population import Individual

D = 1
LO, HI = -3.0, 3.0
LIMITS = {"x": (LO, HI)}
MU_TRUE = 0.4
SIG = 1.0
NOBS = 25
SEED = 20260730

rng_np = np.random.default_rng(SEED)
ybar = float(rng_np.normal(MU_TRUE, SIG / np.sqrt(NOBS)))
# Analytic posterior under a flat prior on the box: N(ybar, SIG^2/NOBS)
POST_M, POST_S = ybar, SIG / np.sqrt(NOBS)


def simulate(mu: float) -> float:
    return abs(float(rng_np.normal(mu, SIG / np.sqrt(NOBS))) - ybar)


def run(n_sims: int, k: int, amis_snapshots: int):
    prop = ABCPMC(
        LIMITS, k=k, kernel="gaussian", scheduler_type="quantile",
        amis_snapshots=amis_snapshots, rng=random.Random(SEED),
    )
    hist = []
    for i in range(n_sims):
        child = prop(hist)
        child.generation = i
        child.loss = simulate(child.position[0])
        hist.append(child)
    return prop, hist


def summarize(pos, w, label):
    m = float(np.sum(w * pos[:, 0]))
    s = float(np.sqrt(np.sum(w * (pos[:, 0] - m) ** 2)))
    ess = float(1.0 / np.sum(w ** 2)) / len(w)
    return dict(label=label, mean=m, sd=s, ess_frac=ess, wmax=float(w.max()))


if __name__ == "__main__":
    N, K = 12000, 100
    prop, hist = run(N, K, amis_snapshots=20)
    n_prior = sum(1 for i in hist if i.tolerance is None)
    print(f"history: n={len(hist)}  bootstrap draws={n_prior} "
          f"(nu_n={n_prior/len(hist):.2e})  analytic posterior "
          f"N({POST_M:.4f}, {POST_S:.4f}^2)")
    n_arch = len(hist) - n_prior
    print(f"delta = 0.5/(m+1) at m=20 is {0.5/21:.4f}; "
          f"nu_n < delta, so the floor binds\n")

    rows = []
    ref = None
    for m in (5, 10, 20, 50, 100, 200, 500, 1000, n_arch):
        pos, w = prop.extract_posterior(hist, n_proposals=m)
        r = summarize(pos, w, m)
        r["delta"] = 0.5 / (min(m, n_arch) + 2)  # m<-#snapshots actually used
        if ref is None:
            ref = w  # m=5 baseline, replaced below
        rows.append((r, w))

    full_w = rows[-1][1]
    print(f"{'m':>7} {'delta':>8} {'post mean':>11} {'post sd':>9} "
          f"{'ESS frac':>9} {'w_max':>9} {'|err mean|':>11} {'TV vs full':>11}")
    for r, w in rows:
        tv = 0.5 * float(np.abs(w - full_w).sum())
        print(f"{r['label']:>7} {r['delta']:>8.4f} {r['mean']:>11.5f} "
              f"{r['sd']:>9.5f} {r['ess_frac']:>9.4f} {r['wmax']:>9.2e} "
              f"{abs(r['mean']-POST_M):>11.5f} {tv:>11.4f}")
