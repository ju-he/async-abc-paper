"""Does a confounded (ridged) target break stabilization?

The production measurement found ||q_tau - q_inf|| ~ tau^-b with b well below
the 1/2 the CLT's rate condition needs, on a run whose bandwidth was still
tightening throughout (min_tol unset). The conjecture is that a pinned
bandwidth lets the archive settle and pushes b past 1/2.

Controlled test: the same 2-D Gaussian-mean benchmark, same seed, same
propagator, run twice -- once with the data-driven schedule and once with
min_tol pinned at the tolerance the unfloored run reaches a third of the way in.
Everything else identical, so the only difference is whether epsilon keeps
moving.
"""
import sys, random, numpy as np
sys.path.insert(0, "/home/juhe/bwSyncShare/Code/async-abc-paper/propulate")
from propulate.propagators.abcpmc import ABCPMC
from propulate.population import Individual

LO, HI = -3.0, 3.0
LIMITS = {"m1": (LO, HI), "m2": (LO, HI)}
MU_TRUE = np.array([0.4, -0.6])
SIG, NOBS, SEED = 1.0, 25, 20260731
N, K, NGRID = 12000, 100, 140


def run(min_tol, n=N, confound=False):
    rng = np.random.default_rng(SEED)
    yobs = rng.normal(MU_TRUE, SIG / np.sqrt(NOBS))
    prop = ABCPMC(LIMITS, k=K, kernel="gaussian", scheduler_type="acceptance_rate",
                  perturbation_scale=0.8, amis_snapshots=20, min_tol=min_tol,
                  rng=random.Random(SEED))
    hist, tols = [], []
    for i in range(n):
        c = prop(hist)
        c.generation = i
        if confound:                      # only the sum is identified: a ridge
            sim = rng.normal(c.position.sum(), SIG / np.sqrt(NOBS))
            c.loss = float(abs(sim - yobs.sum()))
        else:
            sim = rng.normal(c.position, SIG / np.sqrt(NOBS))
            c.loss = float(np.linalg.norm(sim - yobs))
        hist.append(c)
        tols.append(c.tolerance if c.tolerance is not None else np.nan)
    return prop, hist, np.array(tols, dtype=float)


def drift_exponent(prop, hist, label):
    g = np.linspace(LO, HI, NGRID)
    G = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    taus = np.unique(np.geomspace(400, len(hist) - 1, 40).astype(int))
    dens = {}
    for t in taus:
        a = prop._reconstruct_archive(hist[:int(t)], hist[int(t)].tolerance)
        if a is not None:
            dens[int(t)] = np.exp(prop._build_proposal(
                a, hist[int(t)].tolerance).log_mixture_density(G))
    ts = sorted(dens)
    qn = dens[ts[-1]]
    t_a = np.array(ts[:-1], float)
    sup = np.array([float(np.abs(dens[t] - qn).max()) for t in ts[:-1]])
    out = {}
    for name, mask in (("all", t_a > 0), ("tau<=n/3", t_a <= len(hist) / 3)):
        ok = mask & (sup > 0)
        b = -np.polyfit(np.log(t_a[ok]), np.log(sup[ok]), 1)[0]
        out[name] = b
        agg = 1 - b - 0.5
        print(f"  {label:12s} {name:9s} b = {b:6.3f}   aggregate ~ n^{agg:+.3f}   "
              f"{'PASSES (b>1/2)' if b > 0.5 else 'fails'}")
    return out


if __name__ == "__main__":
    print("identified target (both coordinates observed)")
    pI, hI, _ = run(None, confound=False)
    dI = drift_exponent(pI, hI, "identified")
    print("\nconfounded target (only theta1+theta2 observed: a ridge)")
    pC, hC, _ = run(None, confound=True)
    dC = drift_exponent(pC, hC, "confounded")
    print("\nverdict")
    for nm in ("all", "tau<=n/3"):
        print(f"  {nm:9s}: b {dI[nm]:.3f} (identified) -> {dC[nm]:.3f} (confounded)")
    print("  Cellular Potts production run measured b = 0.14-0.33")
