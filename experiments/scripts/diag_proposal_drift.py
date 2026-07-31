"""Does the top-k proposal stabilize, and at what rate?

Two rates matter and they behave differently:

  per-step drift      ||q_{i+1} - q_i||   -- expected O(k/n) by record statistics
  distance to limit   ||q_i - q_inf||     -- what Assumption 4(ii) aggregates,
                                             and the one that binds

Measured on the stored Cellular Potts production history (rerun_20260707),
reconstructing the proposal along the history exactly as extract_posterior does.
Re-analysis only: no simulation, no propagator change.

Reports, against the terminal proposal q_n as a stand-in for q_inf:
  - sup-norm and L1 distance at a log-spaced grid of times
  - the fitted decay exponent b in ||q_tau - q_n|| ~ tau^{-b}
  - the aggregate n^{-1/2} sum_i ||q_i - q_inf||, which Assumption 4(ii) needs
    to vanish -- estimated from the fitted exponent
  - the exact archive turnover count, testing the O(k log n) record prediction
"""
import sys, random, numpy as np, pandas as pd
sys.path.insert(0, "/home/juhe/bwSyncShare/Code/async-abc-paper/propulate")
from propulate.propagators.abcpmc import ABCPMC
from propulate.population import Individual

CSV = ("/tmp/claude-1000/-home-juhe-bwSyncShare-Code-async-abc-paper/"
       "07b2399a-f7fb-4aca-ae08-9ce6bfef146c/scratchpad/cpm/raw_results.csv")
LIMITS = {"division_rate": (0.0, 1.0), "motility": (0.0, 1.0)}
K, PSCALE, NGRID, NTIMES = 100, 0.8, 160, 45


def load(rep=0):
    df = pd.read_csv(CSV, low_memory=False)
    df = df[(df.method == "async_propulate_abc") & (df.replicate == rep)]
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


def archive_turnover(hist, k=K):
    """Exact count of times the top-k membership changes, streaming in order."""
    import heapq
    heap, changes, curve = [], 0, []
    for i, ind in enumerate(hist):
        key = -float(ind.loss)                      # max-heap on loss
        if len(heap) < k:
            heapq.heappush(heap, key); changes += 1
        elif key > heap[0]:
            heapq.heapreplace(heap, key); changes += 1
        if (i + 1) % 500 == 0:
            curve.append((i + 1, changes))
    return changes, curve


if __name__ == "__main__":
    hist = load()
    n = len(hist)
    prop = ABCPMC(LIMITS, k=K, kernel="gaussian", scheduler_type="acceptance_rate",
                  perturbation_scale=PSCALE, amis_snapshots=20, rng=random.Random(0))
    ch, curve = archive_turnover(hist)
    print(f"n = {n} async evaluations, k = {K}")
    print(f"archive membership changes: {ch}  "
          f"(k*log n = {K*np.log(n):.0f}; O(k log n) prediction)")
    for at, c in curve[::max(1, len(curve)//6)]:
        print(f"    by n={at:6d}: {c:5d} changes   k*log n = {K*np.log(at):6.0f}")

    g = np.linspace(0.0, 1.0, NGRID)
    G = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    cell = (1.0 / (NGRID - 1)) ** 2

    def build(tau):
        a = prop._reconstruct_archive(hist[:tau], hist[tau].tolerance)
        if a is None:
            return None
        return prop._build_proposal(a, hist[tau].tolerance)

    taus = np.unique(np.geomspace(600, n - 1, NTIMES).astype(int))
    dens = {}
    for t in taus:
        sn = build(int(t))
        if sn is not None:
            dens[int(t)] = np.exp(sn.log_mixture_density(G))
    ts = sorted(dens)
    qn = dens[ts[-1]]
    print(f"\n{'tau':>7} {'||q_tau-q_n||_inf':>18} {'L1':>10} {'step drift sup':>15}")
    sup, l1 = [], []
    for j, t in enumerate(ts[:-1]):
        d = np.abs(dens[t] - qn)
        s_, l_ = float(d.max()), float(d.sum() * cell)
        sup.append(s_); l1.append(l_)
        drift = (float(np.abs(dens[ts[j+1]] - dens[t]).max())
                 / max(1, ts[j+1] - t))
        print(f"{t:>7} {s_:>18.4f} {l_:>10.4f} {drift:>15.3e}")
    ts_a = np.array(ts[:-1], float); sup_a = np.array(sup)
    ok = sup_a > 0
    b = -np.polyfit(np.log(ts_a[ok]), np.log(sup_a[ok]), 1)[0]
    print(f"\nfitted decay  ||q_tau - q_n||_inf ~ tau^-b  with b = {b:.3f}")
    print(f"Assumption 4(ii) aggregate needs sum_i ||q_i-q_inf|| = o(sqrt n), "
          f"i.e. b > 1/2 for the sum to be o(n^1/2)")
    expo = 1.0 - b
    print(f"  with b = {b:.3f}: sum_i ~ n^{expo:.3f}, so n^-1/2 * sum ~ "
          f"n^{expo - 0.5:+.3f}  -> {'VANISHES' if expo < 0.5 else 'DOES NOT VANISH'}")
