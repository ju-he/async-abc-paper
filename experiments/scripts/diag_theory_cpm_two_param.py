"""Appendix A theory diagnostics on the Cellular Potts production history (two-parameter setup).

Runs, per asynchronous replicate of `cpm_two_param_fixed` (job 14262214, tol_init 0.1, the run
behind every CPM posterior number in the paper):

  1. the fidelity-ratio measurement of diag_denominator_mismatch_cpm.py -- the reported
     (m = 20, prior-floored) denominator against a reference at m = 400 and the observed prior
     share, giving zeta for the floor (a), the quadrature (b) and both, the exact TV between the
     two reported posteriors, mean/width shifts and the ESS fraction;
  2. the stabilization measurement of diag_proposal_drift.py -- archive turnover against the
     k log n record rate, the per-step drift exponent, and the distance-to-limit exponent b of
     Assumption 5(ii) (full range and restricted to the first half), plus a pooled fit over all
     replicates;
  3. the factor-(d) event counts of diag_sampler_fallbacks.py (reject-resample fallbacks and
     exhausted underflow redraws).

Two things differ from the older CPM scripts, which are written for the retired
(division_rate, motility) records: the parameters are (division_rate, cell_volume), and the
stamped proposal-time tolerance is the `proposal_tolerance` column (NaN = bootstrap draw), which
keeps the bootstrap/archive split the old writer erased, so the prior floor (a) is measurable.
History order is arrival order by `sim_end_time`, as in the analysis module.

Re-analysis only: no simulation, no propagator change. About five minutes for five replicates.
Writes experiments/data/diagnostics/theory_cpm_two_param.json.
"""
import argparse, heapq, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "propulate"))
from propulate.propagators.abcpmc import ABCPMC  # noqa: E402
from propulate.population import Individual  # noqa: E402

DEFAULT_CSV = (ROOT / "experiments/data/cpm_two_param_validation/cpm_two_param_fixed"
               / "cellular_potts_two_param/data/raw_results.csv.gz")
OUT = ROOT / "experiments/data/diagnostics/theory_cpm_two_param.json"
LIMITS = {"division_rate": (0.0, 1.0), "cell_volume": (0.0, 1.0)}
K, PSCALE, M_SHIPPED, M_REF = 100, 0.8, 20, 400
NGRID, NTIMES, TAU0 = 160, 45, 600


def load(csv, rep):
    df = pd.read_csv(csv, low_memory=False)
    df = df[(df.method == "async_propulate_abc") & (df.replicate == rep)]
    if df.empty:
        raise SystemExit(f"no async rows for replicate {rep} in {csv}")
    df = df.sort_values("sim_end_time", kind="mergesort").reset_index(drop=True)
    hist = []
    for r in df.itertuples(index=False):
        ind = Individual({"division_rate": float(r.param_division_rate),
                          "cell_volume": float(r.param_cell_volume)}, LIMITS)
        ind.loss = float(r.loss)
        ind.tolerance = None if pd.isna(r.proposal_tolerance) else float(r.proposal_tolerance)
        ind.weight = None if pd.isna(r.weight) else float(r.weight)
        hist.append(ind)
    return hist


def denominator(prop, hist, n_proposals, prior_weight):
    """log qbar_n at every particle: draw-proportional mixture of n_proposals reconstructed
    snapshots plus a prior component of mass prior_weight (None -> the shipped max(nu, delta))."""
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


def weights_from(prop, hist, log_qbar, eps):
    losses = np.array([i.loss for i in hist])
    lw = float(np.log(prop.prior_density)) + prop._kernel_fn.log_weight(losses, eps) - log_qbar
    lw -= lw.max()
    w = np.exp(lw)
    return w / w.sum()


def archive_turnover(hist, k=K):
    """Exact count of top-k membership changes, streaming in history order."""
    heap, changes, curve = [], 0, []
    for i, ind in enumerate(hist):
        key = -float(ind.loss)
        if len(heap) < k:
            heapq.heappush(heap, key); changes += 1
        elif key > heap[0]:
            heapq.heapreplace(heap, key); changes += 1
        if (i + 1) % 500 == 0:
            curve.append((i + 1, changes))
    return changes, curve


def fit_b(ts, sup, hi=None):
    ts, sup = np.asarray(ts, float), np.asarray(sup, float)
    ok = sup > 0
    if hi is not None:
        ok &= ts <= hi
    if ok.sum() < 4:
        return float("nan")
    return float(-np.polyfit(np.log(ts[ok]), np.log(sup[ok]), 1)[0])


def pooled_b(results, frac=None):
    """One exponent for all replicates: common slope, replicate-specific intercepts."""
    X, Y, G = [], [], []
    for r in results:
        for t, s in r["sup_curve"]:
            if s > 0 and (frac is None or t <= frac * r["n"]):
                X.append(np.log(t)); Y.append(np.log(s)); G.append(r["replicate"])
    X, Y, G = map(np.asarray, (X, Y, G))
    D = np.column_stack([X] + [(G == g).astype(float) for g in sorted(set(G))])
    coef, *_ = np.linalg.lstsq(D, Y, rcond=None)
    return float(-coef[0])


def run(csv, rep):
    hist = load(csv, rep)
    n = len(hist)
    n_prior = sum(1 for i in hist if i.tolerance is None)
    nu, delta = n_prior / n, 0.5 / (M_SHIPPED + 1)
    pos = np.stack([i.position for i in hist])
    losses = np.array([i.loss for i in hist])
    n_fallback = sum(1 for i in hist if i.tolerance is not None and i.weight == 1.0)
    n_exhausted = sum(1 for i in hist if i.weight == 0.0)
    prop = ABCPMC(LIMITS, k=K, kernel="gaussian", scheduler_type="acceptance_rate",
                  perturbation_scale=PSCALE, amis_snapshots=M_SHIPPED, rng=random.Random(0))
    eps_sched = min(i.tolerance for i in hist if i.tolerance is not None)
    eps_k = float(np.partition(losses, K - 1)[K - 1])
    print(f"\n=== replicate {rep}: n={n}, bootstrap={n_prior} (nu={nu:.4f}, delta={delta:.4f}, "
          f"floor {'binds' if nu < delta else 'inactive'}); eps_sched={eps_sched:.3g}, eps_k={eps_k:.3g}; "
          f"factor (d): fallbacks={n_fallback}, exhausted redraws={n_exhausted} in {n - n_prior} archive-phase draws")

    # 1. fidelity ratio
    lq_ship, m_s, wp_s = denominator(prop, hist, M_SHIPPED, None)   # as reported: m=20, floored
    lq_q, m_r, _ = denominator(prop, hist, M_REF, wp_s)             # (b) removed, floor kept
    lq_f, _, _ = denominator(prop, hist, M_SHIPPED, nu)             # (a) removed, m kept
    lq_ref, _, _ = denominator(prop, hist, M_REF, nu)               # both removed
    res = {"replicate": rep, "n": n, "n_bootstrap": n_prior, "nu_n": nu, "delta": delta,
           "n_fallback": n_fallback, "n_exhausted_redraws": n_exhausted,
           "m_shipped": m_s, "w_prior_shipped": wp_s, "m_reference": m_r, "w_prior_reference": nu,
           "eps_sched": eps_sched, "eps_k": eps_k, "k": K}
    for eps_name, eps in (("sched", eps_sched), ("kth", eps_k)):
        w_ship, w_ref, w_q = (weights_from(prop, hist, lq, eps) for lq in (lq_ship, lq_ref, lq_q))

        def z(a, b):
            r = np.exp(a - b)
            return float(np.sum(w_ship * np.abs(r - 1.0))), r
        zt, rt = z(lq_ref, lq_ship); zf, _ = z(lq_f, lq_ship); zq, rq = z(lq_q, lq_ship)
        tv_ref = 0.5 * float(np.abs(w_ship - w_ref).sum())
        tv_q = 0.5 * float(np.abs(w_ship - w_q).sum())
        mu_s, mu_r = w_ship @ pos, w_ref @ pos
        sd_s, sd_r = np.sqrt(w_ship @ (pos - mu_s) ** 2), np.sqrt(w_ref @ (pos - mu_r) ** 2)
        ess_s, ess_r = 1 / np.sum(w_ship ** 2) / n, 1 / np.sum(w_ref ** 2) / n
        print(f"  [eps={eps_name} {eps:.3g}] zeta: total={zt:.4f} floor={zf:.4f} quadrature={zq:.4f} "
              f"(TV bound {zt/(1-zt):.4f}); r_total in [{rt.min():.3f},{rt.max():.3f}]; "
              f"exact TV vs reference={tv_ref:.4f} (quadrature only {tv_q:.4f}); "
              f"mean shift/sd=({abs(mu_s[0]-mu_r[0])/sd_r[0]:.4f},{abs(mu_s[1]-mu_r[1])/sd_r[1]:.4f}); "
              f"sd ratio=({sd_s[0]/sd_r[0]:.4f},{sd_s[1]/sd_r[1]:.4f}); ESS frac {ess_s:.4f} vs {ess_r:.4f}")
        res[f"eps_{eps_name}"] = {
            "zeta_total": zt, "zeta_floor": zf, "zeta_quadrature": zq,
            "tv_bound_total": zt / (1 - zt), "r_min_total": float(rt.min()), "r_max_total": float(rt.max()),
            "r_min_quad": float(rq.min()), "r_max_quad": float(rq.max()),
            "tv_exact_total": tv_ref, "tv_exact_quadrature": tv_q,
            "mean_shift_sd": [float(abs(mu_s[j] - mu_r[j]) / sd_r[j]) for j in range(2)],
            "sd_ratio": [float(sd_s[j] / sd_r[j]) for j in range(2)],
            "ess_frac_shipped": float(ess_s), "ess_frac_reference": float(ess_r)}

    # 2. stabilization
    ch, curve = archive_turnover(hist)
    ratios = [c / (K * np.log(at)) for at, c in curve]
    g = np.linspace(0.0, 1.0, NGRID)
    G = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    dens = {}
    for t in np.unique(np.geomspace(TAU0, n - 1, NTIMES).astype(int)):
        a = prop._reconstruct_archive(hist[:int(t)], hist[int(t)].tolerance)
        if a is not None:
            dens[int(t)] = np.exp(prop._build_proposal(a, hist[int(t)].tolerance).log_mixture_density(G))
    ts = sorted(dens)
    qn = dens[ts[-1]]
    sup, drift, tmid = [], [], []
    for j, t in enumerate(ts[:-1]):
        sup.append(float(np.abs(dens[t] - qn).max()))
        drift.append(float(np.abs(dens[ts[j + 1]] - dens[t]).max()) / max(1, ts[j + 1] - t))
        tmid.append(0.5 * (t + ts[j + 1]))
    ts_a = np.array(ts[:-1], float)
    res.update({"archive_changes": ch, "k_log_n": float(K * np.log(n)),
                "turnover_ratio_range": [float(min(ratios)), float(max(ratios))],
                "b_full": fit_b(ts_a, sup), "b_first_half": fit_b(ts_a, sup, hi=n / 2),
                "b_first_third": fit_b(ts_a, sup, hi=n / 3), "drift_exponent": fit_b(tmid, drift),
                "sup_curve": [[int(t), float(s)] for t, s in zip(ts[:-1], sup)]})
    print(f"  turnover {ch} changes (k ln n = {K*np.log(n):.0f}, ratio {min(ratios):.2f}-{max(ratios):.2f}); "
          f"drift ~ tau^-{res['drift_exponent']:.2f}; b_full={res['b_full']:.3f} "
          f"b_half={res['b_first_half']:.3f} b_third={res['b_first_third']:.3f}  [Assumption 5(ii) needs b > 1/2]")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    if not Path(args.csv).exists():
        raise SystemExit(f"history not found: {args.csv}")
    results = [run(args.csv, r) for r in args.replicates]
    summary = {"csv": str(Path(args.csv).relative_to(ROOT)) if str(args.csv).startswith(str(ROOT)) else args.csv,
               "b_pooled_full": pooled_b(results), "b_pooled_first_half": pooled_b(results, 0.5),
               "b_pooled_first_third": pooled_b(results, 1 / 3),
               "archive_phase_draws_total": int(sum(r["n"] - r["n_bootstrap"] for r in results)),
               "n_fallback_total": int(sum(r["n_fallback"] for r in results)),
               "n_exhausted_redraws_total": int(sum(r["n_exhausted_redraws"] for r in results))}
    print("\n=== pooled: b_full={b_pooled_full:.3f} b_first_half={b_pooled_first_half:.3f} "
          "b_first_third={b_pooled_first_third:.3f}; factor (d) events {n_fallback_total}+{n_exhausted_redraws_total} "
          "in {archive_phase_draws_total} archive-phase draws".format(**summary))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "replicates": results}, indent=1) + "\n")
    print(f"wrote {OUT}")
