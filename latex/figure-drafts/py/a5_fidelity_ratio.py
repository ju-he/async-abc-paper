#!/usr/bin/env python3
"""Additional illustration 5: the fidelity ratio r(theta) made visible.

Rebuilds the one-dimensional Gaussian-mean history used by
experiments/scripts/diag_denominator_mismatch.py (same seed, n = 12000,
k = 100), then evaluates on a theta grid

  * the prior and a few of the history-reconstructed snapshot proposals,
  * the reported denominator  qbar_n  (m = 20 snapshots, prior floor), and
  * the draw-mixture reference qbar*_n (M = 400 snapshots, true prior share),

and plots r(theta) = qbar*_n / qbar_n against the reported posterior, so the
reader sees where r departs from one and whether the posterior puts mass
there. Output: ../out/a5_fidelity_ratio.pdf
"""
from pathlib import Path
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from async_abc.plotting import paper_style as ps  # noqa: E402
from propulate.propagators.abcpmc import ABCPMC  # noqa: E402

OUT = HERE.parent / "out"
LO, HI = -3.0, 3.0
LIMITS = {"x": (LO, HI)}
SIG, NOBS, MU_TRUE, SEED = 1.0, 25, 0.4, 20260730
N_SIMS, K = 12000, 100
M_SHIPPED, M_REF = 20, 400


def build_history():
    rng_np = np.random.default_rng(SEED)
    ybar = float(rng_np.normal(MU_TRUE, SIG / np.sqrt(NOBS)))
    prop = ABCPMC(LIMITS, k=K, kernel="gaussian", scheduler_type="quantile",
                  amis_snapshots=M_SHIPPED, rng=random.Random(SEED))
    hist = []
    for i in range(N_SIMS):
        c = prop(hist)
        c.generation = i
        c.loss = abs(float(rng_np.normal(c.position[0], SIG / np.sqrt(NOBS))) - ybar)
        hist.append(c)
    return prop, hist, ybar


def snapshots(prop, hist, n_proposals):
    """The m evenly spaced history-reconstructed proposals extract_posterior uses."""
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
    arch_a = np.asarray(archive_idx)
    seg = np.searchsorted(np.asarray(taus[1:]), arch_a, side="right")
    counts = np.bincount(seg, minlength=len(snaps)).astype(float)
    return snaps, taus, counts, len(archive_idx)


def log_denominator(prop, snaps, counts, prior_weight, grid):
    """log qbar on ``grid`` for the draw-proportional mixture + prior mass."""
    mix_w = np.append(counts * ((1.0 - prior_weight) / counts.sum()), prior_weight)
    comp = np.empty((len(snaps) + 1, len(grid)))
    for s, sn in enumerate(snaps):
        comp[s] = sn.log_mixture_density(grid)
    comp[-1] = np.log(prop.prior_density)
    return logsumexp(comp + np.log(mix_w)[:, None], axis=0)


def main() -> None:
    ps.apply()
    prop, hist, ybar = build_history()
    n = len(hist)
    pos = np.stack([i.position for i in hist])
    losses = np.array([float(i.loss) for i in hist])
    eps = min(i.tolerance for i in hist if i.tolerance is not None)
    n_prior = sum(1 for i in hist if i.tolerance is None)
    nu = n_prior / n

    grid = np.linspace(LO, HI, 1201)[:, None]
    snaps20, taus20, counts20, _ = snapshots(prop, hist, M_SHIPPED)
    snaps400, _, counts400, _ = snapshots(prop, hist, M_REF)
    m = len(snaps20)
    w_floor = max(nu, 0.5 / (m + 1))
    lq_rep = log_denominator(prop, snaps20, counts20, w_floor, grid)
    lq_ref = log_denominator(prop, snaps400, counts400, max(nu, 1e-12), grid)
    r = np.exp(lq_ref - lq_rep)

    # reported posterior weights on the history points (shipped denominator)
    lq_rep_pts = log_denominator(prop, snaps20, counts20, w_floor, pos)
    lw = np.log(prop.prior_density) + prop._kernel_fn.log_weight(losses, eps) - lq_rep_pts
    w = np.exp(lw - lw.max())
    w /= w.sum()
    lq_ref_pts = log_denominator(prop, snaps400, counts400, max(nu, 1e-12), pos)
    zeta = float(np.sum(w * np.abs(np.exp(lq_ref_pts - lq_rep_pts) - 1.0)))
    # the reference-denominator posterior, for the exact total-variation distance
    lw_ref = np.log(prop.prior_density) + prop._kernel_fn.log_weight(losses, eps) - lq_ref_pts
    w_ref = np.exp(lw_ref - lw_ref.max())
    w_ref /= w_ref.sum()
    tv = 0.5 * float(np.abs(w - w_ref).sum())
    # weighted histogram of the reported posterior for the background of panel (b)
    hist_w, edges = np.histogram(pos[:, 0], bins=150, range=(LO, HI), weights=w, density=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=ps.fig_size(0.72, aspect=1.05), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12})
    g = grid[:, 0]
    ax1.axhline(prop.prior_density, color="0.5", lw=1.0, ls="--", label="prior $\\pi$")
    for j, (sn, tau) in enumerate(zip(snaps20, taus20)):
        if j % 4 != 0:
            continue
        ax1.plot(g, np.exp(sn.log_mixture_density(grid)), color="0.75", lw=0.6,
                 label="snapshot proposals $q_{\\tau_s}$" if j == 0 else None)
    ax1.plot(g, np.exp(lq_rep), color=ps.COLORS["async"], lw=1.4,
             label=f"reported denominator $\\bar q_n$ ($m={m}$, prior floor)")
    ax1.plot(g, np.exp(lq_ref), color="black", lw=1.0, ls=(0, (4, 2)),
             label=f"draw mixture $\\bar q^\\star_n$ ($m={len(snaps400)}$, true $\\nu_n$)")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-3, 3e2)
    ax1.set_ylabel("density")
    ax1.legend(frameon=False, fontsize=6.5, loc="upper right")
    ps.panel_tag(ax1, "(a)")

    ax2b = ax2.twinx()
    ax2b.fill_between(0.5 * (edges[1:] + edges[:-1]), hist_w, color=ps.COLORS["async"], alpha=0.15,
                      lw=0, label="reported posterior $\\widehat\\pi_n$")
    ax2b.set_ylim(0, hist_w.max() * 1.15)
    ax2b.set_yticks([])
    ax2b.spines["right"].set_visible(False)
    ax2.plot(g, r, color="black", lw=1.2, label="$r(\\theta)=\\bar q^\\star_n/\\bar q_n$")
    ax2.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax2.set_ylabel("fidelity ratio $r(\\theta)$")
    ax2.set_xlabel("$\\theta$")
    ax2.set_xlim(LO, HI)
    ax2.set_ylim(0, 1.25)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="upper right")
    ax2.text(0.02, 0.04, f"$\\mathbb{{E}}_{{\\widehat\\pi}}|r-1|={zeta:.3f}$;  "
             f"TV(reported, reference) $={100 * tv:.2f}\\%$",
             transform=ax2.transAxes, fontsize=6.5, va="bottom")
    ps.panel_tag(ax2, "(b)")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "a5_fidelity_ratio.pdf")
    print(f"wrote {OUT / 'a5_fidelity_ratio.pdf'}")
    print(f"n={n} bootstrap={n_prior} nu={nu:.3e} floor={w_floor:.4f} m={m} "
          f"eps={eps:.4f} zeta={zeta:.4f} tv={tv:.4f} r range [{r.min():.3f},{r.max():.3f}] "
          f"posterior ~ N({ybar:.3f}, {SIG/np.sqrt(NOBS):.3f}^2)")


if __name__ == "__main__":
    main()
