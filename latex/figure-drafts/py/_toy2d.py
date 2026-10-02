#!/usr/bin/env python3
"""Shared two-dimensional toy behind the algorithm illustrations (a8, a9, a10).

Unit-box prior (pi = 1, so every importance weight is 1/q), a tilted elliptical
discrepancy around a hidden truth, and two miniature runs on it. Both are
*illustrative* miniatures: they follow the structure of the respective
algorithm so that every panel shows a state the algorithm would produce, but
they simplify the implementation (see the departures listed with each run).

* ``run_async``: Algorithm 1 in miniature. Uniform prior draws until k
  particles exist (bootstrap, weight 1), then per call: eps_n =
  min(eps_hist, eps_sched); archive = top-k by rho; adaptation weights
  W_j ~ w_j K_eps(rho_j) (stored proposal-time weight times kernel weight);
  Sigma_n = s Cov_W(A_n); one candidate by parent-and-perturb with in-box
  retry; weight pi / q_bar^on with the balance heuristic over the snapshot
  ring buffer; a snapshot pushed every ``amis_interval`` calls. Every
  particle stores (theta, rho, eps, tau, w).
  Departures from the implementation (none visible in the panels): eps_sched
  jumps to the discrepancy at which 2k particles lie within instead of the
  ESS-retention bisection with its factor-2 cap and once-per-k throttle;
  no eps_0, so the first archive-phase bandwidth is the largest bootstrap
  rho; plain weighted covariance without the bias correction and with a
  fixed jitter; mixture components not truncated to the box (draws are, by
  rejection); no underflow redraws.
* ``run_pmc``: ABC-PMC with a quantile schedule (Del Moral et al. 2012;
  Lenormand et al. 2013; pyABC's default) and a hard threshold (Beaumont et
  al. 2009). Population of N, threshold = alpha-quantile of the previous
  population's discrepancies, Sigma_t = 2 Cov_omega(P_t), candidates drawn,
  simulated and tested one at a time until N are accepted (the rejected
  ones are discarded), weights pi / q_t against the one previous proposal,
  normalized. Run without discrepancy noise so that the drawn acceptance
  region is exact.

Panel helpers (square axes without ticks, ellipses, contour shades, the
kernel-profile inset) live here so all illustrations share one style. Panels
are drawn at the size they are included at, so fonts print at nominal size.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from async_abc.plotting import paper_style as ps  # noqa: E402

OUT = HERE.parent / "out"
STAR = "#111111"
REF_SIZE = 1.25  # inches; marker areas below are given for this panel size
TAG_FS = 7


# ----------------------------------------------------------------- the target
def rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


MU = np.array([0.56, 0.48])
C_TARGET = rot(30) @ np.diag([0.25, 0.11]) ** 2 @ rot(30).T
C_INV = np.linalg.inv(C_TARGET)
RHO_SCALE = 0.10
RHO_NOISE = 0.012


def discrepancy(theta, rng, noise=RHO_NOISE):
    """rho = 0.1 * Mahalanobis distance to the truth + simulation noise."""
    theta = np.atleast_2d(theta)
    d = theta - MU
    m = np.sqrt(np.einsum("ij,jk,ik->i", d, C_INV, d))
    return RHO_SCALE * m + (rng.normal(0, noise, len(theta)) if noise > 0 else 0.0)


def in_box(theta):
    return bool(np.all((theta >= 0.0) & (theta <= 1.0)))


def weighted_cov(x, w):
    mean = w @ x
    d = x - mean
    return (w[:, None] * d).T @ d, mean


# ---------------------------------------------------------------- the mixture
class Proposal:
    """Gaussian mixture with a shared covariance: sum_j W_j N(theta_j, Sigma).

    Components are not truncated to the box (the implementation truncates
    them and evaluates the in-box mass); draws are kept inside by rejection.
    """

    def __init__(self, means, weights, sigma):
        self.means = np.asarray(means, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.L = np.linalg.cholesky(self.sigma)
        self._inv = np.linalg.inv(self.sigma)
        self._norm = 2 * np.pi * np.sqrt(np.linalg.det(self.sigma))

    def pdf(self, theta):
        theta = np.atleast_2d(theta)
        dens = np.zeros(len(theta))
        for m, w in zip(self.means, self.weights):
            d = theta - m
            dens += w * np.exp(-0.5 * np.einsum("ij,jk,ik->i", d, self._inv, d)) / self._norm
        return dens

    def draw(self, rng, z=None, max_tries=200):
        """Parent J ~ W, then theta = theta_J + L z, redrawn until inside the box.

        ``z`` fixes the perturbation (for a reproducible picture); it is
        clipped into the box instead of redrawn.
        """
        j = int(rng.choice(len(self.weights), p=self.weights))
        if z is not None:
            return j, np.clip(self.means[j] + self.L @ np.asarray(z, float), 0.01, 0.99)
        for _ in range(max_tries):
            th = self.means[j] + self.L @ rng.standard_normal(2)
            if in_box(th):
                return j, th
        raise RuntimeError("in-box rejection exhausted; the toy covariance is wider than the box")


# ---------------------------------------------------------- our algorithm
def async_proposal(theta, rho, w, eps, k, s):
    """Archive A = top-k by rho, adaptation weights W_j ~ w_j K_eps(rho_j), Sigma = s Cov_W."""
    arch = np.argsort(rho)[:k]
    kw = w[arch] * np.exp(-0.5 * (rho[arch] / eps) ** 2)
    W = kw / kw.sum()
    cov, mean = weighted_cov(theta[arch], W)
    sigma = s * (cov + 1e-4 * np.eye(2))
    return arch, W, mean, Proposal(theta[arch], W, sigma)


def sched_eps(rho, k):
    """The bandwidth at which 2k evaluated particles lie within (the schedule's target)."""
    return float(np.sort(rho)[min(2 * k, len(rho)) - 1])


def run_async(rng, n_calls=84, k=12, s=2.0, amis_interval=4, S=20):
    theta = np.zeros((0, 2))
    rho = np.zeros(0)
    w = np.zeros(0)
    eps_stamp = np.zeros(0)
    tau = np.zeros(0, dtype=int)
    snaps = []
    since_snapshot = 0
    for n in range(n_calls):
        finite = eps_stamp[np.isfinite(eps_stamp)]
        eps_hist = float(finite.min()) if len(finite) else np.inf
        if len(theta) < k:  # bootstrap: uniform prior draw, weight pi/pi = 1, no bandwidth stamp
            th, wt, eps_n = rng.uniform(0, 1, 2), 1.0, np.inf
        else:
            eps_n = min(eps_hist, sched_eps(rho, k))
            _, _, _, q = async_proposal(theta, rho, w, eps_n, k, s)
            _, th = q.draw(rng)
            denom = (q.pdf(th)[0] + sum(sn.pdf(th)[0] for sn in snaps)) / (1 + len(snaps))
            wt = 1.0 / denom
            since_snapshot += 1
            if since_snapshot >= amis_interval:  # push after weighting, oldest evicted
                snaps.append(q)
                snaps = snaps[-S:]
                since_snapshot = 0
        theta = np.vstack([theta, th])
        rho = np.append(rho, discrepancy(th, rng)[0])
        w = np.append(w, wt)
        eps_stamp = np.append(eps_stamp, eps_n)
        tau = np.append(tau, n)
    return dict(theta=theta, rho=rho, w=w, eps=eps_stamp, tau=tau, snaps=snaps, k=k, s=s)


# ------------------------------------------------------------------ ABC-PMC
def run_pmc(rng, N=12, generations=3, alpha=0.5, s=2.0, noise=0.0):
    """ABC-PMC, quantile schedule, hard threshold; returns the last generation's record."""
    pre = rng.uniform(0, 1, (4 * N, 2))
    eps = float(np.quantile(discrepancy(pre, rng, noise), alpha))
    pop, pr = [], []
    while len(pop) < N:
        th = rng.uniform(0, 1, 2)
        r = discrepancy(th, rng, noise)[0]
        if r < eps:
            pop.append(th)
            pr.append(r)
    pop, pr = np.array(pop), np.array(pr)
    omega = np.full(N, 1.0 / N)
    record = None
    for _ in range(generations):
        eps_new = float(np.quantile(pr, alpha))
        cov, mean = weighted_cov(pop, omega)
        q = Proposal(pop, omega, s * (cov + 1e-4 * np.eye(2)))
        draws, rhos, flags = [], [], []
        while sum(flags) < N:  # propose, simulate, test: one candidate at a time
            _, th = q.draw(rng)
            r = discrepancy(th, rng, noise)[0]
            draws.append(th)
            rhos.append(r)
            flags.append(bool(r < eps_new))
        draws, rhos, flags = np.array(draws), np.array(rhos), np.array(flags)
        acc = draws[flags]
        om_new = 1.0 / q.pdf(acc)
        om_new /= om_new.sum()
        record = dict(pop=pop, omega=omega, rho=pr, eps_new=eps_new, sigma=q.sigma, mean=mean,
                      q=q, draws=draws, rhos=rhos, flags=flags, new_pop=acc, new_omega=om_new,
                      N=N, s=s)
        pop, omega, pr = acc, om_new, rhos[flags]
    return record


# ------------------------------------------------------------- panel helpers
def grid(n=160):
    gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    return gx, gy, np.column_stack([gx.ravel(), gy.ravel()])


def ellipse(center, cov, nsd=1.0, **kw):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    return Ellipse(center, 2 * nsd * np.sqrt(vals[1]), 2 * nsd * np.sqrt(vals[0]), angle=ang, **kw)


class Panel:
    """A square, tick-less axes drawn at ``size`` inches; ``s`` scales marker areas."""

    def __init__(self, size):
        self.size = size
        self.s = (size / REF_SIZE) ** 2
        self.fig = plt.figure(figsize=(size, size))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color("0.45")

    def tag(self, text, color):
        self.ax.text(0.97, 0.05, text, color=color, ha="right", va="bottom", fontsize=TAG_FS,
                     transform=self.ax.transAxes)

    def kernel_inset(self, eps, rho_rug, kind, color, label):
        """The kernel over rho: Gaussian K_eps(rho) or the indicator 1[rho < eps].

        Rug ticks mark the discrepancies of the particles the kernel is applied to.
        """
        small = self.size < 0.8
        ins = self.ax.inset_axes([0.47, 0.06, 0.49, 0.34] if small else [0.52, 0.06, 0.44, 0.30])
        xmax = max(2.4 * eps, float(np.max(rho_rug)) * 1.05)
        x = np.linspace(0, xmax, 300)
        if kind == "gaussian":
            ins.plot(x, np.exp(-0.5 * (x / eps) ** 2), color=color, lw=0.9)
        else:
            ins.plot(x, (x < eps).astype(float), color=color, lw=0.9, drawstyle="steps-post")
        if not small:  # rug ticks are illegible at the strip size
            ins.vlines(rho_rug, 0, 0.16, color=color, lw=0.5, alpha=0.85)
        ins.axvline(eps, color=color, lw=0.5, ls=":")
        ins.set_xlim(0, xmax)
        ins.set_ylim(0, 1.75)
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_facecolor("white")
        for sp in ins.spines.values():
            sp.set_linewidth(0.4)
            sp.set_color("0.5")
        ins.text(eps, 1.7, label, color=color, fontsize=TAG_FS, ha="center", va="top",
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.6))  # inside, on white
        ins.text(0.98, 0.04, r"$\rho$", fontsize=TAG_FS - 1, ha="right", va="bottom",
                 transform=ins.transAxes, color="0.35")
        return ins

    def save(self, name):
        OUT.mkdir(exist_ok=True)
        p = OUT / f"{name}.pdf"
        self.fig.savefig(p, bbox_inches=None, pad_inches=0)
        plt.close(self.fig)
        print(f"wrote {p}")


def tint(color, f):
    """Mix ``color`` with white: f = 0 white, f = 1 the color."""
    r, g, b = to_rgb(color)
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


def contour_levels(dens):
    return np.quantile(dens[dens > dens.max() * 0.02], [0.35, 0.6, 0.8, 0.92])


# panel sizes (inches) matching the include widths in the TikZ wrappers
SIZES = {"": 2.4 / 2.54, "s": 1.6 / 2.54}  # rings: 2.4 cm; strip (a10): 1.6 cm
