#!/usr/bin/env python3
"""Six parameter-space panels for our single-arrival update (a8 ring, a10 strip).

The history comes from ``_toy2d.run_async`` (an illustrative miniature of
Algorithm 1; departures listed there), so every panel shows the state the
algorithm holds at this call:

  1 history          every evaluated particle, shaded by discrepancy (light = far)
  2 bandwidth        the k best in blue; inset: the smooth kernel K_eps(rho) over rho
                     with the archive's rho_j as ticks (no acceptance boundary)
  3 weights          archive members sized by W_j ~ w_j K_eps(rho_j); Sigma_n dotted
  4 proposal         q_n as contours
  5 one draw         a parent J ~ W, perturbed by L_n z, to the candidate theta*
  6 weight           theta* against q_n and the stored snapshots (gray)

Outputs: ../out/a8_panel_1..6.pdf (2.4 cm, for the ring) and
../out/a8s_panel_1..6.pdf (1.6 cm, for the strip).
"""
import numpy as np

import _toy2d as T


def make(size, prefix):
    BLUE = T.ps.COLORS["async"]
    rng = np.random.default_rng(7)
    run = T.run_async(rng)
    theta, rho, w, snaps, k, s = run["theta"], run["rho"], run["w"], run["snaps"], run["k"], run["s"]
    n = len(theta)

    eps_hist = float(np.min(run["eps"][np.isfinite(run["eps"])]))
    eps_n = min(eps_hist, T.sched_eps(rho, k))
    arch, W, mean, q_n = T.async_proposal(theta, rho, w, eps_n, k, s)
    non_arch = np.setdiff1d(np.arange(n), arch)
    gx, gy, G = T.grid()
    dens = q_n.pdf(G).reshape(gx.shape)
    levels = T.contour_levels(dens)
    J, theta_star = q_n.draw(np.random.default_rng(3), z=np.array([1.3, -0.9]))
    parent = theta[arch][J]
    q_on = (q_n.pdf(theta_star)[0] + sum(sn.pdf(theta_star)[0] for sn in snaps)) / (1 + len(snaps))
    w_star = 1.0 / q_on
    shade = 0.25 + 0.7 * (rho - rho.min()) / (rho.max() - rho.min())

    # 1 history, shaded by discrepancy
    p = T.Panel(size)
    p.ax.scatter(theta[:, 0], theta[:, 1], s=11 * p.s, c=[(g, g, g) for g in shade], edgecolors="none")
    p.save(f"{prefix}_panel_1")

    # 2 bandwidth: the k best, and the smooth kernel over rho (no boundary in theta-space)
    p = T.Panel(size)
    p.ax.scatter(theta[non_arch, 0], theta[non_arch, 1], s=8 * p.s, color="0.78", edgecolors="none")
    p.ax.scatter(theta[arch, 0], theta[arch, 1], s=14 * p.s, color=BLUE, edgecolors="none")
    p.kernel_inset(eps_n, rho[arch], "gaussian", BLUE, r"$\epsilon_n$")
    p.save(f"{prefix}_panel_2")

    # 3 adaptation weights (stored weight x kernel weight) and Sigma_n
    p = T.Panel(size)
    p.ax.scatter(theta[non_arch, 0], theta[non_arch, 1], s=8 * p.s, color="0.84", edgecolors="none")
    p.ax.add_patch(T.ellipse(mean, q_n.sigma, nsd=1.0, fill=False, ls=":", lw=0.9, ec=BLUE))
    p.ax.scatter(theta[arch, 0], theta[arch, 1], s=(6 + 90 * W) * p.s, color=BLUE, edgecolors="white",
                 linewidths=0.4)
    p.tag(r"$\Sigma_n$", BLUE)
    p.save(f"{prefix}_panel_3")

    # 4 proposal mixture
    p = T.Panel(size)
    p.ax.contourf(gx, gy, dens, levels=[*levels, dens.max() * 1.01],
                  colors=[T.tint(BLUE, f) for f in (0.25, 0.45, 0.7, 0.95)])
    p.ax.contour(gx, gy, dens, levels=levels, colors=[BLUE], linewidths=0.4)
    p.ax.scatter(theta[arch, 0], theta[arch, 1], s=5 * p.s, color="white", edgecolors=BLUE, linewidths=0.4)
    p.tag(r"$q_n$", BLUE)
    p.save(f"{prefix}_panel_4")

    # 5 one draw: parent J ~ W, perturbation ellipse, candidate
    p = T.Panel(size)
    p.ax.contour(gx, gy, dens, levels=levels, colors=["0.72"], linewidths=0.55)
    p.ax.scatter(theta[arch, 0], theta[arch, 1], s=6 * p.s, color="0.55", edgecolors="none")
    p.ax.add_patch(T.ellipse(parent, q_n.sigma, nsd=1.0, fill=False, ls=":", lw=0.8, ec=BLUE))
    p.ax.scatter(*parent, s=60 * p.s, facecolors="none", edgecolors=BLUE, linewidths=1.2)
    p.ax.annotate("", xy=theta_star, xytext=parent,
                  arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2, shrinkA=3, shrinkB=3))
    p.ax.scatter(*theta_star, marker="*", s=110 * p.s, color=T.STAR, edgecolors="white",
                 linewidths=0.4, zorder=5)
    p.ax.text(theta_star[0] + 0.05, theta_star[1] - 0.04, r"$\theta^\star$", fontsize=T.TAG_FS, va="top")
    p.save(f"{prefix}_panel_5")

    # 6 weight: theta* against q_n and the stored snapshots (no prior term online)
    p = T.Panel(size)
    shown = snaps[::3] + ([snaps[-1]] if (len(snaps) - 1) % 3 else [])
    for sn in shown:
        d = sn.pdf(G).reshape(gx.shape)
        lv = np.quantile(d[d > d.max() * 0.02], [0.6])
        p.ax.contour(gx, gy, d, levels=lv, colors=["0.62"], linewidths=0.6)
    p.ax.contour(gx, gy, dens, levels=levels[1:], colors=[BLUE], linewidths=0.7)
    p.ax.scatter(*theta_star, marker="*", s=90 * p.s, color=T.STAR, edgecolors="white",
                 linewidths=0.4, zorder=5)
    p.tag(r"$w^\star$", "0.2")
    p.save(f"{prefix}_panel_6")

    print(f"[{prefix}] n={n} k={k} eps_n={eps_n:.3f} |S|={len(snaps)} parent={arch[J]} "
          f"theta*={theta_star.round(3)} w*={w_star:.2f} archive rho in "
          f"[{rho[arch].min():.3f}, {rho[arch].max():.3f}] stored w in [{w[arch].min():.2f}, {w[arch].max():.2f}]")


def main() -> None:
    T.ps.apply()
    for suffix, size in T.SIZES.items():
        make(size, f"a8{suffix}")


if __name__ == "__main__":
    main()
