#!/usr/bin/env python3
"""Six parameter-space panels for ABC-PMC (a9 ring, a10 strip).

ABC-PMC with a quantile schedule and a hard threshold, run in miniature by
``_toy2d.run_pmc`` on the same box and discrepancy as the a8 panels, drawn at
one generation update t -> t+1 in the same six positions as ours:

  1 population       the N particles of generation t, sized by omega^(t)
  2 threshold        eps_{t+1} = quantile of the population's rho; the dashed
                     ellipse is the acceptance region; inset: the indicator
                     1[rho < eps_{t+1}] over rho with the population's rho_i as ticks
  3 covariance       same particles, same sizes; Sigma_t = 2 Cov_omega
  4 proposal         q_t as contours
  5 propose, simulate, accept, one at a time: the loop in progress (first 12
                     candidates: kept filled, rejected crossed), the rest not yet drawn
  6 population t+1   all M candidates evaluated: N kept, sized by omega^(t+1);
                     the rejected ones crossed out (discarded)

Outputs: ../out/a9_panel_1..6.pdf (2.4 cm) and ../out/a9s_panel_1..6.pdf (1.6 cm).
"""
import numpy as np

import _toy2d as T

N_SHOWN = 12  # candidates evaluated so far in panel 5


def make(size, prefix):
    RED = T.ps.COLORS["sync"]
    rng = np.random.default_rng(11)
    rec = T.run_pmc(rng)
    pop, omega, rho_t, eps_new, sigma, mean, q = (
        rec[k] for k in ("pop", "omega", "rho", "eps_new", "sigma", "mean", "q"))
    draws, flags, new_pop, new_omega = rec["draws"], rec["flags"], rec["new_pop"], rec["new_omega"]
    gx, gy, G = T.grid()
    dens = q.pdf(G).reshape(gx.shape)
    levels = T.contour_levels(dens)
    region = dict(nsd=eps_new / T.RHO_SCALE, fill=False, ls=(0, (3, 2)), lw=0.8, ec=RED)

    # 1 population t, sized by its importance weights; nothing else survives
    p = T.Panel(size)
    p.ax.scatter(pop[:, 0], pop[:, 1], s=(6 + 90 * omega) * p.s, color=RED, edgecolors="white",
                 linewidths=0.4)
    p.save(f"{prefix}_panel_1")

    # 2 next threshold: the hard acceptance region, and the indicator kernel over rho
    p = T.Panel(size)
    p.ax.add_patch(T.ellipse(T.MU, T.C_TARGET, **region))
    p.ax.scatter(pop[:, 0], pop[:, 1], s=(6 + 90 * omega) * p.s, color=RED, edgecolors="white",
                 linewidths=0.4)
    p.kernel_inset(eps_new, rho_t, "hard", RED, r"$\epsilon_{t+1}$")
    p.save(f"{prefix}_panel_2")

    # 3 weights unchanged; perturbation covariance from them
    p = T.Panel(size)
    p.ax.add_patch(T.ellipse(mean, sigma, nsd=1.0, fill=False, ls=":", lw=0.9, ec=RED))
    p.ax.scatter(pop[:, 0], pop[:, 1], s=(6 + 90 * omega) * p.s, color=RED, edgecolors="white",
                 linewidths=0.4)
    p.tag(r"$\Sigma_t$", RED)
    p.save(f"{prefix}_panel_3")

    # 4 proposal mixture
    p = T.Panel(size)
    p.ax.contourf(gx, gy, dens, levels=[*levels, dens.max() * 1.01],
                  colors=[T.tint(RED, f) for f in (0.25, 0.45, 0.7, 0.95)])
    p.ax.contour(gx, gy, dens, levels=levels, colors=[RED], linewidths=0.4)
    p.ax.scatter(pop[:, 0], pop[:, 1], s=5 * p.s, color="white", edgecolors=RED, linewidths=0.4)
    p.tag(r"$q_t$", RED)
    p.save(f"{prefix}_panel_4")

    # 5 the propose-simulate-accept loop in progress: the first candidates, one at a time
    p = T.Panel(size)
    p.ax.contour(gx, gy, dens, levels=levels, colors=["0.72"], linewidths=0.55)
    p.ax.add_patch(T.ellipse(T.MU, T.C_TARGET, **{**region, "ec": "0.55"}))
    d, f = draws[:N_SHOWN], flags[:N_SHOWN]
    p.ax.scatter(d[~f, 0], d[~f, 1], marker="x", s=18 * p.s, color="0.45", linewidths=0.7)
    p.ax.scatter(d[f, 0], d[f, 1], marker="*", s=70 * p.s, color=RED, edgecolors="white",
                 linewidths=0.3, zorder=5)
    p.ax.scatter(*draws[N_SHOWN], marker="*", s=90 * p.s, color=T.STAR, edgecolors="white",
                 linewidths=0.4, zorder=6)  # the candidate being simulated now
    if size >= 0.8:  # toy-specific counts only at ring size
        p.tag(rf"{f.sum()} of $N$ kept", "0.2")
    p.save(f"{prefix}_panel_5")

    # 6 population t+1: N kept and weighted against q_t; the rest discarded
    p = T.Panel(size)
    p.ax.add_patch(T.ellipse(T.MU, T.C_TARGET, **region))
    rej = draws[~flags]
    p.ax.scatter(rej[:, 0], rej[:, 1], marker="x", s=16 * p.s, color="0.5", linewidths=0.7)
    p.ax.scatter(new_pop[:, 0], new_pop[:, 1], s=(6 + 90 * new_omega) * p.s, color=RED,
                 edgecolors="white", linewidths=0.4, zorder=5)
    if size >= 0.8:
        p.tag(rf"$M={len(draws)}$", "0.2")
    p.save(f"{prefix}_panel_6")

    print(f"[{prefix}] N={rec['N']} eps_t+1={eps_new:.3f} M={len(draws)} kept={flags.sum()} "
          f"acceptance={flags.mean():.2f} shown={N_SHOWN} kept_so_far={f.sum()} "
          f"omega^(t+1) in [{new_omega.min():.3f}, {new_omega.max():.3f}] "
          f"kept_inside_region={bool(np.all(rec['rhos'][flags] < eps_new))}")


def main() -> None:
    T.ps.apply()
    for suffix, size in T.SIZES.items():
        make(size, f"a9{suffix}")


if __name__ == "__main__":
    main()
