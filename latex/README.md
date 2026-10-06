# Paper sources

- `tmlr/` — the TMLR manuscript, the version being worked on (target venue
  since 2026-10-06). Edit `tmlr-article.tex` directly. Its `figures/` and
  `references.bib` are symlinks into `sn-article-template/`, where the figure
  pipeline writes and the bibliography lives.
- `sn-article-template/` — the Springer (sn-jnl) version, frozen at commit
  75dd007 when the TMLR version became the working copy. Still the physical
  home of `figures/` and `sn-bibliography.bib`.
- `figure-drafts/` — TikZ/Python drafts behind the figures.

## Overleaf sync

`overleaf.py` mirrors each paper directory into its own Overleaf project
through Overleaf's git bridge (premium feature). Targets: `springer`
(`sn-article-template/`) and `tmlr` (`tmlr/`). One clone per target lives in
`.overleaf/<target>/` (gitignored); the main repository never sees Overleaf's
history.

One-time setup per target. Create a git authentication token under Account
settings -> Git integration (user `git`, password = token; asked once, then
stored). The project page URL is accepted and translated to the git URL:

    latex/overleaf.py tmlr init https://www.overleaf.com/project/<project-id>
    latex/overleaf.py springer init https://www.overleaf.com/project/<project-id>

Day to day:

    latex/overleaf.py tmlr status        # what each direction would change
    latex/overleaf.py tmlr push          # local -> Overleaf (one commit per push)
    latex/overleaf.py tmlr pull          # Overleaf -> local, then git diff + commit

The `springer` target exists for the frozen Springer version and is normally
left alone. Pulled changes to `figures/` or `references.bib` land in
`sn-article-template/` through the symlinks and so reach both versions. Both Overleaf projects hold their own
copy of those shared files, so after pulling such a change from one target,
push the other (`status` on it lists the pending update).

Guards: `pull` refuses while the paper directory, or anything it symlinks
into, has uncommitted changes (`--force` overrides) and does not delete local
files absent on Overleaf unless `--delete` is passed. `push` refuses when
Overleaf has commits that were never pulled (`--force` overwrites them). Build
artifacts, compiled PDFs, `figures/kit/` (thesis colour scheme), the Springer
`notes/`, `tables/`, `bst/` and placeholder EPS files, and the TMLR README are
not synced; see the exclude lists in `overleaf.py`.
