# TMLR version of the paper

`tmlr-article.tex` is the manuscript (target venue decided 2026-10-06). Edit it
directly, locally or on the TMLR Overleaf project; `../overleaf.py tmlr
push|pull` keeps the two in sync (see `../README.md`). Build with

    latexmk -pdf tmlr-article.tex

Until 2026-10-06 this file was generated from the Springer source
`../sn-article-template/sn-article.tex` by `make_tmlr_version.py`. The
generator was retired when the TMLR version became the working copy (last
version of both at commit 75dd007; `git show 75dd007:latex/tmlr/make_tmlr_version.py`).
The Springer version is frozen at that state and no longer maintained.

## How this version differs from the frozen Springer manuscript

(What the generator used to change; everything else was verbatim at 75dd007.)

- sn-jnl scaffolding -> TMLR: `\documentclass[10pt]{article}` + `\usepackage{tmlr}`;
  `\title[short]{}` / `\author*` / `\affil` / `\abstract{}` / `\keywords{}` ->
  TMLR title, `\author{\name ... \email ... \addr ...}` placeholders and the
  `abstract` environment (no keywords in TMLR). Custom macros and the theorem
  environments of the Springer preamble are carried over.
- Citations: `\cite` -> natbib author-year `\citep`; four textual uses become
  `\citet` / `\citealp` / `\citealt` (the ABC-SMC/PMC parenthetical list in
  §2, "balance heuristic of Veach & Guibas", "surveyed by Bugallo et al.",
  "the convergence argument of Cornuet et al. ... Marin et al. restrict").
- Tables: `\botrule` -> `\bottomrule`.
- Back matter: the Springer Acknowledgments + Declarations block is replaced by
  a short Broader Impact Statement (optional at TMLR; delete if unwanted) and a
  Reproducibility paragraph carrying the data/code availability text;
  Author Contributions / Acknowledgments are left as comments to be filled in
  after acceptance, per TMLR's instructions. The bibliography moves before
  `\appendix`, as in the TMLR template.
- TMLR-audience framing, absent from the Springer version: the first sentence
  names simulation-based inference as the umbrella (Cranmer et al. 2020); §2
  gains a paragraph placing sequential neural SBI (rounds wait for their
  slowest simulation too) and the straggler problem of synchronous SGD
  (Dean et al. 2012, Chen et al. 2016, Hogwild!); §5.2 gains one sentence on
  why the baselines are samplers of the same estimator class.
- Double-blind: "our fork of Propulate" -> "a fork of Propulate". Cluster and
  tool names (JUWELS, cellsInSilico/NAStJA, Propulate) are left as technical
  facts.

## Files

- `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`: the official style files from
  https://github.com/JmlrOrg/tmlr-style-file at commit 7bf90ef (2023-06-30),
  Apache-2.0 (`LICENSE-tmlr-style-file`). TMLR rejects tweaked style files; do
  not edit them.
- `figures/` and `references.bib` are symlinks into `../sn-article-template/`,
  so figures and the bibliography have one source. `tmlr.bst` lowercases
  titles, so proper nouns in the shared .bib are brace-protected
  (`{Bayesian}`, `{Monte Carlo}`, `{pyABC}`, ...).

## TMLR submission notes (author guide, checked 2026-09-22)

- Anonymous by default (`\usepackage{tmlr}` prints "Anonymous authors"); do
  not link to a de-anonymized version. Camera-ready: `[accepted]` plus
  `\month`, `\year`, `\openreview`; preprint: `[preprint]`.
- No page limit, but length must be justified by content; unusually long main
  text delays review. Current build: 40 letter pages, main text 1-15,
  references 16-18, appendices 18-40.
- Broader Impact Statement is mandatory only if the work carries a significant
  risk of harm.
- Supplementary material: one anonymized PDF or ZIP, at most 100 MB.
- Reviewers judge two things: whether the claims are supported by accurate,
  convincing and clear evidence, and whether some of TMLR's audience would be
  interested.
