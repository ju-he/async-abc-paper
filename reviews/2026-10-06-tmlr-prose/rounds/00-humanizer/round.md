# Round 00: humanizer pass

- baseline: a3f2cae (tree clean at start)
- file: latex/tmlr/tmlr-article.tex (main text + appendices, one file)
- pass: academic-humanizer, voice-only brief (templates/humanizer_brief.md); 38 sentence-level
  exact-string edits, file 164,987 -> 164,703 characters; report in report_tmlr-article.md
- patterns cut: "is what makes" clefts (7), economic metaphors (buys/pays/bought/not free/costs
  nothing, 8), intensifiers (exactly/simply/itself/concrete, 4), meta-commentary and announced
  simplicity ("Four measurements follow", "The curves add what a single budget cannot show",
  "one-line calculation", "asks one question", "the paper's central measurement", 6), slogan
  ("This paper removes the generation"), vague verbs (carries/hosts/books, 3), one punchline
  (the k "floor to clear rather than a frontier to negotiate"), British spellings (centred,
  parameterisation x5)
- kept on purpose: content-bearing contrasts (slowest vs average, barrier vs duration ratio,
  containment vs coverage), the author's short lines ("The posterior does not follow.",
  "Identifiability does."), every hedge and limitation, all headings
- build: latexmk clean, 0 LaTeX errors, 0 undefined, 41 pages
- A/B: blinded copies in ab/claude (version_1 = baseline, version_2 = edited) and ab/codex
  (opposite order); keys kept outside the repo

## A/B verdicts (blinded; edited version = version_2 for Claude, version_1 for codex)

| reviewer | prefers | margin | AI-likeness baseline / edited | content check |
|---|---|---|---|---|
| Claude subagent (ab_claude.md) | edited | moderate (~25 of 34 paragraphs) | 3 / 2 | no number/cite/eq change; four framing shifts noted, none a claim about evidence |
| codex gpt-5.6-sol (ab_codex.md) | edited | moderate | 6 / 2 | no number/cite/eq change; flags the removed "exactly", "the one question", "central measurement" as (welcome) de-strengthening |

Per-sentence dissent: Claude preferred the baseline for "Four measurements follow" (abstract
signpost), "hosts" (benchmark-as-subject parallelism), "floor to clear rather than a frontier to
negotiate", and "has headroom" (edited wording added a superlative); codex preferred the baseline
for "raising k is not free". Both reviewers independently flagged the "headroom" sentence.
Decision: the edited file stays; three sentences adjusted (see report, "Post-A/B adjustments");
the four small single-reviewer dissents are left for the evaluation rounds to re-propose if they
recur. Codex's sandbox could not write its file; codex_review.sh captured the report via -o.
- commit: 1d9e2cd
