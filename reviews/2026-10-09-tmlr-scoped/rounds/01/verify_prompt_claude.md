You are verifying a round of prose edits to a LaTeX manuscript. You are one of two independent
verifiers. Your job is to decide, per edit, whether it improved style or understandability
WITHOUT changing the content or the message.

Input:
- the proposals file reviews/2026-10-09-tmlr-scoped/rounds/01/proposals.md: every item with status `applied` lists the exact Before and
  After text and the editor's rationale
- the diff reviews/2026-10-09-tmlr-scoped/rounds/01/round01.diff of the edited files, for context around each change
- the manuscript files themselves if you need wider context:
latex/tmlr/tmlr-article.tex

Definitions:
- `improved`: clearer, shorter, less formulaic, or more natural academic prose; same claims
- `neutral`: no meaningful difference either way
- `regressed`: less clear, less precise, awkward, introduced an error (grammar, LaTeX, broken
  reference, lost antecedent), or removed information the reader needs at that point
- `changed-message`: a claim became stronger or weaker, a result or its qualification was
  altered, a number/citation/label changed, or a limitation was dropped where nothing else
  carries it

Explicitly NOT a message change: removing hedges, intensifiers, rhetorical contrasts, punchline
restatements, meta-commentary, or a caveat that is still stated elsewhere and referenced.
Shortening a limitation to one sentence plus a cross-reference is `improved` or `neutral`, not
`changed-message`, as long as the limitation itself is still asserted.

Also check, for every applied item, that no number, \cite, \ref/\cref, \label or \SI/\num token
was added or removed. Report any such case as `changed-message` even if the prose is fine.

Deliverable (write to reviews/2026-10-09-tmlr-scoped/rounds/01/verify_claude.md if you can write files; a read-only sandbox such as codex prints the whole report as its final message instead, which is captured to that path):

## Verdicts
| id | verdict | reason (one sentence) |
|---|---|---|
one row per applied item, in id order

## Items to revert
list the ids with verdict regressed or changed-message, each with the fix you would prefer
(revert, or a specific alternative wording)

## Overall
Three sentences: did the round as a whole improve the manuscript's voice and readability; did
any pattern of the editor's choices worry you (e.g. systematically flattening good sentences,
over-deleting caveats); would you run another round.
