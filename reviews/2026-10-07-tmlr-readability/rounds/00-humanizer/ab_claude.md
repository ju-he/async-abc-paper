# A/B comparison: version_1.tex vs version_2.tex

Method: `diff` found seven differing lines (68, 193, 226, 494, 525, 605, 696); the remaining 1039 lines are identical. Each site was read with its surrounding paragraph in both versions. Which version is older was not guessed and plays no role below.

## Preference

**Preferred overall: version_1.** Margin: **small** (seven sites, each a few words; the shared text dominates both versions).

Every one of the seven differences is version_2 adding, or substituting in, a word that does rhetorical rather than semantic work: `exact` (recipe), `itself` (the emitted density), `actual` (effect), emphatic `do` (steer), a cleft `What the theory contributes to that run is ...`, and twice the contrast `derived ... rather than assumed` / `derived, not assumed`. None changes a number, citation or result; all shift the register slightly from stating to insisting. Version_1 is the plainer, more active, more agent-explicit reading at six of seven sites and tied at the seventh.

AI-likeness (0 = unmistakably hand-written, 10 = unmistakably machine-written):

- version_1: **3 / 10**. Dense, specific, every quantity bound to a run or a table; the few stylistic tics it has (the `X, not Y` framing in the shared text, e.g. "the scale below which ..., not a threshold above which ...") are in both versions.
- version_2: **4 / 10**. Same base text, plus the emphatic/contrastive additions listed above, which are the pattern most often flagged in machine-assisted academic prose (`itself`, `actual`, `exact`, `do`, and the "not assumed" contrast stated twice).

## Differences

| # | Location (quote) | Preferred | Why |
|---|---|---|---|
| 1 | §Contributions, item 4: "we derive from the execution model the adaptedness" (v1) vs "the adaptedness ... is derived from the execution model rather than assumed" (v2) | v1 | v1 keeps one active subject across both clauses ("we derive ..., and decompose ..."); v2 switches to passive mid-sentence and then back to "we", and the appended "rather than assumed" is a defensive contrast the sentence does not need (the next clause already says what is derived and what is measured). |
| 2 | §Assumptions discussion: "builds to that recipe" vs "builds to that exact recipe" | v1 | "exact" is an empty intensifier; "to that recipe" already means exactly that recipe. |
| 3 | §"Which runs the limits describe", last sentence: "For that run the theory supplies the accounting" vs "What the theory contributes to that run is the accounting" | v1 | v1 is direct with an explicit agent and verb (theory supplies); v2's cleft construction delays the point and "contributes" leans promotional. v1 also mirrors the parallel second clause ("the measurements ... quantify it"). |
| 4a | Appendix, factor (d): "this factor additionally includes the difference" vs "additionally carries the difference" | v1 (slight) | "includes" is the literal relation (the factor is a product that contains this term). "carries" is metaphorical, although the manuscript does use "carried by $r$" elsewhere (Table, class row), so this is close to a tie. |
| 4b | Same paragraph: "The emitted density is the conditional law" vs "The emitted density itself is the conditional law" | v1 | "itself" adds emphasis without content; the sentence already contrasts emitted density with rebuilt snapshots via "so no further gap arises". |
| 5a | Appendix, after Lemma (adapted): "Completion times steer which worker" vs "Completion times do steer which worker" | v1 (slight) | Emphatic "do" is a defensible concessive marker ("they do steer, but only which density"), but the next sentence makes the concession explicit, so the emphasis is redundant; the register asked for favours the plain verb. |
| 5b | Same sentence: "changes which proposal density is used, a predictable choice, and not the law" vs "changes which proposal density is used, which is predictable, and not the law" | v1 | v2 stacks two "which" clauses ("which proposal density ..., which is predictable") and the second has an ambiguous antecedent (the density or the fact that it changes). v1's appositive "a predictable choice" is tighter and uses "predictable" in its filtration sense. |
| 6 | Appendix, after Proposition (censoring): "measures the effect on the posterior mean" vs "measures the actual effect on the posterior mean" | v1 | "actual" is an intensifier. The contrast with the loose worst-case bound is already made by "makes the displayed bound loose ... and the drain-after-deadline experiment ... measures". |
| 7 | Table, execution-model row: "... with $\tilde q_i$ the emitted density." vs "... with $\tilde q_i$ the emitted density, so adaptedness is derived, not assumed." | v1 | The added clause restates what "Lemma ... then gives the adapted filtration" already says and repeats the contrast from item 1; in a status table, the Status column ("holds; implementation property") already carries that message. |

## Content check

No difference changes a number, citation, result or qualification. One pair is borderline and is quoted for the record:

- Item 1, v1: "we derive from the execution model the adaptedness that asynchronous execution requires"
  Item 1, v2: "the adaptedness that asynchronous execution requires is derived from the execution model rather than assumed"
- Item 7, v1: "Lemma~\ref{lem:adapted} then gives the adapted filtration with $\tilde q_i$ the emitted density."
  Item 7, v2: "Lemma~\ref{lem:adapted} then gives the adapted filtration with $\tilde q_i$ the emitted density, so adaptedness is derived, not assumed."

Both versions make the same claim about what the paper does (adaptedness follows from Lemma lem:adapted under Assumption ass:filtration). v2's added "not assumed" is accurate for adaptedness itself but slightly overreaches as a contrast, since the execution model (B1)--(B3) that the derivation rests on is an assumption, and the manuscript labels it as one. This is a shade of emphasis, not a change of claim; I record it so the author can decide whether the contrast is wanted. Otherwise: None.
