# Round 02 verification (Claude)

Scope: the 31 items with status `applied` in `proposals.md`, checked against `round02.diff` and, where the rationale pointed elsewhere in the manuscript, against `latex/tmlr/tmlr-article.tex` (§3.5 line 141, §4 line 179, §5.2 line 235, §6.3 line 292, Appendix A lines 450-476, Conclusion line 385, Table caption line 650).

## Verdicts
| id | verdict | reason (one sentence) |
|---|---|---|
| P-001 | improved | Drops a claim that §6.3 states verbatim one subsection later ("Both ratios increase with simulation cost"), the sentence now carries only the equal-wall-clock conversion, and $4.1\times$ and the Lotka--Volterra statement are intact. |
| P-002 | improved | The practitioner framing is kept in §5.2 ("it is the method a practitioner would run"), so the pointer sentence loses nothing; both `\ref`s stay. |
| P-003 | neutral | "collects" is better than "holds", and "its draw mixture must be asymptotically reflected in the denominator" names the relation precisely, but "the only one that concerns the sampler" is marginally less exact than the original (clause (i) also contains the bandwidth limit, which is a sampler property); no token changed. |
| P-004 | improved | Splits a four-clause sentence; the dropped "which a single-arrival scheme does not do" is stated word for word in §4 (line 179), and both `\citet`s remain. |
| P-005 | improved | The setting gets its own sentence and the result sentence gets a subject; every number, the $r$ interval, the (c) side-condition and the `\ref` survive. |
| P-006 | improved | One result per sentence; the parenthetical becomes a relative clause and the ESS statement a clause of its own, with all four quantities unchanged. |
| P-007 | improved | Splitting at the semicolon lets the "necessary condition, not a bound" caveat stand alone; both draw counts kept. |
| P-008 | improved | One fit per sentence, with the upward-bias reason still attached to the restricted fit; both ranges and both pooled values unchanged. |
| P-009 | improved | A finite-verb sentence for the experiment and the result after it; "differ in $b$ by less than $0.07$" is equivalent to "moves $b$ by less than $0.07$", and both thresholds kept. |
| P-010 | improved | The aside "at no simulation cost" becomes a predicate and the check is named in its own clause; `\S\ref` and $93\%/85\%$ unchanged. |
| P-011 | improved | The result and the list of robustness conditions are separated; every condition, both delay values and both references kept. |
| P-012 | improved | The removed opener is meta-commentary, and the cost-versus-fidelity point is made at Appendix A (b) ("Fixing $m$ therefore trades cost against fidelity") and later in the same paragraph; no token touched. |
| P-013 | improved | "No measurable mean bias" for the unweighted archive is already the first result of the paragraph ("stays ${\approx}0.01$ ... for the unweighted archive"), so the deletion removes a restatement; `\citep{murray2021anytime}` stays. |
| P-014 | changed-message | Flagged under the token rule only: the deleted sentence contained the number token "6" (step 6); the caveat itself survives at §3.5 line 141 ("a stale or empty buffer changes which parents are chosen, not the estimate") and in the Table caption at line 650, so the message is intact. |
| P-015 | improved | Drops the clause that repeats the preceding sentence and names the antecedent ("the two orderings"); "by contrast" reads back across the display to "indexed by proposal time", and the contribution to $r$ is unchanged. |
| P-016 | improved | Two unrelated measurements in two sentences; all five numbers kept. |
| P-017 | improved | "recovered it" states the relation §6.4 reports (the rule recovers the contraction) where "transferred" left the reader to guess; `\S\ref` kept. |
| P-018 | improved | "settings" replaces a colloquial "efficiency knobs" (the Conclusion's "two settings" are a different pair, but the sentence names $k$ and $S$ explicitly, so no confusion); `\S\ref` kept. |
| P-019 | improved | "enters the statements as $r$" says what "is carried as $r$ inside the statements" meant, with the decomposed-and-partly-measured claim unchanged. |
| P-020 | neutral | Removes the cleft "This indexing is why", but the merged sentence now has two "so" clauses in a row before the display, which is no clearer than the original; content and $\mathcal{F}_{i-1}$ chain unchanged. |
| P-026 | improved | Condition, consequence and empirical finding each in one sentence; the deliberately plain "sit at that threshold" is kept and no symbol changed. |
| P-027 | improved | The parenthetical caveat becomes a sentence and the italicized proof condition stands alone; $o(n/\log n)$ and the emphasis kept. |
| P-028 | improved | The dependency chain is easier to follow; "remove the uniform weight bound" is equivalent to the bound going to infinity with $\delta\to0$, and `Lemma~\ref{lem:bounded}` stays. |
| P-029 | neutral | Separating the setup from the two measurements is fine, but "there too" is lost, which was the one link to the CPM result (narrower in the same direction); $0.042$, $0.26\%$, $0.5\%$ unchanged. |
| P-031 | improved | One comparison per benchmark, then the interpretation; all four values kept and the scope ("valid for every $f$") preserved. |
| P-033 | improved | Removes the announced "practical knee" (meta-commentary) while keeping $3\times10^{-4}$, $m=50$ and $2.5\times$ with "At that point" carrying the link. |
| P-034 | improved | The calculation is now in order (generation time, ceiling, ratio, other multipliers); $16/2.004=7.98$, $401\times$, $402\times$, $5\times$, $10\times$ all kept. |
| P-035 | neutral | Correct and clear, but the three short declaratives read staccato where the original single sentence with a colon was already well formed; no content changed. |
| P-036 | improved | "given the same simulation budget" says what "fairly budgeted" meant and matches §5.2 ("Rejection ABC spends the same simulation budget") and §6.2 ("given the same budget"). |
| P-037 | neutral | Evidence-then-interpretation is defensible, but the new first sentence ("The top-$k$ archive alone would under-cover") now follows a sentence about pyABC without the framing that told the reader we are back to the asynchronous estimator; all four coverage values kept. |
| P-038 | improved | One archive size per sentence makes the non-monotone deviations ($0.180$, $0.195$, $0.074$) scannable, and "deviation" is the term the table already uses. |

## Items to revert
- P-014 (changed-message, token rule only). The removed sentence contained the number token "6" (step 6). I checked the two places the rationale points to: §3.5 line 141 states "The kept snapshots are the only state the sampler carries between calls, and a stale or empty buffer changes which parents are chosen, not the estimate", and the Table caption at line 650 says "The running sampler does carry the online snapshot buffer of §3.5". The limitation is therefore still asserted and referenced, and the overview's "Every quantity the sampler uses is recomputed from the history it is handed" is qualified two subsections later. Preferred fix: accept as applied under the declared `allow-token-change` exception. If the token rule is enforced strictly, revert to the Before text.

No item has verdict `regressed`. The three `neutral` Codex splits (P-035, P-037) and P-020/P-029 need no action; if the author wants one tweak, P-037 would read better with "For the asynchronous estimator, the top-$k$ archive alone would under-cover ..." as its first sentence, but the current text is correct.

## Overall
The round improved the manuscript: the Claude items removed genuine duplications (each verified against the place the rationale cited) and the Codex items turned long clause stacks in Appendix A into one-result-per-sentence prose without touching a number, reference or qualification. The one pattern to watch is the Codex splits producing runs of three or four short declaratives (P-028, P-035, P-037, P-038) that read slightly mechanical next to the author's dense register, and P-037's reordering, which dropped the sentence that signalled the return from pyABC to the asynchronous estimator; none of these changed content. I would not run another full round; at most a single read-through of Appendix A and §3.1 for rhythm where newly adjacent short sentences now sit side by side.
