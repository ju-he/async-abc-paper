# Humanizer report: latex/tmlr/tmlr-article.tex (new text since the 2026-10-06 run only)

## H-01: defensive tail 'rather than assumed' in the contribution list

**Before**
```text
with exact consistency when the ratio tends to one; the adaptedness that asynchronous execution requires is derived from the execution model rather than assumed, and we decompose the ratio and measure part of it
```

**After**
```text
with exact consistency when the ratio tends to one; we derive from the execution model the adaptedness that asynchronous execution requires, and decompose the ratio and measure part of it
```

## H-02: intensifier 'exact'

**Before**
```text
which the implementation builds to that exact recipe.
```

**After**
```text
which the implementation builds to that recipe.
```

## H-03: 'What X is Y' cleft

**Before**
```text
What the theory contributes to that run is the accounting of where error enters;
```

**After**
```text
For that run the theory supplies the accounting of where error enters;
```

## H-04: 'itself' as emphasis

**Before**
```text
The emitted density itself is the conditional law of the candidate
```

**After**
```text
The emitted density is the conditional law of the candidate
```

## H-05: vague verb 'carries'

**Before**
```text
Under asynchronous execution this factor additionally carries the difference between
```

**After**
```text
Under asynchronous execution this factor additionally includes the difference between
```

## H-06: emphatic 'do' + three 'which' clauses

**Before**
```text
Completion times do steer which worker proposes next and which prefix it sees. The lemma says that this steering changes \emph{which} proposal density is used, which is predictable, and not the law of the candidate given that density, which is fixed by fresh randomness.
```

**After**
```text
Completion times steer which worker proposes next and which prefix it sees. The lemma says that this steering changes \emph{which} proposal density is used, a predictable choice, and not the law of the candidate given that density, which is fixed by fresh randomness.
```

## H-07: restating closer 'derived, not assumed' (already stated by the contribution list and the lemma)

**Before**
```text
Lemma~\ref{lem:adapted} then gives the adapted filtration with $\tilde q_i$ the emitted density, so adaptedness is derived, not assumed.
```

**After**
```text
Lemma~\ref{lem:adapted} then gives the adapted filtration with $\tilde q_i$ the emitted density.
```

## H-08: intensifier 'actual'

**Before**
```text
measures the actual effect on the posterior mean at $0.013$ or less.
```

**After**
```text
measures the effect on the posterior mean at $0.013$ or less.
```

## Post-A/B adjustment

- H-02 reverted: codex read "exact recipe" as a qualification (the implementation conforms without deviation) and asked to keep it; the brief forbids touching qualifications, so the word is back. Net: 7 edits applied.
