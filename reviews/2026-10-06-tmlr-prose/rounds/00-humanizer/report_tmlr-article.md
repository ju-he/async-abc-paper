# Humanizer pass (stage 1), latex/tmlr/tmlr-article.tex

Voice-only pass per templates/humanizer_brief.md. 37 sentences changed; file length 164987 -> 164703 characters. No number, citation, reference, label, hedge or heading changed (checked by protected-token comparison per edit).

## 1. meta-commentary

**Before**
```text
and find it calibrated where a pyABC baseline with the same kernel and schedule under-covers. Four measurements follow. The barrier's cost,
```

**After**
```text
and find it calibrated where a pyABC baseline with the same kernel and schedule under-covers. The barrier's cost,
```

## 2. slogan -> mechanism

**Before**
```text
This paper removes the generation. Our sampler refits its proposal mixture after every arriving evaluation,
```

**After**
```text
We remove the generation: our sampler refits its proposal mixture after every arriving evaluation,
```

## 3. American spelling

**Before**
```text
A Gaussian mixture centred on the archive members,
```

**After**
```text
A Gaussian mixture centered on the archive members,
```

## 4. 'is what makes' cleft

**Before**
```text
One rule makes this work on a machine where workers never wait for each other: every quantity the sampler uses is recomputed from the history it is handed. There is no population
```

**After**
```text
Every quantity the sampler uses is recomputed from the history it is handed; this rule lets workers run without ever waiting for one another. There is no population
```

## 5. intensifier

**Before**
```text
provides exactly this contract (Appendix~\ref{app:implementation}).
```

**After**
```text
provides this contract (Appendix~\ref{app:implementation}).
```

## 6. clause-stacked sentence split

**Before**
```text
is on the same tightening sequence: the bandwidth in force at call $n$ is the smaller of
```

**After**
```text
is on the same tightening sequence. The bandwidth in force at call $n$ is the smaller of
```

## 7. dramatic 'the one question'

**Before**
```text
and how faithful a stand-in it is becomes the one question of \S\ref{sec:theory}.
```

**After**
```text
and \S\ref{sec:theory} asks how faithful a stand-in it is.
```

## 8. intensifier

**Before**
```text
and a worker that returns from a slow simulation simply finds a larger history than the one it left.
```

**After**
```text
and a worker that returns from a slow simulation finds a larger history than the one it left.
```

## 9. theatrical meta-commentary

**Before**
```text
The analysis asks one question: \emph{is the denominator the density the sample was actually drawn from?} Write
```

**After**
```text
The question is whether the denominator is the density the sample was drawn from. Write
```

## 10. vague verb 'carries'

**Before**
```text
Assumption~\ref{ass:stab} carries everything else. For consistency,
```

**After**
```text
Assumption~\ref{ass:stab} holds the remaining conditions. For consistency,
```

## 11. intensifier

**Before**
```text
This is one concrete reason the method uses smooth kernels.
```

**After**
```text
This is one reason the method uses smooth kernels.
```

## 12. vague verb 'hosts'

**Before**
```text
The Gaussian mean hosts the two studies with injected runtime heterogeneity: a persistently slow worker, and a lognormal multiplier on every evaluation's runtime.
```

**After**
```text
The two studies with injected runtime heterogeneity run on the Gaussian mean: a persistently slow worker, and a lognormal multiplier on every evaluation's runtime.
```

## 13. American spelling

**Before**
```text
The $50^3$ twin and scaling runs used an earlier parameterisation of the same simulator
```

**After**
```text
The $50^3$ twin and scaling runs used an earlier parameterization of the same simulator
```

## 14. economic metaphor 'buys'

**Before**
```text
It is the reference for what an adaptive proposal buys per simulation.
```

**After**
```text
It is the reference for the per-simulation gain of an adaptive proposal.
```

## 15. 'is what' cleft

**Before**
```text
the \emph{equal-wall-clock} ratio is the same quantity at the end of both runs, and the throughput ratio is what separates the two.
```

**After**
```text
the \emph{equal-wall-clock} ratio is the same quantity at the end of both runs, and the two differ by the throughput ratio.
```

## 16. personified evidence

**Before**
```text
The slope of $\epsilon_{(k)}(n)$ in $\log n$ says whether more simulations keep converting into a tighter tolerance.
```

**After**
```text
The slope of $\epsilon_{(k)}(n)$ in $\log n$ shows whether more simulations keep converting into a tighter tolerance.
```

## 17. self-reference to the paper

**Before**
```text
Figure~\ref{fig:barrier} is the paper's central measurement. It covers every configuration on which we ran the barrierized twin:
```

**After**
```text
Figure~\ref{fig:barrier} covers every configuration on which we ran the barrierized twin:
```

## 18. American spelling (caption)

**Before**
```text
and Cellular Potts at $50^3$ ($48$--$384$ workers, earlier parameterisation, utilization ratio)
```

**After**
```text
and Cellular Potts at $50^3$ ($48$--$384$ workers, earlier parameterization, utilization ratio)
```

## 19. American spelling (caption)

**Before**
```text
Cellular Potts $50^3$ (earlier parameterisation, $5.1$\,s; $1800$\,s; log--log)
```

**After**
```text
Cellular Potts $50^3$ (earlier parameterization, $5.1$\,s; $1800$\,s; log--log)
```

## 20. intensifier 'itself'

**Before**
```text
The asynchronous sampler itself scales almost linearly on this workload,
```

**After**
```text
The asynchronous sampler scales almost linearly on this workload,
```

## 21. marketing construction ('what a practitioner gets')

**Before**
```text
Throughput is only instrumental; the tolerance a run reaches is what a practitioner gets. Table~\ref{tab:matched-eps} reads it off every benchmark
```

**After**
```text
Throughput matters only through the tolerance a run reaches. Table~\ref{tab:matched-eps} reads it off every benchmark
```

## 22. meta-commentary

**Before**
```text
and reaches $4.1\times$ on Cellular Potts. The curves add what a single budget cannot show. On the two cheap benchmarks
```

**After**
```text
and reaches $4.1\times$ on Cellular Potts. On the two cheap benchmarks
```

## 23. economic metaphor 'pays'

**Before**
```text
Set against a growing per-simulation cost, that fixed cost decides where the method pays.
```

**After**
```text
Set against a growing per-simulation cost, that fixed cost determines where the method becomes advantageous.
```

## 24. economic metaphor 'not free'

**Before**
```text
The effective sample size is a few multiples of $k$, but raising $k$ is not free:
```

**After**
```text
The effective sample size is a few multiples of $k$, but a larger $k$ has costs:
```

## 25. business metaphor 'headroom'

**Before**
```text
The effective-sample-size ceiling is where the method has headroom.
```

**After**
```text
The effective-sample-size ceiling is the part of the method with the most room for improvement.
```

## 26. economic metaphor 'costs nothing'

**Before**
```text
That restriction costs nothing in the case of most interest, $r\equiv1$.
```

**After**
```text
That restriction is vacuous in the case of most interest, $r\equiv1$.
```

## 27. 'is what makes' cleft

**Before**
```text
Stability is also what makes the theorem usable, because it survives division by a consistent variance estimate:
```

**After**
```text
Stability also makes the theorem usable, because it survives division by a consistent variance estimate:
```

## 28. 'what X turns on is' cleft

**Before**
```text
What the exponent turns on is the inference problem, not the schedule.
```

**After**
```text
The exponent depends on the inference problem, not on the schedule.
```

## 29. 'is what makes' cleft

**Before**
```text
Particles are indexed by \emph{proposal} time, not arrival time. This is what makes the martingale structure survive asynchronous execution:
```

**After**
```text
Particles are indexed by \emph{proposal} time, not arrival time. This indexing is why the martingale structure survives asynchronous execution:
```

## 30. 'is what makes' cleft

**Before**
```text
This is what makes the theorem a statement about the algorithm as implemented rather than about an idealized variant of it:
```

**After**
```text
The theorem is therefore a statement about the algorithm as implemented, not about an idealized variant of it:
```

## 31. economic metaphor 'buy'

**Before**
```text
This step is what the rates buy; the remainder of the proof never touches
```

**After**
```text
This step is where the rates are used; the remainder of the proof never touches
```

## 32. 'is what' cleft

**Before**
```text
there is nothing to synchronize between workers, which is what fits the method to the asynchronous island model
```

**After**
```text
there is nothing to synchronize between workers, which fits the method to the asynchronous island model
```

## 33. vague verb 'books'

**Before**
```text
states its denominator assumption for positive functions and books the discrepancy as factor~(c) of $r$
```

**After**
```text
states its denominator assumption for positive functions and assigns the discrepancy to factor~(c) of $r$
```

## 34. announcing simplicity

**Before**
```text
The prediction is a one-line calculation: the slow worker's recorded evaluation time
```

**After**
```text
The slow worker's recorded evaluation time
```

## 35. economic metaphor 'bought' (caption)

**Before**
```text
the horizontal extent is what the wall clock bought.
```

**After**
```text
the horizontal extent is the extra simulations the wall clock allowed.
```

## 36. economic metaphor 'buy'

**Before**
```text
one might expect to buy calibration with throughput. Over the range we can test, that expectation does not hold.
```

**After**
```text
one might expect to trade throughput for calibration. Over the range we can test, that expectation does not hold.
```

## 37. punchline + economic metaphor

**Before**
```text
The $k$ decision is therefore a floor to clear rather than a frontier to negotiate: go far enough above the small-archive failure region and then stop, because further enlargement buys no calibration and costs throughput monotonically.
```

**After**
```text
The $k$ decision is therefore a floor to clear: go far enough above the small-archive failure region and stop, since further enlargement adds no calibration and costs throughput monotonically.
```

## 38. American spelling

**Before**
```text
parameterisation
```

**After**
```text
parameterization
```
(one further occurrence, in the Appendix E "Two configurations" paragraph)

## Post-A/B adjustments (after both blinded reviews)

- "the part of the method with the most room for improvement" -> "where the method can still
  improve" (the Claude reviewer read the superlative as a strengthening; the business metaphor
  "headroom" stays out).
- "a larger $k$ has costs" -> "raising $k$ has costs" (codex found the nominal subject awkward).
- "itself" restored in "The asynchronous sampler itself scales almost linearly" (it marks the
  switch of subject away from the twin).
