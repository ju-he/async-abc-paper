# Appendix A read-through for rhythm (after round 2)

Requested by the author after both round-2 verifiers noted that the Appendix A splits left runs of short declaratives. Prose-only rejoins where a split had separated a setting from its result or a condition from its consequence; every number, reference and qualification unchanged (checked per edit).

## 1. Measured, in part: merge the setting sentence into the result it introduces

**Before**
```text
The five Cellular Potts production replicates have about $12{,}900$ asynchronous evaluations each across $48$ workers and two correlated dimensions, so that (c) is live (Appendix~\ref{app:cpm}). Both factors are active there: the bootstrap draws
```

**After**
```text
The five Cellular Potts production replicates (Appendix~\ref{app:cpm}) have about $12{,}900$ asynchronous evaluations each across $48$ workers and two correlated dimensions, so that (c) is live, and both factors are active: the bootstrap draws
```

## 2. Measured, in part: three one-clause sentences on the Gaussian history rejoined; 'there too' link to the Cellular Potts result restored

**Before**
```text
On a one-dimensional Gaussian-mean history, the exact $n$-point mixture is computable. Contributions (a) and (b) together give $\widehat\zeta=0.042$. The total-variation distance from the reference that removes (a) and (b) is $0.26\%$, and the estimate is $0.5\%$ narrower.
```

**After**
```text
On a one-dimensional Gaussian-mean history, where the exact $n$-point mixture is computable, contributions (a) and (b) together give $\widehat\zeta=0.042$; the total-variation distance from the reference that removes them is $0.26\%$, and there too the estimate is $0.5\%$ narrower.
```

## 3. Measured, in part: the bound's two comparisons back into one sentence with the general remark

**Before**
```text
The bound \eqref{eq:tilt-bound} is loose against both direct measurements by an order of magnitude. On Cellular Potts it gives $3.6$--$4.0\%$ against $0.2$--$0.5\%$; on the one-dimensional history it gives $4.4\%$ against $0.26\%$. Sup-norm-free bounds valid for every $f$ are generally this loose.
```

**After**
```text
The bound \eqref{eq:tilt-bound} is loose against both direct measurements by an order of magnitude ($3.6$--$4.0\%$ against $0.2$--$0.5\%$ on Cellular Potts, $4.4\%$ against $0.26\%$ on the one-dimensional history), as sup-norm-free bounds valid for every $f$ generally are.
```

## 4. Stabilization: condition, consequence and verdict in one sentence

**Before**
```text
Assumption~\ref{ass:stab}(ii) needs $b>\tfrac12$. Under this condition, the aggregate $n^{-1/2}\sum_{i\le n}\|q_i-q_\infty\|_\infty$ vanishes like $n^{1/2-b}$. The production runs sit at that threshold.
```

**After**
```text
Assumption~\ref{ass:stab}(ii) needs $b>\tfrac12$, at which the aggregate $n^{-1/2}\sum_{i\le n}\|q_i-q_\infty\|_\infty$ vanishes like $n^{1/2-b}$; the production runs sit at that threshold.
```

## 5. Stabilization: the pinned-bandwidth experiment attached to the claim it tests

**Before**
```text
A tightening bandwidth does not move it. We ran a two-dimensional Gaussian-mean benchmark serially under the same propagator, once with the data-driven schedule and once with the bandwidth pinned at the tolerance the first run reached a third of the way in; the two runs differ in $b$ by less than $0.07$, and both satisfy $b>\tfrac12$ at $b=0.61$--$0.76$.
```

**After**
```text
A tightening bandwidth does not move it: we ran a two-dimensional Gaussian-mean benchmark serially under the same propagator, once with the data-driven schedule and once with the bandwidth pinned at the tolerance the first run reached a third of the way in, and the two runs differ in $b$ by less than $0.07$, both satisfying $b>\tfrac12$ at $b=0.61$--$0.76$.
```

## 6. Fixed versus growing m: the growing-m statement and its two qualifications in one sentence

**Before**
```text
Letting $m=m_n\to\infty$ with $V_n/m_n\to0$ removes \emph{this} contribution to $r$. This does not by itself give $r\equiv1$, since the other three factors remain. The proofs go through for any $m_n=o(n/\log n)$ \emph{provided the floor $\delta$ is held fixed rather than tied to $m$}.
```

**After**
```text
Letting $m=m_n\to\infty$ with $V_n/m_n\to0$ removes \emph{this} contribution to $r$ (not $r\equiv1$ itself, since the other three factors remain), and the proofs go through for any $m_n=o(n/\log n)$ \emph{provided the floor $\delta$ is held fixed rather than tied to $m$}.
```

## 7. Fixed versus growing m: four short sentences on the delta-m coupling rejoined as one causal chain

**Before**
```text
The implementation ties them through $\delta=0.5/(m{+}1)$. Growing $m$ would therefore send $\delta\to0$ and remove the uniform weight bound $1/\delta$ of Lemma~\ref{lem:bounded}. That bound makes the Lindeberg step trivial and fixes the constants in the law of large numbers. The rate conditions would have to be redone.
```

**After**
```text
The implementation ties them through $\delta=0.5/(m{+}1)$, so growing $m$ there would send $\delta\to0$ and with it the uniform weight bound $1/\delta$ of Lemma~\ref{lem:bounded}, which makes the Lindeberg step trivial and fixes the constants in the law of large numbers; the rate conditions would have to be redone.
```

## 8. Fixed versus growing m: measurement and its consequence in one sentence

**Before**
```text
On the Gaussian-mean history, the posterior estimate is within $3\times10^{-4}$ in total variation of the exact $n$-point mixture by $m=50$. At that point, a $2.5\times$ more expensive post-hoc pass removes factor (b) almost entirely.
```

**After**
```text
On the Gaussian-mean history the posterior estimate is within $3\times10^{-4}$ in total variation of the exact $n$-point mixture by $m=50$, so a $2.5\times$ more expensive post-hoc pass removes factor (b) almost entirely.
```

