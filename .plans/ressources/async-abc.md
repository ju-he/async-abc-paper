# Steady-State Asynchronous ABC-SMC in Propulate

## 1. Motivation

Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC) and Population Monte Carlo (ABC-PMC) are widely used for likelihood-free inference when simulator-based models make the likelihood intractable. These algorithms typically operate in **synchronous generations**:

1. Sample a population of parameters.
2. Run simulations.
3. Accept those below a tolerance threshold.
4. Update proposal distribution and tolerance.
5. Move to the next generation.

While statistically well-understood, this structure is **not well suited to heterogeneous HPC workloads**:

* All workers must wait for the slowest simulation.
* Simulation runtimes often depend strongly on parameter values.
* Synchronization barriers lead to idle resources and poor scaling.

Optimization frameworks (including Propulate) often address this via **steady-state / asynchronous algorithms**, where individuals are produced and evaluated continuously.

The goal of this approach is therefore:

> Design an **asynchronous, steady-state ABC-SMC-like algorithm** compatible with Propulate's architecture and suitable for HPC environments.

A key constraint is that **Propulate propagators only expose a single entry point**:

```python
__call__(inds: List[Individual]) -> Individual
```

The propagator receives the **entire history of evaluated individuals**, and must compute the next candidate from that history. No persistent archive or assimilation callbacks can be used.

---

# 2. Conceptual Development

## 2.1 Classical ABC-SMC

Classical ABC-SMC defines a sequence of target distributions:

[
\pi_r(\theta) \propto \pi(\theta) L_{\varepsilon_r}(\theta)
]

where

* ( \pi(\theta) ) is the prior
* ( L_{\varepsilon_r}(\theta) ) is the ABC likelihood approximation using tolerance ( \varepsilon_r )

Algorithm structure:

1. Maintain population (P_r) of size (N).
2. Sample proposals from mixture kernel around (P_r).
3. Accept proposals with discrepancy ( \rho \le \varepsilon_r ).
4. Compute importance weights.
5. Build next population (P_{r+1}).

This design is inherently **generation-based** and therefore **synchronous**.

---

## 2.2 Asynchronous Requirements

For HPC use cases we want:

* No global synchronization barrier
* Continuous proposal generation
* Ability to incorporate results as soon as they arrive
* Robust behavior under heterogeneous runtime distributions

These requirements suggest a **steady-state population algorithm**:

* Maintain a rolling population (archive)
* Insert accepted particles continuously
* Remove old or low-quality particles
* Adapt tolerance and proposal distribution online

---

## 2.3 Architectural Constraint: Stateless Propagator

In Propulate, the propagator does **not maintain internal state** across calls.

Instead:

* The entire **evaluated history** is provided to `__call__`.
* The algorithm must reconstruct its state from that history.

Thus the algorithm state must be expressed as **functions of the evaluated history**.

Let

[
\mathcal{H}_n = {x_1, x_2, ..., x_n}
]

be the history of evaluated individuals.

The algorithm must compute:

[
\epsilon_n = T(\mathcal{H}_n)
]

[
A_n = A(\mathcal{H}_n, \epsilon_n)
]

[
q_n(\theta) = Q(\theta \mid A_n)
]

Where:

* (T): tolerance scheduler
* (A): archive selection rule
* (Q): proposal construction

This design makes the algorithm **fully deterministic from history** and compatible with the Propulate API.

---

# 3. Mathematical Formulation

## 3.1 Tolerance Memory and Monotone Guarantee

Each proposed individual stores the effective tolerance active at proposal time in
`Individual.tolerance`. The effective tolerance at any call is reconstructed as:

[
\epsilon_{\text{eff}} = \min_{i : \tau_i \neq \text{None}} \tau_i
]

where ( \tau_i = \texttt{ind.tolerance} ) for each individual in history, falling back to
the constructor's `initial_tol` when history is empty.

The scheduler proposes a new tolerance:

[
\epsilon_{\text{proposed}} = \text{scheduler.compute}(\mathcal{H}_n,\, \epsilon_{\text{eff}})
]

The final effective tolerance enforces monotone decrease:

[
\epsilon_n = \min(\epsilon_{\text{eff}},\, \epsilon_{\text{proposed}})
]

This guarantees ( \epsilon_n ) is non-increasing across calls without any mutable state.

---

## 3.2 Archive Definition

For smooth kernels ($K_\epsilon \neq \mathbf{1}[\rho < \epsilon]$) the archive is
the top-$k$ by lowest discrepancy *with no hard cutoff*; every history member
contributes proportionally through its kernel weight, so excluding far-loss
particles via a hard threshold is unnecessary:

[
A_n = \text{Top}_k\bigl(\{\theta_i \in \mathcal{H}_n\},\ \text{order by } \rho_i\bigr)
]

For the (legacy) hard kernel the classical filter `loss < eps` is retained
as a back-compat option. The archive in either case approximates the current
ABC target $\pi_{\epsilon_n}$ via the kernel-weighted mixture proposal of §3.3.

---

## 3.3 Proposal Distribution

The proposal distribution is a kernel-weighted mixture centered at archive
particles:

[
q_n(\theta) = \sum_{j \in A_n} \tilde W_j^{(n)}\, K_\Sigma(\theta - \theta_j),
\qquad
\tilde W_j^{(n)} \propto w_j \cdot K_{\epsilon_n}(\rho_j)
]

where $w_j$ is the stored *core* importance weight $\pi(\theta_j) / \bar q_{\tau_j}(\theta_j)$
(set at proposal time; see §3.5) and the kernel factor $K_{\epsilon_n}(\rho_j)$
is applied **at use time**, so changes to $\epsilon$ automatically reweight
the archive without re-evaluating the simulator. The perturbation kernel is
Gaussian:

[
K_\Sigma(\theta - \theta_j) = \mathcal{N}(\theta_j,\ \Sigma_n),
\qquad
\Sigma_n = s \cdot \widehat{\mathrm{Cov}}_{\tilde W}(A_n).
]

The smoothing factor $s$ is the ``perturbation_scale`` hyperparameter. The
weighted covariance uses the kernel-effective weights $\tilde W^{(n)}$ rather
than the raw stored weights so the perturbation adapts to the high-likelihood
neighbourhood as the kernel shifts.

For numerical stability with the Gaussian kernel (which can yield
$\tilde W_j^{(n)}$ spanning many decades), all archive-weight arithmetic is
performed in log-space (log-sum-exp normalisation, softmax).

---

## 3.4 Sampling Step

Given the reconstructed proposal:

1. Select parent index $J \sim \text{Categorical}(\tilde W^{(n)})$.
2. Draw $\theta^\star \sim \mathcal{N}(\theta_J,\ \Sigma_n)$ in batched
   reject-resample inside the box.
3. If reject-resample exhausts its budget (very wide kernel relative to
   the box), fall back to a **uniform prior draw**, not boundary clipping.
   This preserves the truncated-density semantics that the importance
   weight assumes.
4. Stamp candidate with $\epsilon_n$ via `child.tolerance = epsilon_n`.
5. Evaluate simulation (outside the propagator).

Note: there is **no inline rejection step** at acceptance: the loss is
evaluated externally by Propulate and the candidate enters history
unconditionally. Archive selection in future calls handles filtering via the
kernel.

---

# 4. Evaluation Framing

For the paper, the primary systems comparison is **quality achieved under a fixed wall-clock budget**,
not only quality after an equal number of simulations.

This matches the intended HPC use case:

* jobs are frequently constrained by queue limits or allocation wall times
* heterogeneous runtimes and persistent stragglers make equal-simulation comparisons less representative
* asynchronous advantage should appear when synchronization barriers create idle time and lost throughput

Fixed-walltime comparisons intentionally measure **end-to-end practical efficiency**:

* simulator throughput
* scheduling overhead
* synchronization overhead
* framework overhead

This is desirable for the paper because the claim is about practical HPC performance, not an abstract sampler in isolation.

Statistical validity remains a separate evaluation axis and must still be checked explicitly via:

* posterior recovery on benchmark models
* simulation-based calibration (SBC)

Walltime alone is therefore **not** treated as a validity metric.
in future calls handles filtering.

---

## 3.5 Streaming-AMIS Importance Weight

A newly proposed particle $\theta^\star$ receives an importance weight under
the *balance heuristic* (Veach 1997; Owen & Zhou 2000) over a sliding ring
buffer $\mathcal{S}$ of past proposals (size $S$, default 20, sampled every
``amis_interval`` calls):

[
w^\star = \frac{\pi(\theta^\star)}{\bar q_n(\theta^\star)},
\qquad
\bar q_n(\theta^\star) = \frac{1}{1 + |\mathcal{S}|}\Bigl[q_n(\theta^\star) + \sum_{s \in \mathcal{S}} q_s(\theta^\star)\Bigr].
]

Each snapshot $q_s$ stores enough state (archive positions, normalised
mixture weights, Cholesky factor of $\Sigma_s$) to evaluate its proposal
density at any new $\theta^\star$ in $O(k \cdot d^2)$.

This is the **streaming limit** of the AMIS scheme (Cornuet, Marin, Mira &
Robert 2012): stage size 1, the proposal mixture updates after every single
arrival, and every accepted particle is implicitly re-weighted against the
cumulative mixture at every subsequent use. Consistency and a CLT follow as
a corollary of the AMIS theorem combined with Wilkinson 2013 smooth-ABC
under explicit conditions (paper §4).

**Setting $|\mathcal{S}| = 0$ recovers the legacy single-current-proposal
weighting** ($w^\star = \pi(\theta^\star) / q_n(\theta^\star)$), preserving
the previous behaviour bit-for-bit.

**Weight staleness is resolved under AMIS.** The previous single-proposal
implementation computed $w_i$ against $q_{\tau_i}$ (the archive at proposal
time) and used that stale weight forever; the AMIS denominator instead
averages over the snapshot buffer, providing a coherent weighting against
the cumulative proposal mixture rather than a moment-of-arrival snapshot.

---

# 4. Relation to Existing Literature

## 4.1 ABC-SMC and ABC-PMC

Relevant work:

* Toni et al. (2008): ABC-SMC
* Beaumont et al. (2009): adaptive ABC-PMC
* Del Moral, Doucet, Jasra (2012): adaptive ABC-SMC

These methods use sequential populations with decreasing tolerances.

Our approach differs by:

* removing generation boundaries
* reconstructing the active population from history
* encoding the tolerance schedule in `Individual.tolerance` fields

---

## 4.2 Adaptive Importance Sampling

With fixed tolerance, the method reduces to **adaptive importance sampling** with evolving proposals.

This connects directly to adaptive PMC methods.

---

## 4.3 Asynchronous Sequential Monte Carlo

Work such as the **Particle Cascade** explores barrier-free SMC algorithms for distributed computing.

These methods:

* process particles asynchronously
* update weights and resampling locally

Our approach shares the **execution model** but differs in its use of ABC likelihoods.

---

## 4.4 Evolutionary and Steady-State Algorithms

Steady-state evolutionary algorithms maintain a rolling population updated continuously.

Our algorithm can be interpreted as a **Bayesian analogue** of these methods.

---

# 5. Alternative Approaches Considered

## 5.1 Mini-Epoch Asynchronous SMC

Define short epochs where proposal and tolerance remain fixed.

Pros:

* Closer to classical SMC theory

Cons:

* Reintroduces (soft) synchronization points — workers stall at epoch
  boundaries while waiting for in-flight stragglers
* Dilutes the headline "no synchronisation barrier" claim

---

## 5.2 Smooth-Kernel ABC + AMIS (**adopted**)

The current design. Replace the hard threshold with a normalised kernel

[
K_\epsilon(\rho) \in \{\,\exp(-\rho^2 / 2\epsilon^2),\ \max(0,\ 1 - \rho^2/\epsilon^2)\,\}
]

and combine with **streaming Adaptive Multiple Importance Sampling** (AMIS,
Cornuet et al. 2012) over a snapshot ring buffer of past proposals. The
combination delivers:

* Continuous archive reweighting (no prior-vs-archive discontinuity).
* Coherent importance weights against the cumulative proposal mixture
  (Veach 1997 balance heuristic), resolving the moving-archive staleness
  of the legacy single-proposal scheme.
* Consistency + CLT as a corollary of AMIS + Wilkinson 2013 (paper §4)
  under explicit conditions on the proposal sequence and bandwidth
  schedule.

The earlier-considered "smooth kernel deviates from classical ABC
rejection" objection no longer applies once AMIS provides the matching
theoretical framework.

---

## 5.3 Fully Online SMC with Reweighting

Adjust weights when tolerance changes; reweight the entire archive at every
$\epsilon$ update.

Pros:

* Theoretically elegant.

Cons:

* Requires re-evaluating $q_\tau$ for every archive member at every call
  — $O(n^2 k d^2)$ cumulatively, infeasible at scale.
* Subsumed by §5.2: the AMIS snapshot buffer is exactly a sparse,
  finite-memory approximation to fully online reweighting that retains
  consistency under standard conditions.

---

## 5.4 Explicit Persistent Archive

Maintain archive state between propagator calls.

Pros:

* Conceptually clean.

Cons:

* Incompatible with Propulate's stateless-propagator interface; would
  require deeper changes to the framework and to crash-recovery
  semantics. The adopted design keeps the archive reconstructible from
  evaluated history.

---

# 6. Advantages of the History-Reconstructed Steady-State Approach

1. Compatible with Propulate's stateless-propagator architecture.
2. Eliminates synchronisation barriers entirely (not just per-generation —
   no within-generation barrier either, unlike pyABC DYN).
3. Smooth kernels remove the prior-vs-archive phase discontinuity that
   complicated the legacy hard-threshold design.
4. AMIS reweighting (§3.5) resolves the moving-archive importance-weight
   staleness with a coherent balance-heuristic denominator.
5. Deterministic reconstruction from evaluated history; a single-float
   tightness floor on the propagator (cache, not algorithmic state)
   makes the monotone-bandwidth guarantee robust to island migration.
6. Consistency + CLT (paper §4) inherited from AMIS + Wilkinson smooth-ABC.
7. Naturally scalable in distributed environments; per-call AMIS cost is
   $O(S \cdot k \cdot d^2)$ at $S = 20$, $k = 100$, $d = 10$ — negligible
   relative to typical simulator cost.

---

# 7. Implementation

## 7.1 Stateless Tolerance Scheduler Contract

Schedulers implement `compute(inds, current_tol) -> float` — a **pure function** of history.

```python
@abstractmethod
def compute(self, inds: List[Individual], current_tol: float) -> float:
    """Propose next tolerance from evaluated history. Must not mutate self."""
```

The `update()` method is kept as a deprecated alias for backward compatibility.

---

## 7.2 `QuantileToleranceScheduler.compute`

Computes the p-th percentile of **accepted-only** losses (individuals with `loss < current_tol`).
Using all-history losses would include prior-phase samples with large losses, biasing the
percentile upward. Filtering to accepted individuals avoids this.

```python
def compute(self, inds, current_tol):
    accepted = [ind for ind in inds if ind.loss < current_tol]
    if len(accepted) < self.population_size + self.additional_needed_inds:
        return current_tol
    return float(np.percentile([ind.loss for ind in accepted], self.percentile))
```

---

## 7.3 `GeometricDecayToleranceScheduler.compute`

Stateless epoch-counting: sort accepted individuals (those with `loss < initial_tol`) by
generation, divide into batches of `population_size + additional_needed_inds`, and apply
one decay step per batch if enough individuals survive the tighter threshold.

```python
def compute(self, inds, current_tol):
    accepted_all = sorted([i for i in inds if i.loss < self.initial_tol],
                          key=lambda i: i.generation)
    tol = self.initial_tol
    batch_size = self.population_size + self.additional_needed_inds
    consumed = 0
    while consumed + batch_size <= len(accepted_all):
        batch = accepted_all[consumed: consumed + batch_size]
        next_tol = self.decay_factor * tol
        if len([i for i in batch if i.loss < next_tol]) >= self.population_size:
            tol = next_tol
        consumed += batch_size
    return tol
```

---

## 7.4 `AcceptanceRateToleranceScheduler.compute`

Uses a sliding window of the `population_size + additional_needed_inds` most recently
evaluated individuals. Returns `current_tol` unchanged when history is too short.

```python
def compute(self, inds, current_tol):
    window_size = self.population_size + self.additional_needed_inds
    if len(inds) < window_size:
        return current_tol
    recent = sorted(inds, key=lambda i: i.generation)[-window_size:]
    rate = len([i for i in recent if i.loss < current_tol]) / len(recent)
    if rate > self.high_rate:
        return current_tol * self.shrink_factor
    elif rate < self.low_rate:
        return current_tol * self.expand_factor
    return current_tol
```

---

## 7.5 `ABC.select_archive`

Pure function returning top-k accepted individuals, sorted by loss:

```python
def select_archive(self, inds, tol):
    accepted = self.filter_by_tolerance(inds, tol)
    return sorted(accepted, key=lambda ind: ind.loss)[:self.k]
```

---

## 7.6 `ABC.__call__` — Stateless Reconstruction

```python
def __call__(self, inds):
    # Reconstruct effective tolerance from stamped history
    tol_from_history = min(
        (ind.tolerance for ind in inds if ind.tolerance is not None),
        default=self.tol,          # self.tol is initial_tol, never mutated
    )
    proposed_tol = self.tolerance_scheduler.compute(inds, tol_from_history)
    effective_tol = min(tol_from_history, proposed_tol)   # monotone guarantee

    # Build archive
    archive = self.select_archive(inds, effective_tol)

    # Prior phase
    if len(archive) < self.k:
        child = Individual(...)
        child.weight = 1.0
        return child

    # Kernel + sample + weight (unchanged structure)
    ...
    child.tolerance = effective_tol   # stamp for future history reconstruction
    child.weight = self.prior_density / denom
    return child
```

---

## 7.7 Covariance Regularization

The kernel covariance matrix is regularized to ensure positive definiteness:

```python
cov += 1e-6 * np.eye(dim)
kernel_cov = 0.5 * (kernel_cov + kernel_cov.T)     # symmetrize
eigs = np.linalg.eigvalsh(kernel_cov)
if eigs.min() <= 0:
    kernel_cov += (-eigs.min() + 1e-8) * np.eye(dim)
```

---

# 8. Final Algorithm

Given evaluated history `inds`:

1. Reconstruct effective tolerance:

   ```
   tol_hist  = min(ind.tolerance for ind in inds if ind.tolerance is not None,
                   default=initial_tol)
   tol_sched = scheduler.compute(inds, tol_hist)
   epsilon_n = min(tol_hist, tol_sched)
   ```

2. Select archive:

   ```
   archive = top_k({theta_i in H : loss_i < epsilon_n})
   ```

3. If `len(archive) < k`:

   * sample from prior, set `weight = 1.0`, return

4. Otherwise:

   * construct mixture proposal kernel from archive
   * sample parent, perturb, clip to bounds
   * stamp `child.tolerance = epsilon_n`
   * compute importance weight `w* = pi(theta*) / q_n(theta*)`

5. Return candidate individual

---

# 9. Interpretation

The resulting algorithm is best interpreted as:

> A **history-adaptive steady-state ABC population sampler** approximating the ABC-SMC target path while eliminating generation-level synchronization.

The tolerance schedule is encoded implicitly in the `Individual.tolerance` field of each
proposed individual, making the full algorithm state recoverable from history alone.

---

# 10. Known Approximations and Limitations

| Issue | Status |
|-------|--------|
| Weight staleness in async execution | **Resolved** under AMIS: the snapshot-buffer balance-heuristic denominator (§3.5) provides a coherent reweighting against the cumulative proposal mixture. Setting `amis_snapshots=0` reverts to the legacy single-proposal approximation. |
| Prior-vs-archive phase discontinuity | **Resolved** under smooth kernels (§3.2): the bootstrap rule uses `len(history) >= k` and every history member contributes through its kernel weight. Hard kernel retains the classical prior-phase guard. |
| Snapshot-buffer truncation | New approximation introduced by AMIS: the buffer has finite size $S$ (default 20), so AMIS denominator approximates the full cumulative mixture only when $S$ is large enough relative to the proposal mixing rate. Theorem (paper §4) requires (C4); empirically tight at $S=20$ across benchmarks. |
| Simulation-time bias | Steady-state design keeps every accepted particle, unlike pyABC DYN's "discard latecomers" rule. Faster-simulating parameter regions may be over-represented in the archive. Measured empirically (runtime-heterogeneity experiment) but not corrected algorithmically. |
| $O(n)$ cost per call over unbounded history | Not addressed; add `max_history` cap if needed for very long runs |
| Categorical/integer search spaces | Not supported; ABC requires continuous (float) limits only |
| Checkpoint granularity mismatch (async vs sync) | Addressed via `checkpoint_strategy="time_uniform"` in `posterior_quality_curve()`, which resamples both method types onto a shared time grid using LOCF |
| Wasserstein metric interpretation | Documented: W1-to-point-mass (mean absolute deviation from truth in 1D); sliced Wasserstein for multi-D |
| Finite-sample unbiasedness (Paige & Wood 2014 sense) | Open. AMIS is consistent but not unbiased for finite $n$. A local-decision-rule extension that recovers finite-sample unbiasedness for ABC is sketched as future work. |

---

# 11. Summary

This design:

* combines smooth-kernel ABC with streaming AMIS reweighting in a
  single-arrival-driven (generation-free) regime
* reconstructs the archive and bandwidth deterministically from evaluated
  history (plus a single-float tightness floor to survive island migration)
* applies the kernel factor $K_\epsilon(\rho)$ at use time so $\epsilon$
  updates reweight the archive without re-evaluating the simulator
* uses the balance heuristic over a finite snapshot ring buffer for
  coherent importance weights against the cumulative proposal mixture
* supports asynchronous HPC execution with no synchronisation barriers
* integrates cleanly into the Propulate stateless-propagator model
* admits a consistency + CLT (paper §4) as a corollary of AMIS + smooth-ABC
  theory under explicit, stated conditions

The result is a **generation-free, single-arrival-driven ABC algorithm**
suitable for large-scale simulator-based inference on heterogeneous
computing environments, with provable guarantees inherited from the
AMIS / smooth-ABC literature.
