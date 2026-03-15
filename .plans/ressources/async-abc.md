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

The active archive is reconstructed from history using strict inequality:

[
A_n = \{\theta_i \in \mathcal{H}_n : \rho_i < \epsilon_n\}
]

A fixed-size top-k subset is selected by loss (ascending):

[
A_n = \text{Top}_k(A_n)
]

The archive therefore approximates the current ABC target:

[
\pi_{\epsilon_n}(\theta)
]

---

## 3.3 Proposal Distribution

The proposal distribution is a mixture kernel centered at archive particles:

[
q_n(\theta) =
\sum_{j=1}^{k} W_j^{(n)} K(\theta \mid \theta_j)
]

with normalized weights

[
W_j^{(n)} = \frac{w_j}{\sum_i w_i}
]

The perturbation kernel is Gaussian:

[
K(\theta \mid \theta_j) = \mathcal{N}(\theta_j, \Sigma_n)
]

where

[
\Sigma_n = s \cdot \widehat{\mathrm{Cov}}_w(A_n)
]

---

## 3.4 Sampling Step

Given the reconstructed proposal:

1. Select parent index (J \sim \text{Categorical}(W^{(n)}))
2. Sample candidate:

[
\theta^\star \sim \mathcal{N}(\theta_J, \Sigma_n)
]

3. Clip to search-space bounds.
4. Stamp candidate with ( \epsilon_n ) via `child.tolerance = epsilon_n`.
5. Evaluate simulation (outside propagator).

Note: there is **no inline rejection step**. The loss is evaluated externally by Propulate
and the candidate enters history regardless of whether `loss < epsilon_n`. Archive selection
in future calls handles filtering.

---

## 3.5 Importance Weight

Accepted particles receive weight

[
w^\star = \frac{\pi(\theta^\star)}{q_n(\theta^\star)}
]

where

[
q_n(\theta^\star) =
\sum_{j=1}^{k}
W_j^{(n)}
\mathcal{N}(\theta^\star; \theta_j, \Sigma_n)
]

This follows the standard ABC-PMC importance weighting scheme.

**Weight staleness (known approximation):** in asynchronous execution the archive can change
between proposal time and result arrival. The stored weight is computed against the archive at
proposal time and is therefore an approximation. This degrades gracefully for slowly-changing
archives and is accepted as the practical trade-off for barrier-free execution.

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

* Introduces synchronization points

---

## 5.2 Smooth-Kernel ABC

Replace hard threshold with kernel weights:

[
K_\epsilon(\rho) = \exp(-\rho^2 / 2\epsilon^2)
]

Pros:

* Enables incremental reweighting

Cons:

* More expensive and deviates from classical ABC rejection

---

## 5.3 Fully Online SMC with Reweighting

Adjust weights when tolerance changes.

Pros:

* Theoretically elegant

Cons:

* Requires reweighting the entire particle archive
* Hard to integrate with the Propulate interface

---

## 5.4 Explicit Persistent Archive

Maintain archive state between proposals.

Pros:

* Conceptually clean

Cons:

* Incompatible with Propulate propagator interface

---

# 6. Advantages of the History-Reconstructed Steady-State Approach

1. Compatible with Propulate architecture
2. Eliminates synchronization barriers
3. Simple integration with existing evolutionary infrastructure
4. Deterministic reconstruction from evaluated history
5. Monotone tolerance guaranteed via `min()` over stored `ind.tolerance` values
6. Naturally scalable in distributed environments

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
| Weight staleness in async execution | Accepted approximation; documented in `__call__` docstring |
| O(n) cost per call over unbounded history | Not addressed; add `max_history` cap if needed for very long runs |
| Categorical/integer search spaces | Not supported; ABC requires continuous (float) limits only |

---

# 11. Summary

This design:

* preserves the mixture proposal and importance weighting structure of ABC-PMC
* reconstructs the active population and tolerance deterministically from evaluated history
* encodes the tolerance schedule in `Individual.tolerance` with a monotone min guarantee
* supports asynchronous HPC execution with no synchronization barriers
* integrates cleanly into the Propulate propagator model

The result is a **steady-state ABC-SMC-inspired algorithm suitable for large-scale simulator-based inference on heterogeneous computing environments**.
