# Paper Design and Experimental Plan for Asynchronous Steady-State ABC-SMC

## 1. Paper Objective

The goal of the paper is to introduce and evaluate an **asynchronous steady-state ABC-SMC algorithm** designed for **HPC environments and heterogeneous simulator workloads**.

The paper must demonstrate three main points:

1. **Statistical validity** – the method produces posterior estimates comparable to established ABC algorithms.
2. **Computational advantages** – asynchronous steady-state execution achieves better posterior quality per wall-clock budget and better utilization under heterogeneous workloads.
3. **Practical usability** – the algorithm works on realistic simulator-based models.

The most important comparison baseline will be **pyABC**, used in two roles:

* `abc_smc_baseline` as the main synchronous pyABC-based comparator for fixed-walltime HPC experiments
* `pyabc_smc` as the external-framework reference for benchmark-validity comparisons

---

# 2. Proposed Paper Structure

## 2.1 Introduction

The introduction should:

* motivate likelihood-free inference
* explain the role of ABC
* describe scaling limitations of synchronous ABC-SMC

Key points to highlight:

* simulator runtimes often vary strongly across parameters
* generation barriers cause idle compute nodes
* asynchronous algorithms eliminate synchronization

Contributions:

1. A **generation-free, single-arrival-driven ABC algorithm** that combines
   smooth-kernel ABC (Wilkinson 2013) with streaming Adaptive Multiple
   Importance Sampling (AMIS, Cornuet et al. 2012). To our knowledge this
   is the first ABC algorithm whose proposal mixture updates after every
   evaluated particle rather than at generation/stage boundaries.
2. A **history-reconstructed particle archive** that makes the algorithm
   stateless (a pure function of the evaluated history) and crash-recoverable.
3. Integration into **Propulate**, exploiting its barrier-free island model
   for true asynchronous execution on HPC.
4. A **consistency + CLT** for the algorithm as a corollary of the AMIS
   theorem under the Wilkinson smooth-likelihood framework, with explicit
   conditions on the proposal sequence and bandwidth schedule.
5. **Empirical evaluation against pyABC** under matched-kernel
   (*apples-to-apples*) settings: pyABC's `UniformAcceptor` is replaced by
   a probabilistic-rejection acceptor using the same K_ε(ρ) as the
   propulate side, isolating the synchronisation regime as the only
   methodological difference between the two baselines.

---

## 2.2 Background

### Approximate Bayesian Computation

Introduce ABC inference:

$$
\pi_\epsilon(\theta|y) \propto \pi(\theta)\, L_\epsilon(\theta),
\qquad
L_\epsilon(\theta) = \mathbb{E}_{\rho \sim p(\rho|\theta)}\bigl[K_\epsilon(\rho)\bigr]
$$

where $\rho$ is the discrepancy between simulated and observed data and
$K_\epsilon$ is the ABC likelihood kernel. Classical ABC uses the hard
indicator $K_\epsilon(\rho) = \mathbf{1}[\rho < \epsilon]$; smooth-kernel ABC
(Wilkinson 2013; Fearnhead & Prangle 2012) replaces this with a continuous
kernel (Gaussian, Epanechnikov) which removes the discontinuity in the
likelihood approximation and admits standard SMC-sampler theory.

Explain:

* simulator-based models
* discrepancy metrics
* hard vs. smooth likelihood kernels
* tolerance / bandwidth schedules

---

### Sequential ABC Algorithms

Discuss the classical generation-staged family:

* **ABC rejection** — embarrassingly parallel but inefficient.
* **ABC-SMC** (Toni et al. 2009; Sisson et al. 2007) — generation-staged
  sequential targeting with tolerance annealing.
* **ABC-PMC** (Beaumont et al. 2009) — Population Monte Carlo with mixture
  proposals built from the previous generation.
* **Adaptive variants**: Del Moral, Doucet & Jasra 2012 (adaptive ε from
  ESS), Drovandi & Pettitt 2011 (acceptance-rate adaptation), Lenormand
  et al. 2013 (APMC, quantile-based adaptation).

Focus on:

* mixture proposal distributions
* importance weights against the previous-generation proposal
* tolerance annealing schedules

Emphasize the **generation-based synchronization constraint**: importance
weights at generation $t$ are computed against a single proposal $q_{t-1}$,
which requires the algorithm to wait for the full generation-$t$ population
before constructing $q_t$.

---

### Adaptive Multiple Importance Sampling

Adaptive Multiple Importance Sampling (AMIS; Cornuet, Marin, Mira & Robert
2012) generalises generation-staged importance sampling: at stage $t$ the
proposal $q_t$ is adapted from past particles, and crucially **all past
particles are re-weighted** against the cumulative proposal mixture

$$
\bar q_n(\theta) = \frac{1}{n}\sum_{j=1}^{n} q_{\tau_j}(\theta)
$$

(the **balance heuristic** of Veach 1997 / Owen & Zhou 2000) rather than
against the proposal under which each was originally drawn. AMIS is consistent
with a CLT under regularity conditions; our paper takes the streaming limit
of this scheme (stage size 1, single-arrival-driven adaptation).

---

### Asynchronous Sequential Monte Carlo

Off-barrier SMC has been studied for state-space models:

* **Particle cascade** (Paige & Wood 2014, NeurIPS) eliminates barrier
  synchronisation in particle filtering using local-decision descendant
  counts; produces an unbiased marginal-likelihood estimator.
* **Anytime Monte Carlo** (Murray, Lee & Jacob 2016) generalises this to a
  broader class of SMC samplers.

None of this prior work targets ABC specifically, and none combines
asynchronous execution with AMIS-style cumulative-mixture reweighting.

---

### Existing Parallel ABC Systems

Distributed ABC frameworks include:

* **pyABC** (Klinger, Rickert & Hasenauer 2018; Schälte et al. 2022) — the
  reference distributed ABC-SMC framework, with two parallelisation
  strategies. Static Scheduling minimises total compute; Dynamic Scheduling
  (DYN) minimises wall-time by sampling on all available hardware until $n$
  particles are accepted *within a generation*, then discarding the
  remaining $m - n$ accepted particles to avoid simulation-time bias. pyABC
  still has a **generation barrier**: workers stall while waiting for the
  in-flight simulations to drain at the boundary.
* **ABCpy** (Dutta et al. 2017) — generic parallel-ABC framework targeting
  HPC.
* **jakeret/abcpmc** — a Python ABC-PMC implementation following Beaumont
  et al. 2009 with `multiprocessing`/MPI; synchronous.

Existing frameworks parallelise simulation but still rely on
**population-level synchronisation**. Our contribution removes the
generation barrier entirely; the trade-off is that the standard generational
SMC convergence theory no longer applies, motivating §4.

---

# 3. Method

Describe the **generation-free, single-arrival-driven ABC algorithm**
combining smooth-kernel ABC with streaming AMIS reweighting.

### History-based state reconstruction

Define evaluated history at call $n$:

$$
\mathcal{H}_n = \bigl\{(\theta_i,\rho_i,\tau_i,w_i)\bigr\}_{i=1}^{n}
$$

where $\theta_i$ is the parameter, $\rho_i$ the discrepancy, $\tau_i$ the
proposal-time index, and $w_i$ the stored core importance weight $\pi/q_{\tau_i}$
(the kernel factor $K_\epsilon(\rho_i)$ is applied separately at use time).

Define the reconstructed archive as the top-$k$ by lowest $\rho$:

$$
A_n = \mathrm{Top}_k\bigl(\{\theta_i : \rho_i \in \text{history}\},\ \text{order by } \rho_i\bigr)
$$

The bandwidth $\epsilon_n$ is reconstructed as the running minimum of
$\{\tau_i\}$ stamped on each evaluated particle, with a propagator-side
tightness floor that survives island migration.

---

### Smooth-kernel proposal mixture

The proposal mixture at call $n$ uses kernel-weighted archive members:

$$
q_n(\theta) = \sum_{j \in A_n} \tilde W_j^{(n)}\, K_\Sigma(\theta - \theta_j),
\qquad
\tilde W_j^{(n)} \propto w_j \cdot K_{\epsilon_n}(\rho_j)
$$

with normalisation $\sum_j \tilde W_j^{(n)} = 1$. The perturbation kernel
$K_\Sigma$ is a Gaussian with covariance $\Sigma_n = s\cdot \widehat{\text{Cov}}_{\tilde W}(A_n)$
estimated from the kernel-weighted archive (the smoothing factor $s$ is the
``perturbation_scale`` hyperparameter). The ABC likelihood kernel
$K_{\epsilon_n}$ is one of:

* **Hard** (classical): $\mathbf{1}[\rho < \epsilon]$.
* **Gaussian** (default): $\exp(-\rho^2 / 2\epsilon^2)$.
* **Epanechnikov**: $\max(0,\, 1 - \rho^2/\epsilon^2)$.

Smooth kernels remove the prior-vs-archive phase discontinuity present in
classical ABC-PMC: every evaluated particle contributes continuously
through its kernel weight.

---

### Streaming AMIS importance weight

A newly proposed particle $\theta^\star$ from $q_{n}$ receives an importance
weight under the *balance heuristic* over a ring buffer of past proposals:

$$
w^\star = \frac{\pi(\theta^\star)}{\bar q_n(\theta^\star)},
\qquad
\bar q_n(\theta^\star) = \frac{1}{1+|\mathcal{S}|}\Bigl[q_n(\theta^\star) + \sum_{s \in \mathcal{S}} q_s(\theta^\star)\Bigr]
$$

where $\mathcal{S}$ is the AMIS snapshot buffer (sliding window of past
proposals, size $S$, sampled every ``amis_interval`` calls). Particles
remain coherent across the moving archive: an importance weight assigned
when a particle was proposed is corrected by the cumulative-mixture
denominator at every subsequent use.

Setting $|\mathcal{S}| = 0$ recovers the single-current-proposal weighting
(equivalent to legacy ABC-PMC); the default is $|\mathcal{S}| = 20$.

---

### Update step

Each invocation of the propagator (one new arrival in the asynchronous
HPC stream):

1. Reconstruct $\epsilon_n$ from history (running minimum + tightness floor).
2. If $|\mathcal{H}_n| < k$: emit a uniform prior draw (bootstrap phase).
3. Otherwise: call the bandwidth scheduler to propose a tighter $\epsilon$;
   apply the monotone-decrease guarantee.
4. Select archive $A_n$: top-$k$ by lowest $\rho$.
5. Compute effective mixture weights $\tilde W^{(n)}$ in log-space via the
   ABC kernel.
6. Build perturbation $\Sigma_n$, factor once via Cholesky.
7. Sample $\theta^\star$ by perturbing a $\tilde W^{(n)}$-weighted parent
   (reject-resample inside the box; fall back to uniform prior draw on
   exhaustion to preserve the truncated-density semantics).
8. Compute $w^\star$ via the AMIS denominator.
9. Append $\theta^\star$ to history; periodically snapshot the current
   proposal into $\mathcal{S}$.

---

### Differences from ABC-SMC

| Property            | Classical ABC-SMC      | Our algorithm                             |
| ------------------- | ---------------------- | ----------------------------------------- |
| Update style        | generation             | event-driven (single-arrival)             |
| Synchronisation     | required               | none                                      |
| Archive             | explicit, generational | reconstructed, sliding top-$k$            |
| ABC likelihood      | hard threshold         | smooth kernel (Gaussian / Epanechnikov)   |
| Importance weight   | against $q_{t-1}$ only | balance heuristic over snapshot buffer    |
| Statelessness       | no                     | yes (pure function of history)            |
| Generation barrier  | yes                    | no (true asynchronous execution)          |

---

# 4. Theoretical Analysis

We state consistency and a CLT for the proposed algorithm as a corollary of
two established results: the AMIS consistency theorem (Cornuet, Marin, Mira
& Robert 2012) and the smooth-likelihood ABC framework (Wilkinson 2013;
Fearnhead & Prangle 2012). The streaming regime (stage size 1,
single-arrival adaptation) is a degenerate case of AMIS stages.

## 4.1 Setup

Let $\pi(\theta)$ be the prior on the parameter $\theta \in \Theta \subset \mathbb{R}^d$,
let $p(\rho \mid \theta)$ be the simulator-induced discrepancy distribution,
and let $K_\epsilon : \mathbb{R}_{\geq 0} \to [0, 1]$ be a normalised ABC
kernel ($K_\epsilon(0) = 1$, $K_\epsilon(\rho) \to 0$ as $\rho/\epsilon \to \infty$).
The **smooth-ABC posterior** at bandwidth $\epsilon$ is

$$
\pi_\epsilon(\theta) \propto \pi(\theta)\, \mathbb{E}_{\rho \sim p(\rho \mid \theta)}[K_\epsilon(\rho)].
$$

The algorithm of §3 produces a stream of particles $\{\theta_i\}_{i \geq 1}$
where $\theta_i$ is drawn from proposal $q_{\tau_i}$ (the archive mixture at
proposal time $\tau_i$), with discrepancy $\rho_i$ measured under
$p(\rho \mid \theta_i)$. Under the balance heuristic over the AMIS snapshot
buffer of size $S$, the importance weight on particle $i$ at call $n$ is

$$
w_i^{\mathrm{bal}}(n) = \frac{\pi(\theta_i)\, K_{\epsilon_n}(\rho_i)}{\bar q_n(\theta_i)},
\qquad
\bar q_n(\theta) = \frac{1}{|\mathcal{S}_n|+1}\Bigl[q_n(\theta) + \sum_{s \in \mathcal{S}_n} q_s(\theta)\Bigr].
$$

The empirical posterior estimator is

$$
\widehat\pi_n(\theta) = \frac{\sum_{i=1}^{n} w_i^{\mathrm{bal}}(n)\, \delta_{\theta_i}(\theta)}{\sum_{i=1}^{n} w_i^{\mathrm{bal}}(n)}.
$$

## 4.2 Conditions

**(C1, bounded proposal-density ratio).** There exist $c, C > 0$ such that
for every proposal $q_\tau$ generated by the algorithm and every
$\theta$ in the support of $\pi$,

$$
c \leq \frac{q_\tau(\theta)}{\pi(\theta)} \leq C.
$$

This rules out proposals that concentrate arbitrarily far from $\pi$.

**(C2, kernel regularity).** The kernel $K_\epsilon$ is non-negative,
integrable in $\rho$, normalised to $K_\epsilon(0) = 1$, and Lipschitz in
$\epsilon$ for every fixed $\rho$.

**(C3, bandwidth schedule).** The bandwidth sequence $\{\epsilon_n\}$ is
monotone non-increasing, satisfies $\epsilon_n - \epsilon_{n+1} \to 0$, and
is bounded below by some $\epsilon_\infty > 0$.

**(C4, snapshot approximation).** The AMIS snapshot mixture $\bar q_n$
approximates the cumulative empirical proposal mixture uniformly: as
$n \to \infty$,

$$
\sup_{\theta \in \Theta}\Bigl| \bar q_n(\theta) - \tfrac{1}{n}\sum_{j=1}^{n} q_{\tau_j}(\theta) \Bigr| \to 0.
$$

This is automatic when the snapshot interval is finite and the proposal
sequence is uniformly continuous in time.

## 4.3 Results

**Theorem 1 (Consistency).** *Under (C1)–(C4), for every continuous
$\pi_{\epsilon_\infty}$-integrable function $f$,*

$$
\int f\, d\widehat\pi_n \xrightarrow{a.s.} \int f\, d\pi_{\epsilon_\infty}
\qquad \text{as } n \to \infty.
$$

*Proof sketch.* The numerator and denominator of $\widehat\pi_n[f]$ are
both sample averages of $w_i^{\mathrm{bal}}(n) f(\theta_i)$ and
$w_i^{\mathrm{bal}}(n)$ respectively over $\theta_i \sim q_{\tau_i}$. Under
(C1)–(C2) the weights have bounded conditional variance; under (C3) the
sequence of targets $\pi_{\epsilon_n}$ converges weakly to
$\pi_{\epsilon_\infty}$; under (C4) the AMIS denominator converges to the
true cumulative proposal mixture. The result then follows from the AMIS
consistency theorem (Cornuet et al. 2012, Theorem 1) applied to the
streaming limit, with the smooth ABC likelihood as the target
(Wilkinson 2013, §2). The Lipschitz condition in (C2) is what makes the
non-stationary target $\{\pi_{\epsilon_n}\}$ tractable: it bounds the
incremental discrepancy between $\pi_{\epsilon_n}$ and $\pi_{\epsilon_{n+1}}$
uniformly in $\theta$.

**Theorem 2 (CLT).** *Under (C1)–(C4) and a finite-second-moment condition
on $w_i^{\mathrm{bal}} f(\theta_i)$,*

$$
\sqrt{n}\Bigl(\int f\, d\widehat\pi_n - \int f\, d\pi_{\epsilon_\infty}\Bigr) \xrightarrow{d} \mathcal{N}(0, \sigma_f^2)
$$

*where $\sigma_f^2$ is the AMIS asymptotic variance (Cornuet et al. 2012,
Theorem 2) evaluated at the streaming limit.*

The CLT gives confidence intervals for posterior expectations — a property
classical generation-staged ABC-PMC does not provide without additional
machinery.

## 4.4 Discussion and limitations

* **(C1) is the strongest condition.** It requires the proposal sequence
  to remain within a bounded density ratio of $\pi$. For an adaptive
  PMC-style archive this holds empirically (the kernel mixture is
  supported on a slowly-moving region of $\Theta$ that always contains the
  posterior mode), but we do not formally guarantee it for the *specific*
  archive evolution rule of §3. A weaker integrable-ratio condition would
  suffice for consistency but break the CLT.
* **$\epsilon_\infty > 0$ in (C3) is a bandwidth floor**, not a tolerance
  floor in the classical sense. The theorem describes the *smooth-ABC
  posterior at the limit bandwidth*, not the exact posterior at
  $\epsilon = 0$. Extending to $\epsilon_\infty = 0$ requires standard
  ABC-SMC-sampler arguments (Del Moral, Doucet & Jasra 2012) which are
  orthogonal to the asynchronicity claim and which we leave to future
  work.
* **Finite-sample unbiasedness in the Paige & Wood 2014 sense is open.**
  AMIS is consistent but not unbiased for finite $n$ because the
  proposals depend on past particles (the adaptation breaks the
  classical IS unbiasedness argument). A Paige-Wood-style local-decision
  rule that recovers finite-sample unbiasedness for ABC is sketched in
  §16 as future work.
* **The snapshot buffer is finite** in implementation ($S = 20$ by
  default). (C4) requires the snapshot interval and buffer size to grow
  appropriately with $n$; in practice the bound is empirically tight at
  fixed $S = 20$ across all benchmarks we test, but adaptive snapshot
  sizing is a natural extension.

This consistency + CLT package is what makes the proposed algorithm a
publishable methodology rather than an engineering construction: it places
the asynchronous, generation-free ABC algorithm inside the AMIS family with
provable guarantees, distinguishing it from prior work that relies on
generation-based SMC theory which does not transfer to the asynchronous
setting.

---

# 5. Implementation

Discuss integration into Propulate.

Important aspects:

* propagator interface
* stateless reconstruction
* AMIS snapshot ring buffer
* log-space arithmetic for smooth-kernel weights
* MPI / distributed execution
* apples-to-apples pyABC acceptor (matched K_ε(ρ))

---

# 6. Experiments

The experimental section should evaluate both:

1. **statistical accuracy**
2. **computational performance**

The suite should be presented in two groups.

Validity evidence:

* Benchmark posterior recovery (Gaussian mean, g-and-k, Lotka-Volterra, Cellular Potts)
* Gaussian sanity check against the analytic posterior target
* Simulation-based calibration (SBC)

HPC performance evidence:

* Runtime heterogeneity experiment (stochastic runtime noise)
* Straggler robustness experiment (persistent slow-worker fault mode)
* Scaling experiments under fixed wall-clock budgets (1–256 cores)

Method-analysis / appendix:

* Sensitivity / hyperparameter analysis (archive size, perturbation scale, tolerance schedule, initial tolerance)
* Ablation analysis

For the walltime-limited HPC experiments, the synchronous baseline uses fixed populations / generations because this yields more interpretable comparisons than contrasting methods that stop according to different epsilon rules.

---

# 7. Benchmark Models

We will use four benchmark problems spanning increasing complexity.

## 6.1 Gaussian Mean Inference

A simple sanity-check model:

$$
y_i \sim \mathcal{N}(\mu, \sigma^2)
$$

Goal:

* verify posterior correctness
* compare posterior recovery and quality-vs-time behavior

Advantages:

* analytic posterior available (under Uniform prior: observed mean clipped to prior bounds)
* easy visualization

---

## 6.2 g-and-k Distribution

A classical ABC benchmark with intractable likelihood.

Properties:

* heavy tails
* skewness
* nonlinear parameter effects

This benchmark is widely used in ABC literature.

Metrics:

* posterior mean error
* Wasserstein distance

---

## 6.3 Lotka-Volterra System

Classic stochastic population dynamics model.

Parameters govern:

* prey growth
* predator interaction
* predator mortality

This model is frequently used in ABC-SMC studies.

Implementation notes:

* Summary statistics are normalized by observed values by default (`normalize_stats=True`), making the distance unitless and balanced across dimensions
* Observed trajectory generation retries on extinction (up to `max_extinction_retries` attempts with incrementing seed)

Evaluation:

* posterior recovery
* quality-vs-time behavior under fixed wall-clock budgets

---

## 6.4 Cellular Potts Model

We will use:

* **cellsinsilico_nastjapy**

This model simulates:

* cell adhesion
* cell migration
* tissue organization

Advantages:

* realistic simulation workloads
* heterogeneous runtimes
* biologically meaningful inference problem

This benchmark is ideal for demonstrating **HPC benefits**.

---

# 8. Baseline Methods

We will compare against:

### Rejection ABC

Baseline likelihood-free method. Supports `max_wall_time_s` for wall-time-limited stopping (consistent with all other methods).

Used only for small problems.

---

### ABC-SMC

Classical population algorithm.

---

### Distributed ABC-SMC

Using **pyABC**.

This provides a strong baseline for distributed ABC.

---

# 9. Statistical Evaluation

Evaluate posterior accuracy.

Metrics:

### Posterior mean error

$$
||\hat{\theta} - \theta^*||
$$

---

### Wasserstein distance

Compare posterior samples using sliced Wasserstein distance for multi-parameter posteriors (POT library, `n_projections=50`); exact 1D Wasserstein for single-parameter cases.

---

### Wasserstein vs. wall-clock time

Track convergence curves: Wasserstein distance (W1-to-point-mass, i.e. mean absolute deviation from truth in 1D; sliced Wasserstein via POT for multi-parameter cases) at fixed checkpoints in simulation count and wall time, comparing async and sync methods.

Checkpoint granularity is equalized across methods via `checkpoint_strategy="time_uniform"` in `posterior_quality_curve()`, which resamples both async and sync records onto a shared evenly-spaced time grid using LOCF (last-observation-carried-forward). This ensures fair comparison between async methods (which produce fine-grained per-event checkpoints) and sync methods (which produce coarse per-generation checkpoints).

---

### Effective Sample Size (ESS)

$$
\text{ESS} = \frac{(\sum w_i)^2}{\sum w_i^2}
$$

Track ESS over time to measure particle diversity.

---

### Credible interval coverage (SBC)

Empirical calibration via simulation-based calibration (SBC):

* Draw θ* from prior
* Run inference given simulated data
* Check whether θ* falls within α-credible intervals with frequency α

Produce rank histograms and empirical coverage tables at levels 0.5, 0.8, 0.9, 0.95. In the current implementation, empirical coverage is the paper-facing SBC summary and includes Wilson-style confidence intervals; rank histograms are retained mainly as diagnostics.

---

# 10. Computational Performance

Measure HPC efficiency.

Metrics:

### Wall-clock time

Time to reach fixed posterior error.

---

### Simulation throughput

Simulations per second.

---

### CPU utilization

Fraction of time workers are active.

---

### Idle worker fraction

Measure synchronization overhead.

---

# 11. Runtime Heterogeneity Experiments

Two complementary experiments characterize the advantage of asynchrony under different failure modes.

### 10.1 Stochastic Runtime Heterogeneity

Artificially introduce runtime variability by wrapping the benchmark simulator with a LogNormal
sleep after each evaluation completes, modelling a slow simulator call:

$$
\text{delay} \sim \text{LogNormal}(\mu, \sigma)
$$

The median delay is controlled by `base_delay_s` in the config: `mu = log(base_delay_s)`.
The sweep parameter `sigma` controls the spread (coefficient of variation).

**Important implementation notes:**
- `mu` is constant (not parameter-dependent); delays are random per evaluation, not correlated
  with `θ`. This models hardware/scheduling jitter rather than stiff-parameter regions.
- Each replicate receives a unique delay seed `stable_seed(base_seed, replicate_idx, sigma)`
  to ensure statistical independence across replicates.
- The sleep is injected *after* the simulation call, so worker timing (from Propulate's
  `evaltime`/`evalperiod`) correctly spans the full busy period including the delay.
- In `--test` mode the sleep is skipped entirely.

Config fields:

```json
{
  "heterogeneity": {
    "distribution": "lognormal",
    "base_delay_s": 1.0,
    "sigma_levels": [0.0, 0.5, 1.0, 1.5, 2.0]
  }
}
```

Then compare:

* synchronous ABC-SMC (`abc_smc_baseline`)
* asynchronous steady-state ABC (`async_propulate_abc`)

Expected result: Async method maintains high utilization while sync method idles at generation
barriers. At high `sigma`, async completes the same simulation budget in substantially less
wall-clock time.

**Idle fraction measurement:**
Two complementary methods are used because the backends record timing differently:
- `worker_idle`: direct per-worker `sim_start_time`/`sim_end_time` span — used for
  `async_propulate_abc` (Propulate records per-evaluation `evaltime`/`evalperiod`).
- `barrier_overhead`: generation-level timing fraction — used for `abc_smc_baseline`
  (which only records generation-start/end, not per-worker intervals).

Both are shown with distinct labels in the idle fraction plots.

**Primary paper figures:**
1. `quality_by_sigma.pdf` — Wasserstein distance vs. wall-clock time, one panel per sigma
   level, async vs. sync overlaid. This is the headline result.
2. `idle_fraction_comparison.pdf` — utilization-loss fraction vs. sigma for each method.
3. `speedup_summary.csv` — median wall-clock span (and speedup ratio vs. `abc_smc_baseline`)
   per sigma level and method; for stating direct speedup claims in the paper.
4. `throughput_over_time.pdf` — simulations/s over time, faceted by sigma.
5. `worker_gantt.pdf` — diagnostic Gantt chart (per-worker timeline).

---

### 10.2 Straggler Tolerance Experiment

Model a persistent structural HPC failure: one worker permanently runs slowly.

Parameters:

* `straggler_rank`: index of the slow worker
* `base_sleep_s`: per-simulation sleep added to normal runtime
* `slowdown_factor`: sweep over {1×, 5×, 10×, 20×}

Expected result: Async ABC routes work to available workers and degrades gracefully; sync ABC-SMC blocks entire generations waiting for the straggler, causing super-linear wall-time degradation.

Metrics:

* throughput vs. slowdown factor
* Gantt chart at worst slowdown level

---

# 12. Scaling Experiments

Run experiments on increasing numbers of cores under fixed wall-clock budgets:

```
1
8
32
128
256
```

Evaluate:

* posterior quality at fixed budget
* attempts at fixed budget
* throughput vs. workers
* efficiency / utilization vs. workers

Core figure family:

```
quality at fixed budget
attempts at fixed budget
throughput vs workers
efficiency vs workers
worker utilization vs workers
```

---

# 13. Sensitivity Analysis

Test robustness to algorithm parameters. The sensitivity grid sweeps four dimensions:

### Archive size

```
k = 50, 100, 200
```

Evaluate effect on posterior accuracy.

---

### Perturbation scale

```
perturbation_scale = 0.4, 0.8, 1.5
```

Controls the bandwidth of the perturbation kernel.

---

### Tolerance scheduling

```
scheduler_type = acceptance_rate, quantile, geometric_decay
```

Compare three adaptive tolerance update strategies.

---

### Initial tolerance (`tol_init` multiplier)

```
tol_init_multiplier = 0.5×, 1×, 2×, 5×
```

Scales the base `tol_init`. This is often the most impactful hyperparameter and was added as a fourth sensitivity dimension.

---

Full grid: 3 × 3 × 3 × 4 = 108 variants; 5 replicates each = 540 runs.

Results are presented as faceted heatmaps (one panel per `tol_init` level).

---

# 14. Ablation Study

Remove components of the algorithm to test importance.

Examples:

* fixed proposal covariance
* fixed tolerance
* no archive truncation

Measure degradation.

---

# 15. Visualization

Recommended figures:

### Algorithm comparison diagram

```
synchronous vs asynchronous
```

---

### Archive evolution

Plot archive particles over time.

---

### Posterior comparison plots

Overlay posterior densities.

---

### Corner plots

Pairwise joint marginals for multi-parameter posteriors. Diagonal: marginal KDE. Off-diagonal: scatter with KDE contours. True parameter values overlaid as reference lines.

---

### Gantt / worker timeline

Horizontal bar chart with one row per worker; colored blocks show individual simulation intervals. Requires per-simulation `sim_start_time` and `sim_end_time` fields. In the current implementation this is treated as a diagnostic plot, not a paper-facing summary plot. For runtime heterogeneity, worker timelines are faceted to avoid method overplotting.

---

### Posterior quality vs. wall-clock time

Wasserstein distance curves per method over wall time. The paper-facing version is now a summary curve over replicates with pointwise 95% confidence bands where possible. Replicate-level traces are emitted separately as diagnostic plots. The convergence module uses `state_kind` labels (`archive_reconstruction`, `generation_population`, `accepted_prefix`) to distinguish the semantics of each method's posterior state at each checkpoint.

---

### Tolerance trajectory

ε over wall-clock time (log scale), sync vs. async overlaid. The paper-facing version is a summary over replicates, aggregated in log-space, with replicate-level variants retained separately for diagnostics.

---

### SBC rank histograms

Uniform rank histogram indicates correct posterior calibration. In the current plotting pipeline this remains useful diagnostically, while coverage tables are the main paper-facing SBC summary.

---

### Straggler throughput curves

Simulation throughput vs. slowdown factor, comparing async and sync methods. The implemented summary uses active simulation span as the canonical denominator and shows mean + 95% confidence intervals over replicates.

---

### Threshold summaries

Wall-clock time, posterior samples, or simulation attempts required to reach a target posterior quality. The paper-facing figures are summary points with confidence intervals. Crossings that occur before a minimum posterior size are ignored.

---

### Audit-controlled paper plots

Paper-facing quality and threshold plots are emitted only when the recorded data pass a benchmark audit. If required information is missing or the run is clearly pathological, the plot metadata records an explicit skip instead of silently producing a misleading figure.

---

### Lotka calibration diagnostic

Lotka-Volterra currently emits an auxiliary diagnostic that estimates fallback/extinction prevalence and recommends a more realistic `tol_init` from the non-extinction loss distribution. This is a data-quality diagnostic, not a paper figure.

---

### Scaling curves

Efficiency vs number of cores.

---

# 16. Discussion

Discuss:

### When asynchronous ABC helps

* heterogeneous simulators
* large clusters
* expensive simulations

---

### When synchronous methods suffice

* homogeneous runtimes
* small clusters

---

# 17. Limitations

We are explicit about what §4's consistency + CLT *does not* give us:

* **Finite-sample unbiasedness in the Paige & Wood 2014 sense is open.**
  AMIS-based estimators are consistent but not unbiased for finite $n$; a
  local-decision rule that recovers finite-sample unbiasedness for ABC is
  sketched as future work.
* **The bandwidth floor $\epsilon_\infty > 0$** in (C3) means the theorem
  characterises the smooth-ABC posterior at the limit bandwidth, not the
  exact posterior at $\epsilon = 0$. Extending requires Del Moral et al.
  2012 SMC-sampler arguments.
* **Condition (C1) (bounded proposal-density ratio)** is assumed for the
  specific archive evolution rule, not proved. Adaptive PMC-style archives
  satisfy this empirically; a formal proof is outside our scope.
* **Simulation-time bias.** Unlike pyABC's "discard latecomers" rule under
  DYN scheduling, our steady-state design keeps every accepted particle.
  Parameter regions with faster simulators may be over-represented in the
  archive. We measure this empirically in the runtime-heterogeneity
  experiment but do not correct for it algorithmically.
* **Snapshot buffer size is fixed** ($S = 20$). (C4) requires snapshot
  size and interval to grow appropriately with $n$ in the asymptotic limit;
  in practice we observe the bound is empirically tight at fixed $S$, but
  adaptive snapshot sizing is a natural extension.

---

# 18. Conclusion

Summarize:

* a generation-free, single-arrival-driven ABC algorithm combining
  smooth-kernel ABC with streaming AMIS reweighting;
* consistency and a CLT as a corollary of AMIS + Wilkinson smooth-ABC,
  under explicit conditions on the proposal sequence and bandwidth
  schedule;
* matched-kernel apples-to-apples evaluation against pyABC isolating
  synchronisation as the sole methodological difference;
* HPC utility on heterogeneous and straggler-prone workloads where the
  generation-barrier cost is substantial.

---

# 19. Expected Contributions

The paper contributes:

1. **A new generation-free ABC algorithm** composed of smooth-kernel ABC
   (Wilkinson 2013) and streaming AMIS reweighting (Cornuet et al. 2012)
   under a single-arrival adaptation regime — to our knowledge the first
   ABC algorithm whose proposal mixture updates after every evaluated
   particle rather than at stage boundaries.
2. **Consistency + CLT** as a corollary of AMIS under the smooth-ABC
   framework, with explicit conditions and an honest accounting of what
   the theorem does and does not give (§4, §17).
3. **Integration into Propulate** as a stateless propagator, with
   apples-to-apples plumbing through the pyABC baselines (matched kernel,
   matched bandwidth schedule, sole methodological difference is the
   synchronisation regime).
4. **Demonstration on realistic simulator-based models** including the
   `cellsInSilico` Cellular Potts model — the headline HPC use case.
5. **Empirical posterior calibration validation via SBC** on a benchmark
   with analytic posterior.
6. **Characterisation of async advantages** under both stochastic and
   persistent (straggler) runtime heterogeneity, with idle-fraction and
   wall-time-to-target-posterior summaries.
7. **Sensitivity and ablation** isolating the contribution of each A+D
   ingredient (hard vs. smooth kernel, with vs. without AMIS reweighting,
   kernel choice).

The results should show that asynchronous, generation-free ABC is a
promising approach for **large-scale simulator-based inference on modern HPC
systems**, with particular strength in environments with heterogeneous or
unreliable worker performance, and with a publishable theoretical backing
inherited from the AMIS / smooth-ABC literature.
