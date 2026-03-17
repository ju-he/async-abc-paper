# Paper Design and Experimental Plan for Asynchronous Steady-State ABC-SMC

## 1. Paper Objective

The goal of the paper is to introduce and evaluate an **asynchronous steady-state ABC-SMC algorithm** designed for **HPC environments and heterogeneous simulator workloads**.

The paper must demonstrate three main points:

1. **Statistical validity** – the method produces posterior estimates comparable to established ABC algorithms.
2. **Computational advantages** – asynchronous steady-state execution improves resource utilization.
3. **Practical usability** – the algorithm works on realistic simulator-based models.

The most important comparison baseline will be **pyABC**, a widely used distributed ABC-SMC framework.

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

1. A **steady-state asynchronous ABC-SMC algorithm**
2. A **history-reconstructed particle archive**
3. Integration into **Propulate**
4. Empirical evaluation against existing frameworks including pyABC

---

## 2.2 Background

### Approximate Bayesian Computation

Introduce ABC inference:

$$
\pi(\theta|y) \propto \pi(\theta)L_\epsilon(\theta)
$$

where (L_\epsilon(\theta)) is the likelihood approximation induced by discrepancy threshold (\epsilon).

Explain:

* simulator-based models
* discrepancy metrics
* tolerance schedules

---

### Sequential ABC Algorithms

Discuss:

* ABC rejection
* ABC-SMC
* ABC-PMC

Focus on:

* mixture proposal distributions
* importance weights
* tolerance annealing

Emphasize the **generation-based synchronization constraint**.

---

### Existing Parallel ABC Systems

Discuss distributed ABC frameworks such as:

* **pyABC**

Explain that existing frameworks parallelize simulation but still rely on **population-level synchronization**.

---

# 3. Method

Describe the **steady-state ABC-SMC approach**.

Sections should include:

### History-based state reconstruction

Define evaluated history:

$$
\mathcal{H}*n = {(\theta_i,\rho_i,w_i)}*{i=1}^n
$$

Define reconstructed archive:

$$
A_n = A(\mathcal{H}_n, \epsilon_n)
$$

---

### Proposal mixture

$$
q_n(\theta) =
\sum_{j=1}^{k} W_j K(\theta|\theta_j)
$$

---

### Update step

Algorithm summary:

1. compute tolerance
2. reconstruct archive
3. construct proposal
4. sample parent
5. perturb and evaluate

---

### Differences from ABC-SMC

| Property        | ABC-SMC    | Steady-state  |
| --------------- | ---------- | ------------- |
| update style    | generation | event-driven  |
| synchronization | required   | none          |
| archive         | explicit   | reconstructed |

---

# 4. Implementation

Discuss integration into Propulate.

Important aspects:

* propagator interface
* stateless reconstruction
* MPI / distributed execution

---

# 5. Experiments

The experimental section should evaluate both:

1. **statistical accuracy**
2. **computational performance**

The full experimental suite consists of:

* Benchmark model evaluations (Gaussian mean, g-and-k, Lotka-Volterra, Cellular Potts)
* Simulation-based calibration (SBC) for posterior validity
* Runtime heterogeneity experiment (stochastic runtime noise)
* Straggler tolerance experiment (persistent slow-worker fault mode)
* Scaling experiments (1–256 cores)
* Sensitivity / hyperparameter analysis (archive size, perturbation scale, tolerance schedule, initial tolerance)

---

# 6. Benchmark Models

We will use four benchmark problems spanning increasing complexity.

## 6.1 Gaussian Mean Inference

A simple sanity-check model:

$$
y_i \sim \mathcal{N}(\mu, \sigma^2)
$$

Goal:

* verify posterior correctness
* compare convergence behavior

Advantages:

* analytic posterior available
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

Evaluation:

* posterior recovery
* number of simulations required

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

# 7. Baseline Methods

We will compare against:

### Rejection ABC

Baseline likelihood-free method.

Used only for small problems.

---

### ABC-SMC

Classical population algorithm.

---

### Distributed ABC-SMC

Using **pyABC**.

This provides a strong baseline for distributed ABC.

---

# 8. Statistical Evaluation

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

Track convergence curves: Wasserstein distance at fixed checkpoints in simulation count and wall time, comparing async and sync methods.

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

Produce rank histograms and empirical coverage tables at levels 0.5, 0.8, 0.9, 0.95.

---

# 9. Computational Performance

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

# 10. Runtime Heterogeneity Experiments

Two complementary experiments characterize the advantage of asynchrony under different failure modes.

### 10.1 Stochastic Runtime Heterogeneity

Artificially introduce runtime variability via LogNormal noise:

$$
t(\theta) \sim \text{LogNormal}(\mu(\theta), \sigma)
$$

Then compare:

* synchronous ABC-SMC
* asynchronous steady-state ABC

Expected result: Async method maintains high utilization while sync method idles at generation barriers.

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

# 11. Scaling Experiments

Run experiments on increasing numbers of cores:

```
1
8
32
128
256
```

Evaluate:

* parallel speedup
* scaling efficiency

Plot:

```
efficiency vs cores
```

---

# 12. Sensitivity Analysis

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

# 13. Ablation Study

Remove components of the algorithm to test importance.

Examples:

* fixed proposal covariance
* fixed tolerance
* no archive truncation

Measure degradation.

---

# 14. Visualization

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

Horizontal bar chart with one row per worker; colored blocks show individual simulation intervals. Requires per-simulation `sim_start_time` and `sim_end_time` fields. Visualizes generation barriers for sync methods and continuous utilization for async.

---

### Posterior quality vs. wall-clock time

Wasserstein distance curves per method over wall time. Shows convergence speed differences between async and sync algorithms.

---

### Tolerance trajectory

ε over wall-clock time (log scale), sync vs. async overlaid. Shows how each method tightens the tolerance schedule.

---

### SBC rank histograms

Uniform rank histogram indicates correct posterior calibration. Shown per method and per parameter.

---

### Straggler throughput curves

Simulation throughput vs. slowdown factor, comparing async and sync methods.

---

### Scaling curves

Efficiency vs number of cores.

---

# 15. Discussion

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

# 16. Limitations

Important to mention:

* theoretical guarantees weaker than classical ABC-SMC
* dependence on archive reconstruction rule
* possible runtime-induced bias

---

# 17. Conclusion

Summarize:

* asynchronous steady-state ABC-SMC removes generation barriers
* statistical accuracy comparable to ABC-SMC
* improved HPC utilization

---

# 18. Expected Contributions

The paper contributes:

1. A new asynchronous ABC algorithm
2. Integration into Propulate
3. Benchmark comparison with **pyABC**
4. Demonstration on realistic models including **cellsinsilico_nastjapy**
5. Empirical posterior calibration validation via SBC
6. Characterization of async advantages under both stochastic and persistent runtime heterogeneity
7. A comprehensive sensitivity analysis including initial tolerance as a key hyperparameter

The results should show that asynchronous steady-state ABC is a promising approach for **large-scale simulator-based inference on modern HPC systems**, with particular strength in environments with heterogeneous or unreliable worker performance.

