# Draft 1 — abstract and introduction (2026-09-21)

Numbers: tab_cpm_production (e716cb5), tab_matched_eps (c495e52), fig_predictor, tab_twin (C1).

## Abstract (≈240 words)

Sequential approximate Bayesian computation advances in generations, and a generation cannot end
before its slowest simulation. On a parallel machine that barrier idles every worker that finished
early, and its cost is the straggler factor of the workload: the expected maximum of W simulation
times over their mean. We introduce a generation-free ABC sampler that updates its proposal after
every arriving evaluation, weights every evaluated particle by a streaming form of adaptive multiple
importance sampling, and reports an estimator that is a pure function of the evaluated history —
recoverable by replay after a crash, and implemented in the barrier-free island model of Propulate.
We prove consistency of the reported estimator, verify its calibration by simulation-based
calibration, and measure four things. (i) The barrier costs what the straggler factor says:
predicted from the asynchronous run's timing alone, the cost matches a barrierized twin of our own
sampler to within 1% under an injected straggler and to within a few per cent on a Cellular Potts
tissue simulator at two sizes, across a range from 1.2x to 400x. (ii) Removing it converts into a
tighter tolerance: on the tissue simulator the asynchronous sampler completes 2.4x the simulations
of a matched synchronous baseline, is 1.5x more efficient per simulation, and reaches a 4.1x tighter
tolerance at equal wall clock — close to 40x tighter than a fairly budgeted rejection sampler.
(iii) The advantage turns on at a few milliseconds per simulation; below that, per-arrival
coordination dominates. (iv) The reported posterior recovers a two-parameter tissue posterior with
the truth covered in every replicate, provided the starting bandwidth is set from the
prior-predictive discrepancy scale; its effective sample size is bounded by a few multiples of the
archive size, which is the method's remaining limit.

## 1 Introduction (≈680 words)

Approximate Bayesian computation (ABC) is the standard route to Bayesian inference when a simulator
can be run but its likelihood cannot be evaluated \cite{beaumont2019abc}. Its sequential forms —
ABC-SMC and ABC-PMC \cite{sisson2007,toni2009abcsmc,beaumont2009adaptive,delmoral2012adaptive} —
move a population of particles through a sequence of shrinking tolerances, and every
implementation we know of does so in generations: the proposal and the tolerance are updated once
the whole population has been simulated. That is a synchronization barrier. On a parallel machine
with W workers, a generation ends when its slowest simulation does, and every worker that finished
earlier waits. The cost is easy to state: if simulation times are drawn from a distribution with
mean μ, a generation of W takes about E[max of W draws], so the barrier costs a factor
E[max_W]/μ — the straggler factor of the workload. It is 1 for a homogeneous simulator and grows
with W and with the spread of the runtime distribution. Simulators that are worth running on such
machines are rarely homogeneous: the cost of a cell-based tissue simulation, for instance, tracks
the number of cells it produces, which varies by orders of magnitude across a prior, and we measure
its runtime coefficient of variation rising from 0.05 to 0.26 as the simulated volume grows.

This paper removes the generation. We propose an ABC sampler in which the proposal mixture is
rebuilt after every arriving evaluation, from a sliding archive of the k best particles evaluated
so far; every evaluated particle is weighted against the mixture of proposals actually used — the
streaming limit of adaptive multiple importance sampling (AMIS) \cite{cornuet2012amis} — and the
estimator the method reports is a pure function of the evaluated history. Nothing has to be
synchronized between workers, so the sampler runs on the barrier-free island model of the
Propulate optimization engine \cite{taubert2023propulate}, and a crashed run is recovered by
replaying its log rather than by checkpointing sampler state. Look-ahead scheduling
\cite{alamoudi2024lookahead} attacks the same idle time by starting the next generation early on
free workers; it keeps the generation, and recovers the 10–50% that the boundary wastes. We remove
the boundary, and recover the factor.

Our contributions are:

1. **A generation-free, single-arrival-driven ABC sampler** combining smooth-kernel ABC with
   streaming AMIS reweighting, whose reported estimator is history-reconstructed and therefore
   replayable, with its implementation in Propulate (§3, §5).
2. **A predictive account of what the barrier costs.** We build a barrierized twin of our own
   sampler — identical in everything but a collective barrier before each proposal — and show that
   its slowdown is predicted from the asynchronous run's timing alone, to within 1% under an
   injected straggler and a few per cent on a real tissue simulator at two sizes, over 1.2x to 400x
   (§7.1). The same prediction lets a production run estimate its own gain from timing data,
   without a wasteful reference run.
3. **The measurement that the gain converts, and where it turns on.** Against a pyABC baseline
   matched on kernel and bandwidth schedule, the tolerance reached per simulation is at least the
   baseline's on every benchmark but the one-dimensional analytic one, so throughput multiplies
   through: 2.4x the simulations, 1.5x per simulation, 4.1x tighter at equal wall clock on the
   tissue simulator (§7.2). Both ratios rise monotonically with the cost of a simulation and cross
   one at a few milliseconds; below that the per-arrival coordination costs more than the barrier
   (§7.3).
4. **Consistency and calibration of the reported estimator, and its limits.** We prove consistency
   for the estimator the experiments report (§4; the proofs and a central limit theorem for
   damped-adaptation variants are in the supplement), verify calibration by simulation-based
   calibration in one and four dimensions and under multimodality, and recover a two-parameter
   Cellular Potts posterior with the truth covered in every replicate. Two settings govern what the
   estimator delivers: the starting bandwidth, which must be set from the prior-predictive
   discrepancy scale on an expensive simulator, and the archive size, which bounds the effective
   sample size at a few multiples of itself (§7.4).

Two limits are stated up front. On simulators cheaper than a few milliseconds the method loses to
a synchronous one, and two of our four benchmarks lie on that side of the line by design. And the
tissue benchmark is a calibration instrument, not a production run: its simulations cost seconds
where production runs cost hours, so its role is to establish the method against a known truth and
to show that its gain is predictable from timing, which is what a production run with an unknown
posterior can use.

The paper is organized as follows. §2 gives the ABC and importance-sampling background, §3 the
method, §4 its theory, §5 the implementation and the two instruments we compare against, §6 the
benchmarks and metrics, §7 the results by claim, and §8–§10 discussion, limitations and
conclusion.
