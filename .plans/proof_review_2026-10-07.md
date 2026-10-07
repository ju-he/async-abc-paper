# Review of the proofs in the TMLR manuscript

Date: 2026-10-07
Manuscript: `latex/tmlr/tmlr-article.tex` at commit `b75ed74` (branch `campaign-tooling`)
Scope: §4 (Theoretical Analysis), Appendix A (CLT and fidelity ratio), Appendix B (Proofs), Table 4 (assumption status)

All line numbers below refer to `latex/tmlr/tmlr-article.tex` at that commit.

---

## 0. Summary

Every proof in Appendix B was checked step by step. No proof is wrong. The
results hold as stated under the assumptions as stated. The review found:

| Class | Count | Examples |
|---|---|---|
| Substantive strengthening available | 4 | Assumption 4 is provable; 5(ii) implies the fidelity clause of 5(i); box claim is false; λ missing from the parameterization |
| Gaps or inconsistencies in individual proofs | 6 | undefined symbol at first use, lemma hypothesis narrower than its use, two definitions of the predictable bandwidth, a typo, small tidy-ups |
| Presentation of Proposition 3 and the floor factor | 4 | state the ζ₀ form first, exact TV identity, relocate the r_floor derivation, a design remark |

Section 1 records what was verified and found correct, so that the
later sections can be read as the complete list of what needs attention.
Sections 2 to 4 give each finding with the reasoning, the consequence for
the paper, and suggested wording where a rewrite is small. Section 5 is a
prioritized action list.

---

## 1. Verification log (what was checked and holds)

### 1.1 Lemma 1 (bounded weights), line 492

Every component of the denominator (eq. 7) is non-negative and the prior
component has weight λ_n ≥ δ, so q̄_n ≥ δ π pointwise. With K_ε ≤ 1 this gives
W_{i,n} ≤ 1/δ. Correct, and correctly noted to need no density-ratio or moment
condition. The lower bound q̄_n ≥ δ π_- > 0 is also what makes 1/q̄_n Lipschitz
in the parameters later.

### 1.2 Lemma 2 (uniform SLLN for bounded martingale differences), line 499

- Azuma–Hoeffding for differences bounded by 2B:
  P(|Σ_{i≤n} D_i| ≥ nt) ≤ 2 exp(−n²t² / (2 n (2B)²)) = 2 exp(−nt²/8B²). Correct.
- An η-net of a compact subset of ℝ^D has at most C_P η^{−D} points. Correct.
- The union bound over a fixed finite net gives a summable sequence in n,
  so Borel–Cantelli applies. Correct, and the parenthetical "the net is fixed"
  is the right thing to emphasize.
- |D^p_i − D^{p'}_i| ≤ 2Λη by Lipschitz continuity of ψ_p and of its
  conditional expectation. Correct.
- The countable intersection over t = 1/r, η = 1/j closes the argument. Correct.

### 1.3 Lemma 3 (conditional-mean identity), line 510

Tower property and Fubini; the integrand is bounded by ‖φ‖_∞/(δπ_-)^ℓ. Correct.
The warning that an F_∞-measurable denominator is not admissible is correct and
is exactly why the CLT centers on predictable proxies.

### 1.4 The parametric class (paragraph at line 524)

- Lipschitz constants of the kernel: Gaussian K_ε(ρ) = exp(−ρ²/2ε²) has
  ∂_ε K = K · ρ²/ε³; with x = ρ²/ε² this is (1/ε) x e^{−x/2}, maximized at
  x = 2, giving 2/(eε). Epanechnikov (1 − ρ²/ε²)_+ has ∂_ε K = 2ρ²/ε³ ≤ 2/ε on
  ρ < ε. Both constants in Assumption 1 and Table 4 are right.
- Truncated Gaussian components with means in a compact box and covariance
  in σ_-²I ⪯ Σ ⪯ σ_+²I have bounded parameter derivatives. Correct.
- The per-axis product normalizer is bounded below by
  Π_d (Φ(w_d/2σ_+) − 1/2) > 0. Correct (but see Finding 2.3 on what it is
  used for).
- Localization on P_{1/j} = [1/j, ε_0] × P_q with countable intersection over
  j, so that no deterministic bandwidth floor is required. Correct and well
  explained; this is a genuinely nice device.

### 1.5 Proof of Theorem 1 (consistency), line 530

- (1/n)Σ W_{i,n} h(θ_i) = (1/n)Σ ψ_{p̂_n}(ξ_i) with p̂_n = (ε_n, q̄_n). By the
  uniform SLLN this equals Γ̄_n(p̂_n) + o(1), where Γ̄_n is the deterministic
  function p ↦ (1/n)Σ_i E_{i−1} ψ_p(ξ_i). Lemma 3 at fixed p gives
  Γ̄_n(p) = ∫ π h L_ε q̄*_n / q_p. Correct, and the warning (line 544) that
  the middle term must not be read as E_{i−1}[ψ_{p̂_n}(ξ_i)] is exactly the
  right one.
- The split against ∫ π h L_{ε_∞} r:
  |∫ π h L_{ε_n} q̄*_n/q̄_n − ∫ π h L_{ε_∞} r|
  ≤ ‖h‖_∞ π_+ |Θ| [ R L_K(1/j) |ε_n − ε_∞| + ‖q̄*_n/q̄_n − r‖_∞ ].
  Uses |L_ε − L_{ε'}| ≤ L_K |ε − ε'| (Jensen through the expectation in ρ),
  L ≤ 1, and the eventual bound R on the ratio. Correct.
- Z_r ≥ r_- Z_{ε_∞} > 0 justifies the division. Correct.

### 1.6 Proof of Theorem 2 (CLT), line 551

- Reduction via Slutsky to S_n → N(0, v²) stably. Correct; Slutsky preserves
  stable convergence.
- Step 1: |S_n − n^{−1/2} Σ V_i| bounded by
  C √n (|ε_n − ε_∞| + ‖q̄_n − q̄_∞‖_∞) + C n^{−1/2} Σ_i (|ε_{i−1} − ε_∞| + ‖q̄_{i−1} − q̄_∞‖_∞),
  with C = ‖g‖_∞ π_+ max{L_K(ε_∞)/(δπ_-), (δπ_-)^{−2}}. The two pieces are the
  kernel difference (Lipschitz, constant L_K(ε_∞) valid because every ε lies
  in [ε_∞, ε_0] by monotonicity) and the reciprocal difference
  |1/q̄_n − 1/q̄_{i−1}| ≤ ‖q̄_n − q̄_{i−1}‖_∞/(δπ_-)². Correct.
- Step 2: c_i = E_{i−1} V_i = ∫ π g (L_{ε_{i−1}} q̃_i/q̄_{i−1} − L_{ε_∞} r).
  The bound |c_i| ≤ C'(|ε_{i−1} − ε_∞| + ‖q̃_i − q̃_∞‖_∞ + ‖q̄_{i−1} − q̄_∞‖_∞)
  needs no sup bound on q̃_i: the kernel term is controlled by
  ∫ q̃_i/q̄_{i−1} ≤ 1/(δπ_-) since q̃_i integrates to one, and the other two
  terms by |Θ|/(δπ_-) and ∫ q̃_∞ ≤ 1 times (δπ_-)^{−2}. The stated dependence
  of C' on ‖g‖_∞, π_±, δ, |Θ|, L_K(ε_∞) is right.
- Step 3(a): |V_i − c_i| ≤ 2π_+‖g‖_∞/(δπ_-) surely, so the conditional
  Lindeberg sum is identically zero for large n. Correct.
- Step 3(b): E_{i−1} V_i² = ∫ π² g² L^{(2)}_{ε_{i−1}} q̃_i / q̄²_{i−1}; the
  squared-kernel Lipschitz constant 2L_K(ε_∞) is right; Cesàro convergence of
  the remainder follows a fortiori from the √n clause. Correct.
- Hall and Heyde (1980), Cor. 3.1 with nested σ-fields (here the filtration
  does not depend on n) gives stable convergence to a mixed normal with
  F_∞-measurable variance. Correct citation and correct use.

### 1.7 Proof of Corollary 2 (confidence intervals), line 599

- Cross terms vanish because W ≤ 1/δ and π̂_n(f) → π^r(f). Correct.
- Uniform SLLN on P_{ε_∞} (deterministic under 5(ii), so no localization) for
  the squared class, then Lemma 3 with ℓ = 2, then the limits
  ε_n → ε_∞, ratio → r, q̄_n → q̄_∞ give
  ∫ π² g² L^{(2)}_{ε_∞} r / q̄_∞ = ∫ π² g² L^{(2)}_{ε_∞} q̃_∞ / q̄²_∞ = v². Correct.
- Stable convergence is equivalent to joint convergence with every
  F_∞-measurable variable, so the pair converges jointly and the continuous
  mapping theorem applies to the ratio. Correct, and the remark that marginal
  mixed-normal convergence would not suffice is right.

### 1.8 Proof of Proposition 3 (tilt bound), line 609

- ζ_0 = π(|r−1|) ≤ η_A π(A) + 1·π(A^c) ≤ η_A + π(A^c). Correct.
- π^r(f) − π(f) = [π(f(r−1)) − π(f) π(r−1)] / π(r); modulus ≤ 2‖f‖_∞ ζ_0/(1−ζ_0)
  using π(r) ≥ 1 − ζ_0. Halving for TV. Correct.
- r_floor algebra: (ν + (1−ν)u)/(δ + (1−δ)u) − 1 = (δ−ν)(u−1)/(δ + (1−δ)u).
  Correct. The branch bounds (u−1)/(δ+(1−δ)u) ≤ 1/(1−δ) for u ≥ 1 and
  (1−u)/(δ+(1−δ)u) ≤ 1/(δ+(1−δ)c) for c ≤ u < 1 are correct.

### 1.9 Other statements checked

- Gaussian kernel: K_ε² = K_{ε/√2}, since exp(−ρ²/ε²) = exp(−ρ²/(2(ε/√2)²)). Correct.
- L^{(2)}_ε ≥ L_ε² by Jensen; the gap is Var(K_ε(ρ) | θ). Correct.
- Rate bookkeeping in §A.1: with ‖q_i − q_∞‖ ~ i^{−b}, Σ_{i≤n} i^{−b} = O(n^{1−b}),
  so n^{−1/2} Σ = O(n^{1/2−b}) → 0 iff b > 1/2. Correct, and consistent with
  the "a = b − 1/2 > 0" parametrization in Step 2.

---

## 2. Substantive strengthening

### 2.1 Assumption 4 (adapted filtration) can be a lemma, not an assumption

**Where.** Assumption 4 (line 184), the Setting paragraph (lines 481–488),
Table 4 row "adapted filtration" (line 635), Limitations (line 382).

**Current text.** The paper asserts that under asynchrony a filtration large
enough to contain in-flight outcomes exists *with q̃_i still the conditional
law of the candidate under it*, calls this "the substantive clause", and
warns that "conditioning on the larger field need not leave the candidate's
law unchanged when completion times steer which rank proposes next."

**Claim.** Under an explicit black-box model of the simulator, this
assumption is provable, and q̃_i equals the emitted proposal density.

**The model (to be stated as an assumption replacing Assumption 4).**

(B1) For each call, given its parameter θ, the simulator returns a pair
(ρ, D) — discrepancy and wall-clock duration — from a law P(·|θ), and these
pairs are conditionally independent across calls given their parameters.

(B2) The sampler's own randomness (parent choice, perturbation, rejection
retries, redraws, bootstrap draws) is drawn fresh at each call,
independently of everything else.

(B3) Which worker proposes next, and at what wall-clock time, is a
deterministic function of earlier proposal times, durations, and worker
identities (ties broken by a fixed rule).

**Lemma (adapted filtration under the black-box model).** Index candidates
by proposal time. Let w_i be the worker that proposes candidate i and let
F_i := σ( (θ_j, ρ_j, D_j, w_j) : j ≤ i ). Then ξ_i = (θ_i, ρ_i) is
F_i-measurable, θ_i | F_{i−1} ~ q̃_i where q̃_i is the proposal density the
implementation emits on the prefix of the history that worker w_i has
received, and ρ_i | F_{i−1} ∨ σ(θ_i) ~ p(·|θ_i).

**Proof sketch.**
1. The proposal time T_i of candidate i is the earliest completion time
   among the busy workers, i.e. a function of (T_j, D_j, w_j)_{j<i}, hence
   F_{i−1}-measurable, by (B3) and induction. So is w_i, and so is the set
   A_i ⊂ {1,…,i−1} of candidates whose results have reached w_i by T_i
   (any message-delivery lag can be added to the filtration in the same way).
2. The emitted proposal density q_i^{em} is a function of the history
   restricted to A_i, so it is F_{i−1}-measurable.
3. By (B2), θ_i is drawn from q_i^{em} with randomness independent of
   F_{i−1}, so the conditional law of θ_i given F_{i−1} is q_i^{em}.
   Hence q̃_i = q_i^{em}.
4. By (B1), (ρ_i, D_i) given θ_i is independent of
   (θ_j, ρ_j, D_j, w_j)_{j<i}, which generate F_{i−1}. Hence
   ρ_i | F_{i−1} ∨ σ(θ_i) ~ p(·|θ_i). ∎

**Why the paper's worry does not bite.** Completion times do steer which
worker proposes next and which prefix it sees. But that steering changes
*which* proposal density is used, which is predictable, not the law of the
candidate *given* that density, which is fixed by fresh randomness. The two
were conflated at line 488.

**When it would fail.** (B1) fails if simulator outputs are coupled across
calls, e.g. shared random seeds or load-dependent numerics; (B2) fails if
the sampler's RNG stream is correlated with runtimes. Both are checkable
implementation properties, which is a better place to carry the risk than
an unverifiable abstract assumption.

**Consequences for the paper.**
- Table 4, row "adapted filtration": "assumed" → "holds under the
  black-box model (B1–B3); the model is an implementation property."
- Line 488: delete the sentence about conditioning on the larger field
  changing the law, replace by a pointer to the lemma.
- The remaining asynchrony effect is the one already routed through r:
  the post-hoc pass reconstructs snapshots from an arrival-ordered log, and
  the arrival-ordered prefix of length τ differs from the proposal-ordered
  prefix by at most W candidates (those in flight at the τ-th arrival).
  This bounds the discrepancy between the two reconstructions by a boundary
  of W particles per snapshot, which is worth stating.
- **Limitations (line 382), length bias.** At any deadline the arrived set is
  {1, …, n'} minus the at most W candidates in flight. So the
  "over-representation of parameter regions that simulate quickly" touches
  at most W of n particles directly, i.e. an O(W/n) effect on the estimator.
  The rest of the completion-time effect is in the proposal sequence q̃_i,
  which importance weighting corrects up to the fidelity ratio. This is a
  theoretical reason for the null result of the runtime-coupled study
  (Appendix F) and should be said there. The current wording concedes more
  than the theory requires.

### 2.2 Assumption 5(ii) implies the fidelity clause of 5(i)

**Where.** Assumption 5 (lines 185–193), the full statement of (ii)
(lines 407–416), Theorem 2 hypothesis (line 418).

**Observation.** Clause (ii) assumes uniform limits q̃_i → q̃_∞ and
q̄_n → q̄_∞ with r = q̃_∞/q̄_∞. Then q̄*_n = (1/n) Σ_{i≤n} q̃_i → q̃_∞
uniformly (Cesàro mean of a uniformly convergent sequence), and since
q̄_∞ ≥ δπ_- > 0,

  ‖q̄*_n/q̄_n − q̃_∞/q̄_∞‖_∞
  ≤ ‖q̄*_n − q̃_∞‖_∞/(δπ_-) + ‖q̃_∞‖_∞ ‖q̄_n − q̄_∞‖_∞/(δπ_-)² → 0,

which is (eq:fidelity) with the same r. (Boundedness of q̃_∞ follows from
the uniform convergence of densities that are bounded on a compact box;
r_- > 0 needs q̃_∞ bounded below, which the paper already requires through
0 < r_- ≤ r.)

**Consequences.**
- Theorem 2 can be stated under Assumptions 1–4 and 5(ii) alone, with a
  one-line remark that (ii) subsumes the ratio clause of (i). This makes the
  logical structure visible: fidelity is the *weakening* of stabilization
  that consistency gets away with.
- The two rate lines of (eq:rate) at line 410: for a non-increasing error
  sequence a_i, n a_n ≤ Σ_{i≤n} a_i, so the second line (partial sum is
  o(√n)) implies the first (a_n = o(n^{−1/2})). The bandwidth error
  |ε_n − ε_∞| is non-increasing by construction (running minimum), so its
  √n clause is redundant and can be dropped. The denominator error
  ‖q̄_n − q̄_∞‖_∞ is not necessarily monotone, so its clause stays, but a
  footnote can say when it is implied.

### 2.3 The "box hypothesis is needed" claim is false

**Where.** Paragraph "The class used below", line 524 onward, the sentences
"Here it matters that the implementation's constant is the product of
per-axis in-box masses, not the exact joint mass … This is where
Assumption 2's box hypothesis is used; compactness of Θ alone would not give
it."

**Why it is false.** For *any* compact Θ ⊂ ℝ^d of positive Lebesgue measure,
the exact in-set mass Z(μ, Σ) = ∫_Θ φ(θ; μ, Σ) dθ is strictly positive
(the Gaussian density is strictly positive) and continuous in (μ, Σ) on the
compact parameter set {μ ∈ Θ} × {σ_-²I ⪯ Σ ⪯ σ_+²I}. A positive continuous
function on a compact set has a positive minimum. So the lower bound
Z ≥ c_0 > 0 holds for the exact joint normalizer as well as the per-axis
product, and needs no box. The same compactness argument bounds the
parameter derivatives of 1/Z.

**What the box is actually needed for.** The implementation's normalizer is
the product of per-axis in-box masses, which is *defined* in terms of a
box. Assumption 2's box hypothesis is therefore what makes "evaluated as the
implementation evaluates them" a well-defined object, not what bounds it
below.

**Suggested replacement.** Keep the explicit constant
Π_d (Φ(w_d/2σ_+) − 1/2) if an explicit constant is wanted, and replace the
two quoted sentences by:

> "The implementation's constant is the product of per-axis in-box masses,
> which is where the box form of Θ in Assumption 2 enters; for this
> constant an explicit lower bound is Π_d (Φ(w_d/2σ_+) − ½) > 0, since each
> mean lies in the box and a half-width lies on one side of it. (The exact
> joint mass is likewise bounded below, by continuity and compactness, so
> nothing in the argument depends on which normalizer is used.)"

### 2.4 The prior share λ_n is missing from the parameterization

**Where.** Line 490: "Elements of Q are parameterized by
p_q = (means, component weights, covariances, α) ranging over a compact set
P_q ⊂ ℝ^D, D = m(kd + k + d(d+1)/2) + m."

**Problem.** The denominator (eq. 7) is
q̄_n = λ_n π + (1−λ_n) Σ_s α_{s,n} q_{τ_s} with λ_n = max(ν_n, δ) ∈ [δ, 1],
which varies with n. The net argument of Lemma 2 covers q̄_n only if λ is a
coordinate of the parameter. As written, P_q does not contain it, so the
uniform bound in the proof of Theorem 1 does not formally cover a varying
prior share.

**Fix.** Add λ ∈ [δ, 1] to p_q and set D = m(kd + k + d(d+1)/2) + m + 1.
The map λ ↦ q̄(θ) is affine with bounded coefficients, so Lipschitz
continuity is immediate and nothing else changes.

---

## 3. Gaps and clarity in individual proofs

### 3.1 L^{(2)} is used before it is defined

Theorem 2 (line 426) and the discussion after it use L^{(2)}_{ε}; the
definition L^{(ℓ)}_ε(θ) := E[K_ε(ρ)^ℓ | θ] appears only inside Lemma 3
(line 511, with ℓ = 1 identified with L). Define L^{(2)} at its first use,
ideally next to (eq:abc-target) or right before Theorem 2.

### 3.2 Lemma 3's hypothesis is narrower than its use

Lemma 3 (line 511) lets "the positive function q ≥ δπ be either
deterministic or F_{i−1}-measurable" but says of the bandwidth only
"ε ∈ [a, ε_0] for some a > 0". Step 2 of the CLT proof applies the lemma
with the random bandwidth ε_{i−1}. State that both ε and q may be
F_{i−1}-measurable. The clause "for some a > 0" is not used by the identity
(it is used only later, for Lipschitz constants) and can go.

### 3.3 Two definitions of the predictable bandwidth

Line 415 defines ε_{i−1} as "the bandwidth … formed from the first i−1
draws". Step 1 (line 562) defines it as "the bandwidth in force at the i-th
proposal". Under asynchrony these differ: the proposing worker holds a
prefix of the first i−1 draws, not all of them. Either choice is
F_{i−1}-measurable (the second via the lemma of §2.1) and either works in
Steps 1 and 2, since only predictability and the rate (eq:rate) are used.
Choose one, use it in both places, and make (eq:rate) refer to it. The
same applies to q̄_{i−1}.

### 3.4 Remark 1 typo

Line 548: "any sampler whose draw mixture q̄_n tracks" should read "whose
draw mixture q̄*_n tracks the denominator q̄_n". As written the sentence
names the denominator as the draw mixture.

### 3.5 Small tidy-ups in the consistency and CLT proofs

- Line 545: "R < ∞ bounds q̄*_n/q̄_n for all large n" can be made concrete:
  R = r_+ + 1 for all n large, by (eq:fidelity).
- Line 581: the parenthetical "Under the sufficient condition that each of
  the three terms is O(i^{−1/2−a}) this holds in each of the cases a < 1/2,
  a = 1/2, a > 1/2" checks nothing non-trivial (any a > 0 works) and
  distracts from the one condition that matters, b > 1/2 in the §A.1
  notation. Either cut it or reduce it to "for an error of order i^{−b} this
  requires b > 1/2".
- Step 3(b): "with (1/n) Σ c_i² → 0 from Step 2" deserves its one-line
  justification: |c_i| ≤ 2π_+‖g‖_∞/(δπ_-) =: C_0, so
  Σ c_i² ≤ C_0 Σ |c_i| = o(√n) = o(n).
- Theorem 1 does not use monotonicity of ε_n, only ε_n → ε_∞ ∈ (0, ε_0];
  with non-monotone convergence one localizes on "ε_n ≥ 1/j for all large
  n". Since the schedule is monotone by construction this is cosmetic, but
  a footnote would widen the theorem to schedules that may loosen.

### 3.6 Hypothesis bookkeeping in the Corollary 2 proof

Line 600 applies Lemma 2 "on P_a with a = ε_∞ (deterministic here, so no
localization is needed)". That is correct only under 5(ii); say "by
Assumption 5(ii)". Also the step "‖q̄_n − q̄_∞‖_∞ → 0 (the last from
(eq:rate))" uses the first rate line, so if §2.2's simplification of
(eq:rate) is adopted, this reference must point at the retained clause.

---

## 4. Proposition 3 and the floor factor

### 4.1 State the ζ₀ form first

Proposition 3 (line 441) is stated for a set A with |r−1| ≤ η_A on A and
|r−1| ≤ 1 off A, with ζ = η_A + π_{ε_∞}(A^c), and only then says "more
sharply, ζ may be replaced by ζ_0 = π_{ε_∞}(|r−1|)". The ζ_0 statement is
both sharper and hypothesis-free (it needs only ζ_0 < 1), and its proof is
the whole of the current proof. The restriction |r−1| ≤ 1 off A buys
nothing: with |r−1| ≤ M off A one gets ζ = η_A + M π(A^c) by the same
line.

**Suggested restructuring.**

> **Proposition 3.** Let ζ_0 := π_{ε_∞}(|r−1|). If ζ_0 < 1 then
> ‖π^r_{ε_∞} − π_{ε_∞}‖_TV ≤ ζ_0/(1−ζ_0).
> **Corollary.** If |r−1| ≤ η_A on A and |r−1| ≤ M off A then
> ζ_0 ≤ η_A + M π_{ε_∞}(A^c) =: ζ, and the bound holds with ζ in place of
> ζ_0 whenever ζ < 1.

### 4.2 An exact identity is available and is what you measure

Since π^r(f) − π(f) = π((f − π(f))(r−1))/π(r) = π((f − π(f))(r − π(r)))/π(r),

  ‖π^r − π‖_TV = π(|r − π(r)|) / (2 π(r)),   exactly.

The direct TV measurements reported in §A.1 (0.2–0.5 % on Cellular Potts,
0.26 % on the Gaussian history) are estimates of this quantity. Quoting the
identity next to the bound explains in one line why the bound is "loose by
an order of magnitude" (it replaces a centered absolute moment by an
uncentered one and divides by 1 − ζ_0 instead of π(r)) and tells the reader
what the untilted weights are estimating.

### 4.3 Relocate the r_floor derivation

The paragraph at line 617 ("The closed form quoted in §A.1(a) …") sits
under "Proof of Proposition 3" but proves nothing in the proposition. It
derives the closed form of factor (a) and a bound of the form η_A for it.
Move it to §A.1(a), or make it a short lemma ("Lemma 4 (prior-floor factor)")
in Appendix B and cite it from §A.1(a) and from Proposition 3's discussion.

### 4.4 Design remark for factor (a)

Factor (a) is identically one whenever the limiting bootstrap share ν_∞ is
at least δ, as the paper notes. In the current design the bootstrap share is
k/n → 0, so factor (a) is always active and r_floor < 1 wherever the proposal
is thinner than the prior. A sampler that draws from the prior with
probability δ at every call (the defensive mixture of Owen and Zhou 2000
applied on the proposal side, not only in the denominator) would make
ν_n → δ' ≥ δ, factor (a) identically one, and the weight bound 1/δ would
then hold against the *true* draw mixture rather than only against the
reconstructed denominator. The proofs need no change for this variant. A
sentence in the "Fixed versus growing m" paragraph (line 475) or in §A.1(a)
would make the trade-off explicit: δ-share of prior draws per call against
removal of the one factor that is measured to narrow the posterior.

---

## 5. Prioritized action list

Ordered by value to the paper; effort noted.

1. **Replace Assumption 4 by the black-box model (B1–B3) and the lemma of
   §2.1.** Medium effort: new assumption text, a ten-line lemma and proof in
   Appendix B, a rewrite of lines 481–488, Table 4 row, and a two-sentence
   change to the length-bias limitation at line 382. Largest payoff: turns
   the one assumption labelled "assumed" in bold into a derived property.
2. **Fix the parameterization (add λ, D + 1).** Trivial. Closes a formal
   gap in the consistency proof.
3. **Correct the box-hypothesis claim (§2.3).** Small rewrite of two
   sentences. Removes a false statement.
4. **Note that 5(ii) implies the ratio clause of 5(i), drop the redundant
   √n clause for the bandwidth (§2.2).** Small. Simplifies Theorem 2's
   hypothesis list.
5. **Define L^{(2)} at first use; widen Lemma 3 to a predictable ε; unify
   the definition of ε_{i−1}; fix the Remark 1 typo (§3.1–3.4).** Trivial each.
6. **Restructure Proposition 3 around ζ_0, add the exact TV identity,
   relocate the r_floor derivation (§4.1–4.3).** Small.
7. **Tidy-ups of §3.5–3.6 and the design remark of §4.4.** Optional.

None of these changes any theorem's conclusion. Items 1 and 4 change what
is assumed; items 2 and 3 repair the proofs' bookkeeping; the rest are
presentation.
