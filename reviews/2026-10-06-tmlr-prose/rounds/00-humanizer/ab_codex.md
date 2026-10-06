I completed the comparison, but the sandbox is read-only, so I could not create `ab_codex.md`. The attempted write was rejected. Here is the requested report:

## Preference

I prefer **version 1 by a moderate margin**. The versions share nearly all their text, but version 1 is generally more formal, literal, and technically precise. Version 2 repeatedly introduces conversational or evaluative framing such as “what the rates buy,” “what the wall clock bought,” “has headroom,” and “the paper’s central measurement.” Version 2 also introduces isolated British spellings into an otherwise American-spelled manuscript.

AI-likeness: **version 1: 2/10; version 2: 6/10**. Version 1 occasionally has dense, clause-stacked sentences. Version 2 adds the more conspicuous AI-like pattern of recasting technical relations as punchy slogans or announcing their significance.

## Differences

There are 31 changed locations; this table covers the 20 most consequential. I omitted punctuation-only changes, minor additions such as “simply” and “itself,” and repeated spelling variants.

| location (quote 5-8 words) | preferred | why |
|---|---|---|
| “find it calibrated where a pyABC baseline” | version 1 | Version 2 inserts “Four measurements follow.” The signpost interrupts an already compact abstract and adds an unnecessary explicit count. |
| “remove the generation: our sampler refits” | version 1 | “We remove” gives the action an explicit agent. Version 2’s “This paper removes” is less direct, although splitting the sentence slightly reduces its load. |
| “Every quantity the sampler uses is recomputed” | version 1 | It states the design rule directly. Version 2’s “One rule makes this work” is vague, rhetorically inflated, and lengthens the sentence before giving the rule. |
| “takes one candidate back, provides this contract” | version 1 | “Provides” is sufficient and evidence-bound. Version 2 adds the empty intensifier “exactly,” strengthening the conformance claim without adding support. |
| “is on the same tightening sequence” | version 1 | The full stop separates consequence from implementation. Version 2’s colon produces a longer clause-stacked sentence. |
| “asks how faithful a stand-in it is” | version 1 | This accurately introduces the theoretical question without claiming that it is the section’s only question. Version 2’s “the one question” is overexclusive. |
| “whether the denominator is the density” | version 1 | The declarative formulation is compact and formal. Version 2 announces and italicizes a question, adding rhetoric but no precision. |
| “one reason the method uses smooth kernels” | version 1 | “One reason” is already specific because the preceding clause names the Lipschitz condition. “Concrete” is an empty intensifier, while “carries everything else” is vaguer than “holds the remaining conditions.” |
| “studies with injected runtime heterogeneity run” | version 1 | “Run on the Gaussian mean” is literal and easy to follow. Version 2 makes the Gaussian mean “host” the studies, an unnecessary personification. |
| “reference for the per-simulation gain of” | version 1 | “Per-simulation gain” names the quantity formally. Version 2’s “what an adaptive proposal buys” is colloquial and introduces a commercial metaphor. |
| “two differ by the throughput ratio” | version 1 | This states the mathematical relation explicitly. Version 2’s “what separates the two” is less exact; “says whether” is also less formal than “shows whether.” |
| “covers every configuration on which we ran” | version 1 | Version 2 inserts “is the paper’s central measurement,” an unsupported editorial valuation that does not help interpret the evidence. |
| “Throughput matters only through the tolerance” | version 1 | Version 1 directly connects the systems metric to the inferential outcome. Version 2 adds practitioner-facing rhetoric and the vague sentence “The curves add what a single budget cannot show.” |
| “cost determines where the method becomes advantageous” | version 1 | This is precise. “Decides where the method pays” personifies the cost and substitutes an informal payoff metaphor. |
| “effective sample size is a few multiples” | version 2 | “Raising \(k\) is not free” gives the action an explicit agent and avoids the awkward subject “a larger \(k\).” The idiom is slightly informal but easier to follow. |
| “part of the method with the most” | version 1 | It states the comparative limitation plainly. “Has headroom” is jargon-like and drops the claim that this component has the *most* room for improvement. |
| “restriction is vacuous in the case” | version 1 | “Vacuous” has a precise mathematical meaning: the restriction imposes no condition when \(r=1\). “Costs nothing” could instead refer to computational or statistical cost. |
| “exponent depends on the inference problem” | version 1 | This is direct technical prose. Version 2’s “What the exponent turns on is…” is a marked, less natural construction. |
| “theorem is therefore a statement about” | version 1 | “Therefore” expresses the inference cleanly. Version 2’s “This is what makes” adds a formulaic causal frame without improving the logic. |
| “floor to clear: go far enough above” | version 1 | Version 1 uses one restrained metaphor. Version 2 extends it into “a frontier to negotiate” and repeats “buy,” making a negative result sound packaged rather than plainly reported. |

## Content check

The numerical results, equations, citations, and substantive empirical findings are unchanged. The following edits do not say quite the same thing:

- Explicit count added:

  - Version 1: no corresponding sentence.
  - Version 2: “Four measurements follow.”

  This adds a numerical organizational claim absent from version 1.

- Contract claim strengthened:

  - Version 1: “provides this contract.”
  - Version 2: “provides exactly this contract.”

  “Exactly” removes latitude and strengthens the qualification.

- Scope of the theory section narrowed:

  - Version 1: “Section … asks how faithful a stand-in it is.”
  - Version 2: “how faithful a stand-in it is becomes the one question of Section …”

  Version 2 claims exclusivity; version 1 identifies a question without excluding others.

- Evaluative priority added:

  - Version 1: “Figure … covers every configuration on which we ran the barrierized twin.”
  - Version 2: “Figure … is the paper’s central measurement. It covers every configuration on which we ran the barrierized twin.”

  Version 2 adds an authorial ranking of the evidence.

- Comparative limitation weakened:

  - Version 1: “The effective-sample-size ceiling is the part of the method with the most room for improvement.”
  - Version 2: “The effective-sample-size ceiling is where the method has headroom.”

  Version 2 no longer says this component has more room for improvement than the others.

- Mathematical qualification made ambiguous:

  - Version 1: “That restriction is vacuous in the case of most interest, \(r\equiv1\).”
  - Version 2: “That restriction costs nothing in the case of most interest, \(r\equiv1\).”

  “Vacuous” says the restriction imposes no condition; “costs nothing” could instead be understood as a claim about computational or statistical cost.