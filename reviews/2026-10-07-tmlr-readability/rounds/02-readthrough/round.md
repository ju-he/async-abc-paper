# Round 02: targeted read-through (no reviewers)

- baseline: ba17440
- scope: the transitions at every new sentence boundary of round 1 (git diff d1ebcc0..ba17440),
  as both round-1 verifiers suggested; Section 3, Limitations, Conclusion, Appendices A-B
- result: the boundaries read cleanly; one fix. The P-036 rewrite opened a sentence with
  "Section~\ref{sec:theory-r}", which is an appendix subsection and the only "Section~\ref" in a
  manuscript that otherwise uses \S\ref (69 times); now "The diagnostic of \S\ref{sec:theory-r}
  assesses ...". No other edit.
- build clean (0 LaTeX errors, 0 undefined), 43 pages

## Stop

Run stopped after round 1 by the stop rule: the Claude evaluator wrote it would not run another
readability round, and both verifiers advise against a further broad round. Final state: 7
humanizer edits + 38 applied round-1 edits (two amended after verification) + 1 read-through fix;
1 reverted, 10 rejected (all duplicates). No claim, number, citation, reference, label or hedge
changed. Two accepted items await the author's confirmation: P-004 (w_n -> lambda_n in Appendix
A(a)) and P-015 ("nine of the ten marginals").
- commit: 82f6602
