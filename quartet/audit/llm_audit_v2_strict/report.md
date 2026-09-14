# Strict independent LLM audit of SIR-4 construction

## Scope and protocol

This is a **single-reviewer LLM audit**, not a human evaluation. The fixed sample contains 50 exported targets selected independently by SHA-256 rank within each field-by-split stratum. It includes 13 targets each from computer science and biology and 12 each from physics and materials science; train and test examples are represented in every field.

For each target, the audit inspected the frozen question, background and hypothesis; every primary inspiration claim; resolved titles and abstracts; relevant citation contexts in the stored target full text; and the first supportable alternative decomposition. A set passed only when (i) the target was faithful, (ii) every source supported the transferable idea attributed to it, (iii) the complete set covered the fixed hypothesis, (iv) each member made a distinct necessary contribution, and (v) an accepted alternative reconstructed the **same** fixed hypothesis. Unsupported or uncertain claims were rejected rather than inferred charitably.

## Results

| Measure | Result | Wilson 95% CI |
|---|---:|---:|
| Target validity | 50/50 (100.0%) | 92.9--100.0% |
| Primary-decomposition validity | 45/50 (90.0%) | 78.6--95.7% |
| Precision of proposed non-unique cases | 32/36 (88.9%) | 74.7--95.6% |
| LLM-confirmed non-uniqueness, all audited targets | 32/50 (64.0%) | 50.1--75.9% |

The nearly balanced sample deliberately gives small fields and test splits more influence than their prevalence in the exported benchmark. Post-stratifying the eight observed field-by-split rates to the exported counts gives a secondary estimate of **52.1%**. This estimate is less stable because each stratum contains only six or seven audited targets; the transparent primary result is 32/50.

## Strict failures

Five primary decompositions were rejected:

1. **Audit 2, mismatch capacity:** a graph-decomposition citation was credited with type-dependent metrics and the method of types beyond the support in its abstract and the target's citation context.
2. **Audit 4, quantum-circuit expressivity:** the Jacobian-rank expressivity method developed by the target was attributed to an earlier general variational-algorithm paper.
3. **Audit 19, synbiotic colorectal-cancer prevention:** the papers did not cover the fixed endpoint's specific barrier-restoration and apoptosis components, so the set was incomplete.
4. **Audit 20, methylation deconvolution:** a gene-expression LTS method was incorrectly described as selecting CpGs; the methylation adaptation belongs to the target.
5. **Audit 45, light-programmable tellurium:** a ferromagnetic source was said to show deterministic switching under linearly polarised light, whereas it reports helicity-dependent switching under circular light and random domains under linear light.

The first, second, fourth and fifth cases are **source-grounding failures**: a plausible target-paper idea was assigned to a citation that did not supply it. The third is a **coverage failure**. These cases show why the strict audit is more informative than simply rerunning the benchmark's semantic checker.

## Interpretation

The audit confirms 32 targets with a valid primary set and at least one distinct valid set for the same frozen endpoint. One such counterexample is sufficient to reject universal paper-level uniqueness; 32/50 additionally supports the directional hypothesis that alternative decompositions are common under the bounded search protocol. The estimate remains a lower bound with respect to search: a target labelled “no alternative found” may still have an unobserved valid set.

These figures should be reported as an **independent LLM audit**. They do not measure human agreement and should not be described as manual or human validation. The fixed sample and row-level decisions in `sample.jsonl` and `verdicts.csv` make the audit reproducible and suitable for later blinded human review.
