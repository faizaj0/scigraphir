#!/usr/bin/env python3
"""Write the adjudicated results for the fixed 50-target strict LLM audit."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT_DIR = HERE / "llm_audit_v2_strict"
SAMPLE = AUDIT_DIR / "sample.jsonl"

# These verdicts were assigned by inspecting the frozen target, every primary
# inspiration claim, the resolved abstracts, the target's citation contexts,
# and the first supportable alternative. Uncertainty is treated as failure.
PRIMARY_FAILURES = {
    2: (
        "The primary set attributes type-dependent decoding metrics and the method "
        "of types to Graph decomposition: A new key to coding theorems. Its abstract "
        "and the target's citation context support an error-exponent/LM-rate result, "
        "not that claimed contribution."
    ),
    4: (
        "The primary set attributes Jacobian-rank dimensional expressivity to The "
        "theory of variational hybrid quantum-classical algorithms. The target paper "
        "presents that geometric/Jacobian analysis as its own contribution; the cited "
        "paper supplies the variational-circuit setting, not the claimed method."
    ),
    19: (
        "The set supports the synbiotic, the AOM/DSS model and general short-chain "
        "fatty-acid effects, but it does not account for the fixed hypothesis's specific "
        "tight-junction, mucus-layer and apoptosis mechanism. Sufficiency therefore "
        "fails under the strict coverage rule."
    ),
    20: (
        "The FARDEEP citation supplies robust least-trimmed-squares deconvolution of "
        "gene-expression profiles. The retained delta incorrectly states that this "
        "source fits subsets of CpGs; that methylation-specific adaptation belongs to "
        "the target paper."
    ),
    45: (
        "The primary delta attributes deterministic reversible reorientation by "
        "linearly polarized light to the cited ferromagnetic study. That work reports "
        "deterministic helicity-dependent switching under circular polarization and "
        "random domains under linear polarization."
    ),
}

RATIONALES = {
    1: "Both sets cover the no-go invariant, higher-dimensional examples and adaptive-gate escape route; D2 changes the locality-preserving classification and Levin--Wen source while reconstructing the same endpoint.",
    3: "Both sets retain the restart transformation and synthesis ingredients; D2 replaces the weighted-graph source with a finite-state competitive-analysis source for the same cycle-ratio role.",
    5: "D2 replaces the Wasserstein-gradient-flow monograph with an equivalent survey while retaining the global-minimum, entropic and privacy ingredients.",
    6: "D2 replaces the monitors-as-memories paper with a distinct monitors paper while retaining multiparty types, higher-order communication and atomic rollback, reconstructing the same causal-consistency claim.",
    7: "The LSTM source supports variable-length sequential encoding into a fixed state; no distinct decomposition was found under the construction budget.",
    8: "D2 changes the generation lower-bound source while preserving universal learning and enumeration components for the same language-identification and generation endpoint.",
    9: "A general blackboard-architecture paper replaces Hearsay-II as the shared-workspace source; language memory and verification components are unchanged.",
    10: "The single Weil-bound inspiration is citation-grounded and coherent with the fixed endpoint; no distinct set was found under the budget.",
    11: "D2 replaces one multi-agent-debate paper with another while retaining consensus arbitration and information gathering under partial observability.",
    12: "The proximal-sampling, convergence and lower-bound sources form a coherent primary explanation; no distinct set was found under the budget.",
    13: "D2 replaces the pCN reference with a covariance-based high-dimensional MCMC source while retaining the Dirichlet-process mixture and Gibbs-sampling components.",
    14: "The sources separately support NRF1 regulation, caveolar claudin-5 internalisation and autophagic handling of claudin-5; no distinct set was found.",
    15: "A pre-Iron-Age Roman reference replaces the Daunian comparison source while the Aegean and Balkan reference panels remain fixed; both sets support the same ancestry comparison.",
    16: "The WRKY binding rule, HKT1;5 sodium transport and NHX antiporter evidence provide complementary components of the fixed mechanism; no distinct set was found.",
    17: "The database and regulator review supply the two m6A gene universes used by the target analysis; no distinct set was found under the budget.",
    18: "The TBC complex, Rheb--mTOR regulation and Wnt-regulated kinesin evidence form a coherent route to the fixed KIF2C mechanism; no distinct set was found.",
    21: "D2 substitutes a general TGF-beta immunity review for a Treg/Th17 review while retaining endothelial, Treg-subset and CD8-exhaustion evidence.",
    22: "The background supplies the PROTAC mechanism; the two papers supply the G4-binding warhead and proximity-labelling evidence. No distinct set was found.",
    23: "D2 replaces one Taguchi-optimisation application with another while retaining mutagenesis, toxin-overproduction and date-waste components.",
    24: "D2 replaces one FMO-2 longevity paper with another while retaining the NHR-49/HLH-30 and bacterial-induction links required by the same endpoint.",
    25: "A different largemouth-bass sex-locus study replaces the original mapping paper while the T2T-Y and transposable-element components remain unchanged.",
    26: "An opah cryptic-species study replaces a general marine-fish barcode study while ASAP remains fixed; both support the same interpretation and delimitation of the John Dory lineages.",
    27: "The cited soft actor--critic paper supplies the continuous-control search method applied to the CFT crossing equations; no distinct set was found.",
    28: "The cited calculation supplies two-loop analytic machinery explicitly extended by the target's penguin calculation; no distinct set was found under the budget.",
    29: "D2 replaces a vortex whispering-gallery source with a broader twisted-spin-wave source while retaining frequency-comb scattering and Penrose amplification.",
    30: "The H I survey, extinction map and simulated spin-temperature relation provide complementary observational ingredients; no distinct set was found.",
    31: "A second neural-network criticality paper replaces the original parameter-diagnostics source while the cross-model-transfer component remains fixed.",
    32: "D2 replaces the derivative-RKHS paper with a standard kernel-approximation monograph while retaining kernel Koopman analysis and the Schrödinger--Kolmogorov transform.",
    33: "A spinning-resonator paper replaces the optomagnonic Sagnac source; whispering-gallery magnon coupling and squeezing remain unchanged.",
    34: "D2 distributes the non-Hermitian quantum-geometric-tensor definition, finite-width dynamics and nonlinear-conductivity method across a distinct complete source set.",
    35: "D2 replaces an interstellar-polarisation correction source with the Serkowski wavelength relation while retaining disc polarimetry, comparison and warp theory.",
    36: "A different pi-flux multicomponent-Hubbard study supplies the DQMC lattice template; sign-free simulation and Ginzburg--Landau ingredients remain fixed.",
    37: "D2 replaces the Gross--Pitaevskii-limit Bogoliubov source with a rigorous excitation-spectrum source while retaining renormalisation and impurity derivation.",
    38: "D2 replaces rotation-measure synthesis with the foundational Faraday-dispersion formulation while retaining scan strategy and ionospheric correction; both reconstruct the same survey endpoint.",
    39: "The primary gives a multiband quantum framework extended by the target to Hall transport; D2 supplies a direct multiband Hall treatment with WAL and EEI fixed.",
    40: "D2 replaces one demonstration of XPD structural sensitivity with another while retaining multiple-scattering simulation, core-level and IrTe2-coordinate components.",
    41: "Two spatial projections of Kubo electrical conductivity provide the same transferable site-decomposition principle used for thermal conductivity.",
    42: "The QSGW, optical ladder, screened-interaction and sound-speed sources cover distinct electronic, optical and mechanical components; no distinct set was found.",
    43: "D2 replaces one momentum-dependent electron--phonon response calculation with another while retaining the NbSe2 validation example.",
    44: "D2 replaces one high-speed bubble-imaging source with another while retaining the same coupled interface-supersaturation model.",
    46: "D2 replaces the HPT precipitation example with a large-strain precipitation study while retaining the Mg--Mn dislocation-nucleation mechanism.",
    47: "D2 replaces the niobium-oxide thermodynamic source with an Ellingham-diagram source while retaining epitaxial growth-window and transport evidence.",
    48: "D2 replaces the CrI3 demonstration with a general magnetic-injection-current treatment while retaining the TBG symmetry and topological-heavy-fermion components.",
    49: "D2 replaces a recent variational phase-field source with the foundational non-isothermal formulation while retaining convex splitting and the cited existence argument.",
    50: "The polar-catastrophe paper and long-range electrostatic neural potential supply the two complementary ingredients of the false-metallisation mechanism; no distinct set was found.",
}

EXPORTED_STRATUM_COUNTS = {
    ("cs", "train"): 5445,
    ("cs", "test"): 1052,
    ("biology", "train"): 3954,
    ("biology", "test"): 827,
    ("physics", "train"): 3118,
    ("physics", "test"): 892,
    ("matsci", "train"): 1304,
    ("matsci", "test"): 331,
}


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return centre - half_width, centre + half_width


def percentage(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%"


def main() -> None:
    rows = [json.loads(line) for line in SAMPLE.open()]
    verdicts = []
    strata = defaultdict(lambda: {"n": 0, "confirmed": 0})

    for row in rows:
        audit_id = row["audit_id"]
        automated_alternative = int(len(row["sets"]) > 1)
        primary_valid = int(audit_id not in PRIMARY_FAILURES)
        confirmed = int(primary_valid and automated_alternative)
        accepted = "D2" if confirmed else ""
        if not primary_valid:
            verdict = "fail_primary"
            rationale = PRIMARY_FAILURES[audit_id]
        elif confirmed:
            verdict = "pass_nonunique"
            rationale = RATIONALES[audit_id]
        else:
            verdict = "pass_no_alternative"
            rationale = RATIONALES[audit_id]

        verdicts.append(
            {
                "audit_id": audit_id,
                "domain": row["domain"],
                "split": row["split"],
                "target_valid": 1,
                "primary_valid": primary_valid,
                "automated_alternative": automated_alternative,
                "accepted_alternative": accepted,
                "confirmed_nonunique": confirmed,
                "verdict": verdict,
                "rationale": rationale,
            }
        )
        stratum = strata[(row["domain"], row["split"])]
        stratum["n"] += 1
        stratum["confirmed"] += confirmed

    with (AUDIT_DIR / "verdicts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=verdicts[0].keys())
        writer.writeheader()
        writer.writerows(verdicts)

    target_valid = sum(row["target_valid"] for row in verdicts)
    primary_valid = sum(row["primary_valid"] for row in verdicts)
    proposed = sum(row["automated_alternative"] for row in verdicts)
    confirmed = sum(row["confirmed_nonunique"] for row in verdicts)
    target_ci = wilson(target_valid, len(verdicts))
    primary_ci = wilson(primary_valid, len(verdicts))
    precision_ci = wilson(confirmed, proposed)
    prevalence_ci = wilson(confirmed, len(verdicts))

    total_population = sum(EXPORTED_STRATUM_COUNTS.values())
    poststratified = sum(
        EXPORTED_STRATUM_COUNTS[key]
        * strata[key]["confirmed"]
        / strata[key]["n"]
        for key in EXPORTED_STRATUM_COUNTS
    ) / total_population

    report = f"""# Strict independent LLM audit of SIR-4 construction

## Scope and protocol

This is a **single-reviewer LLM audit**, not a human evaluation. The fixed sample contains 50 exported targets selected independently by SHA-256 rank within each field-by-split stratum. It includes 13 targets each from computer science and biology and 12 each from physics and materials science; train and test examples are represented in every field.

For each target, the audit inspected the frozen question, background and hypothesis; every primary inspiration claim; resolved titles and abstracts; relevant citation contexts in the stored target full text; and the first supportable alternative decomposition. A set passed only when (i) the target was faithful, (ii) every source supported the transferable idea attributed to it, (iii) the complete set covered the fixed hypothesis, (iv) each member made a distinct necessary contribution, and (v) an accepted alternative reconstructed the **same** fixed hypothesis. Unsupported or uncertain claims were rejected rather than inferred charitably.

## Results

| Measure | Result | Wilson 95% CI |
|---|---:|---:|
| Target validity | {target_valid}/50 ({percentage(target_valid, 50)}) | {100*target_ci[0]:.1f}--{100*target_ci[1]:.1f}% |
| Primary-decomposition validity | {primary_valid}/50 ({percentage(primary_valid, 50)}) | {100*primary_ci[0]:.1f}--{100*primary_ci[1]:.1f}% |
| Precision of proposed non-unique cases | {confirmed}/{proposed} ({percentage(confirmed, proposed)}) | {100*precision_ci[0]:.1f}--{100*precision_ci[1]:.1f}% |
| LLM-confirmed non-uniqueness, all audited targets | {confirmed}/50 ({percentage(confirmed, 50)}) | {100*prevalence_ci[0]:.1f}--{100*prevalence_ci[1]:.1f}% |

The nearly balanced sample deliberately gives small fields and test splits more influence than their prevalence in the exported benchmark. Post-stratifying the eight observed field-by-split rates to the exported counts gives a secondary estimate of **{100*poststratified:.1f}%**. This estimate is less stable because each stratum contains only six or seven audited targets; the transparent primary result is {confirmed}/50.

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
"""
    (AUDIT_DIR / "report.md").write_text(report)

    summary = {
        "n": len(verdicts),
        "target_valid": target_valid,
        "primary_valid": primary_valid,
        "proposed_nonunique": proposed,
        "confirmed_nonunique": confirmed,
        "unweighted_confirmed_rate": confirmed / len(verdicts),
        "poststratified_confirmed_rate": poststratified,
        "strict_primary_failures": sorted(PRIMARY_FAILURES),
    }
    (AUDIT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
