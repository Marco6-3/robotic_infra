# Research / Infrastructure Boundary

Decision date: **2026-09-30**

## What this repository is now

`robotic_infra` is the reusable experimental substrate for future robot-learning projects. It owns simulation, robot/control interfaces, sensing and recording contracts, timing/latency tools, safety/evaluation utilities and reproducibility.

It does **not** own the next paper hypothesis.

The important distinction is:

```text
robotic_infra
  -> provides measurable physical and software primitives

separate research project
  -> chooses the scientific problem
  -> maps the literature
  -> states falsifiable claims
  -> designs project-specific experiments
  -> writes the paper
```

This separation is deliberate. Fast AI-assisted implementation should not be allowed to choose the scientific problem by repeatedly extending whatever experiment was easiest to code next.

## Historical research material

The repository already contains I001/I002, ContactBelief, tactile-history-control, active-tactile-insertion and the execution-time tactile refinement planning note.

These are retained in place because they contain useful negative results, frozen protocols, source snapshots, hashes and regression fixtures. They are **historical evidence**, not the repository's current roadmap.

Do not move or delete them solely for cosmetic separation if doing so breaks links or exact recovery.

## What belongs here

Good additions include:
- reusable robot/simulator adapters;
- tactile, force, camera and proprioceptive observation interfaces;
- multi-rate schedulers and causal timestamp handling;
- delay, jitter, packet-loss and stale-observation injection;
- data collection / replay / evaluation tooling;
- safety monitors and task-independent metrics;
- sensor calibration or common preprocessing that is not paper-specific;
- deterministic fixtures that test whether infrastructure behaves as documented.

## What belongs in a separate research repository

Examples:
- "fresh tactile feedback beats stale chunk-start tactile";
- a new Transformer/GRU/Mamba/tactile cache architecture;
- a paper-specific residual policy;
- a new contact-prediction objective;
- morphology/sensing/control co-design hypotheses;
- ablation tables intended to establish novelty;
- paper figures, narrative claims and submission-specific evaluation.

Those projects may depend on this repository, but their scientific logic should remain outside it.

## Suggested structure for the next research repository

Do not create this structure inside `robotic_infra`; it is shown here only as a migration contract.

```text
<research-project>/
├── RESEARCH_BRIEF.md
├── LITERATURE_MAP.md
├── OPEN_QUESTIONS.md
├── CLAIMS.md
├── protocols/
├── experiments/
├── analysis/
└── paper/
```

Before model implementation, `RESEARCH_BRIEF.md` should state:
1. the problem without architecture names;
2. the physical/system mechanism that may cause it;
3. what the strongest recent work already establishes;
4. the remaining unknown;
5. what result would falsify the claim;
6. why this embodiment/sensing/control setup is an informative test bed.

## Current freeze

The previously proposed execution-time tactile refinement direction is **not deleted**, but it is frozen as a historical planning note pending literature review and a new project-level decision.

No new R0/R1/R2/R3 training, prediction head, architecture search or paper-specific tactile policy should be added to this repository merely because the old roadmap suggested it.

If a future research project needs a generic capability that is missing here—for example a deterministic fresh-vs-stale observation scheduler—implement the generic capability here and keep the scientific comparison in the research repository.
