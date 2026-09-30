# AGENTS.md — robotic_infra scope

## Repository mission

`robotic_infra` is reusable robotics research infrastructure, not an active paper/research project.

Its job is to provide stable, inspectable building blocks for:
- FR3 / MuJoCo / ROS 2 simulation and control;
- robot/action/observation interfaces;
- recording and LeRobot adapters;
- camera/tactile/proprioceptive I/O contracts;
- causal timing, multi-rate scheduling, latency/jitter/dropout injection;
- safety checks, evaluation utilities and reproducibility;
- smoke tests and minimal mechanism tests needed to validate infrastructure.

## Research boundary

Do **not** use this repository as the place where a new research question is invented and then progressively optimized.

Unless the human researcher explicitly changes this boundary, agents must not:
- declare a new "current research direction" in this repository;
- turn an intuition into a paper hypothesis and immediately implement it;
- perform architecture search merely to obtain a positive result;
- weaken a strong baseline, hide proprioception, change labels, or alter task physics to make a proposed method win;
- reinterpret a negative result as a positive claim;
- add new paper-specific models or experiment ladders that are not reusable infrastructure;
- overwrite historical protocols/results in order to match a newer narrative.

A new research project should live in a separate repository with its own literature map, research brief, claims, protocols and paper artifacts. This repository may then expose generic infrastructure that the project imports or reuses.

## Literature-first gate for research requests

If a task would create a new scientific claim rather than improve infrastructure, stop implementation at the boundary of this repository.

Before any new research project begins, the human researcher should own a short brief containing:
1. problem stated without model names;
2. physical/system reason the problem matters;
3. literature map and anchor papers;
4. what is still unknown;
5. evidence that would falsify the hypothesis;
6. why the available embodiment/sensing/control setup is useful for this question.

Only after that gate should a separate research repository implement experiments.

## Historical experiments

The existing `experiments/`, `docs/research-results/` and `docs/research/` material is retained for provenance, reproducibility and regression coverage.

Treat those files as historical evidence/planning snapshots unless a file explicitly says otherwise. Preserve links, frozen protocols, negative results, hashes and replay assumptions. Do not delete or relocate them merely to make the repository look cleaner if that would harm reproducibility.

Infrastructure fixes may update experiment code only when needed to preserve compatibility, contracts or reproducibility. Such changes must not silently strengthen scientific claims.

## Engineering priorities

When changing this repository:
1. preserve public interfaces and recorded data semantics when practical;
2. keep causal timing explicit;
3. keep privileged state out of deployment observations unless clearly marked;
4. make optional simulation/ML dependencies fail or skip cleanly;
5. prefer deterministic tests and pinned dependencies;
6. distinguish smoke tests from scientific evidence;
7. document what was actually executed versus merely planned.

The default question for an agent in this repository is:

> "Does this change make the robotics infrastructure more reliable, reusable or measurable?"

If the answer is mainly "it might become a paper contribution", the work belongs in a separate research project.
