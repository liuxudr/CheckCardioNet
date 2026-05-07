# Changelog

All notable changes to **CheckCardioNet** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-05-06

### Added
- Initial public release as a **prediction-only CLI** for cardio-oncology.
- `checkcardionet score-patient` — single-patient prediction
  (single drug or compare across all candidate ICIs).
- `checkcardionet score-cohort` — batch CSV → ranked drug × patient predictions.
- `checkcardionet list-drugs` — list supported ICI drugs and CVD-risk priors.
- Both inline (`--expr "GENE=val,..."`) and file (`--expr-file expr.csv`)
  expression profile inputs are supported.
- Bundled pre-trained artifacts (no download required):
  - `mr_results.parquet` — Mendelian-randomization causal effects.
  - `bidirectional_scores.parquet` — bidirectionality scores (BDS).
  - `dual_benefit_atlas.parquet` — dual-benefit drug atlas.
- 52-gene immune-checkpoint panel covering adaptive inhibitory, ligands, innate
  (CD47/SIRPα), efferocytosis (MERTK/AXL), and CVD-related axes.
- MIT license + research-use disclaimer.

[Unreleased]: https://github.com/liuxudr/CheckCardioNet/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/liuxudr/CheckCardioNet/releases/tag/v0.1.0
