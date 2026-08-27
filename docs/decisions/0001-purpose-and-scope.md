# ADR 0001 — Purpose and scope

Status: accepted · Date: 2026-08-27

## Context

Public RL repositories usually optimize for one of: breadth (algorithm
zoos), performance (research codebases), or pedagogy (tutorial notebooks).
Mixing goals implicitly produces a repo that serves none of them. This repo
needs its goals stated so future decisions (and future sessions) have a
razor.

## Decision

roborl is an **educational and research repository** with three goals, in
order: (1) learn RL deeply by implementing algorithms from scratch and
verifying them against reference baselines; (2) treat debugging, telemetry
literacy, and experiment methodology as first-class curriculum; (3) build
toward **contact-rich robotic manipulation**, deliberately starting from
classic control. It is explicitly **not** an algorithm zoo, not a framework
for others to build on, and not chasing state-of-the-art results. It is held
to professional engineering standards because the tooling itself teaches.

## Consequences

- Algorithms land only with the full lifecycle (spec, tests, verification,
  report) — fewer algorithms, understood deeply.
- Simplicity wins arguments: readability beats configurability, and
  educational value justifies some duplication (see ADR 0003).
- The roadmap goes classic control → continuous control → goal-conditioned →
  contact-rich manipulation; locomotion tasks appear only as standard
  verification benchmarks along the way.
