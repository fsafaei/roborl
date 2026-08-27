# Contributing

Thanks for your interest! This is primarily a personal learning repository,
but issues and PRs are welcome — especially bug reports, corrections to
docs, and reference-implementation discrepancies found while verifying
algorithms.

## Workflow

1. Branch from `main`: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`,
   `chore/<topic>`.
2. `make setup` once (installs the dev environment and pre-commit hooks).
3. Make small, logically scoped commits with
   [Conventional Commit](https://www.conventionalcommits.org/) messages:
   `feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`.
4. **Run `make check` before every commit** — format check, lint, types,
   fast tests. Never commit red.
5. Open a PR against `main`; CI must be green on both Linux and macOS.

## Quality bar

- **Formatting/linting:** ruff (line length 100, Google-style docstrings).
  Don't weaken rules to get green — fix the code, or add a narrowly scoped
  ignore with a one-line justification comment.
- **Types:** mypy on `src` and `tests`; public functions fully annotated.
  mypy runs in `make check` and CI rather than pre-commit because it needs
  the project's full dependency environment, which pre-commit's isolated
  hook environments would have to duplicate.
- **Tests:** pytest with three markers —
  `unit` (fast, isolated), `smoke` (short end-to-end CPU runs), `slow`
  (excluded from CI). `make test` runs unit+smoke; that suite must stay
  fast (seconds, not minutes).
  **No test may touch the network.** W&B is disabled or offline in all
  tests; reference-data tests run against committed fixtures.
- **Dependencies:** ask before adding any (open an issue), and significant
  additions get an ADR in `docs/decisions/`.
- **Docs:** user-visible changes update the relevant page in `docs/`;
  algorithm work follows [docs/algorithm-lifecycle.md](docs/algorithm-lifecycle.md).

## PR checklist

- [ ] `make check` green locally; CI green
- [ ] Tests added/updated for behavior changes
- [ ] Docs updated where user-visible
- [ ] No unexplained lint/type suppressions
- [ ] No fabricated numbers: every reported result traces to a run and a
      commit (see [docs/benchmarking.md](docs/benchmarking.md))

## Review etiquette

Reviews are part of the curriculum here: comments explain *why*, link to
docs/ADRs where the reasoning lives, and prefer questions over decrees.
Disagreements about policy (thresholds, conventions) end in a PR to the
relevant doc, not a comment thread.
