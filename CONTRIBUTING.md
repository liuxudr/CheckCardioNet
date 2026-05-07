# Contributing to CheckCardioNet

Thank you for considering a contribution!

## Getting started

```bash
git clone https://github.com/liuxudr/CheckCardioNet.git
cd CheckCardioNet
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
checkcardionet list-drugs   # smoke check
```

## Code style

- Python ≥ 3.10, modern type hints (`list[str]`, not `List[str]`).
- Line length: 100.
- Lint:   `ruff check .`
- Format: `ruff format .`
- All new public APIs must have a docstring.

## Tests

- New features must add tests under `tests/`.
- Bug fixes must add a regression test.
- Avoid network or large-data dependencies in unit tests; use small in-memory fixtures.

```bash
pytest --cov=checkcardionet tests/
```

## Pull requests

1. Open an issue first for non-trivial changes so we can agree on the design.
2. Fork → feature branch (`feat/short-name` or `fix/short-name`).
3. Keep PRs focused (one logical change per PR).
4. Include a clear PR description: what changed, why, how it was tested.
5. CI must pass (tests + lint).

## Reporting bugs / requesting features

Open a GitHub Issue with:

- CheckCardioNet version (`pip show checkcardionet`)
- Python version and OS
- Minimal reproducible example
- Expected vs. actual behavior

## Code of Conduct

Be kind, constructive, and respectful. We follow the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
