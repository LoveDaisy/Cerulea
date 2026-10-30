# Reproducible environment

This project uses [uv](https://docs.astral.sh/uv/).  `pyproject.toml` declares
the ranges, `uv.lock` pins the exact versions of every package **including
transitive ones**, and `.python-version` pins the interpreter.

```bash
uv sync                      # install from the lock file
uv run pytest -q             # 193 tests
uv run python -m validation.run_all
```

Nothing else is needed; `uv sync` recreates `.venv` byte-for-byte from the
lock.  Do not `pip install` into the environment — if a package is missing, add
it to `pyproject.toml` and run `uv lock`.

## Updating

```bash
uv lock --upgrade            # bump everything to the newest allowed version
uv run pytest -q
uv run python -m validation.run_all   # the physics numbers must not move
git add uv.lock pyproject.toml
```

The validation suite is the guard rail: a dependency bump that changes
`validation/results.json` materially is a red flag, not a merge.

## Locked versions (at the time of writing)

Python 3.12.11, numpy 2.5.4, scipy 1.18.1, pillow, matplotlib 3.11.2, pytest 8.4.2.
