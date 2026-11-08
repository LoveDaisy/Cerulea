# Reproducible environment

This project uses [uv](https://docs.astral.sh/uv/).  `pyproject.toml` declares
the ranges, `uv.lock` pins the exact versions of every package **including
transitive ones**, and `.python-version` pins the interpreter.

```bash
uv sync                      # install from the lock file
uv run pytest -q             # current regression suite
uv run python -m validation.run_all
```

`uv sync` installs the locked dependency versions for the selected interpreter
and platform. Do not `pip install` into the environment — if a package is
missing, add it to `pyproject.toml` and run `uv lock`.

## Run and understand a current sky

Use the [README's complete CLI scene](README.md#quick-start) to reproduce the
current low-sun image. Add `--approximation spherical-single`, `plane-sos`,
`hybrid-sos` (default) or `spherical-mc` only within their
[declared domains](docs/ARCHITECTURE.md). There is no all-physics switch.

To compare branches and reuse the same result for images/tables:

```bash
uv run python -m experiments.approximations.compare
uv run python -m experiments.approximations.compare \
  --from-npz experiments/approximations/data/representative.npz --publish-figures
```

The first command does real fixed-scene solves and writes only theme-local
outputs. The second redraws the committed spectra, with no solve. See the
[approximation story](docs/physics/APPROXIMATIONS.md) for what the comparisons
can establish and the [catalogue](figures/README.md) for exact parameters.
MC is a budgeted algorithm choice, not automatic accuracy certification.

## Updating

```bash
uv lock --upgrade            # bump everything to the newest allowed version
uv run pytest -q
uv run python -m validation.run_all   # the physics numbers must not move
git add uv.lock pyproject.toml
```

The validation suite is the guard rail: a dependency bump that changes
`validation/results.json` materially is a red flag, not a merge.

## Sources of environment and validation state

[.python-version](.python-version) pins the interpreter and [uv.lock](uv.lock)
pins dependencies. Read these files for exact versions rather than maintaining
a second list here. `validation.run_all` currently exits non-zero for known
physical deviations; the current PASS/FAIL snapshot lives in
[STATUS.md](docs/STATUS.md) and is not duplicated here.

Branch, review, test-selection and merge procedures: [DEVELOPMENT.md](docs/DEVELOPMENT.md).
