# Independent final mechanism review

`review.md` records the independent source/flight/MC composition review. It does
not approve the still-separate stage-four full-ensemble numerical conclusions.

Recompute only the Hamiltonian flight and aerosol phase-moment probes:

```bash
uv run python -m tools.check_curved_hamiltonian
```

The default output is `hamiltonian-probes.json` in this directory. To run from
another working directory without scratchpad files:

```bash
uv run --project /path/to/repo python /path/to/repo/tools/check_curved_hamiltonian.py \
  --out /tmp/curved-hamiltonian.json
```

The command does not run Monte Carlo or modify reference/ensemble data. It uses
Cartesian momentum `p=n*d` and independent DOP853 sphere events to check the
continuous-flight RK implementation. The phase diagnostic integrates the actual
piecewise-constant inverse-CDF cell probabilities against low-order Legendre
polynomials and compares them with the Mie analytic moments. Its conclusions
apply to the explicitly sampled standard-profile rays and rural aerosol cases.

Historical artifacts are preserved separately:

- `historical-initial-probes.py`, `independent-final-probes.json/.log`: exact
  initial checker and results before the source-binding precondition patch.
  The historical program is an evidence snapshot, not the maintained entry point.
- `independent-final-contract.json`: the wrong-table counterexample and independent
  rejection checks after the binding guard was fixed.
- `source_provenance.json`: initial/current runtime source hashes, checker identity,
  portable-run comparison and the guard-only version distinction.
- `inputs.sha256.json`: immutable historical artifacts, review and provenance hashes;
  derived current-run JSON/log are not included in a circular overwrite check.

Current-run outputs `hamiltonian-probes.json/.log` come from the maintained tracked
tool; `portable-hamiltonian-probes.json/.log` preserve the separate run from `/tmp`.
Those two JSONs were compared in full. Their flight/moment arrays also match the
initial run. Source hashes retain the guard-before/after difference explicitly;
the old records were not relabelled as newly executed code.
