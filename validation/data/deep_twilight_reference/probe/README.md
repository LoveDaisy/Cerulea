# Planning probes (frozen, 2026-11-04)

Three owner-authorized read-only probe batches produced during the *planning*
phase of `task-deep-twilight-reference`, before any implementation code
existed. They are migrated here verbatim (byte-preserving copies, original
SHA256 values intact — see `inputs.sha256.json`) from the local scratchpad
directory `scratchpad/task-deep-twilight-reference/probes/`, so that the
protocol's budgets, seeds and precision scales (`../protocol.json`) are
backed by re-runnable evidence outside the scratchpad.

- `bounded_probe.py` + `config.json`, `build.json`, `raw.jsonl`,
  `summary.json`, `single-*.json`, `probe-run.log`, `point-source.npz`:
  small-budget seed/budget variability (48 batches, 1k/4k/16k photons,
  3 wavelengths, 2 depths) and the one planning source-table build
  (75.7 s, label `deep-twilight-plan-probe-450-550-650-clear-v1`).
- `growth_probe.py` + `growth-*.json(l)`, `growth-run.log`: budget growth
  and finite-disk/point-source pairing (96 batches, 100k/400k/1.6M).
- `diagnostic_probe.py` + `diagnostic-*.json(l)`, `diagnostic-run.log`:
  read-only `sys.setprofile` contribution/order diagnostics (16 batches at
  1.6M, plus bit-identity controls).
- `summarize_probes.py` + `verified-probe-summary.json`: independent
  recomputation of both raw sets from the records themselves.

Provenance notes:

- Probe scripts resolve paths from the process working directory (`ROOT =
  Path.cwd()`); they were run from the repository root and read code files
  whose hashes are frozen inside `config.json`. The copies here are records,
  not runnable entry points for the formal benchmark — use
  `tools/deep_twilight_reference.py` for that.
- The probe table `point-source.npz` covers only 450/550/650 nm. It is the
  *planning* table; the formal benchmark loads the archived full81 table
  declared in `../protocol.json`. Do not swap the two.
- Findings quoted in the plan (throughput, conditional half-widths,
  effective-contributor counts, order shares) are all recomputable from
  `*-raw.jsonl`; `verified-probe-summary.json` is the recomputed check.
- Seeds used by the probes (10000+, 20000+, 29999/30000–30007) are disjoint
  from the pre-registered formal seeds (40000+…, bootstrap analysis seed
  90000, cost-probe scratch seeds 901/902, smoke seeds 1000+).
