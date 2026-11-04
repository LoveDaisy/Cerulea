# CSSP completion: bounded feasibility probes

2026-11-04, following the first matched-observation archive at `83ae0eb`.
Scientific interpretation is in
[MATCHED_OBSERVATIONS §4](../../../docs/twilight/MATCHED_OBSERVATIONS.md#4-cssp补齐成本探针需要独立的数据任务).
This packet adds source/metadata evidence without changing spectra, solar axes,
the sky solver or validation reference values.

- `timing/`: pinned acquisition code, request/download provenance and a real
  **DRL 2014-07-17 control MAT**, extracted from a small prefix of the public
  compressed raw archive. It demonstrates actual exposure/serial/temperature
  fields. It is not one of the three CSSP target records, and its settings must
  not be transferred to them. Full raw source is 701,893,601 bytes; the HTTP
  range probe did not provide random access to CSSP members inside gzip.
- `atmosphere/`: actual NOAA KBFD observation, IEM KN38 historical-weather
  representation, site-distance metadata and exact AERONET requests/responses.
  Nearby weather is not a measurement at CSSP. Toronto's later daytime AOD is
  not a measured CSSP twilight column. Empty request results do not establish
  that no other public observation exists.
- `instrument/`: pinned sensitivity-calibration source and provenance. No
  measured angular-response curve was obtained in this bounded probe.
- `inputs.sha256.json`: byte identities of every other local file. Source
  manifests may retain historical paths or list other downloaded source files;
  the packet manifest, rather than those path strings, defines vendored files.

Author source code is MIT-licensed; see `timing/LICENSE-MIT.txt`. The control MAT
comes from the author's public Figshare archive (file4862119). The sampling,
observer-site and source-publication contracts are documented in the preceding
[matched-observation packet](../matched_twilight_observations/README.md).

## Offline check

From the repository root, without a network request or a sky-model run:

```bash
uv run python - <<'PY'
from pathlib import Path
import hashlib, json
from scipy.io import loadmat

p = Path('validation/data/cssp_observation_feasibility')
manifest = json.loads((p / 'inputs.sha256.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == expected, name
m = loadmat(p / 'timing/control-DRL-20140717T184659.mat', simplify_cells=True)['meas']
assert m['integrationTime'] == m['integrationTimeSet'] == 32000
assert m['spd'].shape == (2048, 10)
assert m['describe']['spectrometerInfo']['serialNumber'] == 'USB2+F00926'
rows = json.loads((p / 'atmosphere/noaa-BFD-window.json').read_text())
assert rows[0]['DATE'] == '2014-07-03T08:53:00'
assert 'BR BKN039 OVC090' in rows[0]['REM']
print(len(manifest), 'hashes and raw/control/weather fields verified')
PY
```

This verifies the archived evidence fields. It does not validate the original
instrument clock, certify CSSP cloud conditions, recover target exposures or
complete absolute radiometric calibration.
