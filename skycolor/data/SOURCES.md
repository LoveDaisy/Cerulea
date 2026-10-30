# Data provenance

All tables were extracted once from the sources below and committed as small
CSV files so that the simulation runs offline.  `tools/build_data.py`
regenerates them from the raw files in `data_raw/`.

| File | Quantity | Source |
|---|---|---|
| `cie1931_2deg.csv` | CIE 1931 2° colour matching functions, 1 nm, 360–830 nm | [CVRL](https://cvrl.ioo.ucl.ac.uk/database/data/cmfs/ciexyz31_1.csv) (UCL Colour & Vision Research Laboratory); `data_raw/ciexyz31_1.csv` |
| `solar_am0.csv` | Extraterrestrial solar spectral irradiance, W m⁻² nm⁻¹ | ASTM G173-03 reference spectra, "extraterrestrial" column, via pvlib's `ASTMG173.csv`; `data_raw/ASTMG173.csv` |
| `o3_cross_section.csv` | O₃ absorption cross section, cm² molecule⁻¹ (Hartley–Huggins–Chappuis) | JPL 2006 evaluation, as distributed with [NCAR TUV-x](https://github.com/NCAR/tuv-x) `data/cross_sections/O3_4.nc`; `data_raw/O3_4.nc` |
| `no2_cross_section.csv` | NO₂ absorption cross section, cm² molecule⁻¹, 220 K and 294 K | JPL evaluation, TUV-x `data/cross_sections/NO2_1.nc`; `data_raw/NO2_1.nc` |

Notes
* The O₃ table has no temperature dependence in the visible (the 218 K and
  295 K columns are identical), so the model uses the 295 K values throughout.
  The real Chappuis band is slightly temperature dependent; this is a known
  small bias at stratospheric temperatures.
* The ASTM G173 extraterrestrial spectrum is a synthesis of the Wehrli (1985)
  and Gueymard (2004) spectra; its 400–700 nm integral is 530 W m⁻².
* Rayleigh scattering is computed analytically (Bodhaine et al. 1999) rather
  than tabulated.
