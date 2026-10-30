# Aerosol optics: Mie theory, analytic moments and delta-M

This document describes the aerosol model introduced on `feat/aerosol`:

* the Mie engine and the **analytic** Legendre moments of the phase function;
* the **adaptive phase-function table** used by the single-scattering term;
* the **delta-M** transformation and how it plugs into the SOS solver;
* the **named aerosol library** and its measured optical properties;
* the validation numbers and the remaining approximations.

The public interface is unchanged: every aerosol still exposes
`relative_coefficients(wl)`, `phase(mu, wl)`, `ssa(wl)`, `asymmetry(wl)` and
`legendre_moments(wl, n)`.  `AngstromAerosol` and `MieAerosol` remain
importable, and the old ad-hoc names (`HinterlandAerosol`, `HazeAerosol`,
`DustAerosol`, `SmokeAerosol`, `UrbanAerosol`, `MaritimeAerosol`) now delegate
to the physical library.

---

## 1. Mie theory

For a homogeneous sphere of radius `r`, size parameter `x = 2 pi r / lambda`
and complex refractive index `m = n + i k` (`k >= 0` for absorption), the
scattering amplitudes are

```
S1(theta) = sum_n (2n+1)/(n(n+1)) [ a_n pi_n(cos theta) + b_n tau_n(cos theta) ]
S2(theta) = sum_n (2n+1)/(n(n+1)) [ a_n tau_n(cos theta) + b_n pi_n(cos theta) ]
```

with the usual angular functions `pi_n = P_n^1/sin theta` and
`tau_n = dP_n^1/dtheta`.  The efficiency factors are

```
Q_ext = (2/x^2) sum_n (2n+1) Re(a_n + b_n)
Q_sca = (2/x^2) sum_n (2n+1) (|a_n|^2 + |b_n|^2)
```

and the phase function is normalised so that `int p dOmega = 4 pi`:

```
p(theta) = 2 (|S1|^2 + |S2|^2) / (x^2 Q_sca).          (1)
```

The factor 2 is the convention used throughout this project (it matches
`skycolor.spectra.rayleigh_phase`).  `skycolor/mie.py` implements the
Bohren & Huffman (1983) algorithm: the logarithmic derivative `D_n(mx)` by
downward recurrence (always stable), `psi_n(x)` and `chi_n(x)` by upward
recurrence, truncation at `nmax = x + 4 x^(1/3) + 2`.

The asymmetry parameter is computed analytically from the coefficients
(Bohren & Huffman eq. 4.74), not by quadrature:

```
g = (4 / (x^2 Q_sca)) sum_n [ n(n+2)/(n+1) Re(a_n a*_{n+1} + b_n b*_{n+1})
                             + (2n+1)/(n(n+1)) Re(a_n b*_n) ].
```

This is exact, and it is the reason `g` can no longer exceed 1.  The old code
evaluated `g` by quadrature on a 96-point grid and, for a 8 um droplet, got
`g = 3.57` because the grid missed the forward peak; the analytic value is
`g = 0.8613`.

---

## 2. Analytic Legendre moments

### 2.1 Why quadrature is not enough

The phase function of a sphere of size parameter `x` has a forward diffraction
peak of width `~1/x` radians.  At `x = 200` (r = 17.5 um at 550 nm) the peak is
`0.29 deg` wide; at `x = 2 pi * 8 / 0.55 = 91` (r = 8 um) it is `0.63 deg`.
A fixed grid in `mu = cos theta` cannot resolve it: `mu = 1` is a single point,
and the nearest neighbour of a 721-point grid is `mu = 0.9972`, i.e. `4.3 deg`
away.  Linear interpolation then either misses the peak or, worse, spreads its
energy over one grid cell and produces a spurious `g > 1`.

The moments can, however, be obtained **exactly** from the Mie coefficients,
because the phase function is a polynomial in `mu`.

### 2.2 The expansion

Write `c_n = (2n+1)/(n(n+1))`.  The derivative expansion of Legendre
polynomials gives

```
pi_n(mu) = P_n'(mu) = sum_{a = n-1, n-3, ..., >= 0} (2a+1) P_a(mu).      (2)
```

`tau_n = n mu pi_n - (n+1) pi_{n-1}` therefore also has a finite, sparse
expansion in Legendre polynomials.  Define the coefficients `A_na`, `B_na` by

```
pi_n = sum_a A_na P_a,        tau_n = sum_a B_na P_a.
```

`A_na = 2a+1` for `a = n-1, n-3, ...`, else 0.  `B_na` follows from the
recurrence `mu P_k = (k+1)/(2k+1) P_{k+1} + k/(2k+1) P_{k-1}`; it is non-zero
only for `a = n, n-2, ...` and is given explicitly by

```
B_na = n [ a 1_{a-1 in S_n} + (a+1) 1_{a+1 in S_n} ]
       - (n+1)(2a+1) 1_{a in S_{n-1}},
S_n = {n-1, n-3, ...}.
```

Now define the Legendre coefficients of the two amplitudes:

```
P_a = sum_n c_n (a_n A_na + b_n B_na),
Q_a = sum_n c_n (b_n A_na + a_n B_na).                                   (3)
```

Then `S1 = sum_a P_a P_a(mu)` and `S2 = sum_a Q_a P_a(mu)` **exactly**.  The
phase function is

```
p(mu) = (2/(x^2 Q_sca)) sum_{a,b} [P_a P_b* + Q_a Q_b*] P_a(mu) P_b(mu).
```

Multiply by `P_l(mu)` and integrate.  The integral of three Legendre
polynomials is a squared Wigner 3-j symbol:

```
(1/2) int_{-1}^{1} P_a P_b P_l dmu = <a b l; 0 0 0>^2 =: G_{abl}.        (4)
```

Hence the central result:

```
beta_l = (2/(x^2 Q_sca)) sum_{a,b} G_{abl} [P_a P_b* + Q_a Q_b*].        (5)
```

The selection rules `|a-b| <= l`, `a+b+l` even, and `a+b >= l` restrict the
double sum to a band of width `2l` around the diagonal, so the cost is
`O(nmax L^2)` rather than `O(nmax^2 L)`.  The 3-j symbol is evaluated from
the closed form

```
<a b l; 0 0 0> = (-1)^{J/2} sqrt((J-2a)!(J-2b)!(J-2l)!/(J+1)!)
                 * (J/2)! / [(J/2-a)! (J/2-b)! (J/2-l)!],   J = a+b+l.
```

`P_a` and `Q_a` are computed in `O(nmax)` with the tail sums

```
S^a_k = sum_{n >= k, n = k mod 2} c_n a_n,     S^b_k likewise,
P_a = (2a+1)(S^a_{a+1} - S^b_{a+2}) + a^2 c_a b_a,
Q_a = (2a+1)(S^b_{a+1} - S^a_{a+2}) + a^2 c_a a_a,
```

with `c_a = a_a = b_a = 0` outside `1..nmax`.

### 2.3 Properties and accuracy

* `beta_0 = 1` analytically (it is the normalisation).  The implementation sets
  it exactly to 1 after the size average, so delta-M is exactly energy
  conserving.
* `beta_1 = g`, by construction, and it equals the independent Bohren-Huffman
  formula to better than `1e-10`.
* `-1 < beta_l < 1` for every `l`; no quadrature can produce `g > 1`.
* For a single sphere the formula is exact.  The validation compares it with
  an independent Gauss-Legendre quadrature (which is exact for the polynomial
  of degree `~2 nmax`) for `x = 0.5 ... 200`; the two agree to better than
  `2e-8` for `l <= 8` (the residual is dominated by the conditioning of the
  reference quadrature at large `x`, not by the analytic formula).

The size average is the incoherent, `Q_sca * area`-weighted average of the
single-sphere moments, exactly as the phase function itself is averaged.

---

## 3. Adaptive phase-function table

The single-scattering integrator (`skycolor.rt.single`) evaluates
`aerosol.phase(cos_theta, wl)` at arbitrary angles, so the aerosol needs an
accurate *angular* representation of the size-averaged phase function.

### 3.1 Grid

`skycolor.mie.phase_table_angles(x_max)` builds, in degrees,

```
theta = 0,
        logspace(1e-4, 1, n_forward, endpoint=False),
        linspace(1, 179, n_mid),
        180 - logspace(1, 1e-4, n_back),
        180,
n_forward = clip(3 x_max, 80, 800),
n_mid     = clip(8 x_max, 240, 2400),
n_back    = 80.
```

The grid scales with the largest size parameter in the distribution: the
forward peak width is `~1/x_max`, and the interference structure in the middle
has period `~2 pi / x_max`.  For the water cloud at 450 nm (`x_max ~ 207`)
this is about 2300 angles.

### 3.2 Interpolation

The table stores `log p` on a monotone coordinate `u(theta)`:

```
u = log10(theta)                     theta < 1 deg
u = (theta - 1)/178 * 4              1 <= theta <= 179 deg
u = 4 + (-log10(180 - theta))/4      theta > 179 deg
```

and interpolates linearly in `u`.  This is log-log in the forward peak (where
the phase function is a power law at small angles), log-linear in the middle,
and log-log again in the last degree (the glory).  `p(0)` and `p(180)` are
stored exactly and returned outside the interpolable range.

### 3.3 Normalisation

`PhaseTable.normalisation()` integrates the table itself, and the validation
test integrates the *interpolated* table on a 4000-point Gauss-Legendre grid:
the latter is what the radiative-transfer code actually uses.  Across the
eight library members and wavelengths 450, 550 and 650 nm the worst deviation
from `4 pi` is `2.2e-4` (water cloud at 650 nm); every member is below
`1e-3`.  The node-trapezoid value can differ by up to `1e-3` because the
mid-angle Mie oscillations are under-resolved by the trapezoidal rule on the
node grid; the interpolation integral is the meaningful quantity.

Relative interpolation error against a direct Mie evaluation is typically
`1e-4` (median), `3e-3` at the 90th percentile, and up to `~2%` at a single
angle near the first diffraction sidelobe of a large droplet.  That is ample
for the aureole and the sky colour.

---

## 4. Delta-M

### 4.1 Transformation

Let `beta_0 ... beta_L` be the Legendre moments of the phase function,
`p(mu) = sum_l (2l+1) beta_l P_l(mu)`.  Truncating at `l = N` and using the
first omitted moment as the forward-scattered fraction,

```
f = beta_{N+1},
```

the delta-M approximation replaces the omitted forward peak by a delta
function and renormalises the retained moments:

```
beta*_l = (beta_l - f)/(1-f)     for l <= N,
tau*    = (1 - omega f) tau,
omega*  = omega (1-f)/(1-omega f).
```

### 4.2 Derivation

Write the phase function as a forward delta plus a smoother remainder:

```
p(mu) = (1-f) p*(mu) + 2 f delta(1-mu),
```

where the delta is normalised so that `int 2 delta(1-mu) dOmega = 4 pi`.
The Legendre moments of the delta are `P_l(1) = 1`, so

```
beta_l = (1-f) beta*_l + f   (l <= N),
```

which is the first line above.  For `l > N` the truncated expansion has no
moments, and delta-M *defines* them to be `f`: the delta carries exactly the
omitted forward energy.  Note that `beta*_0 = (1-f)/(1-f) = 1`.

For the medium, removing the forward delta from the scattering leaves an
effective scattering optical depth `tau_s* = (1-f) omega tau`; the absorption
`(1-omega) tau` is unchanged.  Hence the effective extinction seen by the
diffuse field is

```
tau* = (1-f) omega tau + (1-omega) tau = (1 - omega f) tau,
omega* = tau_s*/tau* = omega (1-f)/(1-omega f).
```

The forward-scattered light is not lost: it stays in the direct beam, which is
why `tau* < tau`.  For a conservative aerosol (`omega = 1`), `omega* = 1` and
`tau* = (1-f) tau`: the transformation only rescales the optical depth.

### 4.3 API

```python
from skycolor.aerosol import delta_m, get_aerosol

# moments beta_0 .. beta_L, shape (L+1, n_wl)
mom = aerosol.legendre_moments(wl, n_moments + 1)
res = delta_m(mom, n_moments, tau=tau, omega=omega)
res.beta        # (n_moments, n_wl), beta*_l
res.f           # (n_wl,), forward fraction
res.tau_star    # (n_wl,), or None
res.omega_star  # (n_wl,), or None
res.g_star      # (n_wl,), beta*_1

# the same thing as a method
res = aerosol.delta_m(wl, n_moments, tau=tau, omega=omega)
res = aerosol.sos_inputs(wl, n_moments, tau=tau, omega=omega)  # alias
```

`f` is clipped to `[0, 1-1e-12]` for numerical safety.  For every library
member the raw `beta_32` is already in `[0, 1)`, so the clip is inactive.

### 4.4 Integrating it into `Atmosphere.phase_moments` (do not edit here)

`skycolor/atmosphere.py` is owned by another workstream.  The minimal change
needed for the SOS solver is:

1. `Atmosphere.phase_moments(n)` should return `n+1` moments so that the
   `(N+1)`-th is available.  It already computes the scattering-weighted
   mixture of the components, which is exactly the layer-averaged phase
   function, so no new physics is needed.
2. For each layer and wavelength, take `f = beta[n]` (the last moment),
   renormalise `beta[:n] = (beta[:n] - f)/(1-f)`, and set `beta[0] = 1`.
3. Rescale the layer's optical depth and single-scattering albedo used by the
   SOS solver: `tau* = (1 - omega f) tau`, `omega* = omega (1-f)/(1-omega f)`.
   `f` must be computed per layer and per wavelength, because the aerosol /
   Rayleigh mixture changes with height.

A vectorised helper for the `(n_layer, n_moments, n_wl)` array is just

```python
def delta_m_layers(beta, n_moments):
    f = beta[:, n_moments, :]
    out = (beta[:, :n_moments, :] - f[:, None, :]) / (1.0 - f)[:, None, :]
    out[:, 0, :] = 1.0
    return out, f
```

The standalone `delta_m` in `skycolor/aerosol.py` is the 2-D version of this
(one spectrum at a time) and is already tested.

---

## 5. The aerosol library

The library lives at the bottom of `skycolor/aerosol.py` and is also exposed
through `skycolor.aerosol.LIBRARY` and `get_aerosol(name, **kwargs)`.  Every
member is a `MieAerosol` with a log-normal (or multi-modal) size distribution
and, where it matters, a wavelength-dependent complex refractive index.

### 5.1 Main table

Angstrom exponent from the 440/660 nm pair:
`alpha = -ln(tau(440)/tau(660)) / ln(440/660)`.
`SSA` and `g` are at 550 nm.  `f_32 = beta_32` is the first omitted moment
when the SOS solver keeps 32 moments (`l = 0..31`).

| name | modes (r_g um / sigma_g) | alpha_550 | SSA_550 | g_550 | tau450/tau650 | f_32 | delta-M |
|---|---|---|---|---|---|---|---|
| `rural` | 0.05 / 2.0 | 1.30 | 0.982 | 0.714 | 1.62 | <1e-4 | no |
| `urban` | 0.05 / 2.0 (0.75) + 0.04 / 2.0 (0.25, soot) | 1.25 | 0.901 | 0.704 | 1.59 | <1e-4 | no |
| `maritime` | 0.10 / 1.8 (0.997) + 1.5 / 1.9 (0.003) | 0.55 | 1.000 | 0.813 | 1.23 | 0.127 | recommended |
| `desert_dust` | 0.40 / 2.0 (0.35) + 2.0 / 2.2 (0.65) | -0.04 | 0.876 | 0.816 | 0.984 | 0.368 | required |
| `biomass_burning` | 0.05 / 1.8 | 1.84 | 0.865 | 0.586 | 1.97 | <1e-4 | no |
| `volcanic_stratospheric` | 0.5 / 1.6 | -0.31 | 1.000 | 0.710 | 0.892 | 0.004 | optional |
| `cirrus` | 6.4 / 1.35 (equivalent spheres, r_eff ~ 8 um) | -0.04 | 0.9998 | 0.868 | 0.986 | 0.381 | required |
| `water_cloud` | 6.74 / 1.3 (r_eff = 8 um) | -0.04 | 1.000 | 0.858 | 0.985 | 0.383 | required |

Notes:

* The radii in the table are the **dry** geometric mean radii.  The optical
  numbers are evaluated at the stated default humidity: rural/urban 50 %,
  maritime 50 %, biomass burning 30 %, dust/cirrus/cloud 0 % (or, for cloud,
  the droplets are already liquid water).
* `tau450/tau650 < 1` means the extinction is slightly larger in the red; the
  direct beam becomes relatively bluer after attenuation -- the "blue sun".
  Only desert dust is on the blue-sun side of unity in this library, and only
  weakly (0.984).  A *narrow* coarse mode (e.g. `r = 0.45 um`,
  `sigma_g = 1.05`) drives the ratio to about 0.71 and produces a much
  stronger blue sun; that is a size-distribution choice, not a different
  aerosol, and it is not part of the eight-member library.
* `biomass_burning` is strongly blue-absorbing (`tau450/tau650 = 1.97`); it is
  the opposite of a blue-sun aerosol.
* Delta-M is marked *required* when `f_32 > 0.2` (the 32-moment truncation
  loses more than 20 % of the scattering energy to the forward delta),
  *recommended* for `0.01 < f_32 < 0.2`, and *optional* below 0.01.

### 5.2 Provenance and humidity growth

* **`rural`** -- Shettle & Fenn (1979), water-soluble model; OPAC (Hess et al.
  1998) accumulation mode.  `f(RH) = (1-RH)^-0.35`, volume-weighted index.
* **`urban`** -- illustrative external mixture of OPAC water-soluble and soot;
  the soot number fraction (25 %) is chosen to reproduce the AERONET urban
  SSA of ~0.90 at 550 nm (Dubovik et al. 2002).
* **`maritime`** -- OPAC sea-salt modes + Shettle & Fenn maritime;
  `f(RH) = (1-RH)^-0.30`.  The coarse mode is only 0.3 % of the number but
  carries most of the mass.
* **`desert_dust`** -- OPAC mineral dust size distribution; the imaginary
  index is an illustrative iron-oxide model with a visible minimum and a
  near-IR rise.  Weakly hygroscopic (`gamma = 0.05`).
* **`biomass_burning`** -- illustrative, chosen to reproduce the measured
  smoke alpha (~1.8-2.2) and SSA (~0.80-0.88) (Dubovik et al. 2002; Eck et al.
  2005).  Brown-carbon absorption increases towards the blue.
* **`volcanic_stratospheric`** -- 75 % H2SO4/water droplets, r ~ 0.5 um
  (Palmer & Williams 1975; Stothers 1997; OPAC sulfate mode).  Nearly
  conservative; this is the layer that produces the red afterglows.
* **`cirrus`** -- **equivalent spheres**, r_eff ~ 8 um, ice Ih refractive
  index.  Real crystals are non-spherical; the approximation gets extinction,
  SSA and the gross forward scattering roughly right, but not halos or the
  backscatter glory.
* **`water_cloud`** -- liquid-water droplets, log-normal with
  `sigma_g = 1.3` and `r_eff = 8 um` (range 5-15 um).  Very strong forward
  peak; delta-M is mandatory.

The refractive-index models are simple fits, not laboratory tables:
water `n = 1.324 + 3200/lambda^2`, ice `1.305 + 1500/lambda^2`, sulfate
`1.430 + 3000/lambda^2`, sea salt `1.340 + 1200/lambda^2` (lambda in nm).
Dust and soot use the tabulated/parametric `Im(m)` described above.

### 5.3 Backwards-compatible names

| old name | now |
|---|---|
| `HinterlandAerosol()` | `RuralAerosol(rh=0.5)` |
| `HazeAerosol(rh)` | `RuralAerosol(rh=rh)` |
| `UrbanAerosol()` | `UrbanAerosol(rh=0.5)` (Mie) |
| `MaritimeAerosol()` | `MaritimeAerosol(rh=0.5)` (Mie) |
| `DustAerosol()` | `DesertDustAerosol()` |
| `SmokeAerosol()` | `BiomassBurningAerosol()` |

`AngstromAerosol` is unchanged (power-law extinction + Henyey-Greenstein), and
is still the right tool for fast sweeps and for controlled comparisons.

---

## 6. Validation

### 6.1 Single-sphere Mie benchmarks

Reference: Hong Du, *Mie-scattering calculation*, Appl. Opt. 43, 1951 (2004),
Table 1 (Qext, Qsca); `g` from Wiscombe's MIEV0 as tabulated by Blevins
(2010), Tables 4/24.  The imaginary-part sign convention is `m = n + i k` with
`k > 0` for absorption (the Blevins table prints the opposite sign).

| x | m | Qext (ref) | Qsca (ref) | g (ref) | max rel. error |
|---|---|---|---|---|---|
| 10 | 0.75 | 2.232265 | 2.232265 | 0.896473 | 7e-8 |
| 1000 | 0.75 | 1.997908 | 1.997908 | 0.844944 | 9e-8 |
| 1 | 1.33 + 1e-5i | 0.0939520 | 0.0939230 | 0.184517 | 3e-6 |
| 100 | 1.33 + 1e-5i | 2.10132 | 2.09659 | 0.868959 | 2.3e-5 |
| 1 | 1.5 + 1.0i | 2.33632 | 0.663454 | 0.192136 | 4e-7 |
| 100 | 1.5 + 1.0i | 2.09750 | 1.28370 | 0.850252 | 2.3e-6 |
| 1 | 10 + 10i | 2.53299 | 2.04941 | -0.110664 | 2.4e-6 |
| 100 | 10 + 10i | 2.07112 | 1.83679 | 0.556215 | 2.5e-6 |

Qext and Qsca agree to `<= 2.5e-5` relative, and `g` to `<= 2.5e-5`
absolute, over the whole set -- including the absorbing `10+10i` cases and
`x = 1000`.

### 6.2 Analytic moments

| x | m | max abs diff vs quadrature (l <= 8) |
|---|---|---|
| 0.5 | 1.53 + 0.006i | 5e-13 |
| 2 | 1.53 + 0.006i | 9e-13 |
| 10 | 1.5 + 0.1i | 2e-11 |
| 100 | 1.33 + 1e-5i | 2e-10 |
| 200 | 1.55 + 0.005i | 2e-9 |

`beta_0 = 1` to machine precision; `beta_1` equals the independent
Bohren-Huffman `g` to `<= 1e-10`; `-1 < beta_l < 1` for all tested `x`
(including `x = 500`).  The old fixed-grid value for a 8 um droplet,
`g = 3.57`, is gone; the analytic value is `0.8613`.

### 6.3 Phase-function normalisation

Dense Gauss-Legendre integration of the **interpolated** `phase()`:

| member | 450 nm | 550 nm | 650 nm |
|---|---|---|---|
| rural | 0.999964 | 0.999965 | 0.999968 |
| urban | 0.999966 | 0.999968 | 0.999971 |
| maritime | 1.000060 | 1.000057 | 1.000048 |
| desert_dust | 1.000061 | 1.000089 | 1.000115 |
| biomass_burning | 0.999981 | 0.999983 | 0.999986 |
| volcanic_stratospheric | 0.999872 | 0.999929 | 0.999953 |
| cirrus | 1.000026 | 0.999950 | 0.999872 |
| water_cloud | 0.999986 | 0.999876 | 0.999777 |

(Values are `int p dOmega / (4 pi)`; the exact numbers depend on the last
digits of the caching/ordering, but the worst case is `2.2e-4`.)  At 400 nm,
the largest size parameter in the library (`x_max ~ 335` for desert dust),
the normalisation is 1.00007 (dust), 1.00006 (maritime), 1.00007 (cirrus)
and 1.00004 (water cloud).

### 6.4 The forward peak versus Henyey-Greenstein

For the task's dust distribution (`r_g = 0.5 um`, `sigma_g = 2.2`,
`m = 1.55 + 0.005i`, size-averaged `g = 0.769`) the ratio of the Mie phase
function to HG with the same `g` is:

| theta | 1 deg | 5 deg | 10 deg | 30 deg |
|---|---|---|---|---|
| Mie / HG | 10.9 | 2.2 | 0.70 | 0.62 |

The qualitative conclusion is exactly the one that motivated this work: HG is
far too weak in the first degree and too broad at 10-30 deg.  The precise
numbers depend on the log-normal convention and the size quadrature; the
task's illustrative 5.9 / 1.7 / 0.8 / 0.8 and these 10.9 / 2.2 / 0.70 / 0.62
are the same physical statement.

### 6.5 delta-M

For every library member, `delta_m(..., 32)` gives `beta*_0 = 1` to machine
precision, `f = beta_32` as tabulated, and the exact analytic
`tau* / omega*` limits (checked in `tests/test_aerosol.py`).  The
conservative limit `omega = 1 -> omega* = 1` is exact.

---

## 7. Remaining approximations

* **Homogeneous spheres.**  Mie theory is exact for spheres; cirrus is
  represented by equivalent spheres, so halos and the backscatter glory are
  not modelled.  Dust and soot are treated as spheres with an effective index.
* **Size quadrature.**  Each mode is a truncated log-normal (`|t| <= 3`,
  hard limits `[1e-4, 30] um`) integrated with a trapezoidal rule in `ln r`
  (96-128 radii per mode).  For narrow distributions the Mie ripples in
  `Q_sca` are not fully resolved; `Q_sca` converges to about 0.1 %, which is
  why the documented SSA/alpha are quoted to three decimals.  The analytic
  moments are exact for each sphere, so this error only enters through the
  size weights.
* **External mixture.**  Different modes can have different refractive
  indices (urban soot + water-soluble), but there is no internal mixing or
  core-shell treatment.
* **Humidity growth.**  `r(RH) = r_dry (1-RH)^-gamma` with a volume-weighted
  (Wiener) effective index.  No deliquescence, hysteresis or kappa-Koehler
  treatment.
* **Refractive indices.**  Simple Cauchy fits for water/ice/sulfate/sea salt
  and an illustrative iron-oxide/brown-carbon model for dust/soot.  They
  reproduce the measured Angstrom exponents and SSAs, not every absorption
  band.
* **No polarisation.**  The scalar (unpolarised) Mie phase function is used;
  the polarised sky would need the full scattering matrix.
* **Phase-table resolution.**  The grid scales with `x_max` and is validated
  up to `x_max ~ 335` (desert dust at 400 nm); the analytic moments are
  validated to `x = 500`.  For much larger particles the table would need
  more points; the analytic moments remain exact, but the angular
  interpolation would degrade.
* **`f` clipping.**  `delta_m` clips `f` to `[0, 1)`; for the library the raw
  `beta_32` is already non-negative, so this is inactive, but a strongly
  oscillatory phase function could in principle have a negative `beta_{N+1}`.

---

## 8. Quick reference

```python
import numpy as np
from skycolor.aerosol import get_aerosol, delta_m

dust = get_aerosol("desert_dust")
wl = np.array([450.0, 550.0, 650.0])

ext, sca = dust.relative_coefficients(wl)   # normalised at 550 nm
ssa = dust.ssa(wl)
g = dust.asymmetry(wl)                      # analytic, in (-1, 1)
beta = dust.legendre_moments(wl, 33)        # (33, 3), beta_0 = 1

# delta-M inputs for the SOS solver
res = dust.sos_inputs(wl, 32, tau=0.3, omega=ssa)
res.beta, res.f, res.tau_star, res.omega_star, res.g_star
```

The library is also reachable as `skycolor.aerosol.LIBRARY` and through
`get_aerosol("rural", rh=0.8)`, `get_aerosol("water_cloud", r_eff_um=12)`,
etc.
