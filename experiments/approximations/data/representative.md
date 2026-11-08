# Actual production calls

Production version: `6627c000a558c5503e38c32814c158b6cce22ac7`

Order: plane, hybrid, single, MC seeds 7/17/27 (30°), then 3°.
First call cold; later calls share warmed optical caches. No fair speed ranking.
MC: qualitative / budget-limited; concentration diagnostics not provided.
stderr includes smoother bias floor, not total physical error or Lv confidence.

Medium: `{'preset': 'clear', 'aod550': 0.05, 'ozone_du': 300.0, 'no2_column': 0.0, 'refraction': True, 'z_bounds_km': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.5, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 38.0, 41.0, 44.0, 47.0, 50.0, 53.0, 56.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 150.0, 160.0, 175.0, 200.0], 'albedo': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 'observer_km': 0.0, 'aerosol': 'HinterlandAerosol; exponential H=1.5 km'}`

Display: `{'camera': 'perspective', 'size': (640, 320), 'hfov_deg': 100.0, 'pitch_deg': 14.0, 'yaw_deg': 0.0, 'exposure': 0.03, 'tonemap': 'aces', 'adaptation': None, 'sun_disk': False, 'ground': False}`

## e30_plane-sos

Wall time: 16.007754 s

```text
approximation: plane-sos (method 'clear')
sun: zenith 60.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  multiple: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
  single: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
effective: sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}
sos: orders=10 converged=yes
```

options: `{'sun_zenith_deg': 60.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': True, 'approximation': 'plane-sos', 'sos': {}}`

effective: `{'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}}`

diagnostics: `{'orders': 10, 'converged': True}`

approximations: `{'multiple': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}, 'single': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}}`

## e30_hybrid-sos

Wall time: 9.718124 s

```text
approximation: hybrid-sos (method 'clear')
sun: zenith 60.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  single: rt.single.sky_radiance (source_refraction=True)
    scattering: single scattering, exact spherical-shell quadrature
    solar_source: unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)
    horizontal_field: applied to view-path coefficients at every point
    refraction: solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)
  multiple: rt.multiple.solve (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
effective: single {'n_gauss': 3, 'chunk': 512}; sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}
sos: orders=10 converged=yes
```

options: `{'sun_zenith_deg': 60.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': True, 'approximation': 'hybrid-sos', 'n_gauss': 3, 'chunk': 512, 'sos': {}}`

effective: `{'single': {'n_gauss': 3, 'chunk': 512}, 'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 10, 'converged': True}}`

diagnostics: `{'orders': 10, 'converged': True}`

approximations: `{'single': {'kernel': 'rt.single.sky_radiance', 'approximations': {'scattering': 'single scattering, exact spherical-shell quadrature', 'solar_source': 'unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)', 'horizontal_field': 'applied to view-path coefficients at every point', 'refraction': 'solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)'}, 'source_refraction': True}, 'multiple': {'kernel': 'rt.multiple.solve', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}}`

## e30_spherical-single

Wall time: 8.399124 s

```text
approximation: spherical-single (method 'clear')
sun: zenith 60.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  single: rt.single.sky_radiance (source_refraction=True)
    scattering: single scattering, exact spherical-shell quadrature
    solar_source: unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)
    horizontal_field: applied to view-path coefficients at every point
    refraction: solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)
effective: single {'n_gauss': 3, 'chunk': 512}
```

options: `{'sun_zenith_deg': 60.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': False, 'approximation': 'spherical-single', 'n_gauss': 3, 'chunk': 512}`

effective: `{'single': {'n_gauss': 3, 'chunk': 512}}`

diagnostics: `{}`

approximations: `{'single': {'kernel': 'rt.single.sky_radiance', 'approximations': {'scattering': 'single scattering, exact spherical-shell quadrature', 'solar_source': 'unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)', 'horizontal_field': 'applied to view-path coefficients at every point', 'refraction': 'solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)'}, 'source_refraction': True}}`

## e30_spherical-mc_s7

Wall time: 22.229264 s

```text
approximation: spherical-mc (method 'twilight')
sun: zenith 60.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  single: rt.single.sky_radiance (source_refraction=True)
    scattering: single scattering, exact spherical-shell quadrature
    solar_source: unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)
    horizontal_field: applied to view-path coefficients at every point
    refraction: solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)
  multiple: rt.twilight.multiple_scattered (source_refraction=True)
    scattering: split estimator: exact spherical first order + MC higher orders
    solar_source: unified finite disk for both terms; full and anchor wavelength subsets share one geometry per medium token
    horizontal_field: first-order term only
    ms_geometry: Monte Carlo assumes a horizontally uniform atmosphere
    mc_refraction: none (straight-line photon transport)
    ss_refraction: unified refracted solar source in the first-order term
effective: single {'n_gauss': 3, 'chunk': 512}; mc {'quality': 'low', 'seed': 7, 'grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000}, 'n_zen_run': 3, 'n_az_run': 4, 'n_anchor_run': 5, 'n_photons_run': 200004, 'bandwidth': 0.001}
mc: photons_run=200004 anchors_run=5 coarse grid 3x4 seed 7
mc stderr (W m-2 sr-1 nm-1): median 6.127e-04, max 2.920e-03 -- composite estimate: the MC standard error propagated through the angular smoother and wavelength interpolation plus a ~3% smoother resolution-bias floor (in quadrature); not the total physical error of the model, and the photon budget is not tail coverage
```

options: `{'sun_zenith_deg': 60.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': True, 'approximation': 'spherical-mc', 'n_gauss': 3, 'chunk': 512, 'quality': 'low', 'seed': 7, 'grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000}}`

effective: `{'single': {'n_gauss': 3, 'chunk': 512}, 'mc': {'quality': 'low', 'seed': 7, 'grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000}, 'n_zen_run': 3, 'n_az_run': 4, 'n_anchor_run': 5, 'n_photons_run': 200004, 'bandwidth': 0.001}}`

diagnostics: `{'undetected_wl': [], 'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000, 'anchor_wl': [380.0, 480.0, 580.0, 680.0, 780.0], 'bandwidth': 0.001, 'n_anchor_run': 5, 'n_photons_run': 200004, 'n_zen_run': 3, 'n_az_run': 4, 'coarse_zenith': [0.0, 0.0, 0.0, 0.0, 83.0, 83.0, 83.0, 83.0, 86.0, 86.0, 86.0, 86.0], 'coarse_azimuth': [0.0, 60.0, 120.0, 180.0, 0.0, 60.0, 120.0, 180.0, 0.0, 60.0, 120.0, 180.0], 'coarse_values': [[0.01976569070435713, 0.018590541315013226, 0.019515705529020953, 0.01977577670698964, 0.0630523628040831, 0.053489214249751685, 0.05023028330889635, 0.053642360276763695, 0.0626815234106683, 0.053121545380464846, 0.049893136408527064, 0.05340369642449406], [0.01098467463133934, 0.010110715893389275, 0.010702342202575587, 0.010549765940342883, 0.08063967420934053, 0.061433483356943576, 0.054684362301325014, 0.05907744246014777, 0.09733599794040738, 0.07018765403493231, 0.06296248897723529, 0.06669617427689668], [0.0033921837760817336, 0.0037386710940546207, 0.003128222264845038, 0.0033275505619578946, 0.042146144172606766, 0.02794864480832337, 0.023511906664310803, 0.025318711494762038, 0.05320096838672556, 0.03592182670127528, 0.030161193021824535, 0.033465387204250144], [0.0015051284711790334, 0.0015531785061103146, 0.0013188553574232768, 0.0013834330071399192, 0.020857580957335795, 0.014039061947529744, 0.010968351460961501, 0.011980903528715628, 0.03006488609971367, 0.01945145702805061, 0.0158557793065027, 0.01694641135105052], [0.0006189110153721056, 0.0006233325329185943, 0.0007327145095516268, 0.0005672476668586584, 0.010068672745563922, 0.0068737736603010306, 0.005642047640271508, 0.005805393355444642, 0.015001610729652183, 0.010317378519377737, 0.00827378845501203, 0.008188482577128599]], 'coarse_errors': [[0.0004133840003810578, 0.00038194203908202177, 0.0003986140004252608, 0.0004101468986164778, 0.0006538637237018704, 0.0005692372550166918, 0.0005440401737440699, 0.0005632565243319974, 0.0006374879368126777, 0.0005626315216578772, 0.0005343324731321145, 0.0005618561656753179], [0.00038557546044791784, 0.0003667636607890722, 0.0003771527136131023, 0.00037605834294534315, 0.0011600754148653998, 0.0008656874298019289, 0.0007415613686865002, 0.0008081488422763069, 0.001272692305534443, 0.0008642682918212409, 0.000788866141153973, 0.0008235374201549916], [0.00019338311080205383, 0.00020614386147474367, 0.00017396651129862893, 0.00018810483434334656, 0.0008516252118566295, 0.0005414253868761097, 0.0004187477217115497, 0.0004507974235978597, 0.0009442757124495092, 0.0005724693367260069, 0.0004623663388056717, 0.0005174155487132494], [0.00011503062769935507, 0.0001393035761033633, 0.0001030058940229865, 0.00010383590038978862, 0.0005818809100361484, 0.00035396439296359077, 0.00026387918752601464, 0.00027736629649088726, 0.0006822838236943401, 0.00040622600665633074, 0.0003034529829147502, 0.0003286112180737187], [6.974185052040433e-05, 6.890531449113252e-05, 8.522663823260491e-05, 6.0158570281876724e-05, 0.00034931524623897566, 0.00022081195667874638, 0.0001587755359053956, 0.00017863511684214323, 0.0004350741847178615, 0.00026094382346367526, 0.0001845590373357346, 0.00018976554932747646]]}`

approximations: `{'single': {'kernel': 'rt.single.sky_radiance', 'approximations': {'scattering': 'single scattering, exact spherical-shell quadrature', 'solar_source': 'unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)', 'horizontal_field': 'applied to view-path coefficients at every point', 'refraction': 'solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)'}, 'source_refraction': True}, 'multiple': {'kernel': 'rt.twilight.multiple_scattered', 'approximations': {'scattering': 'split estimator: exact spherical first order + MC higher orders', 'solar_source': 'unified finite disk for both terms; full and anchor wavelength subsets share one geometry per medium token', 'horizontal_field': 'first-order term only', 'ms_geometry': 'Monte Carlo assumes a horizontally uniform atmosphere', 'mc_refraction': 'none (straight-line photon transport)', 'ss_refraction': 'unified refracted solar source in the first-order term'}, 'source_refraction': True}}`

## e30_spherical-mc_s17

Wall time: 23.088575 s

```text
approximation: spherical-mc (method 'twilight')
sun: zenith 60.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  single: rt.single.sky_radiance (source_refraction=True)
    scattering: single scattering, exact spherical-shell quadrature
    solar_source: unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)
    horizontal_field: applied to view-path coefficients at every point
    refraction: solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)
  multiple: rt.twilight.multiple_scattered (source_refraction=True)
    scattering: split estimator: exact spherical first order + MC higher orders
    solar_source: unified finite disk for both terms; full and anchor wavelength subsets share one geometry per medium token
    horizontal_field: first-order term only
    ms_geometry: Monte Carlo assumes a horizontally uniform atmosphere
    mc_refraction: none (straight-line photon transport)
    ss_refraction: unified refracted solar source in the first-order term
effective: single {'n_gauss': 3, 'chunk': 512}; mc {'quality': 'low', 'seed': 17, 'grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000}, 'n_zen_run': 3, 'n_az_run': 4, 'n_anchor_run': 5, 'n_photons_run': 200004, 'bandwidth': 0.001}
mc: photons_run=200004 anchors_run=5 coarse grid 3x4 seed 17
mc stderr (W m-2 sr-1 nm-1): median 6.088e-04, max 2.929e-03 -- composite estimate: the MC standard error propagated through the angular smoother and wavelength interpolation plus a ~3% smoother resolution-bias floor (in quadrature); not the total physical error of the model, and the photon budget is not tail coverage
```

options: `{'sun_zenith_deg': 60.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': True, 'approximation': 'spherical-mc', 'n_gauss': 3, 'chunk': 512, 'quality': 'low', 'seed': 17, 'grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000}}`

effective: `{'single': {'n_gauss': 3, 'chunk': 512}, 'mc': {'quality': 'low', 'seed': 17, 'grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000}, 'n_zen_run': 3, 'n_az_run': 4, 'n_anchor_run': 5, 'n_photons_run': 200004, 'bandwidth': 0.001}}`

diagnostics: `{'undetected_wl': [], 'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000, 'anchor_wl': [380.0, 480.0, 580.0, 680.0, 780.0], 'bandwidth': 0.001, 'n_anchor_run': 5, 'n_photons_run': 200004, 'n_zen_run': 3, 'n_az_run': 4, 'coarse_zenith': [0.0, 0.0, 0.0, 0.0, 83.0, 83.0, 83.0, 83.0, 86.0, 86.0, 86.0, 86.0], 'coarse_azimuth': [0.0, 60.0, 120.0, 180.0, 0.0, 60.0, 120.0, 180.0, 0.0, 60.0, 120.0, 180.0], 'coarse_values': [[0.01985041610296683, 0.018943404389184845, 0.019798603351956896, 0.01969901207129225, 0.062324998589320495, 0.0542606253014809, 0.05062370900814098, 0.05487519165345297, 0.06328958163282639, 0.052404599411435854, 0.05038820837566779, 0.05343146381501733], [0.009874716246461709, 0.01009870541285963, 0.010736570679836584, 0.01071217073799542, 0.07982495467573482, 0.061163913647577034, 0.05519226776702642, 0.059089213072773324, 0.09763766117853749, 0.07140816569098542, 0.06289808331989975, 0.06685840252106709], [0.003289941412928738, 0.003464696773046884, 0.003314249766585441, 0.003259582257329519, 0.040032623141356034, 0.02820167653507533, 0.02384124428569854, 0.025481847093691053, 0.0543267439510759, 0.03492772899730698, 0.03074657141990268, 0.033277625893541975], [0.001580463989710819, 0.0013619557908080338, 0.0013854592686367165, 0.0013866418917205282, 0.020408837978977915, 0.013860188827183978, 0.01124691718889826, 0.012165517059657523, 0.02871908048090577, 0.02003729256578546, 0.015621553965184219, 0.016631540931673596], [0.0006532231228166531, 0.0005530495840856293, 0.0006108420320606556, 0.0006002256933205474, 0.01029764323807778, 0.0071470475039792595, 0.0054280177887349335, 0.0054362140613161735, 0.015052724915719427, 0.010165560454263657, 0.008246422669361817, 0.00864441351814837]], 'coarse_errors': [[0.00040221962962864115, 0.0003925119870926162, 0.000413555466055348, 0.0004008006786562229, 0.0006356889196321064, 0.0005677848730094474, 0.0005439948934607917, 0.0005721957992813681, 0.0006514564512817933, 0.0005571936784856885, 0.0005382988397536931, 0.000567436507900802], [0.0003667899921724661, 0.0003689354566781137, 0.00037511994975470005, 0.0003844782390730152, 0.0011317187822248974, 0.000839559557911069, 0.0007610081816817102, 0.000800195252286493, 0.0012708397015338882, 0.0008817008368393935, 0.0007950074749741875, 0.0008187478766766823], [0.00017508541136540597, 0.0001825448282907318, 0.0001687124967026112, 0.00017583162216996894, 0.0008058235408418648, 0.0005181862961580461, 0.00042836968561869816, 0.00046744356027395056, 0.0009582624742386841, 0.0005620721988661245, 0.00048140561753827525, 0.0005098587626692505], [0.00012177743186631937, 0.00011289716444130436, 0.00011264528008000308, 0.00010868058181583345, 0.0005611103302699422, 0.00035940045743104493, 0.000254229270862167, 0.0002870972787840978, 0.0006654982740657252, 0.0004280280638708364, 0.0002875865516587152, 0.00030887714570506037], [6.260719189739877e-05, 4.880882206061142e-05, 7.3858111840487e-05, 6.406152796791297e-05, 0.00036573308123774806, 0.00024152661697770094, 0.00015815452178062874, 0.0001583429895433206, 0.00042799969569780924, 0.0002582766854826072, 0.0001966845894321618, 0.0002139213457274124]]}`

approximations: `{'single': {'kernel': 'rt.single.sky_radiance', 'approximations': {'scattering': 'single scattering, exact spherical-shell quadrature', 'solar_source': 'unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)', 'horizontal_field': 'applied to view-path coefficients at every point', 'refraction': 'solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)'}, 'source_refraction': True}, 'multiple': {'kernel': 'rt.twilight.multiple_scattered', 'approximations': {'scattering': 'split estimator: exact spherical first order + MC higher orders', 'solar_source': 'unified finite disk for both terms; full and anchor wavelength subsets share one geometry per medium token', 'horizontal_field': 'first-order term only', 'ms_geometry': 'Monte Carlo assumes a horizontally uniform atmosphere', 'mc_refraction': 'none (straight-line photon transport)', 'ss_refraction': 'unified refracted solar source in the first-order term'}, 'source_refraction': True}}`

## e30_spherical-mc_s27

Wall time: 23.299156 s

```text
approximation: spherical-mc (method 'twilight')
sun: zenith 60.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  single: rt.single.sky_radiance (source_refraction=True)
    scattering: single scattering, exact spherical-shell quadrature
    solar_source: unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)
    horizontal_field: applied to view-path coefficients at every point
    refraction: solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)
  multiple: rt.twilight.multiple_scattered (source_refraction=True)
    scattering: split estimator: exact spherical first order + MC higher orders
    solar_source: unified finite disk for both terms; full and anchor wavelength subsets share one geometry per medium token
    horizontal_field: first-order term only
    ms_geometry: Monte Carlo assumes a horizontally uniform atmosphere
    mc_refraction: none (straight-line photon transport)
    ss_refraction: unified refracted solar source in the first-order term
effective: single {'n_gauss': 3, 'chunk': 512}; mc {'quality': 'low', 'seed': 27, 'grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000}, 'n_zen_run': 3, 'n_az_run': 4, 'n_anchor_run': 5, 'n_photons_run': 200004, 'bandwidth': 0.001}
mc: photons_run=200004 anchors_run=5 coarse grid 3x4 seed 27
mc stderr (W m-2 sr-1 nm-1): median 6.089e-04, max 2.887e-03 -- composite estimate: the MC standard error propagated through the angular smoother and wavelength interpolation plus a ~3% smoother resolution-bias floor (in quadrature); not the total physical error of the model, and the photon budget is not tail coverage
```

options: `{'sun_zenith_deg': 60.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': True, 'approximation': 'spherical-mc', 'n_gauss': 3, 'chunk': 512, 'quality': 'low', 'seed': 27, 'grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000}}`

effective: `{'single': {'n_gauss': 3, 'chunk': 512}, 'mc': {'quality': 'low', 'seed': 27, 'grid': {'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000}, 'n_zen_run': 3, 'n_az_run': 4, 'n_anchor_run': 5, 'n_photons_run': 200004, 'bandwidth': 0.001}}`

diagnostics: `{'undetected_wl': [], 'n_zen': 3, 'n_az': 4, 'n_anchor': 5, 'n_photons': 200000, 'anchor_wl': [380.0, 480.0, 580.0, 680.0, 780.0], 'bandwidth': 0.001, 'n_anchor_run': 5, 'n_photons_run': 200004, 'n_zen_run': 3, 'n_az_run': 4, 'coarse_zenith': [0.0, 0.0, 0.0, 0.0, 83.0, 83.0, 83.0, 83.0, 86.0, 86.0, 86.0, 86.0], 'coarse_azimuth': [0.0, 60.0, 120.0, 180.0, 0.0, 60.0, 120.0, 180.0, 0.0, 60.0, 120.0, 180.0], 'coarse_values': [[0.019382576756628626, 0.019729777287559567, 0.01968184703098281, 0.018707251903840174, 0.06266239594766958, 0.05421726337703548, 0.051947378635186496, 0.054443152535509166, 0.06380865787014978, 0.053575914352036125, 0.04991857325407674, 0.053681320460580655], [0.01019016435171384, 0.009715245628076769, 0.01014843224023761, 0.010161950306995588, 0.08045089531797475, 0.05952066424509269, 0.055148596808020714, 0.060035289920027955, 0.09621712936829077, 0.06954922134203657, 0.06228856839600842, 0.06774989730424198], [0.003145196500239015, 0.0033907135246906738, 0.0038080269622643606, 0.0031661749087148554, 0.03940596706764746, 0.027515314841368058, 0.02365752806544423, 0.02492972456838479, 0.05404128394259568, 0.03707780507332758, 0.031072944879683474, 0.03259751340973289], [0.0013347374869425754, 0.0015887227681618875, 0.0011687099928242265, 0.0011848686052782459, 0.021436785563714443, 0.013137354502456072, 0.011438254717909971, 0.011766544790642302, 0.029400356008534405, 0.01977342935862108, 0.015982881728181925, 0.01683950489908332], [0.0006511940016416966, 0.0006492373116235408, 0.000461758678865335, 0.0006054596591762694, 0.009268379057730234, 0.007337705485573464, 0.005509446127540732, 0.00562802296252123, 0.014964046576079333, 0.010383302503133116, 0.008371464845807466, 0.008682900949225006]], 'coarse_errors': [[0.00040173977223711835, 0.0004086407491260571, 0.000402614908672469, 0.0003858925112927429, 0.0006268761451503737, 0.0005693481148174636, 0.000558488112264673, 0.0005758745068068768, 0.0006554013907190548, 0.000575083631036613, 0.0005353505110527066, 0.0005644546778348274], [0.0003643156163113852, 0.00034710472671276314, 0.00035276217639491957, 0.0003665626781639209, 0.0011415455390808322, 0.0008218018616673891, 0.0007555629709765906, 0.000811810999333372, 0.0012668431855319224, 0.0008715267036440591, 0.0007802785240536319, 0.000859891728286795], [0.00017444364407394114, 0.00018398108728217704, 0.00020089185343448104, 0.00017152413613571999, 0.0008040844391581295, 0.000519408246907094, 0.000411807697977829, 0.00045092554674930395, 0.0009597115254563282, 0.0006103979855863105, 0.00046777588757319075, 0.0004978822277825586], [0.00012456491337961263, 0.00013040967252785185, 9.011479607370622e-05, 8.385840107392616e-05, 0.0005980003221691616, 0.00032821914269741994, 0.00027189104225423404, 0.000272079196411674, 0.0006709074652576575, 0.0004111008699036289, 0.0003057658902270876, 0.00032335489165123894], [6.978108788052214e-05, 6.330072648585951e-05, 4.9503789751631676e-05, 6.996652752992788e-05, 0.0003309941058192704, 0.0002377206540275118, 0.00015341411970288407, 0.0001617354626924865, 0.00042117736224197595, 0.00027894837587428656, 0.00018438372897542897, 0.00020848582094197733]]}`

approximations: `{'single': {'kernel': 'rt.single.sky_radiance', 'approximations': {'scattering': 'single scattering, exact spherical-shell quadrature', 'solar_source': 'unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)', 'horizontal_field': 'applied to view-path coefficients at every point', 'refraction': 'solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)'}, 'source_refraction': True}, 'multiple': {'kernel': 'rt.twilight.multiple_scattered', 'approximations': {'scattering': 'split estimator: exact spherical first order + MC higher orders', 'solar_source': 'unified finite disk for both terms; full and anchor wavelength subsets share one geometry per medium token', 'horizontal_field': 'first-order term only', 'ms_geometry': 'Monte Carlo assumes a horizontally uniform atmosphere', 'mc_refraction': 'none (straight-line photon transport)', 'ss_refraction': 'unified refracted solar source in the first-order term'}, 'source_refraction': True}}`

## e3_plane-sos

Wall time: 0.696856 s

```text
approximation: plane-sos (method 'clear')
sun: zenith 87.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  multiple: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
  single: rt.multiple.solve(single_view=True) (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
effective: sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 9, 'converged': True}
sos: orders=9 converged=yes
```

options: `{'sun_zenith_deg': 87.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': True, 'approximation': 'plane-sos', 'sos': {}}`

effective: `{'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 9, 'converged': True}}`

diagnostics: `{'orders': 9, 'converged': True}`

approximations: `{'multiple': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}, 'single': {'kernel': 'rt.multiple.solve(single_view=True)', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}}`

## e3_hybrid-sos

Wall time: 9.545410 s

```text
approximation: hybrid-sos (method 'clear')
sun: zenith 87.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  single: rt.single.sky_radiance (source_refraction=True)
    scattering: single scattering, exact spherical-shell quadrature
    solar_source: unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)
    horizontal_field: applied to view-path coefficients at every point
    refraction: solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)
  multiple: rt.multiple.solve (source_refraction=True)
    scattering: orders >= 2, plane-parallel discrete ordinates (delta-M)
    domain: ground fixed at z_bounds[0]; atmosphere.elevation_km is not read
    horizontal_field: not applied
    solar_source: unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments
    refraction: solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)
effective: single {'n_gauss': 3, 'chunk': 512}; sos {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 9, 'converged': True}
sos: orders=9 converged=yes
```

options: `{'sun_zenith_deg': 87.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': True, 'approximation': 'hybrid-sos', 'n_gauss': 3, 'chunk': 512, 'sos': {}}`

effective: `{'single': {'n_gauss': 3, 'chunk': 512}, 'sos': {'n_mu': 32, 'n_modes': 12, 'n_moments': 32, 'tol': 0.001, 'max_orders': 40, 'delta_m': True, 'n_moments_optics': 32, 'orders_run': 9, 'converged': True}}`

diagnostics: `{'orders': 9, 'converged': True}`

approximations: `{'single': {'kernel': 'rt.single.sky_radiance', 'approximations': {'scattering': 'single scattering, exact spherical-shell quadrature', 'solar_source': 'unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)', 'horizontal_field': 'applied to view-path coefficients at every point', 'refraction': 'solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)'}, 'source_refraction': True}, 'multiple': {'kernel': 'rt.multiple.solve', 'approximations': {'scattering': 'orders >= 2, plane-parallel discrete ordinates (delta-M)', 'domain': 'ground fixed at z_bounds[0]; atmosphere.elevation_km is not read', 'horizontal_field': 'not applied', 'solar_source': 'unified finite disk, per-beam apparent direction and amplitude projected onto the Fourier source; raw-extinction sun T with delta-M reserved for the diffuse moments', 'refraction': 'solar side fully refracted; plane-parallel diffuse transport stays straight (retained approximation)'}, 'source_refraction': True}}`

## e3_spherical-single

Wall time: 8.745905 s

```text
approximation: spherical-single (method 'clear')
sun: zenith 87.000 deg, azimuth 0.000 deg (input direction pre-refraction)
source-side refraction: on
components:
  single: rt.single.sky_radiance (source_refraction=True)
    scattering: single scattering, exact spherical-shell quadrature
    solar_source: unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)
    horizontal_field: applied to view-path coefficients at every point
    refraction: solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)
effective: single {'n_gauss': 3, 'chunk': 512}
```

options: `{'sun_zenith_deg': 87.0, 'sun_azimuth_deg': 0.0, 'refraction': True, 'single': True, 'multiple': False, 'approximation': 'spherical-single', 'n_gauss': 3, 'chunk': 512}`

effective: `{'single': {'n_gauss': 3, 'chunk': 512}}`

diagnostics: `{}`

approximations: `{'single': {'kernel': 'rt.single.sky_radiance', 'approximations': {'scattering': 'single scattering, exact spherical-shell quadrature', 'solar_source': 'unified finite disk (apparent directions, continuous refracted columns, occultation, local gain)', 'horizontal_field': 'applied to view-path coefficients at every point', 'refraction': 'solar side fully refracted; straight viewing rays and geometric view geometry (retained approximation)'}, 'source_refraction': True}}`
