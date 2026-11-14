"""Cross-kernel Mie agreement: ``cloud.py`` vs ``mie.py`` (audit fork-guard).

``skycolor.cloud`` deliberately reimplements the Mie kernel instead of
importing ``skycolor.mie`` (cloud module docstring: a cloud model must not
silently change when an aerosol model is refactored).  A deliberate fork
drifts when one side is fixed alone, so this test pins the two kernels to
each other: qext/qsca from both implementations must agree on a common
(x, m) grid spanning the series regime through deep-recurrence territory
(both sides truncate at the shared Bohren & Huffman criterion
``n_max = x + 4x**(1/3) + 2``).

Tier declaration (author-at-write-time, DEVELOPMENT.md "Test tiers"):
``kind:contract`` (kernel-to-kernel agreement anchor, small numeric grid)
and ``cost:`` est. <0.5 s (32 vectorised pairs) -- fast tier, no marker.
"""
import numpy as np

from skycolor import cloud, mie

#: size parameters: small (few series terms) through large; the aerosol
#: library itself operates below x ~ 200 (mie.py docstring), 300 exercises
#: the recurrences a bit beyond that.
_X = np.array([1.0, 2.5, 5.0, 10.0, 21.0, 50.0, 120.0, 300.0])
#: refractive indices: weak contrast, liquid water, ice, absorbing.
_M = [1.05 + 0.0j, 1.333 + 0.0j, 1.31 + 0.0j, 1.5 + 0.1j]

#: Declared tolerance (measured 2026-11-14): the two kernels arrange the same
#: Bohren & Huffman recurrences differently, so agreement is close-but-not-
#: bitwise -- worst observed relative deviation on this grid is 3.9e-05
#: (x = 120, m = water); the live worst deviation is printed on failure.
#: 1e-4 leaves ~2.6x headroom over the measured value while staying far
#: tighter than any physical use of qext/qsca; a real single-side kernel fix
#: moves coefficients by orders of magnitude more than this.
RTOL = 1e-4


def test_cloud_and_library_mie_kernels_agree():
    """qext/qsca from both kernels agree on the shared (x, m) grid."""
    # the library batch API is per-pair (equal-length 1-D x, m): flatten the
    # grid x-major, m-minor and mirror that order on the cloud side.
    x_pairs = np.repeat(_X, len(_M))
    m_pairs = np.tile(np.asarray(_M), len(_X))
    library = mie.mie_coefficients_batch(x_pairs, m_pairs)
    cloud_flat = np.stack([cloud.mie_coefficients(_X, m)["qext"] for m in _M]).T \
        .ravel(), np.stack([cloud.mie_coefficients(_X, m)["qsca"] for m in _M]).T.ravel()
    cloud_qext, cloud_qsca = cloud_flat
    de = np.abs(cloud_qext - library.qext)
    ds = np.abs(cloud_qsca - library.qsca)
    assert np.allclose(cloud_qext, library.qext, rtol=RTOL, atol=1e-12) and \
        np.allclose(cloud_qsca, library.qsca, rtol=RTOL, atol=1e-12), (
        f"kernel fork drifted: worst pair (x={x_pairs[int(np.argmax(de + ds))]}, "
        f"m={m_pairs[int(np.argmax(de + ds))]}): "
        f"max |dqext| = {de.max():.3e} (rel {de.max() / np.abs(cloud_qext).max():.3e}), "
        f"max |dqsca| = {ds.max():.3e} (rel {ds.max() / np.abs(cloud_qsca).max():.3e})")
