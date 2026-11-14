"""Shared helpers for the experiment themes.

Themes own their output: every script derives its ``figures/`` and ``out/``
directories from its own ``__file__`` (see the ``FIG``/``OUT`` pattern in the
theme scripts), and :func:`save` takes an explicit destination path -- there
is no global root-figures default and no import-time directory creation.
Production-representative figures are written to the root ``figures/`` only
by explicit choice (task-4 curation), never by these helpers' defaults.
"""
from __future__ import annotations

import pathlib

import numpy as np

from skycolor import color
from skycolor.color import K_M
from skycolor.constants import WL_COARSE

ROOT = pathlib.Path(__file__).resolve().parents[1]

WL = WL_COARSE


def save(fig, path):
    """Save a matplotlib figure to an explicit destination path."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"  figure -> {shown}")
    return path


def strip(images, gap=4):
    """Vertically concatenate images with a thin separator."""
    if gap <= 0:
        return np.concatenate(images, axis=0)
    sep = np.zeros((gap, images[0].shape[1], 3))
    out = []
    for i, im in enumerate(images):
        if i:
            out.append(sep)
        out.append(im)
    return np.concatenate(out, axis=0)


def row(images, gap=4):
    if gap <= 0:
        return np.concatenate(images, axis=1)
    sep = np.zeros((images[0].shape[0], gap, 3))
    out = []
    for i, im in enumerate(images):
        if i:
            out.append(sep)
        out.append(im)
    return np.concatenate(out, axis=1)


def chromaticity_of(rad, wl=WL):
    xyz = color.spectrum_to_xyz(rad, wl)
    return color.chromaticity(xyz), K_M * xyz[..., 1]


def panel_label(rgb, text, size=1):
    """Draw a small text label onto an image (no font dependency: 3x5 bitmap)."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(color.to_uint8(rgb)).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 8, 12], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im, dtype=float) / 255.0
