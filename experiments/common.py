"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skycolor import color, render  # noqa: E402
from skycolor.constants import WL_COARSE  # noqa: E402

FIG = ROOT / "figures"
OUT = ROOT / "experiments" / "out"
FIG.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

WL = WL_COARSE


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  figure -> {path.relative_to(ROOT)}")
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
    return color.chromaticity(xyz), 683.0 * xyz[..., 1]


def panel_label(rgb, text, size=1):
    """Draw a small text label onto an image (no font dependency: 3x5 bitmap)."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(color.to_uint8(rgb)).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 8, 12], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im, dtype=float) / 255.0
