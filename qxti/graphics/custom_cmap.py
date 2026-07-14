"""Custom QXTI colormaps.

Sequential
----------
    custom_cmap   : turquoise → steel-blue → grey-purple → coral → peach
    custom_cmap_r : reversed

Diverging (white at zero)
-------------------------
    custom_cmap_div   : turquoise (neg) → white (0) → peach (pos)
    custom_cmap_div_r : reversed

Usage
-----
    from qxti.graphics.custom_cmap import custom_cmap, custom_cmap_div

    plt.imshow(data, cmap=custom_cmap)         # sequential
    plt.imshow(data, cmap=custom_cmap_div)     # diverging, white at 0
    plt.imshow(data, cmap='qxti_custom')       # by name
    plt.imshow(data, cmap='qxti_custom_div')   # diverging by name

All variants are registered with matplotlib on import.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# RGB 0-255 anchor colours for the sequential map (low → high)
_COLORS_RGB = [
    (108, 194, 189),   # 0.00 — turquoise
    (89,  129, 158),   # 0.25 — steel blue
    (124, 122, 162),   # 0.50 — grey-purple
    (246, 126, 125),   # 0.75 — coral pink
    (255, 192, 167),   # 1.00 — peach
]

_COLORS_NORM = [(r / 255, g / 255, b / 255) for r, g, b in _COLORS_RGB]

# Sequential
custom_cmap = LinearSegmentedColormap.from_list("qxti_custom", _COLORS_NORM, N=256)
custom_cmap_r = custom_cmap.reversed()

# Diverging: turquoise (negative) → white (zero) → peach (positive)
# Slightly saturated mid-tones for stronger colour punch while keeping white at 0.
_DIV_COLORS = [
    (108 / 255, 194 / 255, 189 / 255),   # turquoise       (most negative)
    (155 / 255, 215 / 255, 211 / 255),   # mid turquoise   (stronger than before)
    (1.0,       1.0,       1.0      ),   # white           (zero)
    (255 / 255, 208 / 255, 188 / 255),   # mid peach       (stronger than before)
    (255 / 255, 175 / 255, 140 / 255),   # deep peach      (most positive)
]

custom_cmap_div = LinearSegmentedColormap.from_list("qxti_custom_div", _DIV_COLORS, N=256)
custom_cmap_div_r = custom_cmap_div.reversed()


def _register() -> None:
    for cmap in (custom_cmap, custom_cmap_r, custom_cmap_div, custom_cmap_div_r):
        try:
            mpl.colormaps.register(cmap=cmap, name=cmap.name)
        except (AttributeError, ValueError):
            try:
                import matplotlib.pyplot as plt
                plt.register_cmap(cmap=cmap)
            except Exception:
                pass


_register()
