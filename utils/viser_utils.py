import numpy as np
from matplotlib import colormaps


def define_track_colors(pts, colormap='turbo'):
    """
    Determines colors for each point in a set of 2D points using a colormap.

    Parameters:
    - pts: List of points [(x, y, z), ...]
    - colormap: Name of the colormap to use

    Returns:
    - colors: List of colors for each point
    """
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    if np.all(maxs == mins):
        maxs = mins + 1
    pts_norm = (pts - mins) / (maxs - mins)
    orders = np.argsort(np.argsort([np.square(pt).sum() for pt in pts_norm])) / max((len(pts) - 1), 1)
    if len(orders) == 1:
        orders = np.array([0.5])
    return np.array([colormaps[colormap](order)[:3] for order in orders])
