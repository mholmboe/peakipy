"""
Manual baseline defined by user-specified control points. 
"""

import numpy as np

try:
    from scipy.interpolate import CubicSpline
    HAS_CSPLINE = True
except Exception:
    HAS_CSPLINE = False


def manual_baseline(x, y, points=None, interp='linear', **kwargs):
    """
    Build a baseline from user control points and interpolate across x.

    Parameters
    ----------
    x : array_like
        X-axis data (used as evaluation grid)
    y : array_like
        Y-axis data (not used for baseline generation but kept for API consistency)
    points : list of tuple
        Control points as (x, y). Must have at least 2 points.
    interp : str, optional
        'linear' or 'cubic'. Cubic requires scipy CubicSpline; falls back to linear.

    Returns
    -------
    baseline : ndarray
        Interpolated baseline
    corrected : ndarray
        y - baseline
    points_sorted : ndarray
        Sorted control points array of shape (n, 2)
    """
    if points is None or len(points) < 2:
        raise ValueError("Manual baseline requires at least 2 control points.")

    pts = np.array(points, dtype=float)
    # Sort by x to avoid spline errors
    pts = pts[np.argsort(pts[:, 0])]
    px, py = pts[:, 0], pts[:, 1]

    x = np.asarray(x)
    # Interpolate
    if interp == 'cubic' and HAS_CSPLINE and len(pts) >= 3:
        cs = CubicSpline(px, py, extrapolate=True)
        baseline = cs(x)
    else:
        # Default / fallback linear
        baseline = np.interp(x, px, py, left=py[0], right=py[-1])

    corrected = np.asarray(y) - baseline
    return baseline, corrected, pts
