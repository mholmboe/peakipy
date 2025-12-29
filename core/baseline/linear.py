"""
Linear baseline correction. 
"""

import numpy as np


def linear_baseline(x, y, endpoints='auto', slope=None, intercept=None):
    """
    Linear baseline correction (y = mx + b).
    
    Parameters
    ----------
    x : array_like
        X-axis data
    y : array_like
        Y-axis data
    endpoints : str or tuple, optional
        If 'auto': use first and last points
        If tuple (x1, x2): use points nearest to these x values
        If 'manual': would require GUI selection (not implemented here)
    slope : float, optional
        If provided together with intercept, use these values directly and skip fitting.
    intercept : float, optional
        Intercept to use when slope is provided.
    
    Returns
    -------
    baseline : ndarray
        Linear baseline
    corrected : ndarray
        Baseline-corrected data (y - baseline)
    slope : float
        Baseline slope (m)
    intercept : float
        Baseline intercept (b)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Use manual slope/intercept if provided
    if slope is not None and intercept is not None:
        baseline = slope * x + intercept
        corrected = y - baseline
        return baseline, corrected, slope, intercept
    
    if endpoints == 'auto':
        # Use first and last 5% of points (more robust than single points)
        n_edge = max(1, len(x) // 20)
        x_fit = np.concatenate([x[:n_edge], x[-n_edge:]])
        y_fit = np.concatenate([y[:n_edge], y[-n_edge:]])
    
    elif isinstance(endpoints, tuple) and len(endpoints) == 2:
        # Find points closest to specified x values
        x1, x2 = endpoints
        idx1 = np.argmin(np.abs(x - x1))
        idx2 = np.argmin(np.abs(x - x2))
        x_fit = np.array([x[idx1], x[idx2]])
        y_fit = np.array([y[idx1], y[idx2]])
    
    else:
        raise ValueError("endpoints must be 'auto' or tuple of (x1, x2)")
    
    # Fit linear baseline
    coeffs = np.polyfit(x_fit, y_fit, 1)
    slope, intercept = coeffs
    
    baseline = slope * x + intercept
    corrected = y - baseline
    
    return baseline, corrected, slope, intercept


def two_point_baseline(x, y, x1, x2):
    """
    Linear baseline using two specified x points.
    
    Parameters
    ----------
    x : array_like  
        X-axis data
    y : array_like
        Y-axis data
    x1 : float
        First x coordinate
    x2 : float
        Second x coordinate
    
    Returns
    -------
    baseline : ndarray
        Linear baseline
    corrected : ndarray
        Baseline-corrected data
    """
    baseline, corrected, _, _ = linear_baseline(x, y, endpoints=(x1, x2))
    return baseline, corrected
