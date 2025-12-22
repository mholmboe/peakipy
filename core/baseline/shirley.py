"""
Shirley background correction for XPS data analysis.
"""

import numpy as np


def shirley_background(x, y, tol=1e-5, max_iter=50, start_offset=0.0, end_offset=0.0):
    """
    Shirley background correction algorithm.
    
    The Shirley background assumes that the background at any point is proportional
    to the total peak area above that point. This is commonly used in X-ray
    Photoelectron Spectroscopy (XPS) to correct for inelastic scattering.
    
    Parameters
    ----------
    x : array_like
        X-axis data (binding energy or kinetic energy)
    y : array_like
        Y-axis data (intensity)
    tol : float, optional
        Convergence tolerance, default 1e-5
    max_iter : int, optional
        Maximum number of iterations, default 50
    
    Returns
    -------
    baseline : ndarray
        Shirley background
    corrected : ndarray
        Background-corrected data (y - baseline)
    
    References
    ----------
    Shirley, D. A. (1972). High-Resolution X-Ray Photoemission Spectrum of 
    the Valence Bands of Gold. Physical Review B, 5(12), 4709.
    
    Notes
    -----
    The algorithm iteratively computes the background based on the cumulative
    integral of the (signal - background) from each point to the end of the
    spectrum.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Ensure data is sorted by x (typically binding energy decreases)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Initialize background with linear interpolation between endpoints
    background = np.linspace(y_sorted[0], y_sorted[-1], len(y_sorted))
    
    # Get endpoint values
    y_start = y_sorted[0] + start_offset
    y_end = y_sorted[-1] + end_offset
    
    # Iterative Shirley algorithm
    for iteration in range(max_iter):
        # Compute the integral from each point to the end
        # Background at point i is proportional to integral from i to end
        cumulative = np.zeros_like(y_sorted)
        
        # Calculate cumulative sum from right to left (end to start)
        signal = y_sorted - background
        signal = np.maximum(signal, 0)  # Only positive contributions
        
        # Trapezoidal integration from each point to the end
        for i in range(len(y_sorted) - 1, -1, -1):
            if i == len(y_sorted) - 1:
                cumulative[i] = 0
            else:
                dx = x_sorted[i+1] - x_sorted[i]
                cumulative[i] = cumulative[i+1] + 0.5 * (signal[i] + signal[i+1]) * abs(dx)
        
        # Normalize cumulative to range [y_end, y_start]
        total_area = cumulative[0]
        
        if total_area > 0:
            # New background proportional to cumulative integral
            new_background = y_end + (y_start - y_end) * cumulative / total_area
        else:
            # If no area above background, use linear baseline
            new_background = background
        
        # Check convergence
        diff = np.max(np.abs(new_background - background))
        background = new_background
        
        if diff < tol:
            break
    
    # Restore original order
    background_unsorted = np.zeros_like(background)
    background_unsorted[sort_idx] = background
    
    corrected = y - background_unsorted
    
    return background_unsorted, corrected


def shirley_background_simple(x, y, tol=1e-5, max_iter=50, start_offset=0.0, end_offset=0.0):
    """
    Simplified Shirley background (assumes x is already sorted).
    
    This version is faster but requires x to be monotonically increasing
    or decreasing.
    
    Parameters
    ----------
    x : array_like
        X-axis data (must be sorted)
    y : array_like
        Y-axis data
    tol : float, optional
        Convergence tolerance
    max_iter : int, optional
        Maximum iterations
    
    Returns
    -------
    baseline : ndarray
        Shirley background
    corrected : ndarray
        Background-corrected data
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Check if we need to reverse (XPS typically has decreasing binding energy)
    if x[-1] < x[0]:
        x = x[::-1]
        y = y[::-1]
        reversed_data = True
    else:
        reversed_data = False
    
    # Initialize
    background = np.linspace(y[0] + start_offset, y[-1] + end_offset, len(y))
    y_start = y[0] + start_offset
    y_end = y[-1] + end_offset
    
    # Iterate
    for iteration in range(max_iter):
        signal = np.maximum(y - background, 0)
        
        # Cumulative integral from right to left
        cumsum = np.cumsum(signal[::-1])[::-1]
        
        # Scale cumulative to background endpoints
        if cumsum[0] > 0:
            background = y_end + (y_start - y_end) * cumsum / cumsum[0]
        
        # Check convergence
        if iteration > 0 and np.max(np.abs(background - prev_background)) < tol:
            break
        
        prev_background = background.copy()
    
    corrected = y - background
    
    # Reverse back if needed
    if reversed_data:
        background = background[::-1]
        corrected = corrected[::-1]
    
    return background, corrected


def active_shirley_background(x, y, tol=1e-5, max_iter=50, endpoints='auto', start_offset=0.0, end_offset=0.0):
    """
    Shirley background with automatic or manual endpoint selection.
    
    Parameters
    ----------
    x : array_like
        X-axis data
    y : array_like
        Y-axis data
    tol : float, optional
        Convergence tolerance
    max_iter : int, optional
        Maximum iterations
    endpoints : str or tuple, optional
        If 'auto': use data endpoints
        If tuple (i_start, i_end): use these indices for endpoints
    
    Returns
    -------
    baseline : ndarray
        Shirley background
    corrected : ndarray
        Background-corrected data
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if isinstance(endpoints, tuple) and len(endpoints) == 2:
        # Use specified endpoint indices
        i_start, i_end = endpoints
        # Create subset
        x_roi = x[i_start:i_end+1]
        y_roi = y[i_start:i_end+1]
        
        # Calculate Shirley on ROI
        bg_roi, corr_roi = shirley_background(x_roi, y_roi, tol, max_iter, start_offset=start_offset, end_offset=end_offset)
        
        # Extend to full range (linear outside ROI)
        baseline = np.zeros_like(y)
        baseline[i_start:i_end+1] = bg_roi
        
        # Linear extrapolation before start
        if i_start > 0:
            baseline[:i_start] = bg_roi[0]
        
        # Linear extrapolation after end
        if i_end < len(y) - 1:
            baseline[i_end+1:] = bg_roi[-1]
        
        corrected = y - baseline
    else:
        # Use full range
        baseline, corrected = shirley_background(x, y, tol, max_iter, start_offset=start_offset, end_offset=end_offset)
    
    return baseline, corrected
