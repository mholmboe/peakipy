"""
Polynomial baseline correction.
"""

import numpy as np


def polynomial_baseline(x, y, degree=2, exclude_regions=None):
    """
    Polynomial baseline correction using least squares fitting.
    
    Parameters
    ----------
    x : array_like
        X-axis data
    y : array_like
        Y-axis data
    degree : int, optional
        Polynomial degree (0-5 recommended), default 2
    exclude_regions : list of tuples, optional
        List of (x_min, x_max) regions to exclude from baseline fit
        (e.g., to exclude peak regions)
    
    Returns
    -------
    baseline : ndarray
        Fitted polynomial baseline
    corrected : ndarray
        Baseline-corrected data (y - baseline)
    coeffs : ndarray
        Polynomial coefficients
    
    Examples
    --------
    >>> # Fit quadratic baseline, excluding peak region
    >>> baseline, corrected, coeffs = polynomial_baseline(
    ...     x, y, degree=2, exclude_regions=[(50, 70)]
    ... )
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Create mask for excluded regions
    mask = np.ones(len(x), dtype=bool)
    
    if exclude_regions is not None:
        for x_min, x_max in exclude_regions:
            mask &= ~((x >= x_min) & (x <= x_max))
    
    # Fit polynomial to non-excluded points
    coeffs = np.polyfit(x[mask], y[mask], degree)
    
    # Evaluate polynomial over entire range
    baseline = np.polyval(coeffs, x)
    corrected = y - baseline
    
    return baseline, corrected, coeffs


def robust_polynomial_baseline(x, y, degree=2, max_iter=10, sigma=1.5):
    """
    Robust polynomial baseline using iterative sigma clipping.
    
    Fits polynomial iteratively, excluding outliers above the baseline.
    Useful when peak regions are not known in advance.
    
    Parameters
    ----------
    x : array_like
        X-axis data
    y : array_like
        Y-axis data
    degree : int, optional
        Polynomial degree, default 2
    max_iter : int, optional
        Maximum iterations, default 10
    sigma : float, optional
        Sigma threshold for outlier rejection, default 3.0
    
    Returns
    -------
    baseline : ndarray
        Fitted polynomial baseline
    corrected : ndarray
        Baseline-corrected data
    coeffs : ndarray
        Polynomial coefficients (highest degree first)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    mask = np.ones(len(x), dtype=bool)
    coeffs = None
    
    for i in range(max_iter):
        # Fit polynomial
        coeffs = np.polyfit(x[mask], y[mask], degree)
        baseline = np.polyval(coeffs, x)
        residuals = y - baseline
        
        # Calculate threshold: keep points below baseline + sigma*std
        threshold = sigma * np.std(residuals[mask])
        new_mask = residuals < threshold
        
        # Check convergence
        if np.array_equal(mask, new_mask):
            break
        
        mask = new_mask
    
    corrected = y - baseline
    
    return baseline, corrected, coeffs
