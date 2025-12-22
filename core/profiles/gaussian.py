"""
Gaussian profile function.
"""

import numpy as np


def gaussian(x, amplitude, center, sigma):
    """
    Gaussian peak profile.
    
    Parameters
    ----------
    x : array_like
        Independent variable (x-axis data)
    amplitude : float
        Peak amplitude (height)
    center : float
        Peak center position (mean, μ)
    sigma : float
        Peak width (standard deviation, σ)
    
    Returns
    -------
    array_like
        Gaussian peak values at x positions
    
    Notes
    -----
    Mathematical form: f(x) = A * exp(-((x - μ)² / (2σ²)))
    
    FWHM (Full Width at Half Maximum) = 2.355 * sigma
    """
    return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))
