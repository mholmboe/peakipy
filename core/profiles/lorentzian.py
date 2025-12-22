"""
Lorentzian profile function.
"""

import numpy as np


def lorentzian(x, amplitude, center, gamma):
    """
    Lorentzian (Cauchy) peak profile.
    
    Parameters
    ----------
    x : array_like
        Independent variable (x-axis data)
    amplitude : float
        Peak amplitude (height)
    center : float
        Peak center position (x₀)
    gamma : float
        Half-width at half-maximum (HWHM)
    
    Returns
    -------
    array_like
        Lorentzian peak values at x positions
    
    Notes
    -----
    Mathematical form: f(x) = A * (γ² / ((x - x₀)² + γ²))
    
    FWHM (Full Width at Half Maximum) = 2 * gamma
    """
    return amplitude * (gamma**2 / ((x - center)**2 + gamma**2))
