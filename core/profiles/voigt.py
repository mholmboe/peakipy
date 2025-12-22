"""
Voigt profile function.
"""

import numpy as np
from scipy.special import wofz


def voigt(x, amplitude, center, sigma, gamma):
    """
    Voigt peak profile (convolution of Gaussian and Lorentzian).
    
    Parameters
    ----------
    x : array_like
        Independent variable (x-axis data)
    amplitude : float
        Peak amplitude (height)
    center : float
        Peak center position
    sigma : float
        Gaussian width (standard deviation)
    gamma : float
        Lorentzian width (HWHM)
    
    Returns
    -------
    array_like
        Voigt peak values at x positions
    
    Notes
    -----
    Uses the Faddeeva function (scipy.special.wofz) for accurate computation.
    The Voigt profile is a convolution of Gaussian and Lorentzian profiles,
    commonly used in spectroscopy where both thermal (Gaussian) and lifetime
    (Lorentzian) broadening are present.
    """
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def pseudo_voigt(x, amplitude, center, sigma, gamma, fraction=0.5):
    """
    Pseudo-Voigt profile (weighted sum of Gaussian and Lorentzian).
    
    This is a computationally faster approximation of the true Voigt profile.
    
    Parameters
    ----------
    x : array_like
        Independent variable (x-axis data)
    amplitude : float
        Peak amplitude (height)
    center : float
        Peak center position
    sigma : float
        Gaussian width (standard deviation)
    gamma : float
        Lorentzian width (HWHM)
    fraction : float, optional
        Mixing fraction (0 = pure Gaussian, 1 = pure Lorentzian), default 0.5
    
    Returns
    -------
    array_like
        Pseudo-Voigt peak values at x positions
    """
    from .gaussian import gaussian
    from .lorentzian import lorentzian
    
    gauss = gaussian(x, amplitude, center, sigma)
    lorentz = lorentzian(x, amplitude, center, gamma)
    
    return (1 - fraction) * gauss + fraction * lorentz
