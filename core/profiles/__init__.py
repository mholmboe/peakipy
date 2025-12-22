"""
Profile functions for peak fitting.

This module provides standard peak profile functions (Gaussian, Lorentzian, Voigt)
and utilities for managing custom profiles.
"""

from .gaussian import gaussian
from .lorentzian import lorentzian
from .voigt import voigt, pseudo_voigt
from .custom import load_custom_profile


# Profile registry - maps profile names to functions
PROFILE_REGISTRY = {
    'gaussian': gaussian,
    'lorentzian': lorentzian,
    'voigt': voigt,
    'pseudo_voigt': pseudo_voigt,
}


def get_profile(name):
    """
    Get profile function by name.
    
    Parameters
    ----------
    name : str
        Profile name (e.g., 'gaussian', 'lorentzian', 'voigt')
    
    Returns
    -------
    callable
        Profile function
    
    Raises
    ------
    KeyError
        If profile name not found in registry
    """
    if name not in PROFILE_REGISTRY:
        raise KeyError(f"Profile '{name}' not found. Available: {list_profiles()}")
    return PROFILE_REGISTRY[name]


def list_profiles():
    """
    List all available profile names.
    
    Returns
    -------
    list
        List of available profile names
    """
    return list(PROFILE_REGISTRY.keys())


def register_profile(name, func):
    """
    Register a custom profile function.
    
    Parameters
    ----------
    name : str
        Profile name
    func : callable
        Profile function with signature: func(x, **params)
    """
    PROFILE_REGISTRY[name] = func


__all__ = [
    'gaussian',
    'lorentzian', 
    'voigt',
    'pseudo_voigt',
    'load_custom_profile',
    'get_profile',
    'list_profiles',
    'register_profile',
    'PROFILE_REGISTRY',
]
