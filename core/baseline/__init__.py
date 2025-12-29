"""
Baseline correction methods for profile fitting. 

This module provides various baseline correction algorithms:
- AsLS (Asymmetric Least Squares)
- Polynomial fitting
- Linear fitting
- Rolling ball algorithm
"""

from .asls import asls_baseline, asls_baseline_with_params
from .polynomial import polynomial_baseline, robust_polynomial_baseline
from .linear import linear_baseline, two_point_baseline
from .rolling_ball import rolling_ball_baseline, adaptive_rolling_ball
from .shirley import shirley_background, shirley_background_simple, active_shirley_background
from .manual import manual_baseline


# Baseline method registry
BASELINE_METHODS = {
    'asls': asls_baseline_with_params,
    'polynomial': robust_polynomial_baseline,
    'linear': linear_baseline,
    'rolling_ball': rolling_ball_baseline,
    'shirley': shirley_background,
    'manual': manual_baseline,
}


def apply_baseline(method, x, y, **params):
    """
    Apply baseline correction using specified method.
    
    Parameters
    ----------
    method : str
        Baseline method name: 'asls', 'polynomial', 'linear', 'rolling_ball'
    x : array_like
        X-axis data
    y : array_like
        Y-axis data
    **params
        Method-specific parameters
    
    Returns
    -------
    baseline : ndarray
        Fitted baseline
    corrected : ndarray
        Baseline-corrected data
    
    Examples
    --------
    >>> # AsLS baseline
    >>> baseline, corrected = apply_baseline('asls', x, y, lam=1e5, p=0.01)
    >>> 
    >>> # Polynomial baseline
    >>> baseline, corrected, coeffs = apply_baseline('polynomial', x, y, degree=2)
    >>> 
    >>> # Linear baseline
    >>> baseline, corrected, slope, intercept = apply_baseline('linear', x, y)
    >>> 
    >>> # Rolling ball baseline
    >>> baseline, corrected = apply_baseline('rolling_ball', x, y, radius=50)
    """
    if method not in BASELINE_METHODS:
        raise ValueError(f"Unknown baseline method: {method}. "
                        f"Available: {list(BASELINE_METHODS.keys())}")
    
    func = BASELINE_METHODS[method]
    return func(x, y, **params)


def list_baseline_methods():
    """
    List all available baseline correction methods.
    
    Returns
    -------
    list
        List of method names
    """
    return list(BASELINE_METHODS.keys())


__all__ = [
    'asls_baseline',
    'asls_baseline_with_params',
    'polynomial_baseline',
    'robust_polynomial_baseline',
    'linear_baseline',
    'two_point_baseline',
    'rolling_ball_baseline',
    'adaptive_rolling_ball',
    'shirley_background',
    'shirley_background_simple',
    'active_shirley_background',
    'manual_baseline',
    'apply_baseline',
    'list_baseline_methods',
    'BASELINE_METHODS',
]
