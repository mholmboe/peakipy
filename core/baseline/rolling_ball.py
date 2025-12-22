"""
Rolling ball baseline correction.
"""

import numpy as np
from scipy.ndimage import grey_opening, minimum_filter1d


def rolling_ball_baseline(x, y, radius=50, mode='morphological'):
    """
    Rolling ball baseline correction using morphological operations.
    
    This method simulates rolling a ball under the spectrum. The ball's
    trajectory forms the baseline. Effective for broad background signals.
    
    Parameters
    ----------
    x : array_like
        X-axis data (not used in computation, for API consistency)
    y : array_like
        Y-axis data
    radius : int or float, optional
        Ball radius in number of points. Larger radius = smoother baseline.
        Default: 50
    mode : str, optional
        Algorithm mode: 'morphological' (faster) or 'explicit' (more accurate)
        Default: 'morphological'
    
    Returns
    -------
    baseline : ndarray
        Rolling ball baseline
    corrected : ndarray
        Baseline-corrected data (y - baseline)
    
    Notes
    -----
    The morphological opening operation is equivalent to rolling a ball
    underneath the data: erosion followed by dilation.
    """
    y = np.asarray(y)
    radius = int(radius)
    
    if mode == 'morphological':
        # Use morphological opening (fast, approximate)
        # grey_opening = minimum filter followed by maximum filter
        structure_size = 2 * radius + 1
        baseline = grey_opening(y, size=structure_size)
    
    elif mode == 'explicit':
        # Explicit rolling ball algorithm (slower, more accurate)
        baseline = _rolling_ball_explicit(y, radius)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'morphological' or 'explicit'")
    
    corrected = y - baseline
    
    return baseline, corrected


def _rolling_ball_explicit(y, radius):
    """
    Explicit rolling ball baseline using minimum filter.
    
    Parameters
    ----------
    y : ndarray
        Y-axis data
    radius : int
        Ball radius in points
    
    Returns
    -------
    baseline : ndarray
        Baseline
    """
    # Create ball structuring element
    # Ball equation: y_ball = sqrt(r^2 - x^2) - r (inverted parabola approximation)
    x_ball = np.arange(-radius, radius + 1)
    y_ball = np.sqrt(np.maximum(0, radius**2 - x_ball**2)) - radius
    
    # Pad data
    y_padded = np.pad(y, radius, mode='edge')
    
    # Subtract ball from data
    y_shifted = y_padded - y_ball[radius]
    
    # Apply minimum filter
    baseline_padded = minimum_filter1d(y_shifted, size=2*radius+1, mode='constant', cval=np.inf)
    
    # Add ball back
    baseline_padded += y_ball[radius]
    
    # Remove padding
    baseline = baseline_padded[radius:-radius]
    
    return baseline


def adaptive_rolling_ball(x, y, min_radius=10, max_radius=100, n_trials=5):
    """
    Automatically determine optimal ball radius.
    
    Tries multiple radii and selects based on residual flatness.
    
    Parameters
    ----------
    x : array_like
        X-axis data
    y : array_like
        Y-axis data
    min_radius : int, optional
        Minimum ball radius to try, default 10
    max_radius : int, optional
        Maximum ball radius to try, default 100
    n_trials : int, optional
        Number of radii to try, default 5
    
    Returns
    -------
    baseline : ndarray
        Best rolling ball baseline
    corrected : ndarray
        Baseline-corrected data
    best_radius : int
        Selected optimal radius
    """
    y = np.asarray(y)
    
    # Try different radii
    radii = np.linspace(min_radius, max_radius, n_trials, dtype=int)
    scores = []
    
    for radius in radii:
        baseline, corrected = rolling_ball_baseline(x, y, radius=radius)
        
        # Score: want residuals to be mostly positive (no negative dips)
        # and relatively flat (low std of positive part)
        positive_mask = corrected > 0
        if np.sum(positive_mask) > 0:
            score = np.std(corrected[positive_mask]) - np.mean(np.abs(corrected[~positive_mask]))
        else:
            score = np.inf
        
        scores.append(score)
    
    # Select best radius
    best_idx = np.argmin(scores)
    best_radius = radii[best_idx]
    
    # Recompute with best radius
    baseline, corrected = rolling_ball_baseline(x, y, radius=best_radius)
    
    return baseline, corrected, best_radius
