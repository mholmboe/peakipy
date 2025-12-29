"""
Peak parameter initialization methods. 

Provides different strategies for initializing component parameters
(center, width, amplitude) before fitting.
"""

import numpy as np


def init_with_gmm(x, y, n_components, baseline=None, max_samples=20000, random_state=42):
    """
    Initialize component parameters using Gaussian Mixture Model.
    
    Uses EM algorithm to find likely peak locations based on the
    intensity-weighted distribution of x-values. Particularly effective
    for overlapping peaks.
    
    Parameters
    ----------
    x : array_like
        X data
    y : array_like
        Y data (raw or baseline-corrected)
    n_components : int
        Number of components to initialize
    baseline : array_like, optional
        Baseline to subtract. If None, uses 5th percentile
    max_samples : int, optional
        Max samples for GMM fitting (for performance)
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    list of dict or None
        List of parameter dicts: [{'center': c, 'width': w, 'amplitude': a}, ...]
        Returns None if GMM fails (caller should fallback to evenly_spaced)
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None  # sklearn not available
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Baseline subtraction
    if baseline is None:
        y_corrected = y - np.percentile(y, 5)
    else:
        y_corrected = y - baseline
    
    y_corrected = np.clip(y_corrected, 0, None)
    
    # Check if data is essentially zero
    if y_corrected.sum() < 1e-12:
        return None
    
    # Normalize to probability mass
    prob = y_corrected / (y_corrected.sum() + 1e-12)
    
    # Sample x-coordinates weighted by intensity
    n_samples = min(max_samples, len(x) * 50)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(x), size=n_samples, p=prob)
    X_samples = x[indices].reshape(-1, 1)
    
    # Fit GMM
    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state,
            init_params='k-means++',
            max_iter=200
        )
        gmm.fit(X_samples)
        
        # Extract parameters
        centers = gmm.means_.ravel()
        widths = np.sqrt(gmm.covariances_.ravel())
        
        # Estimate amplitudes from data at discovered centers
        amplitudes = np.interp(centers, x, y_corrected)
        
        # Sort by center position
        order = np.argsort(centers)
        
        return [
            {
                'center': float(centers[i]),
                'width': float(widths[i]),
                'amplitude': float(amplitudes[i])
            }
            for i in order
        ]
    
    except Exception:
        # GMM fitting failed
        return None


def init_evenly_spaced(x, n_components, y=None):
    """
    Initialize components evenly spaced across x-range.
    
    Simple, fast initialization method that distributes components
    uniformly across the data range. Works well for well-separated peaks.
    
    Parameters
    ----------
    x : array_like
        X data
    n_components : int
        Number of components to initialize
    y : array_like, optional
        Y data for amplitude estimation. If None, uses default amplitude.
        
    Returns
    -------
    list of dict
        List of parameter dicts: [{'center': c, 'width': w, 'amplitude': a}, ...]
    """
    x = np.asarray(x)
    x_min, x_max = np.min(x), np.max(x)
    x_range = x_max - x_min
    
    if x_range == 0:
        x_range = 1.0  # Avoid division by zero
    
    # Estimate amplitude from data if provided
    if y is not None:
        y = np.asarray(y)
        y_max = np.max(y)
        y_min = np.min(y)
        # Use peak height divided by number of components as initial amplitude
        base_amplitude = max((y_max - y_min) / n_components, y_max * 0.5)
    else:
        base_amplitude = 100.0  # Fallback default
    
    params = []
    for i in range(n_components):
        center = x_min + (i + 1) / (n_components + 1) * x_range
        width = x_range / (4 * n_components)
        params.append({
            'center': float(center),
            'width': float(width),
            'amplitude': float(base_amplitude)
        })
    
    return params

