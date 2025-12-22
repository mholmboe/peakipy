"""
Data preprocessing utilities.
"""

import numpy as np
from scipy.signal import savgol_filter


def remove_outliers_zscore(x, y, threshold=3.0):
    """
    Remove outliers using z-score method.
    
    Parameters
    ----------
    x : array_like
        X-axis data
    y : array_like
        Y-axis data
    threshold : float, optional
        Z-score threshold for outlier detection, default 3.0
    
    Returns
    -------
    x_clean : ndarray
        X data with outliers removed
    y_clean : ndarray
        Y data with outliers removed
    mask : ndarray (bool)
        Boolean mask indicating which points were kept
    """
    y = np.asarray(y)
    z_scores = np.abs((y - np.mean(y)) / np.std(y))
    mask = z_scores < threshold
    
    return np.asarray(x)[mask], y[mask], mask


def remove_outliers_iqr(x, y, factor=1.5):
    """
    Remove outliers using Interquartile Range (IQR) method.
    
    Parameters
    ----------
    x : array_like
        X-axis data
    y : array_like
        Y-axis data
    factor : float, optional
        IQR factor for outlier bounds, default 1.5
    
    Returns
    -------
    x_clean : ndarray
        X data with outliers removed
    y_clean : ndarray
        Y data with outliers removed
    mask : ndarray (bool)
        Boolean mask indicating which points were kept
    """
    y = np.asarray(y)
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    mask = (y >= lower_bound) & (y <= upper_bound)
    
    return np.asarray(x)[mask], y[mask], mask


def smooth_data(y, window_length=11, polyorder=3):
    """
    Smooth data using Savitzky-Golay filter.
    
    Parameters
    ----------
    y : array_like
        Y-axis data to smooth
    window_length : int, optional
        Length of filter window (must be odd), default 11
    polyorder : int, optional
        Order of polynomial fit, default 3
    
    Returns
    -------
    y_smooth : ndarray
        Smoothed Y data
    """
    if window_length % 2 == 0:
        window_length += 1  # Must be odd
    
    if window_length > len(y):
        window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
    
    return savgol_filter(y, window_length, polyorder)


def normalize_data(y, method='minmax'):
    """
    Normalize data to a standard range.
    
    Parameters
    ----------
    y : array_like
        Y-axis data to normalize
    method : str, optional
        Normalization method: 'minmax' (0-1) or 'zscore' (mean=0, std=1)
    
    Returns
    -------
    y_norm : ndarray
        Normalized Y data
    params : dict
        Normalization parameters for inverse transform
    """
    y = np.asarray(y)
    
    if method == 'minmax':
        y_min = np.min(y)
        y_max = np.max(y)
        y_norm = (y - y_min) / (y_max - y_min)
        params = {'min': y_min, 'max': y_max, 'method': 'minmax'}
    
    elif method == 'zscore':
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_norm = (y - y_mean) / y_std
        params = {'mean': y_mean, 'std': y_std, 'method': 'zscore'}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return y_norm, params


def crop_roi(x, y, x_min=None, x_max=None):
    """
    Crop data to region of interest (ROI).
    
    Parameters
    ----------
    x : array_like
        X-axis data
    y : array_like
        Y-axis data
    x_min : float or None, optional
        Minimum x value (inclusive)
    x_max : float or None, optional
        Maximum x value (inclusive)
    
    Returns
    -------
    x_roi : ndarray
        X data in ROI
    y_roi : ndarray
        Y data in ROI
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    mask = np.ones(len(x), dtype=bool)
    
    if x_min is not None:
        mask &= (x >= x_min)
    
    if x_max is not None:
        mask &= (x <= x_max)
    
    return x[mask], y[mask]


def interpolate_data(x, y, x_min=None, x_max=None, step=None, n_points=None):
    """
    Interpolate data to regular grid.
    
    Parameters
    ----------
    x : array_like
        X-axis data (must be monotonic)
    y : array_like
        Y-axis data
    x_min : float, optional
        Minimum x value for interpolation. If None, use min(x)
    x_max : float, optional
        Maximum x value for interpolation. If None, use max(x)
    step : float, optional
        Step size for interpolation. If None and n_points is None, use original spacing
    n_points : int, optional
        Number of points for interpolation. Ignored if step is provided
    
    Returns
    -------
    x_interp : ndarray
        Interpolated X values
    y_interp : ndarray
        Interpolated Y values
    
    Examples
    --------
    >>> # Interpolate to regular 0.1 spacing
    >>> x_new, y_new = interpolate_data(x, y, step=0.1)
    >>> 
    >>> # Interpolate to 500 points
    >>> x_new, y_new = interpolate_data(x, y, n_points=500)
    """
    from scipy.interpolate import interp1d
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Determine range
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    
    # Create interpolation grid
    if step is not None:
        # Use specified step size
        x_interp = np.arange(x_min, x_max + step/2, step)
    elif n_points is not None:
        # Use specified number of points
        x_interp = np.linspace(x_min, x_max, n_points)
    else:
        # Use original data range with average spacing
        avg_spacing = np.mean(np.diff(np.sort(x)))
        x_interp = np.arange(x_min, x_max + avg_spacing/2, avg_spacing)
    
    # Create interpolation function
    interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Interpolate
    y_interp = interp_func(x_interp)
    
    return x_interp, y_interp
