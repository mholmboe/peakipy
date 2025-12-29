"""
Asymmetric Least Squares (AsLS) baseline correction. 
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def asls_baseline(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares Smoothing for baseline correction.
    
    This method fits a smooth baseline by iteratively reweighting the data points.
    Points below the baseline get higher weights, while points above get lower weights,
    making it effective for separating peaks from baseline.
    
    Parameters
    ----------
    y : array_like
        Y-axis data (spectrum/chromatogram)
    lam : float, optional
        Smoothness parameter. Larger values = smoother baseline.
        Typical range: 1e2 to 1e9. Default: 1e5
    p : float, optional
        Asymmetry parameter (0 < p < 1). Smaller values = more asymmetric
        (higher weight for points below baseline).
        Typical range: 0.001 to 0.1. Default: 0.01
    niter : int, optional
        Number of iterations, default 10
    
    Returns
    -------
    baseline : ndarray
        Fitted baseline
    corrected : ndarray
        Baseline-corrected data (y - baseline)
    
    References
    ----------
    Eilers, P.H.C., Boelens, H.F.M., 2005. Baseline Correction with Asymmetric
    Least Squares Smoothing. Leiden University Medical Centre Report.
    """
    y = np.asarray(y)
    L = len(y)
    
    # Difference matrix (second derivative)
    # Create sparse identity and difference matrices
    D = sparse.eye(L, format='csc')
    D = D[1:] - D[:-1]  # First difference
    D = D[1:] - D[:-1]  # Second difference
    
    # Penalty term: lambda * D^T * D
    D = lam * D.T.dot(D)
    
    # Initialize weights
    w = np.ones(L)
    
    # Iterative reweighting
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L, format='csc')
        Z = W + D
        z = spsolve(Z, w * y)
        
        # Update weights: higher weight for points below baseline
        w = p * (y > z) + (1 - p) * (y <= z)
    
    baseline = z
    corrected = y - baseline
    
    return baseline, corrected


def asls_baseline_with_params(x, y, lam=1e5, p=0.01, niter=10):
    """
    AsLS baseline with x-axis included (for consistency with other methods).
    
    Parameters
    ----------
    x : array_like
        X-axis data (not used in computation, but included for API consistency)
    y : array_like
        Y-axis data
    lam : float, optional
        Smoothness parameter, default 1e5
    p : float, optional
        Asymmetry parameter, default 0.01
    niter : int, optional
        Number of iterations, default 10
    
    Returns
    -------
    baseline : ndarray
        Fitted baseline
    corrected : ndarray
        Baseline-corrected data
    """
    return asls_baseline(y, lam, p, niter)
