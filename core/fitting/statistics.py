"""
Goodness-of-fit statistics calculator. 
"""

import numpy as np


def calculate_statistics(y_data, y_fit, n_params):
    """
    Calculate goodness-of-fit statistics.
    
    Parameters
    ----------
    y_data : array_like
        Experimental Y data
    y_fit : array_like
        Fitted Y data
    n_params : int
        Number of fitting parameters
    
    Returns
    -------
    stats : dict
        Dictionary containing various fit statistics:
        - 'r_squared': R² (coefficient of determination)
        - 'adj_r_squared': Adjusted R²
        - 'chi_squared': Chi-squared
        - 'reduced_chi_squared': Reduced chi-squared
        - 'rmse': Root mean square error
        - 'aic': Akaike Information Criterion
        - 'bic': Bayesian Information Criterion
    """
    y_data = np.asarray(y_data)
    y_fit = np.asarray(y_fit)
    
    n = len(y_data)
    residuals = y_data - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    
    # R-squared
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Adjusted R-squared
    if n > n_params + 1:
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_params - 1)
    else:
        adj_r_squared = r_squared
    
    # Chi-squared
    chi_squared = ss_res
    
    # Reduced chi-squared
    dof = n - n_params  # degrees of freedom
    reduced_chi_squared = chi_squared / dof if dof > 0 else np.inf
    
    # RMSE
    rmse = np.sqrt(ss_res / n)
    
    # AIC (Akaike Information Criterion)
    # AIC = n*ln(SS_res/n) + 2*k
    if ss_res > 0:
        aic = n * np.log(ss_res / n) + 2 * n_params
    else:
        aic = -np.inf
    
    # BIC (Bayesian Information Criterion)
    # BIC = n*ln(SS_res/n) + k*ln(n)
    if ss_res > 0:
        bic = n * np.log(ss_res / n) + n_params * np.log(n)
    else:
        bic = -np.inf
    
    stats = {
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'chi_squared': chi_squared,
        'reduced_chi_squared': reduced_chi_squared,
        'rmse': rmse,
        'aic': aic,
        'bic': bic,
        'n_data': n,
        'n_params': n_params,
        'dof': dof,
    }
    
    return stats


def extract_lmfit_statistics(result):
    """
    Extract statistics from lmfit ModelResult object.
    
    Parameters
    ----------
    result : lmfit.ModelResult
        Result from lmfit fitting
    
    Returns
    -------
    stats : dict
        Dictionary of statistics
    """
    stats = {
        'r_squared': 1 - result.residual.var() / np.var(result.data) if hasattr(result, 'data') else None,
        'chi_squared': result.chisqr,
        'reduced_chi_squared': result.redchi,
        'aic': result.aic,
        'bic': result.bic,
        'n_data': result.ndata,
        'n_params': result.nvarys,
        'dof': result.nfree,
    }
    
    # Add adjusted R²
    if stats['r_squared'] is not None:
        n = stats['n_data']
        k = stats['n_params']
        if n > k + 1:
            stats['adj_r_squared'] = 1 - (1 - stats['r_squared']) * (n - 1) / (n - k - 1)
        else:
            stats['adj_r_squared'] = stats['r_squared']
    
    # Add RMSE
    stats['rmse'] = np.sqrt(result.chisqr / result.ndata)
    
    return stats


def format_statistics(stats):
    """
    Format statistics for display.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary
    
    Returns
    -------
    str
        Formatted statistics string
    """
    lines = []
    lines.append("=== Fit Statistics ===")
    lines.append(f"R² = {stats.get('r_squared', 0):.6f}")
    lines.append(f"Adj. R² = {stats.get('adj_r_squared', 0):.6f}")
    lines.append(f"RMSE = {stats.get('rmse', 0):.6e}")
    lines.append(f"χ² = {stats.get('chi_squared', 0):.6e}")
    lines.append(f"Reduced χ² = {stats.get('reduced_chi_squared', 0):.6f}")
    lines.append(f"AIC = {stats.get('aic', 0):.2f}")
    lines.append(f"BIC = {stats.get('bic', 0):.2f}")
    lines.append(f"N data = {stats.get('n_data', 0)}")
    lines.append(f"N parameters = {stats.get('n_params', 0)}")
    lines.append(f"Degrees of freedom = {stats.get('dof', 0)}")
    
    return '\n'.join(lines)
