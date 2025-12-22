"""
Model builder for composite profile fitting using lmfit.
"""

from lmfit import Model, CompositeModel
from ..profiles import get_profile


def build_composite_model(components, baseline_info=None):
    """
    Build lmfit composite model from list of profile components.
    
    Parameters
    ----------
    components : list of dict
        List of component definitions.
    baseline_info : dict, optional
        Baseline information for parametric methods:
        - 'type': 'polynomial' or 'linear'
        - 'degree': int (for polynomial)
    
    Returns
    -------
    model : lmfit.CompositeModel or lmfit.Model
        Composite model
    params : lmfit.Parameters
        Initial parameters
    """
    if not components:
        raise ValueError("Must provide at least one component")
    
    models = []
    
    # 1. Add peak components
    for comp in components:
        profile_type = comp['type']
        prefix = comp.get('prefix', '')
        profile_func = get_profile(profile_type)
        model = Model(profile_func, prefix=prefix)
        models.append(model)
    
    # 2. Add parametric baseline if requested
    if baseline_info:
        bl_type = baseline_info.get('type')
        if bl_type == 'polynomial':
            degree = baseline_info.get('degree', 2)
            from lmfit.models import PolynomialModel
            bl_model = PolynomialModel(degree=degree, prefix='bl_')
            models.append(bl_model)
        elif bl_type == 'linear':
            from lmfit.models import LinearModel
            bl_model = LinearModel(prefix='bl_')
            models.append(bl_model)
    
    # Combine models
    if len(models) == 1:
        composite_model = models[0]
    else:
        composite_model = models[0]
        for model in models[1:]:
            composite_model = composite_model + model
    
    # Create parameters with hints
    params = composite_model.make_params()
    
    # Set initial values and bounds from component specs
    for comp in components:
        prefix = comp.get('prefix', '')
        comp_params = comp.get('params', {})
        
        for param_name, value in comp_params.items():
            full_name = f"{prefix}{param_name}"
            if full_name in params:
                if isinstance(value, dict):
                    params[full_name].set(**value)
                else:
                    params[full_name].set(value=value)
        
        # Apply positive bounds for width and amplitude parameters
        for param_suffix in ['sigma', 'gamma']:
            full_name = f"{prefix}{param_suffix}"
            if full_name in params:
                params[full_name].set(min=1e-6)  # Must be positive
        
        amp_name = f"{prefix}amplitude"
        if amp_name in params:
            params[amp_name].set(min=0)  # Amplitude must be non-negative
    
    # Initialize baseline parameters to 0 (lmfit defaults to -inf which causes NaN)
    if baseline_info:
        bl_type = baseline_info.get('type')
        if bl_type == 'polynomial':
            degree = baseline_info.get('degree', 2)
            for i in range(degree + 1):
                param_name = f'bl_c{i}'
                if param_name in params:
                    params[param_name].set(value=0)
        elif bl_type == 'linear':
            if 'bl_slope' in params:
                params['bl_slope'].set(value=0)
            if 'bl_intercept' in params:
                params['bl_intercept'].set(value=0)
                    
    return composite_model, params


def set_parameter_bounds(params, bounds_dict):
    """
    Set parameter bounds for fitting.
    
    Parameters
    ----------
    params : lmfit.Parameters
        Parameters object
    bounds_dict : dict
        Dictionary mapping parameter names to (min, max) tuples
    
    Examples
    --------
    >>> bounds = {
    ...     'g1_amplitude': (0, 200),
    ...     'g1_sigma': (1, 20),
    ... }
    >>> set_parameter_bounds(params, bounds)
    """
    for param_name, (min_val, max_val) in bounds_dict.items():
        if param_name in params:
            params[param_name].set(min=min_val, max=max_val)


def set_parameter_constraints(params, constraints):
    """
    Set parameter constraints/expressions.
    
    Parameters
    ----------
    params : lmfit.Parameters
        Parameters object
    constraints : dict
        Dictionary mapping parameter names to constraint expressions
    
    Examples
    --------
    >>> # Fix ratio between two peaks
    >>> constraints = {
    ...     'g2_amplitude': '0.8 * g1_amplitude',
    ... }
    >>> set_parameter_constraints(params, constraints)
    """
    for param_name, expr in constraints.items():
        if param_name in params:
            params[param_name].set(expr=expr)
