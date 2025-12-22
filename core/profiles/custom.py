"""
Custom profile loader for user-defined peak functions.
"""

import json
import numpy as np


def load_custom_profile(filepath):
    """
    Load custom profile definition from .txt or .json file.
    
    Parameters
    ----------
    filepath : str
        Path to custom profile definition file
    
    Returns
    -------
    callable
        Profile function that can be used for fitting
    
    Notes
    -----
    Expected JSON format:
    {
      "name": "custom_peak",
      "function": "amplitude * np.exp(-x/decay)",
      "parameters": ["amplitude", "decay"],
      "description": "Exponential decay peak"
    }
    
    The function string will be evaluated with numpy (np) available.
    """
    with open(filepath, 'r') as f:
        if filepath.endswith('.json'):
            profile_def = json.load(f)
        else:
            # Simple text format: one line with function expression
            profile_def = {
                'name': 'custom',
                'function': f.read().strip(),
                'parameters': []
            }
    
    func_str = profile_def['function']
    params = profile_def.get('parameters', [])
    
    # Create the function dynamically
    # This uses eval with a restricted namespace for safety
    def custom_function(x, **kwargs):
        # Build local namespace with numpy and parameters
        local_vars = {'np': np, 'x': x}
        local_vars.update(kwargs)
        
        try:
            return eval(func_str, {"__builtins__": {}}, local_vars)
        except Exception as e:
            raise ValueError(f"Error evaluating custom profile: {e}")
    
    custom_function.__name__ = profile_def.get('name', 'custom')
    custom_function.__doc__ = profile_def.get('description', 'Custom profile function')
    
    return custom_function, params
