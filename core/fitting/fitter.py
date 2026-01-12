"""
Main profile fitter class using lmfit. 
"""

import numpy as np
from lmfit import minimize
from .model_builder import build_composite_model
from .statistics import extract_lmfit_statistics
from ..baseline import apply_baseline


class ProfileFitter:
    """
    Main class for multi-component profile fitting.
    
    Attributes
    ----------
    x : ndarray
        X-axis data
    y : ndarray
        Y-axis data (original)
    y_corrected : ndarray
        Baseline-corrected Y data
    components : list
        List of profile components
    baseline_method : str or None
        Baseline correction method
    baseline_params : dict
        Baseline correction parameters
    baseline : ndarray or None
        Fitted baseline
    baseline_raw : ndarray or None
        Baseline on the original (unnormalized) scale
    result : lmfit.ModelResult or None
        Fitting result
    """
    
    def __init__(self, x_data, y_data):
        """
        Initialize ProfileFitter.
        
        Parameters
        ----------
        x_data : array_like
            X-axis data
        y_data : array_like
            Y-axis data
        """
        self.x = np.asarray(x_data)
        self.y_raw = np.asarray(y_data)
        # Working copy that may be normalized; keep y as alias for backward compatibility
        self.y = self.y_raw.copy()
        self.y_work = self.y  # deprecated alias retained
        self.y_corrected = self.y.copy()
        
        self.components = []
        self.baseline_method = None
        self.baseline_params = {}
        self.baseline = None
        self.baseline_raw = None
        self.norm_factor = 1.0  # Normalization factor for intensity scaling
        self.normalized = False
        
        # Negative penalty settings
        self.negative_penalty_enabled = False
        self.negative_penalty_weight = 100.0
        
        # Baseline optimization setting
        self.optimize_baseline = False  # If True, optimize polynomial baseline with peaks
        self.baseline_range = (None, None)  # Sub-range for baseline calculation
        
        self.result = None
        self._model = None
        self._params = None
    
    def add_component(self, profile_type, prefix=None, **initial_params):
        """
        Add a profile component to the fit.
        
        Parameters
        ----------
        profile_type : str
            Profile type ('gaussian', 'lorentzian', 'voigt', etc.)
        prefix : str, optional
            Parameter prefix (auto-generated if not provided)
        **initial_params
            Initial parameter values (amplitude, center, sigma, etc.)
        
        Examples
        --------
        >>> fitter.add_component('gaussian', amplitude=100, center=50, sigma=5)
        >>> fitter.add_component('lorentzian', prefix='l1_', amplitude=80, center=70, gamma=3)
        """
        if prefix is None:
            # Auto-generate prefix
            prefix = f"{profile_type[0]}{len(self.components)+1}_"
        
        component = {
            'type': profile_type,
            'prefix': prefix,
            'params': initial_params
        }
        
        self.components.append(component)
    
    def remove_component(self, index):
        """Remove component by index."""
        if 0 <= index < len(self.components):
            self.components.pop(index)
            # Invalidate cached model
            self._model = None
            self._params = None
    
    def clear_components(self):
        """Remove all components."""
        self.components = []
        # Invalidate cached model
        self._model = None
        self._params = None
    
    def set_baseline(self, method, **params):
        """
        Set baseline correction method and parameters.
        
        Parameters
        ----------
        method : str
            Baseline method ('asls', 'polynomial', 'linear', 'rolling_ball')
        **params
            Method-specific parameters
        
        Examples
        --------
        >>> fitter.set_baseline('asls', lam=1e5, p=0.01)
        >>> fitter.set_baseline('polynomial', degree=2)
        """
        self.baseline_method = method
        self.baseline_params = params

    # ===================== Baseline Helpers =====================
    def _baseline_range_indices(self):
        """Return index mask for the configured baseline range."""
        x_min, x_max = self.baseline_range
        mask = np.ones_like(self.x, dtype=bool)
        if x_min is not None:
            mask &= (self.x >= x_min)
        if x_max is not None:
            mask &= (self.x <= x_max)
        indices = np.where(mask)[0]
        if len(indices) < 2:
            indices = np.arange(len(self.x))
        return indices

    def _flatten_range(self, baseline_sub, indices):
        """Extend a sub-range baseline flat to full length."""
        if len(indices) == len(self.x):
            return baseline_sub
        baseline_full = np.zeros_like(self.y_work)
        baseline_full[indices] = baseline_sub
        if indices[0] > 0:
            baseline_full[:indices[0]] = baseline_sub[0]
        if indices[-1] < len(self.x) - 1:
            baseline_full[indices[-1]+1:] = baseline_sub[-1]
        return baseline_full

    def _compute_baseline_with_range(self, y_source, params=None):
        """Compute baseline on configured range, flatten outside, return (baseline, corrected)."""
        if self.baseline_method is None:
            baseline = np.zeros_like(y_source)
            return baseline, y_source.copy()

        params = params if params is not None else self.baseline_params
        indices = self._baseline_range_indices()
        res = apply_baseline(self.baseline_method, self.x[indices], y_source[indices], **params)
        if len(res) < 1:
            raise ValueError("Baseline method returned empty result")
        baseline_sub = res[0]
        baseline_full = self._flatten_range(baseline_sub, indices)
        corrected = y_source - baseline_full
        return baseline_full, corrected

    def _seed_baseline_params_from_curve(self):
        """Seed baseline params from stored baseline_raw for parametric baselines."""
        if self.baseline_raw is None or self.baseline_method not in ['linear', 'polynomial']:
            return
        try:
            x_centered = self.x - np.mean(self.x)
            if self.baseline_method == 'linear':
                coeffs = np.polyfit(x_centered, self.baseline_raw, 1)
                self.baseline_params['slope'] = coeffs[0]
                self.baseline_params['intercept'] = coeffs[1] - coeffs[0] * np.mean(self.x)
            else:
                degree = self.baseline_params.get('degree', 2)
                coeffs_desc = np.polyfit(x_centered, self.baseline_raw, degree)
                coeffs_asc = list(coeffs_desc[::-1])
                # adjust intercept back to original x scale
                self.baseline_params['coeffs'] = coeffs_asc
                self.baseline_params['degree'] = degree
        except Exception:
            pass
    
    def apply_baseline_correction(self):
        """
        Apply baseline correction to data.
        Supports calculation on a sub-range with flat extensions.
        
        Returns
        -------
        baseline : ndarray
            Fitted baseline
        corrected : ndarray
            Baseline-corrected data
        """
        baseline, corrected = self._compute_baseline_with_range(self.y_work, self.baseline_params)
        self.baseline = baseline
        self.baseline_raw = baseline.copy()
        self.y_corrected = corrected
        return self.baseline, self.y_corrected
    
    def enable_negative_penalty(self, penalty_weight=100.0):
        """
        Enable penalty for negative baseline-corrected intensities.
        
        Instead of clipping negative values, this adds a penalty term during
        fitting that discourages the model from producing negative values.
        
        Parameters
        ----------
        penalty_weight : float
            Weight multiplier for penalty term (default: 100.0)
            Higher values = stronger penalty against negative values
        """
        self.negative_penalty_enabled = True
        self.negative_penalty_weight = penalty_weight
    
    def normalize_raw_data(self):
        """
        Normalize raw experimental data (self.y) to have maximum intensity of 1.0.
        
        This MUST be called BEFORE baseline correction if you want the baseline
        calculated on the normalized scale.
        """
        if self.normalized:
            return
        max_intensity = np.max(self.y_work)
        if max_intensity > 0:
            self.norm_factor = max_intensity
            self.y_work = self.y_work / max_intensity
            self.y = self.y_work
            # Reset y_corrected to match new scale (it will be updated by baseline correction anyway)
            self.y_corrected = self.y_work.copy()
            self.normalized = True
        else:
            self.norm_factor = 1.0
            
    def normalize_intensity(self):
        """
        Deprecated: Use normalize_raw_data() instead for pre-baseline normalization.
        
        Normalize baseline-corrected data to have maximum intensity of 1.0.
        Baseline stays on the original scale (baseline_raw) per user expectation.
        """
        max_intensity = np.max(self.y_corrected)
        if max_intensity > 0:
            # Only update if we haven't already normalized
            if self.norm_factor == 1.0:
                self.norm_factor = max_intensity
            
            self.y_corrected = self.y_corrected / max_intensity
        else:
            if self.norm_factor == 1.0:
                self.norm_factor = 1.0
    
    def fit(self, method='leastsq', max_nfev=None, skip_baseline_correction=False, iter_cb=None, **fit_kws):
        """
        Execute the fitting procedure.
        
        Parameters
        ----------
        method : str, optional
            Fitting method: 'leastsq', 'least_squares', 'differential_evolution', etc.
            Default: 'leastsq' (Levenberg-Marquardt)
        max_nfev : int, optional
            Maximum number of function evaluations
        skip_baseline_correction : bool, optional
            If True, skip baseline correction (assumes it's already been applied)
        **fit_kws
            Additional keyword arguments for lmfit.minimize
        
        Returns
        -------
        result : lmfit.ModelResult or lmfit.MinimizerResult
            Fitting result object
        """
        use_simultaneous = self.optimize_baseline and not skip_baseline_correction and self.baseline_method is not None
        y_data = self.y_work

        if not self.components:
            raise ValueError("No components added. Use add_component() first.")
            
        # Determine if we need baseline parameters in the model
        is_parametric_baseline = self.baseline_method in ['polynomial', 'linear']
        baseline_info = None
        if use_simultaneous and is_parametric_baseline:
            baseline_info = {'type': self.baseline_method}
            if self.baseline_method == 'polynomial':
                baseline_info['degree'] = self.baseline_params.get('degree', 2)

        # Check if we need to (re)build the model and parameters
        # We rebuild if:
        # 1. No model/params exist
        # 2. We need parametric baseline but current model doesn't have it
        # 3. We have parametric baseline but don't want it anymore
        has_bl_params = self._params is not None and ('bl_c0' in self._params or 'bl_slope' in self._params)
        needs_bl_params = baseline_info is not None
        
        # If we already have a baseline from a prior run, use it to seed baseline parameters
        if use_simultaneous and is_parametric_baseline and self.baseline_raw is not None:
            self._seed_baseline_params_from_curve()

        if self._model is None or self._params is None or has_bl_params != needs_bl_params:
            from .model_builder import build_composite_model
            self._model, self._params = build_composite_model(self.components, baseline_info)
        
        # Calculate data-driven bounds for baseline parameters
        y_span = np.max(y_data) - np.min(y_data)
        x_span = np.max(self.x) - np.min(self.x)
        x_mean = np.mean(self.x)
        
        # Seed linear baseline params from UI if provided with bounds
        if self.baseline_method == 'linear' and self._params is not None:
            # Slope bounds: limit to reasonable range based on data
            max_slope = 2 * y_span / x_span if x_span > 0 else 1.0
            if 'bl_slope' in self._params:
                slope_val = self.baseline_params.get('slope', 0)
                self._params['bl_slope'].set(value=slope_val, min=-max_slope, max=max_slope)
            if 'bl_intercept' in self._params:
                intercept_val = self.baseline_params.get('intercept', 0)
                max_intercept = 2 * np.max(np.abs(y_data))
                self._params['bl_intercept'].set(value=intercept_val, min=-max_intercept, max=max_intercept)
        
        # Seed polynomial baseline params if provided with bounds
        if self.baseline_method == 'polynomial' and self._params is not None:
            degree = self.baseline_params.get('degree', baseline_info.get('degree') if baseline_info else 2)
            coeffs = self.baseline_params.get('coeffs')
            # Calculate coefficient bounds based on data scale
            max_c0 = 2 * np.max(np.abs(y_data))  # intercept bound
            for i in range(degree + 1):
                pname = f'bl_c{i}'
                if pname in self._params:
                    init_val = coeffs[i] if coeffs and i < len(coeffs) else 0
                    # Higher order coefficients should be smaller
                    max_ci = max_c0 / (x_span ** i) if x_span > 0 else max_c0
                    self._params[pname].set(value=init_val, min=-max_ci, max=max_ci)
        
        # Add AsLS hyper-parameters if needed (keeping wide bounds for flexibility)
        if use_simultaneous and self.baseline_method == 'asls':
            if 'bl_log_lam' not in self._params:
                init_lam = self.baseline_params.get('lam', 1e5)
                log_lam = np.log10(init_lam)
                # Wide bounds - AsLS benefits from flexibility
                self._params.add('bl_log_lam', value=log_lam, min=2, max=10)
                init_p = self.baseline_params.get('p', 0.01)
                self._params.add('bl_p', value=init_p, min=0.0001, max=0.5)
        elif 'bl_log_lam' in self._params:
            # Remove AsLS parameters if optimizing simultaneously is turned off
            del self._params['bl_log_lam']
            del self._params['bl_p']
        
        # Add rolling-ball hyper-parameter if needed
        if use_simultaneous and self.baseline_method == 'rolling_ball':
            if 'bl_radius' not in self._params:
                init_radius = self.baseline_params.get('radius', 20)
                self._params.add('bl_radius', value=init_radius, min=1, max=max(len(self.x) // 2, 2))
        elif 'bl_radius' in self._params:
            del self._params['bl_radius']
        
        # Add manual baseline control-point parameters if needed with bounds
        if use_simultaneous and self.baseline_method == 'manual':
            # remove old ones if point count changed
            to_delete = [k for k in self._params if k.startswith('bl_mpt_')]
            for k in to_delete:
                del self._params[k]
            points = self.baseline_params.get('points', [])
            # Bound control points to ±2x the data span from their initial y values
            max_y_delta = 2 * y_span
            for i, (_, y_val) in enumerate(points):
                self._params.add(f'bl_mpt_{i}', value=y_val, 
                                min=y_val - max_y_delta, max=y_val + max_y_delta)
        else:
            to_delete = [k for k in self._params if k.startswith('bl_mpt_')]
            for k in to_delete:
                del self._params[k]
        
        # Add Shirley endpoint offsets if needed
        if use_simultaneous and self.baseline_method == 'shirley':
            span = max(np.max(y_data) - np.min(y_data), 1.0)
            if 'bl_start_offset' not in self._params:
                self._params.add('bl_start_offset', value=self.baseline_params.get('start_offset', 0.0),
                                 min=-2*span, max=2*span)
            if 'bl_end_offset' not in self._params:
                self._params.add('bl_end_offset', value=self.baseline_params.get('end_offset', 0.0),
                                 min=-2*span, max=2*span)
        else:
            for name in ['bl_start_offset', 'bl_end_offset']:
                if name in self._params:
                    del self._params[name]

        if use_simultaneous:
            # Local state for progress tracking
            last_bl = np.zeros_like(self.x)
            
            def _apply_range_flat(baseline_array):
                """Apply baseline_range by flattening outside mask."""
                x_min, x_max = self.baseline_range
                mask = (self.x >= (x_min if x_min is not None else -np.inf)) & \
                       (self.x <= (x_max if x_max is not None else np.inf))
                indices = np.where(mask)[0]
                if len(indices) < 2:
                    return baseline_array
                bl_full = baseline_array.copy()
                if indices[0] > 0:
                    bl_full[:indices[0]] = baseline_array[indices[0]]
                if indices[-1] < len(self.x) - 1:
                    bl_full[indices[-1]+1:] = baseline_array[indices[-1]]
                return bl_full
            
            def _compute_baseline(residual_y, params_for_baseline):
                """Calculate baseline on the configured range and extend flat outside."""
                bl_full, _ = self._compute_baseline_with_range(residual_y, params_for_baseline)
                return bl_full

            # Objective function for simultaneous fitting
            def objective(params):
                y_peaks = self._model.eval(params, x=self.x)
                if is_parametric_baseline:
                    # Parametric: extract baseline from params, apply calc range, remove baseline from peak sum
                    if self.baseline_method == 'polynomial':
                        degree = self.baseline_params.get('degree', 2)
                        coeffs = [params[f'bl_c{i}'].value for i in range(degree + 1)]  # ascending order c0 + c1*x ...
                        bl_raw = np.polynomial.polynomial.polyval(self.x, coeffs)
                    elif self.baseline_method == 'linear':
                        bl_raw = params['bl_slope'].value * self.x + params['bl_intercept'].value
                    else:
                        bl_raw = np.zeros_like(y_data)

                    bl_full = _apply_range_flat(bl_raw)
                    peaks_only = y_peaks - bl_raw  # remove baseline part from composite eval
                    residual = y_data - (peaks_only + bl_full)
                else:
                    # Non-parametric baseline: iterative calculation using apply_baseline
                    y_residual_for_baseline = y_data - y_peaks
                    
                    # Use current optimized baseline parameters if present
                    current_params = self.baseline_params.copy()
                    if 'bl_log_lam' in params:
                        current_params['lam'] = 10**params['bl_log_lam'].value
                    if 'bl_p' in params:
                        current_params['p'] = params['bl_p'].value
                    if 'bl_radius' in params:
                        current_params['radius'] = params['bl_radius'].value
                    if 'bl_start_offset' in params:
                        current_params['start_offset'] = params['bl_start_offset'].value
                    if 'bl_end_offset' in params:
                        current_params['end_offset'] = params['bl_end_offset'].value
                    if self.baseline_method == 'manual':
                        pts = self.baseline_params.get('points', [])
                        if pts:
                            new_pts = []
                            for i, (px, _) in enumerate(pts):
                                pname = f'bl_mpt_{i}'
                                y_val = params[pname].value if pname in params else pts[i][1]
                                new_pts.append((px, y_val))
                            current_params['points'] = new_pts

                    bl_full = _compute_baseline(y_residual_for_baseline, current_params)
                    residual = y_data - (y_peaks + bl_full)
                if self.negative_penalty_enabled:
                    if not is_parametric_baseline:
                        negative_mask = (y_data - bl_full) < 0
                        if np.any(negative_mask):
                            total_abs_res = np.sum(np.abs(residual))
                            penalty = (self.negative_penalty_weight/100.0)*total_abs_res*(np.sum(negative_mask)/len(y_data))
                            residual[negative_mask] += penalty / np.sum(negative_mask)
                
                # Store current baseline for callback
                nonlocal last_bl
                last_bl = bl_full
                
                return residual

            def wrapped_iter_cb(params, iter, resid, *args, **kws):
                if iter_cb:
                    iter_cb(self.x, last_bl)

            # Run optimization
            from lmfit import minimize
            self.result = minimize(objective, self._params, method=method, max_nfev=max_nfev, 
                                  iter_cb=wrapped_iter_cb if iter_cb else None, **fit_kws)
            
            # Post-optimization: finalize baseline and corrected data
            if is_parametric_baseline:
                # Extract baseline from model result
                if self.baseline_method == 'polynomial':
                    degree = self.baseline_params.get('degree', 2)
                    coeffs = [self.result.params[f'bl_c{i}'].value for i in range(degree + 1)]  # ascending
                    bl_raw = np.polynomial.polynomial.polyval(self.x, coeffs)
                    # Update parameters for GUI sync
                    self.baseline_params['coeffs'] = coeffs
                    self.baseline_params['degree'] = degree
                elif self.baseline_method == 'linear':
                    slope = self.result.params['bl_slope'].value
                    intercept = self.result.params['bl_intercept'].value
                    bl_raw = slope * self.x + intercept
                    self.baseline_params['slope'] = slope
                    self.baseline_params['intercept'] = intercept
                else:
                    bl_raw = np.zeros_like(y_data)

                self.baseline = _apply_range_flat(bl_raw)
                self.baseline_raw = self.baseline.copy()
                
                y_peaks = self._model.eval(self.result.params, x=self.x) - bl_raw
                self.result.best_fit = y_peaks
                self.y_corrected = y_data - self.baseline
                self.result.residual = self.y_corrected - y_peaks
            else:
                # Non-parametric (AsLS, rolling ball, Shirley)
                y_peaks = self._model.eval(self.result.params, x=self.x)
                
                # Update baseline parameters if they were optimized
                current_params = self.baseline_params.copy()
                if self.baseline_method == 'asls':
                    if 'bl_log_lam' in self.result.params:
                        current_params['lam'] = 10**self.result.params['bl_log_lam'].value
                        self.baseline_params['lam'] = current_params['lam']
                    if 'bl_p' in self.result.params:
                        current_params['p'] = self.result.params['bl_p'].value
                        self.baseline_params['p'] = current_params['p']
                elif self.baseline_method == 'rolling_ball':
                    if 'bl_radius' in self.result.params:
                        current_params['radius'] = self.result.params['bl_radius'].value
                        self.baseline_params['radius'] = current_params['radius']
                elif self.baseline_method == 'shirley':
                    if 'bl_start_offset' in self.result.params:
                        current_params['start_offset'] = self.result.params['bl_start_offset'].value
                        self.baseline_params['start_offset'] = current_params['start_offset']
                    if 'bl_end_offset' in self.result.params:
                        current_params['end_offset'] = self.result.params['bl_end_offset'].value
                        self.baseline_params['end_offset'] = current_params['end_offset']
                elif self.baseline_method == 'manual':
                    pts = self.baseline_params.get('points', [])
                    new_pts = []
                    for i, (px, _) in enumerate(pts):
                        pname = f'bl_mpt_{i}'
                        y_val = self.result.params[pname].value if pname in self.result.params else pts[i][1]
                        new_pts.append((px, y_val))
                    if new_pts:
                        current_params['points'] = new_pts
                        self.baseline_params['points'] = new_pts

                self.baseline, _ = self._compute_baseline_with_range(y_data - y_peaks, current_params)
                self.baseline_raw = self.baseline.copy()
                self.y_corrected = y_data - self.baseline
                self.result.best_fit = y_peaks
                self.result.residual = self.y_corrected - y_peaks
            
            # Helper for component evaluation
            def eval_wrapper(x=None):
                if x is None: x = self.x
                return self._model.eval_components(params=self.result.params, x=x)
            self.result.eval_components = eval_wrapper

        else:
            # Traditional mode: Fixed baseline
            if not skip_baseline_correction:
                self.apply_baseline_correction()
            
            if self.negative_penalty_enabled:
                def penalty_objective(params):
                    y_peaks = self._model.eval(params, x=self.x)
                    residual = self.y_corrected - y_peaks
                    negative_mask = self.y_corrected < 0
                    if np.any(negative_mask):
                        total_abs_res = np.sum(np.abs(residual))
                        negative_count = np.sum(negative_mask)
                        penalty = (self.negative_penalty_weight/100.0) * total_abs_res * (negative_count/len(y_data))
                        residual[negative_mask] += penalty / negative_count
                    return residual
                
                from lmfit import minimize
                self.result = minimize(penalty_objective, self._params, method=method, max_nfev=max_nfev, **fit_kws)
                self.result.best_fit = self._model.eval(self.result.params, x=self.x)
                self.result.residual = self.y_corrected - self.result.best_fit
                
                def eval_wrapper(x=None):
                    if x is None: x = self.x
                    return self._model.eval_components(params=self.result.params, x=x)
                self.result.eval_components = eval_wrapper
            else:
                self.result = self._model.fit(self.y_corrected, self._params, x=self.x, method=method, max_nfev=max_nfev, **fit_kws)

        return self.result
    
    def evaluate_components(self, x=None):
        """
        Evaluate individual components at given x values.
        
        Parameters
        ----------
        x : array_like, optional
            X values to evaluate at. If None, use self.x
        
        Returns
        -------
        components_dict : dict
            Dictionary mapping component prefixes to evaluated curves
        """
        if self.result is None:
            raise ValueError("No fit result available. Run fit() first.")
        
        if x is None:
            x = self.x
        
        components_dict = {}
        
        for comp in self.components:
            prefix = comp['prefix']
            comp_name = f"{comp['type']}_{prefix}"
            
            # Evaluate this component
            components_dict[comp_name] = self.result.eval_components(x=x)[prefix]
        
        return components_dict
    
    def get_statistics(self):
        """
        Calculate fit statistics.
        
        Returns
        -------
        stats : dict
            Dictionary containing fit quality metrics
        """
        if self.result is None:
            raise ValueError("No fit result available. Run fit() first.")
        
        from .statistics import calculate_statistics
        
        # For MinimizerResult, we manually added best_fit and residual
        # Make sure they exist before calculating statistics
        if not hasattr(self.result, 'best_fit') or not hasattr(self.result, 'residual'):
            # Return default stats if attributes missing
            return {
                'r_squared': 0.0,
                'adj_r_squared': 0.0,
                'rmse': 0.0,
                'chi_squared': 0.0,
                'reduced_chi_squared': 0.0,
                'aic': 0.0,
                'bic': 0.0
            }
        
        stats = calculate_statistics(
            self.y_corrected,
            self.result.best_fit,
            len(self.result.params)
        )
        
        # Include baseline parameters in stats for easy access
        if self.baseline_method:
            stats['baseline_method'] = self.baseline_method
            stats['baseline_params'] = self.baseline_params.copy() if self.baseline_params else {}
            stats['baseline_optimize'] = self.optimize_baseline
            if self.baseline_range and self.baseline_range != (None, None):
                stats['baseline_range'] = self.baseline_range
            
            # Add baseline coefficients from pre-calculated baseline
            fitted_bl_params = {}
            
            # Polynomial/linear coefficients from baseline_result (always pre-calculated)
            if hasattr(self, 'baseline_result') and self.baseline_result:
                if 'coefficients' in self.baseline_result:
                    coeffs = self.baseline_result['coefficients']
                    for i, c in enumerate(coeffs):
                        fitted_bl_params[f'c{i}'] = c
                if 'slope' in self.baseline_result:
                    fitted_bl_params['slope'] = self.baseline_result['slope']
                if 'intercept' in self.baseline_result:
                    fitted_bl_params['intercept'] = self.baseline_result['intercept']
            
            # Add optimized parameters if simultaneous mode was used
            if self.optimize_baseline and self.result and hasattr(self.result, 'params'):
                params = self.result.params
                if 'bl_log_lam' in params:
                    fitted_bl_params['lam (optimized)'] = 10**params['bl_log_lam'].value
                if 'bl_p' in params:
                    fitted_bl_params['p (optimized)'] = params['bl_p'].value
                if self.baseline_method == 'polynomial':
                    degree = self.baseline_params.get('degree', 2)
                    coeffs = []
                    for i in range(degree + 1):
                        pname = f'bl_c{i}'
                        if pname in params:
                            coeffs.append(params[pname].value)
                    if coeffs:
                        for i, c in enumerate(coeffs):
                            fitted_bl_params[f'c{i} (optimized)'] = c
                elif self.baseline_method == 'linear':
                    if 'bl_slope' in params:
                        fitted_bl_params['slope (optimized)'] = params['bl_slope'].value
                    if 'bl_intercept' in params:
                        fitted_bl_params['intercept (optimized)'] = params['bl_intercept'].value
                elif self.baseline_method == 'rolling_ball':
                    if 'bl_radius' in params:
                        fitted_bl_params['radius (optimized)'] = params['bl_radius'].value
                elif self.baseline_method == 'shirley':
                    if 'bl_start_offset' in params:
                        fitted_bl_params['start_offset (optimized)'] = params['bl_start_offset'].value
                    if 'bl_end_offset' in params:
                        fitted_bl_params['end_offset (optimized)'] = params['bl_end_offset'].value
                elif self.baseline_method == 'manual':
                    pts = self.baseline_params.get('points', [])
                    for i, (px, _) in enumerate(pts):
                        pname = f'bl_mpt_{i}'
                        if pname in params:
                            fitted_bl_params[f'pt{i}_y (optimized)'] = params[pname].value
                            fitted_bl_params[f'pt{i}_x'] = px
            
            if fitted_bl_params:
                stats['fitted_baseline_params'] = fitted_bl_params
            
        return stats
    
    def get_fit_report(self):
        """
        Get detailed fit report.
        
        Returns
        -------
        str
            Fit report string
        """
        if self.result is None:
            raise ValueError("No fit result available. Run fit() first.")
        
        # Check if result has fit_report method (ModelResult does, MinimizerResult doesn't)
        report = ""
        if self.optimize_baseline and self.baseline_method is not None:
            report += "[[ SIMULTANEOUS BASELINE OPTIMIZATION ENABLED ]]\n"
            report += f"Method: {self.baseline_method}\n"
            if self.baseline_range[0] is not None or self.baseline_range[1] is not None:
                report += f"Range: {self.baseline_range}\n"
            
            # Add summary of optimized values
            report += "Fitted Parameters:\n"
            if self.baseline_method == 'asls':
                report += f"  - Lambda: {self.baseline_params.get('lam', 0):.2e}\n"
                report += f"  - p:      {self.baseline_params.get('p', 0):.4f}\n"
            elif self.baseline_method == 'polynomial':
                degree = self.baseline_params.get('degree', 0)
                coeffs = self.baseline_params.get('coeffs', [])
                report += f"  - Degree: {degree}\n"
                for i, c in enumerate(coeffs):
                    report += f"  - c{len(coeffs)-1-i}:   {c:.4f}\n" # polyval uses highest degree first
            elif self.baseline_method == 'linear':
                report += f"  - Slope:     {self.baseline_params.get('slope', 0):.4f}\n"
                report += f"  - Intercept: {self.baseline_params.get('intercept', 0):.4f}\n"
            elif self.baseline_method == 'rolling_ball':
                report += f"  - Radius:    {self.baseline_params.get('radius', 0)}\n"
            elif self.baseline_method == 'shirley':
                report += f"  - Tol:       {self.baseline_params.get('tol', 0):.2e}\n"
                report += f"  - Max Iter:  {self.baseline_params.get('max_iter', 0)}\n"
                if 'start_offset' in self.baseline_params:
                    report += f"  - Start Offset: {self.baseline_params.get('start_offset', 0):.4f}\n"
                if 'end_offset' in self.baseline_params:
                    report += f"  - End Offset:   {self.baseline_params.get('end_offset', 0):.4f}\n"
            report += "\n"
        
        # Component summary with relative weights
        if self.result is not None and self.components:
            params = self.result.params
            # total amplitude across components
            total_amp = 0.0
            amps = []
            for comp in self.components:
                prefix = comp['prefix']
                amp_name = f"{prefix}amplitude"
                amp_val = params[amp_name].value if amp_name in params else 0.0
                amps.append(amp_val)
                total_amp += amp_val

            report += "[[ COMPONENTS ]]\n"
            report += f"Total Amplitude: {total_amp:.6g}\n\n"
            
            for idx, comp in enumerate(self.components):
                prefix = comp['prefix']
                ctype = comp['type']
                amp_name = f"{prefix}amplitude"
                center_name = f"{prefix}center"
                sigma_name = f"{prefix}sigma"
                gamma_name = f"{prefix}gamma"
                amp_val = params[amp_name].value if amp_name in params else 0.0
                center_val = params[center_name].value if center_name in params else 0.0
                sigma_val = params[sigma_name].value if sigma_name in params else None
                gamma_val = params[gamma_name].value if gamma_name in params else None
                rel = amp_val / total_amp if total_amp > 0 else 0.0
                report += f"Comp {idx+1} ({ctype}):\n"
                report += f"  - Center:    {center_val:.6g}\n"
                report += f"  - Weight:    {rel*100:.2f}%\n"
                if sigma_val is not None:
                    report += f"  - Sigma:     {sigma_val:.6g}\n"
                if gamma_val is not None:
                    report += f"  - Gamma:     {gamma_val:.6g}\n"
            report += "\n"

        if hasattr(self.result, 'fit_report') and callable(self.result.fit_report):
            report += self.result.fit_report()
            return report
        else:
            # Generate custom fit report for MinimizerResult
            from lmfit import fit_report
            report += fit_report(self.result)
            return report
