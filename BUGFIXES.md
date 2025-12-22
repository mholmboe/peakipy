# Stabilization and Bug Fixes

## PySide6 Transition Fixes

### 1. Vanishing Fit Results (FIXED ✓)
**Problem:** After a fit completed, the results (plot and stats) would sometimes vanish or be overwritten by a blank evaluation.
**Root Cause:** The `_sync_sliders_to_results` method was updating sliders, which emitted signals that triggered `update_live_preview`, which in turn cleared the fit plot for a live preview.
**Solution:** Implemented signal blocking in `main_window.py` around the slider sync logic:
```python
self.control_panel.blockSignals(True)
self.control_panel.component_panel.blockSignals(True)
try:
    self._sync_sliders_to_results(result)
finally:
    self.control_panel.component_panel.blockSignals(False)
    self.control_panel.blockSignals(False)
```

### 2. Live Preview Initialization (FIXED ✓)
**Problem:** Plotting individual components on application start caused clutter.
**Solution:** Added "Live Preview" toggles to both baseline and component panels. Components/baselines are only overlaid when these toggles are active.

### 3. Component Weight/Amplitude Sync (FIXED ✓)
**Problem:** After a fit, the amplitude sliders and weight sliders did not reflect the fitted results correctly.
**Solution:** Updated `_sync_sliders_to_results` to calculate fractional weights (`fitted_amplitude / total_area`) and set a common total amplitude across all components, making the manual adjustment intuitive.

### 4. Dynamic Import Errors (FIXED ✓)
**Problem:** `NameError` and `AttributeError` during baseline method switches in the Qt version.
**Solution:** Ensured all widgets (like `method_combo`) are correctly re-initialized and all PySide6 classes are properly imported in `baseline_panel.py` and `component_panel.py`.

## Core Logic Fixes (Legacy)

### 1. Dimension Mismatch Error (FIXED ✓)
**Problem:** `x and y must have same first dimension` error during ROI cropping.
**Solution:** Updated logic to ensure the `ProfileFitter` and `PlotPanel` always use the same processed data buffers.

### 2. Error Logging System (NEW ✓)
**Implementation:**
- Created `utils/logger.py` with comprehensive logging utility.
- Logs written to `logs/profile_fitting_YYYYMMDD_HHMMSS.log`.

## Files Reference (PySide6)
- `gui_qt/main_window.py`: Application orchestration.
- `gui_qt/control_panel.py`: Unified control panel.
- `gui_qt/plot_panel.py`: Matplotlib canvas integration.
- `gui_qt/results_panel.py`: Fit statistics display.

## Recent Fixes (December 2024)

### 5. Component Count Desync (FIXED ✓)
**Problem:** Changing the number of components (e.g., 3 → 2) still used the old component count for fitting.
**Root Cause:** The fitter cached its model (`_model`, `_params`) and didn't invalidate on component changes.
**Solution:** Added cache invalidation in `clear_components()` and `remove_component()` in `fitter.py`.

### 6. Polynomial Baseline Coefficients Not Displayed (FIXED ✓)
**Problem:** Polynomial/linear baseline coefficients weren't shown in statistics.
**Root Cause:** `robust_polynomial_baseline()` wasn't returning coefficients.
**Solution:** Updated function to return `(baseline, corrected, coeffs)` and added `baseline_result` storage in fitter.

### 7. Polynomial Model Initialization Error (FIXED ✓)
**Problem:** Simultaneous polynomial baseline fitting caused NaN errors.
**Root Cause:** lmfit's `PolynomialModel` initializes coefficients to `-inf`.
**Solution:** Added explicit initialization of `bl_c0`, `bl_c1`, etc. to `0` in `model_builder.py`.

### 8. Parameter Range Constraints (FIXED ✓)
**Problem:** Component parameters (center, sigma, gamma) could be set to unreasonable values.
**Solution:** Added `set_data_range()` methods to dynamically constrain parameters to loaded data range.

### 9. Non-Negative Amplitude/Width Bounds (FIXED ✓)
**Problem:** Fitting could produce negative sigma/gamma values.
**Solution:** Added explicit `min=1e-6` bounds for sigma/gamma and `min=0` for amplitude in `model_builder.py`.
