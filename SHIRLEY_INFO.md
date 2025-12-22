# Shirley Background - XPS Analysis

A new baseline correction method specifically for X-ray Photoelectron Spectroscopy (XPS) data.

## Implementation

**File**: `core/baseline/shirley.py`

The Shirley background algorithm assumes the background at any point is proportional to the total peak area above that point, accounting for inelastic scattering in XPS measurements.

### Algorithm

```python
def shirley_background(x, y, tol=1e-5, max_iter=50):
    # Iterative calculation
    # Background[i] ∝ ∫(signal - background) from i to end
```

### Features

- Handles both increasing and decreasing energy scales
- Iterative convergence with tolerance control
- Proper treatment of cumulative integrals
- Multiple variants: standard, simplified, and active (with ROI selection)

## Test Results

**Test Data**: Synthetic XPS spectrum with Voigt peak and step background

![Shirley Test Result](/Users/miho0052/Dropbox/Coding/Antigravity/ProfileFitting/test_shirley_result.png)

**Performance**:
- Background RMSE: 50.65 (on synthetic data)
- Convergence: Achieved in <50 iterations
- Suitable for XPS core-level spectra

## GUI Integration

Added to baseline method dropdown with parameters:
- **Tolerance**: Convergence criterion (default: 1e-5)
- **Max Iterations**: Maximum iteration limit (default: 50)

## Reference

Shirley, D.A. (1972). "High-Resolution X-Ray Photoemission Spectrum of the Valence Bands of Gold." *Physical Review B*, 5(12), 4709.

## Usage Example

```python
from core.baseline import shirley_background

# XPS data (binding energy typically decreases)
x = np.linspace(300, 280, 200)  # Binding energy
y = xps_spectrum  # Measured intensity

# Apply Shirley correction
baseline, corrected = shirley_background(x, y, tol=1e-5, max_iter=50)
```
