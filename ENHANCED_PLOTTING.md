# Multi-Layer Visualization

The plotting panel provides high-fidelity, real-time feedback of the fitting process, leveraging the Matplotlib backend integrated into PySide6.

## Visual Components

### 1. Data Layers
- **Experimental Data (Baseline-Subtracted)**: Dark gray circles showing the corrected data that is actually fitted.
- **Total Fit Curve**: Solid red line showing the peak-only model on the same baseline-subtracted scale.
- **Fitted Baseline**: Dash-dot gray line shown after fit for reference; not added to the plotted “Total Fit”. Manual baselines also show orange control-point markers.
- **Residuals**: A separate bottom panel showing `Measured - Calculated` on the corrected scale.
- **Confidence Bands**: Shaded gray region at ±2σ in the residuals panel.

### 2. Live Preview Visualization
When the **Live Preview** toggle is enabled in the control panels:
- **Baseline Preview**:
  - The estimated baseline is shown as a **dashed gray line**.
  - The baseline-corrected data is shown as **orange dots**.
  - This allows you to verify subtraction quality before running a full fit.
- **Component Preview**:
  - Individual peaks are shown as dashed lines in distinct colors.
  - Colors cycle through: `Red`, `Purple`, `Blue`, `Green`, `Orange`.
  - Peak shapes update instantly as you move the Center, Amplitude, or Width sliders.

## Interactive Features
- **High-DPI Support**: Plots remain crisp on Retina and 4K displays.
- **Dynamic Resizing**: The plot adapts its layout as the control panel is resized.
- **Publication Ready**: Clean aesthetics with proper LaTeX-style labels for mathematical symbols (σ, γ).
