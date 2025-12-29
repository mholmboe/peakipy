# Modernized UI (PySide6)

The application has been rebuilt from the ground up using PySide6 (Qt 6), providing a more responsive, stable, and aesthetically pleasing experience.

## New Architectural Features

### 1. Unified Control Sidebar
All fitting controls are now hosted in a single, resizable sidebar. This ensures that even with complex multi-peak models, the UI remains organized.

## Technical Layout
The controls follow a logical top-to-bottom data flow:
1.  **Data Source**: File operations (Load/Export). 
2.  **Data Preprocessing**: Range and interpolation.
3.  **Baseline Correction**: Background subtraction with independent **Calculation Range** selection and **Simultaneous Optimization** support for baseline parameters (ASLS λ/p, polynomial & linear coefficients, rolling-ball radius, Shirley endpoint offsets). Linear now includes slope/intercept sliders for manual seeding. A **Manual** mode lets you click up to 15 control points (linear or cubic interpolation) to define a custom baseline and optionally optimize the point heights.
4.  **Profile Components**: Peak parameterization and manual placement.
5.  **Fitting Options**: Global constraints (Normalize, Non-Negative).

### 2. Live Preview Engine
A first-class feature in the new GUI is the **Live Preview** toggle (available for both Baselines and Components).
- Adjusting a slider emits a signal that instantly updates the plot.
- Performance is optimized via a pseudo-evaluation layer that doesn't block the main UI thread.

### 3. Smart Initialization Row
The **Profile Components** panel now features a dedicated row for peak initialization:
- **Evenly Spaced**: Automatic placement based on the data range.
- **GMM (Gaussian Mixture Model)**: Smart peak detection using statistical clustering.
- **⟳ Re-initialize**: Reset peak parameters to their automatic guesses at any time.

### 4. Resizable Dock Widgets
The GUI uses a dock-based architecture, allowing the plotting area to expand as needed while keeping control groups accessible.

## Technical Improvements
- **Signal/Slot Architecture**: Robust event handling replaces the fragile Tkinter `trace_add` system.
- **Custom Parameter Controls**: Unified widgets for Center, Amplitude, Width, and Weight that synchronize a high-precision slider with a double-value spinbox.
- **Automatic Scaling**: Support for High-DPI displays.

## Standard User Workflow
1. **Load Data** → UI auto-populates X-ranges.
2. **Preprocessing** (Optional) → Crop or interpolate.
3. **Set Baseline** → Turn on **Live Preview** and slide parameters; for Manual, click the plot to place 2–15 control points (linear/cubic) and optionally optimize their heights.
4. **Add Peaks** → Choose a profile and use **Live Preview** for manual placement of centers/widths/amplitudes.
5. **Fit & Export** → Run fit (with optional simultaneous baseline refinement) and export results + data.
