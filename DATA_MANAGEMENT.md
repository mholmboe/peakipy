# Data Management and Preprocessing

## Overview
The application handles the full data lifecycle from import and noise reduction to final fit results export, now unified in a modern sidebar-based control panel.

## Key Features

### 1. Robust Data Import
**Auto Header Detection**
- Automatically skips non-numeric header lines and detects comments starting with `#`.
- Gracefully handles irregular spacing and multiple delimiters (tabs, spaces, commas).
- **📁 Load Data**: Located in the **Data Source** group at the top of the sidebar.

- **Calculation Range**: Independently define the sub-range for baseline calculation. If your data spans [10, 90] but you only want to fit the baseline between [20, 80], the application will calculate the curve in that window and extend it as a flat line outside.
- **Simultaneous Optimization**: Baseline parameters can be refined during every iteration of the peak fitting process (ASLS λ/p, polynomial & linear coefficients, rolling-ball radius, Shirley endpoint offsets). Linear mode supports manual slope/intercept seeds that can then be optimized. A manual click-defined baseline (up to 15 points, linear/cubic interpolation) can also be optimized by adjusting control-point heights. This allows the baseline to adapt and results in a more accurate fit for complex experimental data.
- **Live Preview**: Instantly see both the calculated baseline and the orange corrected data markers.
- **Baseline Coefficients**: For polynomial and linear baselines, the calculated coefficients are displayed in the statistics panel after fitting.
- **Real-time Feedback**: During a fit, the plot updates sequentially as the algorithm converges. This "visual progress" allows you to verify the baseline's evolution in real-time.

### 3. Integrated Export System
**💾 Export Results**
Generates two synchronized text files for every fit:
- **`*_results.txt`**: Complete fit report, statistics (R², RMSE, AIC, BIC), and all parameter values with uncertainties.
- **`*_data.txt`**: Tab-separated columnar data for publication plotting.
  - Columns: `X`, `Y_Exp`, `Y_Fit`, `Residual`, `Comp_1`...`Comp_N`.

## Standard Workflow

1.  **Import**: Click **📁 Load Data** in the top section.
2.  **Filter**: Adjust X-ranges in **Data Preprocessing** and click **Apply Data Processing**.
3.  **Correct**: Select a baseline method and toggle **Live Preview** to verify the subtraction. For Manual, click the plot to place 2–15 control points (linear/cubic); for Linear/Polynomial, adjust slope/intercept or degree; enable **Optimize Simultaneously** if needed.
4.  **Fit**: Configure peak components and click **🚀 Run Fit** (baseline parameters can be optimized together with peaks).
5.  **Export**: Click **💾 Export Results** to save your work.

## Technical Layout
The controls follow a logical top-to-bottom data flow:
1.  **Data Source**: File operations (Load/Export).
2.  **Data Preprocessing**: Range and interpolation.
3.  **Baseline Correction**: Background subtraction with independent **Calculation Range** selection.
4.  **Profile Components**: Peak parameterization and manual placement.
5.  **Fitting Options**: Global constraints (Normalize, Non-Negative).
