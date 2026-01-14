"""
Main Window for PySide6 Profile Fitting Application. 
"""

import sys
import os
import json
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QDockWidget, QStatusBar, QFileDialog, QMessageBox, QApplication,
                               QProgressDialog, QMenu)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import Qt

from core.data_import import load_data_file
from core.fitting import ProfileFitter
from core.data_preprocessing import crop_roi, interpolate_data, remove_outliers_zscore, remove_outliers_iqr, smooth_data
from core.fitting.model_builder import build_composite_model
from core.fitting.initializers import init_with_gmm, init_evenly_spaced
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSettings, QSize, QPoint, QRunnable, QThreadPool, QObject, Slot
from PySide6.QtGui import QIcon

class FitWorker(QThread):
    """Background thread for fitting."""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(np.ndarray, np.ndarray) # x, baseline progress

    def __init__(self, fitter, skip_baseline_correction=True):
        super().__init__()
        self.fitter = fitter
        self.skip_baseline_correction = skip_baseline_correction

    def run(self):
        try:
            def iter_callback(x, baseline):
                self.progress.emit(x, baseline)
                
            result = self.fitter.fit(skip_baseline_correction=self.skip_baseline_correction, iter_cb=iter_callback)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class EvalWorkerSignals(QObject):
    """Signals for EvalWorker."""
    finished = Signal(object, object)  # y_eval, stats
    error = Signal(str)

class EvalWorker(QRunnable):
    """Background runnable for model evaluation."""
    def __init__(self, fitter, components):
        super().__init__()
        self.fitter = fitter
        self.components = components
        self.signals = EvalWorkerSignals()

    @Slot()
    def run(self):
        try:
            from core.fitting.model_builder import build_composite_model
            from core.fitting.statistics import calculate_statistics
            
            model, params = build_composite_model(self.components)
            y_eval = model.eval(params, x=self.fitter.x)
            stats = calculate_statistics(self.fitter.y_corrected, y_eval, len(params))
            self.signals.finished.emit(y_eval, stats)
        except Exception as e:
            self.signals.error.emit(str(e))


from .plot_panel import QtPlotPanel
from .control_panel import QtControlPanel
from .results_panel import QtResultsPanel

class QtMainWindow(QMainWindow):
    """
    Main Application Window (PySide6).
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PeakiPy Baseline/Profile Fitting App")
        self.resize(1450, 950)
        
        # Data storage
        self.x_data_raw = None
        self.y_data_raw = None
        self.x_data = None
        self.y_data = None
        self.fitter = None
        self.current_data_file = None  # Track loaded file for sessions
        self.fit_progress_dialog = None
        
        # Thread pool for non-blocking evaluation
        self.threadpool = QThreadPool()
        
        # Persistence (must be before _create_menu for recent files)
        self.settings = QSettings("ExpLab", "ProfileFitting")
        
        self._create_actions()
        self._setup_ui()
        self._create_menu()
        self._connect_signals()
        
        self._load_settings()
        
        # Live Preview Debouncer
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self.update_live_preview)
        
        # Manual baseline state
        self.manual_points = []
        self.manual_editing = False
        
        self.statusBar().showMessage("Ready")

    def _setup_ui(self):
        # Central Plot Panel
        self.plot_panel = QtPlotPanel()
        self.setCentralWidget(self.plot_panel)
        
        # Control Panel Dock
        self.control_dock = QDockWidget("Controls", self)
        self.control_dock.setObjectName("control_dock")
        self.control_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.control_panel = QtControlPanel(self)
        scroll.setWidget(self.control_panel)
        
        self.control_dock.setWidget(scroll)
        self.control_dock.setMinimumWidth(280)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.control_dock)
        
        # Results Panel Dock
        self.results_dock = QDockWidget("Results", self)
        self.results_dock.setObjectName("results_dock")
        self.results_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.results_panel = QtResultsPanel(self)
        self.results_dock.setWidget(self.results_panel)
        self.results_dock.setMinimumWidth(300)
        self.addDockWidget(Qt.RightDockWidgetArea, self.results_dock)
        
        # Persistence: Nesting and Tabs
        self.setDockNestingEnabled(True)

    def _connect_signals(self):
        # Control Panel signals
        self.control_panel.preprocessingChanged.connect(self.apply_preprocessing)
        self.control_panel.evaluateRequested.connect(self.run_evaluation)
        self.control_panel.fitRequested.connect(self.run_fit)
        self.control_panel.loadDataRequested.connect(self.load_file)
        self.control_panel.exportRequested.connect(self.export_results)
        self.control_panel.component_panel.auto_init_btn.clicked.connect(self.auto_initialize)
        
        # Settings changes for Live Preview
        self.control_panel.baseline_panel.baseline_changed.connect(self.handle_settings_changed)
        self.control_panel.component_panel.componentsChanged.connect(self.handle_settings_changed)

        # Manual baseline buttons
        self.control_panel.baseline_panel.manual_start_btn.clicked.connect(self._start_manual_baseline_edit)
        self.control_panel.baseline_panel.manual_clear_btn.clicked.connect(self._clear_manual_baseline)

    def handle_settings_changed(self):
        """Trigger debounced live preview."""
        self._preview_timer.start(300) # 300ms debounce

    # ============ Manual Baseline Helpers ============
    def _start_manual_baseline_edit(self):
        """Enable manual baseline point capture on the plot."""
        if not self.fitter:
            self.statusBar().showMessage("Load and process data before editing baseline.")
            return
        max_pts = int(self.control_panel.baseline_panel.manual_max_points.value())
        self.manual_editing = True
        self.statusBar().showMessage(f"Manual baseline: click up to {max_pts} points.")
        self.plot_panel.start_manual_capture(self._on_manual_click, max_pts)

    def _clear_manual_baseline(self):
        """Clear manual baseline points and preview."""
        self.manual_points = []
        self.control_panel.baseline_panel.set_manual_points([])
        self.plot_panel.clear_baseline_preview()
        self.plot_panel.stop_manual_capture()
        self.manual_editing = False
        self.statusBar().showMessage("Manual baseline cleared.")

    def _on_manual_click(self, x, y):
        """Handle manual baseline point placement."""
        if not self.manual_editing:
            return
        max_pts = int(self.control_panel.baseline_panel.manual_max_points.value())
        if len(self.manual_points) >= max_pts:
            self.statusBar().showMessage(f"Max manual points reached ({max_pts}).")
            return
        self.manual_points.append((x, y))
        # sort by x
        self.manual_points = sorted(self.manual_points, key=lambda p: p[0])
        self.control_panel.baseline_panel.set_manual_points(self.manual_points)
        if len(self.manual_points) >= 2:
            try:
                interp = self.control_panel.baseline_panel.manual_interp.currentText().lower()
                from core.baseline import apply_baseline
                baseline, corrected, _ = apply_baseline('manual', self.fitter.x, self.fitter.y, 
                                                        points=self.manual_points, interp=interp)
                self.plot_panel.update_manual_baseline_preview(self.fitter.x, self.fitter.y, baseline, corrected, self.manual_points)
            except Exception as e:
                self.statusBar().showMessage(f"Manual baseline error: {e}")
        else:
            # just show control points
            self.plot_panel.update_manual_baseline_preview(self.fitter.x, self.fitter.y, np.zeros_like(self.fitter.y), self.fitter.y, self.manual_points)

    def update_live_preview(self):
        """Perform non-destructive preview update."""
        if not self.fitter: return
        
        # 1. Check component preview (tends to clear axes)
        comp_preview = self.control_panel.component_panel.live_preview_check.isChecked()
        if comp_preview:
            # run_evaluation calls plot_panel.update_fit_plot which clears axes
            self.run_evaluation()
            
        # 2. Check baseline preview (adds overlay)
        bl_panel = self.control_panel.baseline_panel
        bl_enabled = bl_panel.group_box.isChecked()
        bl_preview = bl_panel.live_preview_check.isChecked()
        
        if not bl_enabled or not bl_preview:
            # Only clear if we didn't just clear via run_evaluation
            if not comp_preview:
                self.plot_panel.clear_baseline_preview()
        else:
            try:
                config = bl_panel.get_config()
                if config:
                    from core.baseline import apply_baseline
                    
                    x, y = self.fitter.x, self.fitter.y
                    
                    # Simultaneous Preview: subtract current peak model
                    if config["optimize"]:
                        try:
                            # Evaluate current model with slider parameters
                            self._sync_fitter()
                            model, params = build_composite_model(self.fitter.components)
                            y_peaks = model.eval(params, x=x)
                            y_for_baseline = y - y_peaks
                        except Exception as e:
                            print(f"DEBUG: Peak eval failed for simultaneous preview: {e}")
                            y_for_baseline = y
                    else:
                        y_for_baseline = y

                    xmin, xmax = self.control_panel.get_calc_range()
                    # ... masking logic
                    mask = np.ones_like(x, dtype=bool)
                    if xmin is not None: mask &= (x >= xmin)
                    if xmax is not None: mask &= (x <= xmax)
                    indices = np.where(mask)[0]
                    
                    if len(indices) < 2:
                        indices = np.arange(len(x))
                    
                    x_sub = x[indices]
                    y_sub = y_for_baseline[indices] # Use residual for baseline
                    
                    try:
                        res = apply_baseline(config["method"], x_sub, y_sub, **config["params"])
                        bl_full = np.zeros_like(x)
                        bl_full[indices] = res[0]
                        
                        # Pad edges
                        if indices[0] > 0:
                            bl_full[:indices[0]] = res[0][0]
                        if indices[-1] < len(x) - 1:
                            bl_full[indices[-1]+1:] = res[0][-1]
                            
                        # Corrected data is ALWAYS y_raw - baseline
                        corr = y - bl_full
                        
                        # Apply visual normalization if enabled
                        opts = self.control_panel.get_fitting_options()
                        if opts["normalize"]:
                            norm_range = opts.get("normalize_range", (None, None))
                            mask = np.ones_like(x, dtype=bool)
                            if norm_range[0] is not None: mask &= (x >= norm_range[0])
                            if norm_range[1] is not None: mask &= (x <= norm_range[1])
                            
                            c_subset = corr[mask] if np.any(mask) else corr
                            c_max = np.max(c_subset) if len(c_subset) > 0 else 0
                            
                            if c_max > 0:
                                corr = corr / c_max

                        self.plot_panel.update_baseline_preview(x, y, bl_full, corr)
                    except Exception as inner_e:
                        print(f"DEBUG: apply_baseline failed: {inner_e}")
                        self.statusBar().showMessage(f"Baseline error: {str(inner_e)}")
                else:
                    self.plot_panel.clear_baseline_preview()
            except Exception as e:
                print(f"DEBUG: Preview Error: {e}")
                self.statusBar().showMessage(f"Baseline preview error: {str(e)}")

    def _create_actions(self):
        # File Actions
        self.load_action = QAction("&Load Data File...", self)
        self.load_action.setShortcut("Ctrl+O")
        self.load_action.triggered.connect(self.load_file)
        
        self.export_action = QAction("&Export Results...", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.triggered.connect(self.export_results)
        
        self.export_plot_action = QAction("Export &Plot...", self)
        self.export_plot_action.setShortcut("Ctrl+P")
        self.export_plot_action.triggered.connect(self.export_plot)
        
        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)
        
        # Session Actions
        self.save_session_action = QAction("&Save Session...", self)
        self.save_session_action.setShortcut("Ctrl+S")
        self.save_session_action.triggered.connect(self.save_session)
        
        self.load_session_action = QAction("&Load Session...", self)
        self.load_session_action.setShortcut("Ctrl+Shift+O")
        self.load_session_action.triggered.connect(self.load_session)

    def _create_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.load_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_session_action)
        file_menu.addAction(self.load_session_action)
        
        # Recent Files submenu
        self.recent_menu = QMenu("Recent Files", self)
        file_menu.addMenu(self.recent_menu)
        self._update_recent_files_menu()
        
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)
        file_menu.addAction(self.export_plot_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.control_dock.toggleViewAction())
        view_menu.addAction(self.results_dock.toggleViewAction())
        
        help_menu = menubar.addMenu("&Help")
        about_action = help_menu.addAction("&About")
        about_action.triggered.connect(self.show_about)

    def load_file(self):
        """Open file dialog and load data."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "Text files (*.txt);;All files (*)"
        )
        
        if filename:
            try:
                self._load_data_file_internal(filename)
                self.statusBar().showMessage(f"Loaded {len(self.x_data)} points")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def apply_preprocessing(self):
        """Apply ROI, outlier removal, interpolation, smoothing, and normalization."""
        if self.x_data_raw is None:
            return
            
        config = self.control_panel.get_preprocessing_config()
        try:
            x_proc, y_proc = self.x_data_raw.copy(), self.y_data_raw.copy()
            
            # Step 1: Outlier removal (before other processing)
            if config.get("outlier_method", "none") != "none":
                threshold = config.get("outlier_threshold", 3.0)
                if config["outlier_method"] == "zscore":
                    x_proc, y_proc, _ = remove_outliers_zscore(x_proc, y_proc, threshold)
                elif config["outlier_method"] == "iqr":
                    x_proc, y_proc, _ = remove_outliers_iqr(x_proc, y_proc, threshold)
            
            # Step 2: Crop to ROI
            x_proc, y_proc = crop_roi(x_proc, y_proc, config["x_min"], config["x_max"])
            
            # Step 3: Interpolation
            if config["interpolate"]:
                x_proc, y_proc = interpolate_data(x_proc, y_proc, config["x_min"], config["x_max"], 
                                                step=config["step"])
            
            # Step 4: Smoothing (after interpolation for uniform spacing)
            if config.get("smoothing", False):
                window = config.get("smooth_window", 11)
                order = config.get("smooth_order", 3)
                y_proc = smooth_data(y_proc, window_length=window, polyorder=order)
            
            # Step 5: Normalization
            # Step 5: Normalization
            if self.control_panel.norm_check.isChecked():
                # Use range from inputs
                opts = self.control_panel.get_fitting_options()
                norm_range = opts.get("normalize_range", (None, None))
                
                y_subset = y_proc
                if norm_range[0] is not None or norm_range[1] is not None:
                     mask = np.ones_like(x_proc, dtype=bool)
                     if norm_range[0] is not None: mask &= (x_proc >= norm_range[0])
                     if norm_range[1] is not None: mask &= (x_proc <= norm_range[1])
                     if np.any(mask):
                         y_subset = y_proc[mask]
                
                y_max = np.max(y_subset) if len(y_subset) > 0 else 0
                if y_max > 0:
                    y_proc = y_proc / y_max
            
            self.x_data, self.y_data = x_proc, y_proc
            self.fitter = ProfileFitter(self.x_data, self.y_data)
            self.manual_points = []
            self.control_panel.baseline_panel.set_manual_points([])
            self.plot_panel.clear_baseline_preview()
            
            # Update component parameter ranges based on data
            x_min, x_max = np.min(self.x_data), np.max(self.x_data)
            self.control_panel.component_panel.set_data_range(x_min, x_max)
            
            self.plot_panel.plot_data(self.x_data, self.y_data)
            self.statusBar().showMessage(f"Processed: {len(self.x_data)} points")
        except Exception as e:
            QMessageBox.warning(self, "Processing Error", str(e))

    def _sync_fitter(self):
        """Transfer UI settings to the ProfileFitter object."""
        if not self.fitter: return
        
        # 1. Clear components
        self.fitter.clear_components()
        
        # 2. Add components
        comp_configs = self.control_panel.component_panel.get_config()
        for i, config in enumerate(comp_configs):
            profile = config["type"]
            prefix = f"c{i+1}_"
            
            # Apply weight to amplitude
            amp = config["amplitude"] * config["weight"]
            
            if profile == "voigt":
                self.fitter.add_component(profile, prefix=prefix, center=config["center"], 
                                         amplitude=amp, sigma=config["sigma"], gamma=config["gamma"])
            else:
                self.fitter.add_component(profile, prefix=prefix, center=config["center"], 
                                         amplitude=amp, **{("sigma" if profile == "gaussian" else "gamma"): config["width"]})

        # 3. Baseline
        base_config = self.control_panel.baseline_panel.get_config()
        if base_config:
            if base_config["method"] == "manual" and (not base_config["params"].get("points") or len(base_config["params"].get("points")) < 2):
                raise ValueError("Manual baseline requires at least 2 points. Click points on the plot first.")
            self.fitter.set_baseline(base_config["method"], **base_config["params"])
            self.fitter.baseline_range = self.control_panel.get_calc_range()
            self.fitter.optimize_baseline = base_config["optimize"]
        else:
            self.fitter.baseline_method = None
            self.fitter.baseline_range = (None, None)
            self.fitter.optimize_baseline = False

        # 4. Global options
        opts = self.control_panel.get_fitting_options()
        self.fitter._non_negative = opts["non_negative"]

    def run_evaluation(self):
        """Evaluate model without fitting (uses thread pool for responsiveness)."""
        if not self.fitter: return
        
        try:
            # Reset fitter state to original data before each evaluation
            # This prevents cumulative state corruption during live preview
            self.fitter.y_work = self.fitter.y_raw.copy()
            self.fitter.y = self.fitter.y_work
            self.fitter.y_corrected = self.fitter.y_work.copy()
            self.fitter.normalized = False
            self.fitter.baseline = None
            self.fitter.baseline_raw = None
            
            self._sync_fitter()
            
            # Apply normalization/baseline
            # Apply normalization/baseline
            opts = self.control_panel.get_fitting_options()
            norm_range = opts.get("normalize_range", (None, None))
            if opts["normalize"]: self.fitter.normalize_raw_data(norm_range)
            if self.fitter.baseline_method: self.fitter.apply_baseline_correction()
            
            curr_comps = [c.copy() for c in self.fitter.components]
            if opts["normalize"]:
                # Manual intensity normalization to sync components
                x = self.fitter.x
                mask = np.ones_like(x, dtype=bool)
                if norm_range[0] is not None: mask &= (x >= norm_range[0])
                if norm_range[1] is not None: mask &= (x <= norm_range[1])
                y_subset = self.fitter.y_corrected[mask] if np.any(mask) else self.fitter.y_corrected
                max_val = np.max(y_subset) if len(y_subset) > 0 else 0
                
                if max_val > 0:
                    scale = 1.0 / max_val
                    self.fitter.y_corrected *= scale
                    if self.fitter.norm_factor == 1.0: self.fitter.norm_factor = max_val
                    for c in curr_comps:
                        if 'params' in c and 'amplitude' in c['params']:
                            c['params']['amplitude'] *= scale
            
            self.statusBar().showMessage("Evaluating...")
            
            # Create worker and run in thread pool
            worker = EvalWorker(self.fitter, curr_comps)
            worker.signals.finished.connect(self._on_eval_complete)
            worker.signals.error.connect(self._on_eval_error)
            self.threadpool.start(worker)
            
        except Exception as e:
            QMessageBox.critical(self, "Evaluation Error", str(e))

    def _on_eval_complete(self, y_eval, stats):
        """Handle evaluation completion."""
        self.statusBar().showMessage("Evaluation Complete")
        
        # Build components for plotting
        comps = {}
        for i, c in enumerate(self.fitter.components):
            m_i, p_i = build_composite_model([c])
            comps[f"Comp {i+1}"] = m_i.eval(p_i, x=self.fitter.x)
        
        # Show baseline-subtracted preview (consistent scale)
        y_plot = self.fitter.y_corrected
        y_fit_total = y_eval
        residual = self.fitter.y_corrected - y_eval

        baseline_for_plot = self.fitter.baseline_raw if getattr(self.fitter, "baseline_raw", None) is not None else self.fitter.baseline
        self.plot_panel.update_fit_plot(self.fitter.x, y_plot, y_fit_total, 
                                      residual, components=comps, 
                                      baseline=baseline_for_plot, show_fit=False,
                                      y_raw=self.y_data)
        self.results_panel.display_results(stats, "Evaluation only (no fit report)")

    def _on_eval_error(self, message):
        """Handle evaluation error."""
        self.statusBar().showMessage("Evaluation Failed")
        QMessageBox.critical(self, "Evaluation Error", message)


    def run_fit(self):
        """Run full fitting optimization in background thread."""
        if not self.fitter: return
        
        self.statusBar().showMessage("Fitting...")
        self.control_panel.setEnabled(False) # Prevent changes during fit
        
        # Show progress dialog
        self.fit_progress_dialog = QProgressDialog("Optimizing fit...", "Cancel", 0, 0, self)
        self.fit_progress_dialog.setWindowTitle("Fitting")
        self.fit_progress_dialog.setWindowModality(Qt.WindowModal)
        self.fit_progress_dialog.setMinimumDuration(500)  # Show after 500ms
        self.fit_progress_dialog.setValue(0)
        
        try:
            self._sync_fitter()
            
            opts = self.control_panel.get_fitting_options()
            norm_range = opts.get("normalize_range", (None, None))
            if opts["normalize"]: self.fitter.normalize_raw_data(norm_range)
            
            # Always pre-apply baseline for consistent plotting
            if self.fitter.baseline_method:
                self.fitter.apply_baseline_correction()
                
            # Note: normalize_intensity removed here as it's handled post-fit

            if opts["non_negative"]: self.fitter.enable_negative_penalty()

            # Background thread execution
            skip_bl = not self.fitter.optimize_baseline  # allow simultaneous optimization to recompute baseline
            self.fit_worker = FitWorker(self.fitter, skip_baseline_correction=skip_bl)
            self.fit_worker.finished.connect(self._on_fit_complete)
            self.fit_worker.error.connect(self._on_fit_error)
            self.fit_worker.progress.connect(self._on_fit_progress)
            self.fit_progress_dialog.canceled.connect(self._on_fit_canceled)
            self.fit_worker.start()
            
        except Exception as e:
            self._on_fit_error(str(e))

    def _on_fit_canceled(self):
        """Handle fit cancellation by user."""
        if hasattr(self, 'fit_worker') and self.fit_worker.isRunning():
            self.fit_worker.terminate()
            self.fit_worker.wait(1000)
        self.control_panel.setEnabled(True)
        self.statusBar().showMessage("Fit Cancelled")

    def _on_fit_complete(self, result):
        if self.fit_progress_dialog:
            self.fit_progress_dialog.close()
        self.control_panel.setEnabled(True)
        self.statusBar().showMessage("Fit Completed")
        
        try:
            # Post-fit Normalization Scaling
            opts = self.control_panel.get_fitting_options()
            if opts["normalize"]:
                norm_range = opts.get("normalize_range", (None, None))
                x = self.fitter.x
                mask = np.ones_like(x, dtype=bool)
                if norm_range[0] is not None: mask &= (x >= norm_range[0])
                if norm_range[1] is not None: mask &= (x <= norm_range[1])
                y_subset = self.fitter.y_corrected[mask] if np.any(mask) else self.fitter.y_corrected
                max_val = np.max(y_subset) if len(y_subset) > 0 else 0
                
                if max_val > 0:
                    scale = 1.0 / max_val
                    self.fitter.y_corrected *= scale
                    result.best_fit *= scale
                    result.residual *= scale
                    # Scale parameters
                    for pname in result.params:
                        if 'amplitude' in pname: # Check if parameter is amplitude
                            result.params[pname].value *= scale
                            if result.params[pname].stderr:
                                result.params[pname].stderr *= scale

            # Update UI
            stats = self.fitter.get_statistics()
            report = self.fitter.get_fit_report()
            
            # Map component evals for plotter - use available keys from result
            comp_evals = result.eval_components(x=self.fitter.x)
            plot_comps = {}
            for key in comp_evals.keys():
                if key.startswith('c') and key.endswith('_'):
                    # Extract component number from key (e.g., 'c1_' -> 1)
                    comp_num = key[1:-1]  
                    plot_comps[f"Comp {comp_num}"] = comp_evals[key]

            # Plot baseline-subtracted experimental data with peak-only fit (consistent scale)
            y_plot = self.fitter.y_corrected
            y_fit_total = result.best_fit
            residual_plot = result.residual

            # Plot original-scale baseline if available
            baseline_for_plot = self.fitter.baseline_raw if getattr(self.fitter, "baseline_raw", None) is not None else self.fitter.baseline
            self.plot_panel.update_fit_plot(self.fitter.x, y_plot, y_fit_total, 
                                          residual_plot, components=plot_comps, 
                                          baseline=baseline_for_plot, show_fit=True,
                                          y_raw=self.y_data)
            self.results_panel.display_results(stats, report)
            
            # Sync sliders with fitted results - BLOCK SIGNALS to prevent feedback loop
            self.control_panel.blockSignals(True)
            self.control_panel.component_panel.blockSignals(True)
            try:
                self._sync_sliders_to_results(result)
            finally:
                self.control_panel.component_panel.blockSignals(False)
                self.control_panel.blockSignals(False)
            
        except Exception as e:
            QMessageBox.critical(self, "UI Update Error", str(e))

    def _on_fit_error(self, message):
        if self.fit_progress_dialog:
            self.fit_progress_dialog.close()
        self.control_panel.setEnabled(True)
        self.statusBar().showMessage("Fit Failed")
        QMessageBox.critical(self, "Fit Error", message)

    def _on_fit_progress(self, x, baseline):
        """Update plot with intermediate baseline state."""
        # Calculate corrected data based on intermediate baseline
        corr = self.fitter.y - baseline
        self.plot_panel.update_baseline_preview(x, self.fitter.y, baseline, corr)

    def auto_initialize(self):
        """Auto-initialize component parameters."""
        if not self.fitter: return
        
        n = self.control_panel.component_panel.num_peaks_spin.value()
        method = self.control_panel.component_panel.get_init_method()
        
        # Determine data for init
        y_init = self.fitter.y_corrected if self.fitter.baseline_method else self.y_data
        
        params = None
        if method == "gmm":
            params = init_with_gmm(self.x_data, y_init, n)
            if params is None:
                self.statusBar().showMessage("GMM failed, falling back to evenly spaced")
        
        if params is None:
            params = init_evenly_spaced(self.x_data, n, y_init)
        
        try:
            self.control_panel.component_panel.set_component_params(params)
            self.statusBar().showMessage(f"Parameters re-initialized ({method})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Initialization Error", f"Failed to set parameters: {str(e)}\n\nCheck console for details.")

    def export_results(self):
        """Save results and data to files."""
        if not self.fitter or not hasattr(self.fitter, 'result'):
            QMessageBox.warning(self, "Export", "No fit results to export.")
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "Text files (*.txt)")
        if filename:
            try:
                # Logic copied from Streamlit Dual Export implementation
                base = os.path.splitext(filename)[0]
                results_file = f"{base}_results.txt"
                data_file = f"{base}_data.txt"
                
                # 1. Results
                with open(results_file, 'w') as f:
                    f.write(self.fitter.get_fit_report())
                    f.write("\n\nStatistics:\n")
                    for k, v in self.fitter.get_statistics().items():
                        f.write(f"{k}: {v}\n")
                
                # 2. Data
                x = self.fitter.x
                y_exp = self.fitter.y_corrected
                y_fit = self.fitter.result.best_fit
                res = self.fitter.result.residual
                comp_evals = self.fitter.result.eval_components(x=x)
                comp_keys = sorted([k for k in comp_evals.keys() if k.startswith('c')])
                
                header = "X\tY_Exp\tY_Fit\tResidual" + "".join([f"\tComp_{i+1}" for i in range(len(comp_keys))])
                with open(data_file, 'w') as f:
                    f.write(header + "\n")
                    for i in range(len(x)):
                        line = f"{x[i]:.6e}\t{y_exp[i]:.6e}\t{y_fit[i]:.6e}\t{res[i]:.6e}"
                        for k in comp_keys:
                            line += f"\t{comp_evals[k][i]:.6e}"
                        f.write(line + "\n")
                
                QMessageBox.information(self, "Export", f"Exported successfully to:\n{results_file}\n{data_file}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def export_plot(self):
        """Save plot to image file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        if filename:
            try:
                self.plot_panel.fig.savefig(filename)
                self.statusBar().showMessage(f"Plot saved to {os.path.basename(filename)}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save plot: {str(e)}")

    def _sync_sliders_to_results(self, result):
        """Update UI controls with fitted parameters."""
        profile = self.control_panel.component_panel.profile_combo.currentText().lower()
        params = result.params
        
        # Calculate total amplitude for relative weighting
        total_amp = 0
        fitted_amps = []
        for i in range(len(self.fitter.components)):
            prefix = f"c{i+1}_"
            amp_val = params[f"{prefix}amplitude"].value if f"{prefix}amplitude" in params else 0
            fitted_amps.append(amp_val)
            total_amp += amp_val
            
        new_params = []
        for i in range(len(self.fitter.components)):
            prefix = f"c{i+1}_"
            p = {"type": profile}
            if f"{prefix}center" in params: p["center"] = params[f"{prefix}center"].value
            
            # Set relative weight and common total amplitude
            p["amplitude"] = total_amp
            p["weight"] = fitted_amps[i] / total_amp if total_amp > 0 else 0
            
            if profile == "voigt":
                if f"{prefix}sigma" in params: p["sigma"] = params[f"{prefix}sigma"].value
                if f"{prefix}gamma" in params: p["gamma"] = params[f"{prefix}gamma"].value
            else:
                w_param = f"{prefix}sigma" if profile == "gaussian" else f"{prefix}gamma"
                if w_param in params: p["width"] = params[w_param].value
            
            new_params.append(p)
            
        self.control_panel.component_panel.set_component_params(new_params)
        
        # Sync baseline parameters
        if self.fitter.baseline_method:
            self.control_panel.baseline_panel.set_baseline_params(
                self.fitter.baseline_method, self.fitter.baseline_params
            )

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About", 
                         "Baseline and Profile Fitting Application\nVersion 2.0 (PySide6)\n\n"
                         "A modern GUI for multi-component peak fitting.")

    def _load_settings(self):
        """Restore window geometry and state."""
        geom = self.settings.value("geometry")
        if geom: self.restoreGeometry(geom)
        
        state = self.settings.value("windowState")
        if state: self.restoreState(state)
        
        # Enforce visibility of docks on startup to prevent "lost panel" issues
        if self.control_dock: self.control_dock.setVisible(True)
        if self.results_dock: self.results_dock.setVisible(True)

    def closeEvent(self, event):
        """Save settings on close."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        event.accept()

    # ==================== Session Management ====================
    
    def save_session(self):
        """Save current session to a .pffit JSON file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "Profile Fitting Session (*.pffit);;All files (*)"
        )
        if not filename:
            return
            
        if not filename.endswith('.pffit'):
            filename += '.pffit'
            
        session = {
            "version": "2.0",
            "data_file": self.current_data_file,
            "preprocessing": self.control_panel.get_preprocessing_config(),
            "baseline": self.control_panel.baseline_panel.get_config(),
            "components": self.control_panel.component_panel.get_config(),
            "fitting_options": self.control_panel.get_fitting_options()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(session, f, indent=2, default=str)
            self.statusBar().showMessage(f"Session saved to {os.path.basename(filename)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save session: {e}")

    def load_session(self):
        """Load session from a .pffit JSON file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "Profile Fitting Session (*.pffit);;All files (*)"
        )
        if not filename:
            return
            
        try:
            with open(filename, 'r') as f:
                session = json.load(f)
            
            # Load data file if specified
            if session.get("data_file") and os.path.exists(session["data_file"]):
                self._load_data_file_internal(session["data_file"])
            
            # Apply preprocessing
            if session.get("preprocessing"):
                pre = session["preprocessing"]
                self.control_panel.x_min.setText(str(pre.get("x_min", "auto")) if pre.get("x_min") else "auto")
                self.control_panel.x_max.setText(str(pre.get("x_max", "auto")) if pre.get("x_max") else "auto")
                self.control_panel.interp_check.setChecked(pre.get("interpolate", False))
                self.control_panel.interp_step.setText(str(pre.get("step", "auto")) if pre.get("step") else "auto")
            
            # Apply baseline config
            if session.get("baseline"):
                bl = session["baseline"]
                if bl.get("method"):
                    self.control_panel.baseline_panel.set_baseline_params(bl["method"], bl.get("params", {}))
                    self.control_panel.baseline_panel.simultaneous_check.setChecked(bl.get("optimize", False))
            
            # Apply fitting options
            if session.get("fitting_options"):
                opts = session["fitting_options"]
                self.control_panel.norm_check.setChecked(opts.get("normalize", True))
                self.control_panel.penalty_check.setChecked(opts.get("non_negative", True))
            
            self.statusBar().showMessage(f"Session loaded from {os.path.basename(filename)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load session: {e}")

    def _load_data_file_internal(self, filename):
        """Internal method to load a data file (used by load_file and load_session)."""
        self.x_data_raw, self.y_data_raw = load_data_file(filename)
        self.x_data = self.x_data_raw.copy()
        self.y_data = self.y_data_raw.copy()
        
        self.fitter = ProfileFitter(self.x_data, self.y_data)
        self.manual_points = []
        self.control_panel.baseline_panel.set_manual_points([])
        self.current_data_file = filename
        self.plot_panel.plot_data(self.x_data, self.y_data, title=f"Data: {os.path.basename(filename)}")
        
        # Update control panel ranges
        self.control_panel.x_min.setText(f"{self.x_data.min():.4f}")
        self.control_panel.x_max.setText(f"{self.x_data.max():.4f}")
        
        # Add to recent files
        self._add_to_recent_files(filename)

    # ==================== Recent Files ====================
    
    def _update_recent_files_menu(self):
        """Populate the Recent Files submenu."""
        self.recent_menu.clear()
        recent = self.settings.value("recentFiles", [])
        
        if not recent:
            action = self.recent_menu.addAction("(No recent files)")
            action.setEnabled(False)
            return
            
        for filepath in recent[:10]:  # Max 10 recent files
            action = self.recent_menu.addAction(os.path.basename(filepath))
            action.setData(filepath)
            action.triggered.connect(self._open_recent_file)
        
        self.recent_menu.addSeparator()
        clear_action = self.recent_menu.addAction("Clear Recent Files")
        clear_action.triggered.connect(self._clear_recent_files)

    def _add_to_recent_files(self, filepath):
        """Add a file to the recent files list."""
        recent = self.settings.value("recentFiles", [])
        if filepath in recent:
            recent.remove(filepath)
        recent.insert(0, filepath)
        self.settings.setValue("recentFiles", recent[:10])
        self._update_recent_files_menu()

    def _open_recent_file(self):
        """Open a file from the recent files menu."""
        action = self.sender()
        if action:
            filepath = action.data()
            if os.path.exists(filepath):
                try:
                    self._load_data_file_internal(filepath)
                    self.statusBar().showMessage(f"Loaded {len(self.x_data)} points from recent file")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
            else:
                QMessageBox.warning(self, "File Not Found", f"File no longer exists:\n{filepath}")

    def _clear_recent_files(self):
        """Clear the recent files list."""
        self.settings.setValue("recentFiles", [])
        self._update_recent_files_menu()
