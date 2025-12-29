"""
Baseline Correction Panel for PySide6. 
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QCheckBox, 
                               QComboBox, QStackedWidget, QFormLayout, QSpinBox,
                               QLineEdit)
from PySide6.QtCore import Signal, Qt
from .widgets.parameter_control import ParameterControl

class QtBaselinePanel(QWidget):
    """
    Panel for baseline correction configuration.
    """
    # Signals for orchestration
    baseline_changed = Signal()
    preview_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        self.group_box = QGroupBox("Baseline Correction")
        self.group_box.setCheckable(True)
        self.group_box.setChecked(True)
        self.group_box.toggled.connect(self.baseline_changed)
        
        layout = QVBoxLayout(self.group_box)
        
        # Method selection
        form = QFormLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItems(["AsLS", "Polynomial", "Linear", "Rolling Ball", "Shirley", "Manual"])
        self.method_combo.setMinimumWidth(120)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("Method:", self.method_combo)
        
        self.live_preview_check = QCheckBox("Live Preview")
        self.live_preview_check.setToolTip("Updates plot as you adjust parameters")
        self.live_preview_check.toggled.connect(self.baseline_changed)
        
        self.simultaneous_check = QCheckBox("Optimize Simultaneously")
        self.simultaneous_check.setToolTip("Optimize baseline parameters together with peaks (ASLS λ/p, polynomial & linear coefficients, rolling-ball radius, Shirley endpoint offsets).")
        self.simultaneous_check.toggled.connect(self.baseline_changed)
        
        form.addRow(self.live_preview_check)
        form.addRow(self.simultaneous_check)
        
        layout.addLayout(form)
        
        # Stacked widget for method-specific parameters
        self.params_stack = QStackedWidget()
        
        # 1. AsLS Params
        asls_widget = QWidget()
        asls_layout = QVBoxLayout(asls_widget)
        asls_layout.setContentsMargins(0, 10, 0, 0)
        # lam is log10 in control panel
        self.asls_lam = ParameterControl("Smoothness (λ)", 2.0, 9.0, 5.0, decimals=1)
        self.asls_p = ParameterControl("Asymmetry (p)", 0.001, 0.1, 0.01, step=0.001, decimals=3)
        self.asls_niter = ParameterControl("Iterations", 1, 50, 10, decimals=0)
        
        for w in [self.asls_lam, self.asls_p, self.asls_niter]:
            asls_layout.addWidget(w)
            w.valueChanged.connect(self.baseline_changed)
        self.params_stack.addWidget(asls_widget)
        
        # 2. Polynomial Params
        poly_widget = QWidget()
        poly_layout = QVBoxLayout(poly_widget)
        self.poly_degree = ParameterControl("Degree", 0, 5, 2, decimals=0)
        self.poly_degree.valueChanged.connect(self.baseline_changed)
        poly_layout.addWidget(self.poly_degree)
        self.params_stack.addWidget(poly_widget)
        
        # 3. Linear (slope/intercept)
        linear_widget = QWidget()
        linear_layout = QVBoxLayout(linear_widget)
        linear_layout.setContentsMargins(0, 10, 0, 0)
        self.linear_slope = ParameterControl("Slope", -1e3, 1e3, 0.0, step=0.1, decimals=3)
        self.linear_intercept = ParameterControl("Intercept", -1e3, 1e3, 0.0, step=0.1, decimals=3)
        for w in [self.linear_slope, self.linear_intercept]:
            linear_layout.addWidget(w)
            w.valueChanged.connect(self.baseline_changed)
        self.params_stack.addWidget(linear_widget)
        
        # 4. Rolling Ball
        rb_widget = QWidget()
        rb_layout = QVBoxLayout(rb_widget)
        self.rb_radius = ParameterControl("Radius", 1, 200, 20, decimals=0)
        self.rb_radius.valueChanged.connect(self.baseline_changed)
        rb_layout.addWidget(self.rb_radius)
        self.params_stack.addWidget(rb_widget)
        
        # 5. Shirley
        shirley_widget = QWidget()
        shirley_layout = QVBoxLayout(shirley_widget)
        self.shirley_tol = ParameterControl("Tolerance", 0.0, 1e-4, 1e-6, step=1e-7, decimals=7)
        self.shirley_iter = ParameterControl("Max Iterations", 1, 100, 50, decimals=0)
        shirley_layout.addWidget(self.shirley_tol)
        shirley_layout.addWidget(self.shirley_iter)
        self.shirley_tol.valueChanged.connect(self.baseline_changed)
        self.shirley_iter.valueChanged.connect(self.baseline_changed)
        self.params_stack.addWidget(shirley_widget)

        # 6. Manual
        manual_widget = QWidget()
        manual_layout = QVBoxLayout(manual_widget)
        manual_layout.setContentsMargins(0, 10, 0, 0)
        self.manual_points = []
        self.manual_max_points = ParameterControl("Max Points", 2, 15, 5, decimals=0)
        self.manual_interp = QComboBox()
        self.manual_interp.addItems(["Linear", "Cubic"])
        from PySide6.QtWidgets import QPushButton
        self.manual_start_btn = QPushButton("Edit Baseline (click plot)")
        self.manual_clear_btn = QPushButton("Clear Points")
        manual_layout.addWidget(self.manual_max_points)
        manual_layout.addWidget(self.manual_interp)
        manual_layout.addWidget(self.manual_start_btn)
        manual_layout.addWidget(self.manual_clear_btn)
        for w in [self.manual_max_points, self.manual_interp]:
            if hasattr(w, 'valueChanged'):
                w.valueChanged.connect(self.baseline_changed)
            elif hasattr(w, 'currentIndexChanged'):
                w.currentIndexChanged.connect(self.baseline_changed)
        self.params_stack.addWidget(manual_widget)
        
        layout.addWidget(self.params_stack)
        
        main_layout.addWidget(self.group_box)

    def _on_method_changed(self, index):
        self.params_stack.setCurrentIndex(index)
        self.baseline_changed.emit()

    def get_config(self):
        """Return dict of baseline configuration."""
        if not self.group_box.isChecked():
            return None
        
        method = self.method_combo.currentText().lower().replace(" ", "_")
        params = {}
        
        if method == "asls":
            params = {
                "lam": 10**self.asls_lam.value(),
                "p": self.asls_p.value(),
                "niter": int(self.asls_niter.value())
            }
        elif method == "polynomial":
            params = {"degree": int(self.poly_degree.value())}
        elif method == "rolling_ball":
            params = {"radius": int(self.rb_radius.value())}
        elif method == "shirley":
            params = {
                "tol": self.shirley_tol.value(),
                "max_iter": int(self.shirley_iter.value())
            }
        elif method == "linear":
            params = {
                "slope": self.linear_slope.value(),
                "intercept": self.linear_intercept.value()
            }
        elif method == "manual":
            params = {
                "points": getattr(self, "manual_points", []),
                "interp": self.manual_interp.currentText().lower(),
                "max_points": int(self.manual_max_points.value())
            }

        return {
            "method": method, 
            "params": params,
            "optimize": self.simultaneous_check.isChecked()
        }

    def set_baseline_params(self, method, params):
        """Update UI controls with fitted baseline parameters."""
        # Find index for method
        idx = self.method_combo.findText(method.title().replace("_", " "))
        if idx >= 0:
            self.method_combo.setCurrentIndex(idx)
            
        if method == "asls":
            if "lam" in params:
                # Value is 10^lam, we need log10 for the slider
                import numpy as np
                self.asls_lam.setValue(np.log10(params["lam"]))
            if "p" in params: self.asls_p.setValue(params["p"])
            if "niter" in params: self.asls_niter.setValue(params["niter"])
        elif method == "polynomial":
            if "degree" in params: self.poly_degree.setValue(params["degree"])
        elif method == "rolling_ball":
            if "radius" in params: self.rb_radius.setValue(params["radius"])
        elif method == "shirley":
            if "tol" in params: self.shirley_tol.setValue(params["tol"])
            if "max_iter" in params: self.shirley_iter.setValue(params["max_iter"])
        elif method == "linear":
            if "slope" in params: self.linear_slope.setValue(params["slope"])
            if "intercept" in params: self.linear_intercept.setValue(params["intercept"])
        elif method == "manual":
            self.manual_points = params.get("points", [])
            if "interp" in params:
                idx = self.manual_interp.findText(params["interp"].title())
                if idx >= 0:
                    self.manual_interp.setCurrentIndex(idx)
            if "max_points" in params:
                self.manual_max_points.setValue(params["max_points"])

    def set_manual_points(self, points):
        """Update stored manual points (list of (x, y))."""
        self.manual_points = points
