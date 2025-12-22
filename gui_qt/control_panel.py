"""
Unified Control Panel for PySide6.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QGroupBox, 
                               QFormLayout, QLineEdit, QCheckBox, QHBoxLayout, QLabel,
                               QScrollArea, QFrame)
from PySide6.QtCore import Signal
try:
    import qtawesome as qta
    HAS_ICONS = True
except ImportError:
    HAS_ICONS = False
from .baseline_panel import QtBaselinePanel
from .component_panel import QtComponentPanel

class QtControlPanel(QWidget):
    """
    Unified panel hosting all fitting controls.
    """
    fitRequested = Signal()
    evaluateRequested = Signal()
    preprocessingChanged = Signal()
    loadDataRequested = Signal()
    exportRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(250)
        self._setup_ui()

    def _setup_ui(self):
        # Main layout with scroll area
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        
        # 0. Global File Actions
        file_group = QGroupBox("Data Source")
        file_layout = QHBoxLayout(file_group)
        self.load_btn = QPushButton("Load Data")
        if HAS_ICONS:
            self.load_btn.setIcon(qta.icon('mdi.folder-open'))
        self.load_btn.clicked.connect(self.loadDataRequested)
        self.export_btn = QPushButton("Export Results")
        if HAS_ICONS:
            self.export_btn.setIcon(qta.icon('mdi.content-save'))
        self.export_btn.clicked.connect(self.exportRequested)
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.export_btn)
        layout.addWidget(file_group)

        # 1. Data Treatment (compact 2-row layout)
        pre_group = QGroupBox("Data Treatment")
        pre_layout = QVBoxLayout(pre_group)
        
        # Row 1: X Min, X Max, Calc Min, Calc Max
        row1 = QHBoxLayout()
        self.x_min = QLineEdit("auto")
        self.x_min.setMaximumWidth(75)
        self.x_max = QLineEdit("auto")
        self.x_max.setMaximumWidth(75)
        self.calc_x_min = QLineEdit("auto")
        self.calc_x_min.setMaximumWidth(75)
        self.calc_x_min.setPlaceholderText("auto")
        self.calc_x_max = QLineEdit("auto")
        self.calc_x_max.setMaximumWidth(75)
        self.calc_x_max.setPlaceholderText("auto")
        
        row1.addWidget(QLabel("X:"))
        row1.addWidget(self.x_min)
        row1.addWidget(QLabel("-"))
        row1.addWidget(self.x_max)
        row1.addSpacing(8)
        row1.addWidget(QLabel("Calc:"))
        row1.addWidget(self.calc_x_min)
        row1.addWidget(QLabel("-"))
        row1.addWidget(self.calc_x_max)
        row1.addStretch()
        pre_layout.addLayout(row1)
        
        # Row 2: Interp + Step, Norm, Apply
        row2 = QHBoxLayout()
        self.interp_check = QCheckBox("Interp.")
        self.interp_step = QLineEdit("auto")
        self.interp_step.setMaximumWidth(50)
        self.interp_step.setEnabled(False)
        self.interp_check.toggled.connect(self.interp_step.setEnabled)
        row2.addWidget(self.interp_check)
        row2.addWidget(self.interp_step)
        row2.addSpacing(8)
        self.norm_check = QCheckBox("Norm.")
        self.norm_check.setChecked(True)
        row2.addWidget(self.norm_check)
        row2.addStretch()
        self.apply_pre_btn = QPushButton("Apply")
        row2.addWidget(self.apply_pre_btn)
        self.apply_pre_btn.clicked.connect(self.preprocessingChanged)
        pre_layout.addLayout(row2)
        
        layout.addWidget(pre_group)
        
        # 2. Baseline
        self.baseline_panel = QtBaselinePanel()
        layout.addWidget(self.baseline_panel)
        
        # 3. Fitting (Buttons ABOVE components)
        fitting_group = QGroupBox("Fitting")
        fitting_layout = QVBoxLayout(fitting_group)
        
        # Run actions row: Non-negative + Evaluate + Run Fit (at top)
        btn_layout = QHBoxLayout()
        self.penalty_check = QCheckBox("Non-negative data")
        self.penalty_check.setChecked(True)
        btn_layout.addWidget(self.penalty_check)
        btn_layout.addStretch()
        
        self.eval_btn = QPushButton("Evaluate")
        self.eval_btn.setFixedSize(120, 30)
        self.eval_btn.setStyleSheet("QPushButton { border: 1px solid #c0c0c0; border-radius: 4px; }")
        if HAS_ICONS:
            self.eval_btn.setIcon(qta.icon('mdi.eye'))
        self.eval_btn.clicked.connect(self.evaluateRequested)
        
        self.fit_btn = QPushButton("Run Fit")
        self.fit_btn.setFixedSize(120, 30)
        self.fit_btn.setStyleSheet("QPushButton { background-color: #2060a0; color: white; border: none; border-radius: 4px; font-weight: bold; }")
        if HAS_ICONS:
            self.fit_btn.setIcon(qta.icon('mdi.rocket-launch', color='white'))
        self.fit_btn.clicked.connect(self.fitRequested)
        
        btn_layout.addWidget(self.eval_btn)
        btn_layout.addWidget(self.fit_btn)
        fitting_layout.addLayout(btn_layout)
        
        # Component panel below buttons
        self.component_panel = QtComponentPanel()
        fitting_layout.addWidget(self.component_panel)
        
        layout.addWidget(fitting_group)
        
        scroll.setWidget(scroll_content)
        outer_layout.addWidget(scroll)

    def get_preprocessing_config(self):
        """Return dict of preprocessing settings."""
        return {
            "x_min": None if self.x_min.text() == "auto" else float(self.x_min.text()),
            "x_max": None if self.x_max.text() == "auto" else float(self.x_max.text()),
            "interpolate": self.interp_check.isChecked(),
            "step": None if self.interp_step.text() == "auto" else float(self.interp_step.text())
        }

    def get_fitting_options(self):
        """Return dict of global fitting options."""
        return {
            "normalize": self.norm_check.isChecked(),
            "non_negative": self.penalty_check.isChecked()
        }

    def get_calc_range(self):
        """Return the calculation range (xmin, xmax) for baseline."""
        try:
            xmin = None if self.calc_x_min.text().strip() in ("auto", "") else float(self.calc_x_min.text())
            xmax = None if self.calc_x_max.text().strip() in ("auto", "") else float(self.calc_x_max.text())
        except ValueError:
            xmin, xmax = None, None
        return (xmin, xmax)
