"""
Results Display Panel for PySide6.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QTextEdit, QFrame, QGridLayout, QGroupBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class MetricCard(QFrame):
    """A small card to display a single metric."""
    def __init__(self, label, value, parent=None):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setFrameShape(QFrame.StyledPanel)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.label = QLabel(label)
        self.label.setStyleSheet("font-size: 10px; font-weight: bold;")
        self.label.setAlignment(Qt.AlignCenter)
        
        self.value = QLabel(value)
        self.value.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa;")
        self.value.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.label)
        layout.addWidget(self.value)

    def setValue(self, text):
        self.value.setText(text)
class QtResultsPanel(QWidget):
    """
    Panel for displaying fit results and statistics.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Statistics Grid
        stats_group = QGroupBox("Fit Quality")
        stats_layout = QGridLayout(stats_group)
        
        self.r2_card = MetricCard("R²", "---")
        self.rmse_card = MetricCard("RMSE", "---")
        self.chi2_card = MetricCard("Red. χ²", "---")
        self.aic_card = MetricCard("AIC", "---")
        
        stats_layout.addWidget(self.r2_card, 0, 0)
        stats_layout.addWidget(self.rmse_card, 0, 1)
        stats_layout.addWidget(self.chi2_card, 1, 0)
        stats_layout.addWidget(self.aic_card, 1, 1)
        
        layout.addWidget(stats_group)
        
        # Detailed Report
        layout.addWidget(QLabel("Full Fit Report:"))
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Courier New", 10))
        self.report_text.setMinimumHeight(200)
        layout.addWidget(self.report_text, 1) # Add stretch factor

    def display_results(self, stats, report):
        """Update the panel with new results."""
        if not stats:
            self.clear()
            self.report_text.setPlainText(report)
            return

        self.r2_card.setValue(f"{stats.get('r_squared', 0):.4f}")
        self.rmse_card.setValue(f"{stats.get('rmse', 0):.2e}")
        self.chi2_card.setValue(f"{stats.get('reduced_chi_squared', 0):.2e}")
        self.aic_card.setValue(f"{stats.get('aic', 0):.1f}")
        
        # Build report with baseline info (single consolidated section)
        full_report = report
        if 'baseline_method' in stats:
            bl_info = f"\n\n[[Baseline]]\nMethod: {stats['baseline_method']}"
            if stats.get('baseline_optimize'):
                bl_info += " (simultaneous optimization)"
            if stats.get('baseline_range'):
                bl_info += f"\nRange: {stats['baseline_range']}"
            # Initial parameters
            if stats.get('baseline_params'):
                bl_info += "\nInitial params:"
                for key, val in stats['baseline_params'].items():
                    bl_info += f"\n  {key}: {val}"
            # Fitted parameters (polynomial coefficients, etc.)
            if stats.get('fitted_baseline_params'):
                bl_info += "\nFitted params:"
                for key, val in stats['fitted_baseline_params'].items():
                    if isinstance(val, float):
                        bl_info += f"\n  {key}: {val:.6g}"
                    else:
                        bl_info += f"\n  {key}: {val}"
            full_report += bl_info
        
        self.report_text.setPlainText(full_report)
        # Scroll to top
        self.report_text.verticalScrollBar().setValue(0)

    def clear(self):
        """Clear all results."""
        self.r2_card.setValue("---")
        self.rmse_card.setValue("---")
        self.chi2_card.setValue("---")
        self.aic_card.setValue("---")
        self.report_text.clear()
