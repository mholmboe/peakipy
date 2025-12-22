"""
Widget for a single peak component in PySide6.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                               QComboBox, QLabel, QPushButton)
from PySide6.QtCore import Signal, Qt
from .parameter_control import ParameterControl

class ComponentGroup(QGroupBox):
    """
    Controls for a single peak component.
    """
    settingsChanged = Signal()
    removeRequested = Signal()

    def __init__(self, index, parent=None):
        super().__init__(f"Component {index + 1}", parent)
        self.index = index
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Weight row
        top_row = QHBoxLayout()
        self.weight_slider = ParameterControl("Weight", 0.0, 2.0, 1.0, decimals=2)
        self.weight_slider.valueChanged.connect(self.settingsChanged)
        top_row.addWidget(self.weight_slider)
        
        layout.addLayout(top_row)
        
        # Center, Amplitude, Width
        self.center = ParameterControl("Center", 0.0, 10000.0, 0.0, decimals=4)
        self.amplitude = ParameterControl("Amplitude", 0.0, 1e6, 100.0, decimals=2)
        
        # Width depends on profile
        self.width_param = ParameterControl("Sigma (σ)", 0.001, 500.0, 5.0, decimals=4)
        
        # Voigt has two widths
        self.gamma_param = ParameterControl("Gamma (γ)", 0.001, 500.0, 5.0, decimals=4)
        self.gamma_param.setVisible(False)
        
        for p in [self.center, self.amplitude, self.width_param, self.gamma_param]:
            layout.addWidget(p)
            p.valueChanged.connect(self.settingsChanged)

    def set_profile_type(self, profile):
        """Update parameter visibility based on global profile type."""
        profile = profile.lower()
        if profile == "voigt":
            self.width_param.label.setText("Sigma (σ)")
            self.gamma_param.setVisible(True)
        else:
            self.width_param.label.setText("Sigma (σ)" if profile == "gaussian" else "Gamma (γ)")
            self.gamma_param.setVisible(False)
        self.settingsChanged.emit()

    def get_config(self, profile):
        """Return dict of component configuration for a specific profile type."""
        profile = profile.lower()
        config = {
            "type": profile,
            "center": self.center.value(),
            "amplitude": self.amplitude.value(),
            "weight": self.weight_slider.value()
        }
        
        if profile == "voigt":
            config["sigma"] = self.width_param.value()
            config["gamma"] = self.gamma_param.value()
        else:
            config["width"] = self.width_param.value()
            
        return config

    def set_values(self, **kwargs):
        """Programmatically set widget values."""
        self.blockSignals(True)
        if "weight" in kwargs: self.weight_slider.setValue(kwargs["weight"])
        if "center" in kwargs: self.center.setValue(kwargs["center"])
        if "amplitude" in kwargs: self.amplitude.setValue(kwargs["amplitude"])
        if "width" in kwargs: self.width_param.setValue(kwargs["width"])
        if "sigma" in kwargs: self.width_param.setValue(kwargs["sigma"])
        if "gamma" in kwargs: self.gamma_param.setValue(kwargs["gamma"])
        self.blockSignals(False)
        self.settingsChanged.emit()

    def set_data_range(self, x_min, x_max):
        """Update parameter constraints based on loaded data range."""
        data_range = x_max - x_min
        max_width = data_range / 2  # Width shouldn't exceed half the data range
        
        # Update center range
        self.center.set_range(x_min, x_max)
        
        # Update width ranges (sigma and gamma)
        self.width_param.set_range(0.001, max_width)
        self.gamma_param.set_range(0.001, max_width)
