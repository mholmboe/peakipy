"""
Custom synchronized Slider + Spinbox widget for parameter control. 
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QSlider, QDoubleSpinBox, QLabel
from PySide6.QtCore import Qt, Signal

class ParameterControl(QWidget):
    """
    A widget that synchronizes a QSlider and a QDoubleSpinBox.
    """
    valueChanged = Signal(float)

    def __init__(self, label="", min_val=0.0, max_val=100.0, default=0.0, step=None, decimals=2, parent=None):
        super().__init__(parent)
        
        # Default step to 1.0 if it's an integer control (decimals=0)
        if step is None:
            step = 1.0 if decimals == 0 else 0.01

        self.min_val = min_val
        self.max_val = max_val
        self.multiplier = 10**decimals
        
        self._setup_ui(label, min_val, max_val, default, step, decimals)

    def _setup_ui(self, label, min_val, max_val, default, step, decimals):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if label:
            self.label = QLabel(label)
            self.label.setMinimumWidth(80)
            layout.addWidget(self.label)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val * self.multiplier), int(max_val * self.multiplier))
        self.slider.setValue(int(default * self.multiplier))
        
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setDecimals(decimals)
        self.spinbox.setValue(default)
        self.spinbox.setFixedWidth(80)
        
        # Connections for synchronization
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)
        
        layout.addWidget(self.slider)
        layout.addWidget(self.spinbox)

    def _on_slider_changed(self, value):
        float_val = value / self.multiplier
        if abs(self.spinbox.value() - float_val) > 1e-7:
            self.spinbox.blockSignals(True)
            self.spinbox.setValue(float_val)
            self.spinbox.blockSignals(False)
            self.valueChanged.emit(float_val)

    def _on_spinbox_changed(self, value):
        int_val = int(value * self.multiplier)
        if self.slider.value() != int_val:
            self.slider.blockSignals(True)
            self.slider.setValue(int_val)
            self.slider.blockSignals(False)
            self.valueChanged.emit(value)

    def value(self):
        return self.spinbox.value()

    def setValue(self, value):
        self.spinbox.setValue(value)

    def set_range(self, min_val, max_val):
        """Dynamically update the min/max range for both slider and spinbox."""
        self.min_val = min_val
        self.max_val = max_val
        
        # Update spinbox range
        self.spinbox.blockSignals(True)
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.blockSignals(False)
        
        # Update slider range
        self.slider.blockSignals(True)
        self.slider.setRange(int(min_val * self.multiplier), int(max_val * self.multiplier))
        self.slider.blockSignals(False)
        
        # Clamp current value to new range
        current = self.spinbox.value()
        if current < min_val:
            self.setValue(min_val)
        elif current > max_val:
            self.setValue(max_val)
