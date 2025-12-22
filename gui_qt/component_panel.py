"""
Component Management Panel for PySide6.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QScrollArea, QFrame, QLabel, QSpinBox, QComboBox,
                               QRadioButton, QButtonGroup, QCheckBox)
from PySide6.QtCore import Signal
from .widgets.component_group import ComponentGroup

class QtComponentPanel(QWidget):
    """
    Panel for managing multiple peak components.
    """
    componentsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.components = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Top controls: Number of peaks, Auto-init
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("N peaks:"))
        self.num_peaks_spin = QSpinBox()
        self.num_peaks_spin.setRange(1, 20)
        self.num_peaks_spin.setValue(1)
        self.num_peaks_spin.valueChanged.connect(self._sync_components)
        top_layout.addWidget(self.num_peaks_spin)
        
        top_layout.addWidget(QLabel("Peak Type:"))
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["Gaussian", "Lorentzian", "Voigt"])
        self.profile_combo.setMinimumWidth(100)
        top_layout.addWidget(self.profile_combo)
        
        self.live_preview_check = QCheckBox("Live Preview")
        self.live_preview_check.setToolTip("Updates plot as you adjust components")
        self.live_preview_check.toggled.connect(self.componentsChanged)
        top_layout.addWidget(self.live_preview_check)
        
        layout.addLayout(top_layout)

        # Row 2: Initialization Method
        init_row = QHBoxLayout()
        self.init_group = QButtonGroup(self)
        self.even_radio = QRadioButton("Evenly Spaced")
        self.gmm_radio = QRadioButton("GMM (Smart)")
        self.gmm_radio.setChecked(True)
        self.init_group.addButton(self.even_radio)
        self.init_group.addButton(self.gmm_radio)
        
        init_row.addWidget(QLabel("Init Method:"))
        init_row.addWidget(self.even_radio)
        init_row.addWidget(self.gmm_radio)
        
        init_row.addStretch()
        
        self.auto_init_btn = QPushButton("⟳ Re-initialize")
        init_row.addWidget(self.auto_init_btn)
        
        layout.addLayout(init_row)
        
        # Container for component groups (no embedded scroll - parent will scroll)
        self.comp_container = QFrame()
        self.comp_layout = QVBoxLayout(self.comp_container)
        self.comp_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.comp_container)
        
        # Initialize with one component
        self._sync_components(1)

    def _sync_components(self, count):
        """Create or remove component groups to match count."""
        # Add new ones
        while len(self.components) < count:
            comp = ComponentGroup(len(self.components))
            comp.set_profile_type(self.profile_combo.currentText()) # Sync new comp
            comp.settingsChanged.connect(self.componentsChanged)
            comp.weight_slider.valueChanged.connect(self._on_weight_changed)
            # Apply data range if set
            if hasattr(self, '_data_range') and self._data_range:
                comp.set_data_range(*self._data_range)
            self.components.append(comp)
            self.comp_layout.insertWidget(len(self.components) - 1, comp)
            
        # Remove excess
        while len(self.components) > count:
            comp = self.components.pop()
            comp.deleteLater()
        
        # Normalize weights for new component set
        self._normalize_weights()
        self.componentsChanged.emit()

    def _on_weight_changed(self):
        """Normalize all weights to sum to 1 when any weight changes."""
        self._normalize_weights()

    def _normalize_weights(self):
        """Adjust all component weights so they sum to 1."""
        if not self.components:
            return
        
        # Get current weights
        weights = [c.weight_slider.value() for c in self.components]
        total = sum(weights)
        
        if total <= 0:
            # If all zeros, distribute equally
            equal_weight = 1.0 / len(self.components)
            for c in self.components:
                c.weight_slider.blockSignals(True)
                c.weight_slider.setValue(equal_weight)
                c.weight_slider.blockSignals(False)
        else:
            # Normalize to sum to 1
            for c, w in zip(self.components, weights):
                c.weight_slider.blockSignals(True)
                c.weight_slider.setValue(w / total)
                c.weight_slider.blockSignals(False)

    def _on_profile_type_changed(self):
        """Sync all components to the new profile type."""
        profile = self.profile_combo.currentText()
        for comp in self.components:
            comp.set_profile_type(profile)
        self.componentsChanged.emit()

    def get_config(self):
        """Return list of component configurations using the global profile type."""
        profile = self.profile_combo.currentText()
        return [c.get_config(profile) for c in self.components]

    def set_component_params(self, params_list):
        """Set parameters for all components from a list of dicts."""
        if not params_list: return
        
        # Determine profile type from first item if present
        if "type" in params_list[0]:
            idx = self.profile_combo.findText(params_list[0]["type"].capitalize())
            if idx >= 0: 
                self.profile_combo.blockSignals(True)
                self.profile_combo.setCurrentIndex(idx)
                self.profile_combo.blockSignals(False)
        
        self.num_peaks_spin.blockSignals(True)
        self.num_peaks_spin.setValue(len(params_list))
        self.num_peaks_spin.blockSignals(False)
        
        self._sync_components(len(params_list))
        
        for comp, params in zip(self.components, params_list):
            comp.set_values(**params)
        
        self.componentsChanged.emit()
        
    def get_init_method(self):
        """Return the selected initialization method string."""
        if self.even_radio.isChecked():
            return "even"
        return "gmm"

    def set_data_range(self, x_min, x_max):
        """Update parameter constraints for all components based on data range."""
        self._data_range = (x_min, x_max)  # Store for new components
        for comp in self.components:
            comp.set_data_range(x_min, x_max)
