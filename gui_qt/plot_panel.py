"""
Plotting panel for PySide6 using Matplotlib. 
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

class QtPlotPanel(QWidget):
    """
    Matplotlib-based plotting panel for Qt.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._bg_cache = None  # For blitting
        self._blit_enabled = True

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create figure and canvas
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        
        # Main plot and residuals plot
        # Using constrained_layout for better spacing
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        
        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.fig.tight_layout()
        
        # Connect resize event to invalidate cache
        self.canvas.mpl_connect('resize_event', self._on_resize)
        
        # Manual baseline editing
        self._manual_cid = None
        self._manual_click_handler = None
        self._manual_max_points = None
        self.manual_point_markers = None

    def _on_resize(self, event):
        """Invalidate background cache on resize."""
        self._bg_cache = None

    def _cache_background(self):
        """Cache the current axes background for blitting."""
        self.canvas.draw()
        self._bg_cache = self.canvas.copy_from_bbox(self.ax1.bbox)

    def _set_ylim_from_data(self, *arrays, pad_frac=0.1):
        """Set y-limits with a margin based on provided arrays (handles negatives)."""
        vals = []
        for arr in arrays:
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.size:
                vals.append(arr.ravel())
        if not vals:
            return
        data = np.concatenate(vals)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return
        vmin, vmax = data.min(), data.max()
        span = vmax - vmin
        pad = pad_frac * max(abs(vmin), abs(vmax), span, 1e-9)
        # Ensure we always allow a bit below zero for negative baselines
        floor = min(vmin - pad, -0.1, -0.1 * abs(vmax))
        ceil = vmax + pad
        if ceil <= floor:
            ceil = floor + max(span, pad, 1e-3)
        self.ax1.set_ylim(floor, ceil)

    def _finalize_axes(self, *arrays, legend_loc='best', draw=True, cache=True):
        """Apply legend/grid/ylim updates and refresh canvas/cache."""
        self._set_ylim_from_data(*arrays)
        self.ax1.legend(loc=legend_loc, fontsize=9)
        self.ax1.grid(True, alpha=0.3)
        if draw:
            self.canvas.draw()
            if cache:
                self._bg_cache = self.canvas.copy_from_bbox(self.ax1.bbox)

    def plot_data(self, x, y, title="Experimental Data"):
        """Plot raw experimental data."""
        self.ax1.clear()
        self.ax1.plot(x, y, 'o', markersize=4, label='Data', alpha=0.6, color='#34495e')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Intensity')
        self.ax1.set_title(title, fontweight='bold')
        self.ax2.clear()
        self.ax2.axis('off')
        self._finalize_axes(y, draw=True, cache=True)

    def update_fit_plot(self, x, y_exp, y_fit, residuals, components=None, baseline=None, show_fit=True, y_raw=None):
        """Update plot with fit results."""
        self.ax1.clear()
        self.ax2.clear()
        self.ax2.axis('on')
        
        # Main plot
        if y_raw is not None:
            self.ax1.plot(x, y_raw, 'o', markersize=3, label='Original Data', alpha=0.35, color='#2c3e50')
        self.ax1.plot(x, y_exp, 'o', markersize=4, label='Experimental Data', alpha=0.6, color='#34495e')
        if show_fit:
            self.ax1.plot(x, y_fit, '-', linewidth=2.5, label='Total Fit', color='#c0392b')
        
        # Plot components
        if components:
            colors = ['#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#f39c12']
            for i, (name, y_comp) in enumerate(components.items()):
                self.ax1.plot(x, y_comp, '--', linewidth=1.5, 
                             label=name, color=colors[i % len(colors)], alpha=0.8)
        
        # Plot baseline
        if baseline is not None:
            self.ax1.plot(x, baseline, '-.', linewidth=1.5, label='Baseline', color='#7f8c8d', alpha=0.7)
            
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Intensity')
        self.ax1.set_title('Fit Result', fontweight='bold')
        comp_arrays = components.values() if components else []
        self._finalize_axes(y_exp, y_fit if show_fit else None, baseline, *(comp_arrays if components else []), y_raw, draw=False, cache=False)

        # Residuals plot
        self.ax2.plot(x, residuals, 'o', markersize=3, color='#27ae60', alpha=0.7)
        self.ax2.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=1.5)
        
        # ±2σ bands
        residual_std = np.std(residuals)
        self.ax2.axhline(y=2*residual_std, color='#95a5a6', linestyle=':', linewidth=1)
        self.ax2.axhline(y=-2*residual_std, color='#95a5a6', linestyle=':', linewidth=1)
        self.ax2.fill_between(x, -2*residual_std, 2*residual_std, color='#bdc3c7', alpha=0.2)
        
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Residuals')
        self.ax2.set_title(f'Difference Curve (±2σ = ±{2*residual_std:.2e})', fontsize=10)
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        # Invalidate cache after fit results
        self._bg_cache = None

    def update_baseline_preview(self, x, y_raw, baseline, corrected):
        """Update plot with baseline preview using blitting for performance."""
        # Remove old preview lines if they exist
        if hasattr(self, 'bl_preview_line'):
            try:
                self.bl_preview_line.remove()
                self.corr_preview_line.remove()
            except:
                pass

        # Use blitting for fast updates if cache available
        if self._blit_enabled and self._bg_cache is not None:
            # Restore background
            self.canvas.restore_region(self._bg_cache)
            
            # Create new preview lines
            self.bl_preview_line, = self.ax1.plot(x, baseline, '--', color='#7f8c8d', 
                                                 linewidth=1.5, label='Baseline (preview)', alpha=0.8)
            self.corr_preview_line, = self.ax1.plot(x, corrected, 'o', markersize=3, 
                                                   color='#f39c12', label='Corrected (preview)', alpha=0.4)
            
            # Draw only the new artists
            self.ax1.draw_artist(self.bl_preview_line)
            self.ax1.draw_artist(self.corr_preview_line)
            
            # Blit the updated region
            self.canvas.blit(self.ax1.bbox)
        else:
            # Fallback to full redraw
            self.bl_preview_line, = self.ax1.plot(x, baseline, '--', color='#7f8c8d', 
                                                 linewidth=1.5, label='Baseline (preview)', alpha=0.8)
            self.corr_preview_line, = self.ax1.plot(x, corrected, 'o', markersize=3, 
                                                   color='#f39c12', label='Corrected (preview)', alpha=0.4)

        # Ensure limits, legend, and cache update regardless of branch
        self._finalize_axes(y_raw, baseline, corrected, draw=True, cache=True)

    def clear_baseline_preview(self):
        """Remove baseline preview lines from plot."""
        if hasattr(self, 'bl_preview_line'):
            try:
                self.bl_preview_line.remove()
                del self.bl_preview_line
            except:
                pass
        if hasattr(self, 'corr_preview_line'):
            try:
                self.corr_preview_line.remove()
                del self.corr_preview_line
            except:
                pass
        if self.manual_point_markers:
            try:
                self.manual_point_markers.remove()
            except Exception:
                pass
            self.manual_point_markers = None
        # Refresh legend and redraw
        self.ax1.legend(loc='best', fontsize=9)
        self.canvas.draw_idle()
        # Re-cache background
        self._bg_cache = self.canvas.copy_from_bbox(self.ax1.bbox)

    def clear(self):
        """Clear all plots."""
        self.ax1.clear()
        self.ax2.clear()
        self.ax2.axis('off')
        self._bg_cache = None
        self.canvas.draw()

    # ================= Manual Baseline Helpers =================
    def start_manual_capture(self, click_handler, max_points):
        """Enable manual point capture on the main axes."""
        self.stop_manual_capture()
        self._manual_click_handler = click_handler
        self._manual_max_points = max_points
        self._manual_cid = self.canvas.mpl_connect('button_press_event', self._on_manual_click)

    def stop_manual_capture(self):
        """Disable manual point capture."""
        if self._manual_cid is not None:
            self.canvas.mpl_disconnect(self._manual_cid)
            self._manual_cid = None
        self._manual_click_handler = None
        self._manual_max_points = None

    def _on_manual_click(self, event):
        if event.inaxes != self.ax1:
            return
        if self._manual_click_handler:
            self._manual_click_handler(event.xdata, event.ydata)

    def update_manual_baseline_preview(self, x, y_raw, baseline, corrected, points):
        """Show manual baseline, corrected data, and control points."""
        self.update_baseline_preview(x, y_raw, baseline, corrected)
        # Add control points markers
        px = [p[0] for p in points]
        py = [p[1] for p in points]
        if self.manual_point_markers:
            try:
                self.manual_point_markers.remove()
            except Exception:
                pass
        self.manual_point_markers = self.ax1.scatter(px, py, color='#d35400', s=40, zorder=4, label='Baseline Ctrl Pts')
        self.ax1.legend(loc='best', fontsize=9)
        self.canvas.draw_idle()
