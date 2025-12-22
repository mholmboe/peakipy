"""
Entry point for the PySide6 version of the Profile Fitting Application.
"""

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from gui_qt.main_window import QtMainWindow

def load_theme(app):
    """Load the application theme from QSS file."""
    theme_path = os.path.join(os.path.dirname(__file__), 'gui_qt', 'theme.qss')
    if os.path.exists(theme_path):
        with open(theme_path, 'r') as f:
            app.setStyleSheet(f.read())

def main():
    # Enable High-DPI scaling
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    
    # Set premium font only if available (avoid font lookup warnings)
    if QFontDatabase.hasFamily("Inter"):
        app.setFont(QFont("Inter", 10))
    elif QFontDatabase.hasFamily("Segoe UI"):
        app.setFont(QFont("Segoe UI", 10))
    # else: use Qt/system default font
    
    # Load modern dark theme
    load_theme(app)
    
    window = QtMainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
