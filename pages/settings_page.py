# pages/settings_page.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

class SettingsPage(QWidget):
    def __init__(self, hub, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lbl = QLabel("Settings (coming soon)")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color:#CCC; font-size:18px;")
        lay.addStretch(1); lay.addWidget(lbl); lay.addStretch(1)
