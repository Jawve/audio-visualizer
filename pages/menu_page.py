# pages/menu_page.py
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy

# ⬇️ add this import to use the same animated background as Visualizer
from backgrounds import AnimatedBackground


class MenuPage(QWidget):
    startVisualizer = Signal()
    openSettings = Signal()
    exitApp = Signal()

    def __init__(self):
        super().__init__()

        # Let the background show through any child widgets
        self.setAttribute(Qt.WA_StyledBackground, False)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Root layout (no margins so background fills fully)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Animated background (same as Visualizer page)
        self.bg = AnimatedBackground(mode="animated", parent=self)
        root.addWidget(self.bg)

        # Transparent overlay that holds the actual menu UI
        overlay = QWidget(self.bg)
        overlay.setAttribute(Qt.WA_TranslucentBackground, True)
        overlay_lay = QVBoxLayout(overlay)
        overlay_lay.setAlignment(Qt.AlignCenter)
        overlay_lay.setContentsMargins(24, 24, 24, 24)
        overlay_lay.setSpacing(18)

        # Mount overlay into the background’s layout
        bg_layout = QVBoxLayout(self.bg)
        bg_layout.setContentsMargins(0, 0, 0, 0)
        bg_layout.setSpacing(0)
        bg_layout.addWidget(overlay)

        # --- UI content (unchanged, but now sits over the animated bg) ---
        title = QLabel("Audio Visual Sandbox")
        title.setStyleSheet("font-size: 26px; font-weight: 700; color: #EEE;")
        title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        btn_vis = QPushButton("Visualizer")
        btn_set = QPushButton("Settings")
        btn_exit = QPushButton("Exit")

        for b in (btn_vis, btn_set, btn_exit):
            b.setMinimumWidth(240)
            b.setMinimumHeight(44)
            b.setStyleSheet("""
                QPushButton {
                  background:#2A2340; color:#EEE; border: 1px solid #4b4267; border-radius: 10px;
                }
                QPushButton:hover { background:#332a4f; }
            """)

        overlay_lay.addWidget(title)
        overlay_lay.addSpacing(24)
        overlay_lay.addWidget(btn_vis)
        overlay_lay.addWidget(btn_set)
        overlay_lay.addWidget(btn_exit)

        # Wire signals
        btn_vis.clicked.connect(self.startVisualizer)
        btn_set.clicked.connect(self.openSettings)
        btn_exit.clicked.connect(self.exitApp.emit)
