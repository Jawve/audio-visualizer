# app.py
import sys
from PySide6 import QtGui
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from audio_hub import AudioHub
from pages.menu_page import MenuPage
from pages.settings_page import SettingsPage
from pages.visualizer_page import VisualizerPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visual Sandbox")
        self.resize(1200, 780)

        self.stack = QStackedWidget(); self.setCentralWidget(self.stack)

        self.hub = AudioHub()                 # lightweight stub; emits level arrays
        self.menu = MenuPage()
        self.visualizer = VisualizerPage(self.hub)
        self.settings = SettingsPage(self.hub)

        self.stack.addWidget(self.menu)       # 0
        self.stack.addWidget(self.visualizer) # 1
        self.stack.addWidget(self.settings)   # 2

        self.menu.startVisualizer.connect(lambda: self.stack.setCurrentIndex(1))
        self.menu.openSettings.connect(lambda: self.stack.setCurrentIndex(2))
        self.menu.exitApp.connect(self.safe_exit)

        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=lambda: self.stack.setCurrentIndex(0))
        QtGui.QShortcut(QtGui.QKeySequence("F1"), self, activated=lambda: self.stack.setCurrentIndex(2))

    def safe_exit(self):
        try: self.hub.terminate()
        finally: self.close()

def main():
    app = QApplication(sys.argv); app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QtGui.QPalette.Window, QColor(18,15,24))
    pal.setColor(QtGui.QPalette.Button, QColor(32,28,44))
    pal.setColor(QtGui.QPalette.ButtonText, QColor(240,240,245))
    pal.setColor(QtGui.QPalette.Base, QColor(24,20,34))
    pal.setColor(QtGui.QPalette.Text, QColor(235,235,240))
    app.setPalette(pal)

    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
