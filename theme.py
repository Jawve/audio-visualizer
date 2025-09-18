# theme.py
from PySide6.QtGui import QColor

def color_hex(a, b, c, alpha=1.0):
    return QColor(int(a), int(b), int(c), int(alpha*255))

PRIMARY_PURPLE = color_hex(128, 58, 180)     # #803AB4
ACCENT_ORANGE = color_hex(255, 140, 0)       # #FF8C00
DARK_BG       = color_hex(10, 8, 14)         # #0A080E
BEIGE         = color_hex(245, 242, 232)     # #F5F2E8

DOT_OVERLAY_ALPHA = 0.08
