# widgets/bars_widget.py
import math, time, random
from typing import List
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QPainterPath
from PySide6.QtWidgets import QWidget

class BarsWidget(QWidget):
    """
    Rounded, mirrored bar visualizer.
    - call update_levels(list[0..1]) with nbars values
    - set_mirrored(True/False)
    - set_colormap('viridis' | 'plasma')
    """
    def __init__(self, parent=None, bar_count: int = 48, smoothing: float = 0.65):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._bar_count = int(bar_count)
        self._smooth = float(smoothing)
        self._levels = [0.0]*self._bar_count
        self._smoothed = [0.0]*self._bar_count
        self._mirrored = True
        self._round = 6.0
        self._cmap = "viridis"  # default; UI can switch to 'plasma'

        # Idle animation timer (runs even without audio to keep it lively)
        self._idle = True
        self._t0 = time.time()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_idle)
        self._timer.start(33)

    # --------- API expected by page ----------
    def set_mirrored(self, on: bool):
        self._mirrored = bool(on); self.update()

    def set_bar_count(self, n: int):
        n = max(8, int(n))
        self._bar_count = n
        self._levels = [0.0]*n
        self._smoothed = [0.0]*n
        self.update()

    def set_smoothing(self, s: float):
        self._smooth = max(0.0, min(0.99, float(s)))

    def set_colormap(self, name: str):
        name = (name or "").lower()
        self._cmap = "plasma" if name == "plasma" else "viridis"
        self.update()

    def update_levels(self, values: List[float]):
        # values in 0..1
        n = min(len(values), self._bar_count)
        for i in range(n):
            self._levels[i] = max(0.0, min(1.0, float(values[i])))
        # enable active mode (stop idle shape snaps)
        self._idle = False
        self.update()

    # --------- internals ----------
    def _tick_idle(self):
        if not self._idle:
            # after some seconds of no updates, return to idle
            if time.time() - self._t0 > 1.0:
                self._idle = True
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        w = self.width(); h = self.height()
        xpad = max(6, int(0.01*w))
        ypad = max(4, int(0.05*h))
        inner_h = max(8, h - 2*ypad)

        # compute bar rectangles
        gap = 0.25  # fraction of bar slot
        slot = (w - 2*xpad) / max(1, self._bar_count)
        bw = slot * (1.0 - gap)
        r = min(self._round, bw*0.45)

        # smoothing
        a = self._smooth
        for i, v in enumerate(self._levels):
            self._smoothed[i] = a*self._smoothed[i] + (1.0 - a)*v
        levels = self._smoothed

        # simple colormap
        def color_for(v: float) -> QColor:
            # 0..1 → gradient
            if self._cmap == "plasma":
                # deep purple→magenta→yellow
                return QColor.fromHsvF(0.85 - 0.85*v, 1.0, 0.95, 1.0)
            else:
                # viridis-ish: blue→green→yellow
                # hue ~ 0.65→0.15
                return QColor.fromHsvF(0.65 - 0.50*v, 0.85, 0.95, 1.0)

        # draw bars
        base_y = ypad + inner_h/2.0
        for i, v in enumerate(levels):
            x0 = xpad + i*slot + (slot - bw)/2.0
            full = inner_h/2.0
            h_up = max(1.0, v*full)
            # rounded rectangle upwards
            path = QPainterPath()
            path.addRoundedRect(QRectF(x0, base_y - h_up, bw, h_up), r, r)
            c = color_for(v)
            p.setPen(Qt.NoPen); p.setBrush(c)
            p.drawPath(path)
            if self._mirrored:
                # mirror downward (keep seamless center)
                path2 = QPainterPath()
                path2.addRoundedRect(QRectF(x0, base_y, bw, h_up), r, r)
                p.drawPath(path2)

        p.end()

    # called by page to enter idle again if needed
    def note_idle_tick(self):
        self._t0 = time.time()
