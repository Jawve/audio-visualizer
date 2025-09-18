# backgrounds.py
import math, time, random
import numpy as np
from PySide6.QtCore import QTimer, QPointF, QPoint, Qt
from PySide6.QtGui import QPainter, QLinearGradient, QColor
from PySide6.QtWidgets import QWidget
from theme import PRIMARY_PURPLE, ACCENT_ORANGE, DOT_OVERLAY_ALPHA, color_hex

def cubic_bezier(p0, p1, p2, p3, u):
    """Evaluate cubic Bezier at u in [0,1]."""
    v = 1.0 - u
    return (v*v*v)*p0 + 3*v*v*u*p1 + 3*v*u*u*p2 + (u*u*u)*p3

def ease_in_out(u):
    """Smooth ease for progress/alpha."""
    return 0.5 * (1.0 - math.cos(math.pi * max(0.0, min(1.0, u))))

class AnimatedBackground(QWidget):
    """
    Animated gradient + soft floating circles + 'swooping' Ben-Day dots:
    dots spawn occasionally, follow a curvy Bezier path across the screen, and vanish.
    mode: "animated" | "purple"
    """
    def __init__(self, mode="animated", parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        self.mode = mode

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 fps

        # ----- Big floating blobs (unchanged) -----
        self.circles = []
        rng = np.random.default_rng(42)
        for _ in range(24):
            r = rng.uniform(40, 140)
            x = rng.uniform(0, 1); y = rng.uniform(0, 1)
            dx = rng.uniform(-0.15, 0.15); dy = rng.uniform(-0.15, 0.15)
            hue_t = rng.uniform(0, 1)
            self.circles.append([x, y, dx, dy, r, hue_t])

        # ----- Swooping Ben-Day dots -----
        self.mouse_pos = QPoint(-10_000, -10_000)
        self._last_t = time.time()

        # parameters
        self.dot_count          = 120           # total particles (not all visible at once)
        self.dot_radius_min     = 3.0
        self.dot_radius_max     = 6.0
        self.dot_breathe_amp    = 2.0           # extra radius pulse
        self.dot_breathe_rate   = (1.2, 2.8)    # Hz-ish (radians/sec scalars)
        self.path_duration_s    = (2.2, 4.5)    # how long a swoop takes
        self.spawn_delay_s      = (1.0, 6.0)    # off-screen wait between flights
        self.curve_mag_x        = (0.15, 0.35)  # curvature as fraction of width
        self.curve_mag_y        = (0.08, 0.25)  # curvature as fraction of height
        self.vertical_jitter_px = 6.0           # small natural jitter around path
        self.jitter_rate        = 0.7

        # mouse avoidance (subtle, smoothed)
        self.avoid_radius       = 140.0
        self.avoid_push         = 16.0
        self.avoid_smooth       = 0.85  # 0..1 (closer to 1 = smoother, slower)

        self._dots = []  # particles with path state; built lazily on first paint

    def setMode(self, mode: str):
        self.mode = mode
        self.update()

    def mouseMoveEvent(self, ev):
        self.mouse_pos = ev.pos()

    # ----------------- Swoop particles -----------------
    def _ensure_dots(self, w: int, h: int):
        if self._dots:
            return
        now = time.time()
        self._dots = [self._make_dot(now, w, h, initial=True) for _ in range(self.dot_count)]

    def _make_dot(self, now: float, w: int, h: int, initial=False):
        """Create a new dot with a planned future swoop (or immediate if initial)."""
        # direction: -1 = right->left, +1 = left->right
        direction = random.choice((-1, 1))

        # start/end X slightly offscreen for clean entry/exit
        margin_x = 80
        start_x = -margin_x if direction > 0 else (w + margin_x)
        end_x   =  w + margin_x if direction > 0 else (-margin_x)

        # choose a vertical band for start/end (spread across the canvas)
        band_top = int(h * random.uniform(0.05, 0.95))
        # end Y differs slightly
        band_bottom = min(h-1, max(0, band_top + random.randint(-int(0.2*h), int(0.2*h))))

        p0 = np.array([float(start_x), float(band_top)], dtype=np.float32)
        p3 = np.array([float(end_x),   float(band_bottom)], dtype=np.float32)

        # Curvature control points: offset from the line in a natural way
        cx_mag = random.uniform(*self.curve_mag_x) * w
        cy_mag = random.uniform(*self.curve_mag_y) * h

        # direction-aware control points (produce natural S-curves)
        # place p1 near 1/3 progress, p2 near 2/3 progress, with perpendicular-ish offsets
        mid_dir = np.array([p3[0]-p0[0], p3[1]-p0[1]], dtype=np.float32)
        L = np.hypot(mid_dir[0], mid_dir[1]) + 1e-6
        tdir = mid_dir / L
        nrm = np.array([-tdir[1], tdir[0]], dtype=np.float32)  # perpendicular

        p1 = p0 + tdir * (0.33*L) + nrm * (random.uniform(-1, 1) * cy_mag) + np.array([random.uniform(-cx_mag, cx_mag), 0], dtype=np.float32)
        p2 = p0 + tdir * (0.66*L) + nrm * (random.uniform(-1, 1) * cy_mag) + np.array([random.uniform(-cx_mag, cx_mag), 0], dtype=np.float32)

        # timing
        duration = random.uniform(*self.path_duration_s)
        # spawn time: initial dots can be delayed so not all appear at once
        delay = (0.0 if initial else random.uniform(*self.spawn_delay_s))
        start_time = now + delay

        # visuals
        base_r = random.uniform(self.dot_radius_min, self.dot_radius_max)
        breathe_rate = random.uniform(*self.dot_breathe_rate)
        breathe_phase = random.uniform(0, 2*math.pi)
        jitter_phase = random.uniform(0, 2*math.pi)

        # smoothed avoidance memory
        return {
            "p0": p0, "p1": p1, "p2": p2, "p3": p3,
            "start": start_time, "dur": duration,
            "base_r": base_r,
            "breathe_rate": breathe_rate, "breathe_phase": breathe_phase,
            "jitter_phase": jitter_phase,
            "ay_smooth": 0.0, "ax_smooth": 0.0,
        }

    def _advance_and_draw_dots(self, p: QPainter, w: int, h: int, dt: float, t: float):
        if not self._dots:
            return

        base_alpha = int(255 * DOT_OVERLAY_ALPHA)
        mx = float(self.mouse_pos.x()); my = float(self.mouse_pos.y())
        r2 = self.avoid_radius * self.avoid_radius

        p.setPen(Qt.NoPen)

        for i in range(len(self._dots)):
            d = self._dots[i]
            u = (t - d["start"]) / d["dur"]

            if u < 0.0:
                # hasn't spawned yet
                continue
            if u >= 1.0:
                # finished; schedule a new swoop with random delay
                self._dots[i] = self._make_dot(t, w, h, initial=False)
                continue

            # eased param for smooth motion
            ue = ease_in_out(u)

            # bezier path pos
            p0, p1, p2, p3 = d["p0"], d["p1"], d["p2"], d["p3"]
            pos = cubic_bezier(p0, p1, p2, p3, ue)

            # vertical jitter & breathing radius
            d["jitter_phase"] += self.jitter_rate * dt
            jy = self.vertical_jitter_px * math.sin(d["jitter_phase"])

            d["breathe_phase"] += d["breathe_rate"] * dt
            rad = d["base_r"] + self.dot_breathe_amp * (0.5 + 0.5*math.sin(d["breathe_phase"]))

            # alpha fade in/out with the same easing
            alpha = int(base_alpha * (0.3 + 0.7 * (math.sin(math.pi * u) ** 1.6)))  # zero at ends, peak mid-flight
            col = QColor(255, 255, 255, max(0, min(255, alpha)))
            p.setBrush(col)

            # subtle mouse avoidance (smoothed), small both-axes
            dx = pos[0] - mx; dy = (pos[1] + jy) - my
            d2 = dx*dx + dy*dy
            ax_tgt = ay_tgt = 0.0
            if d2 < r2:
                dmag = math.sqrt(d2) + 1e-6
                strength = (self.avoid_radius - dmag) / self.avoid_radius
                push = self.avoid_push * strength
                ax_tgt = (dx / dmag) * push
                ay_tgt = (dy / dmag) * push

            # smooth the avoidance to avoid jolts
            d["ax_smooth"] = self.avoid_smooth * d["ax_smooth"] + (1.0 - self.avoid_smooth) * ax_tgt
            d["ay_smooth"] = self.avoid_smooth * d["ay_smooth"] + (1.0 - self.avoid_smooth) * ay_tgt

            draw_x = pos[0] + d["ax_smooth"]
            draw_y = pos[1] + jy + d["ay_smooth"]

            p.drawEllipse(QPointF(draw_x, draw_y), rad, rad)

    # ----------------- Painting -----------------
    def _paint_purple(self, p: QPainter):
        h = self.height()
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(26, 15, 30))   # #1A0F1E
        grad.setColorAt(1.0, QColor(46, 0, 62))    # #2E003E
        p.fillRect(self.rect(), grad)

    def _paint_animated(self, p: QPainter):
        w = self.width(); h = self.height()
        now = time.time(); dt = max(0.0, min(0.05, now - self._last_t))
        self._last_t = now

        # background gradient
        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0.0, color_hex(18, 14, 28))
        grad.setColorAt(0.5 + 0.08*math.sin(now*0.7), color_hex(30, 18, 45))
        grad.setColorAt(1.0, color_hex(12, 10, 20))
        p.fillRect(self.rect(), grad)

        # big blobs (unchanged)
        for i in range(len(self.circles)):
            x, y, dx, dy, r, hue_t = self.circles[i]
            x += dx*0.0015; y += dy*0.0015
            if x < -0.2: x = 1.2
            if x > 1.2: x = -0.2
            if y < -0.2: y = 1.2
            if y > 1.2: y = -0.2
            self.circles[i][0] = x; self.circles[i][1] = y

            mix = 0.5 + 0.5*math.sin(2*math.pi*(hue_t + 0.05*now))
            col = QColor(
                int(ACCENT_ORANGE.red()  *(1-mix) + PRIMARY_PURPLE.red()  *mix),
                int(ACCENT_ORANGE.green()*(1-mix) + PRIMARY_PURPLE.green()*mix),
                int(ACCENT_ORANGE.blue() *(1-mix) + PRIMARY_PURPLE.blue() *mix),
                48
            )
            p.setPen(Qt.NoPen); p.setBrush(col)
            p.drawEllipse(QPointF(x*w, y*h), r, r)

        # swooping dots
        self._ensure_dots(w, h)
        self._advance_and_draw_dots(p, w, h, dt, now)

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        if self.mode == "purple":
            self._paint_purple(p)
        else:
            self._paint_animated(p)
        p.end()
