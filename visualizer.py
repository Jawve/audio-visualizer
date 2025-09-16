# visualizer.py — sleek real-time loopback spectrum with custom header/footer and working theme toggle
# Requires: PyAudioWPatch, numpy, matplotlib
#   pip install PyAudioWPatch numpy matplotlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.colors as mcolors
import pyaudiowpatch as pyaudio

from audio_core import (
    resolve_device,
    get_stream,
    chunks_numpy_mono,
)

# ---------------- DSP ----------------
def magnitude_bars(x: np.ndarray, sr: int, nfft=4096, fmax=20000, nbars=48) -> np.ndarray:
    """Windowed FFT -> magnitude per bar in dBFS (clamped [-60, 0])."""
    N = min(len(x), nfft)
    if N < 64:
        return np.full(nbars, -60.0, dtype=np.float32)

    w = np.hanning(N)
    X = np.fft.rfft(x[:N] * w)
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    mags = np.abs(X)
    mags = mags[freqs <= fmax]

    splits = np.array_split(mags, nbars)
    bars_lin = np.array([s.mean() if len(s) else 0.0 for s in splits], dtype=np.float32)
    bars_db = 20.0 * np.log10(bars_lin + 1e-9)
    return np.clip(bars_db, -60.0, 0.0)

# ---------------- UI helpers ----------------
class SleekUI:
    """
    Single Axes covering the whole figure (0..1 x 0..1).
    We draw header/footer/body and all bars inside this Axes.
    This avoids widget quirks and guarantees click handling works.
    """
    def __init__(self, fig, nbars: int):
        self.fig = fig
        self.ax = fig.add_axes([0, 0, 1, 1])  # full-canvas axes
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")

        # Layout
        self.HEADER_H = 0.00
        self.FOOTER_H = 0.0
        self.BODY_Y   = self.FOOTER_H
        self.BODY_H   = 1.0 - self.HEADER_H - self.FOOTER_H
        self.MARGIN_X = 0.02  # inset margins for body
        self.nbars = nbars

        # Theme colors
        self.COLOR_BLACK = (0.0, 0.0, 0.0, 1.0)
        self.COLOR_BEIGE = (0.96, 0.95, 0.90, 1.0)

        # Body background (toggle between beige/black)
        self.body_is_black = False
        self.body_rect = Rectangle((0, self.BODY_Y), 1.0, self.BODY_H,
                                   facecolor=self.COLOR_BEIGE, edgecolor=None, zorder=0)
        self.ax.add_patch(self.body_rect)

        # Inner plot area where bars live (respect left/right margins)
        self.plot_x0 = self.MARGIN_X
        self.plot_x1 = 1.0 - self.MARGIN_X
        self.plot_y0 = self.BODY_Y + 0.02   # small vertical inset
        self.plot_y1 = self.BODY_Y + self.BODY_H - 0.02

        # Pre-create bar rectangles
        self.cmap = plt.colormaps["viridis"]
        self.norm = mcolors.Normalize(vmin=-60.0, vmax=0.0)

        self.bar_width = (self.plot_x1 - self.plot_x0) / self.nbars * 0.9
        self.bar_gap   = (self.plot_x1 - self.plot_x0) / self.nbars * 0.1
        self.bars = []
        for i in range(self.nbars):
            x = self.plot_x0 + i * ((self.plot_x1 - self.plot_x0) / self.nbars) + self.bar_gap * 0.5
            # start with zero height at bottom of plot area
            rect = Rectangle((x, self.plot_y0), self.bar_width, 0.0,
                             facecolor=self.cmap(self.norm(-60.0)), edgecolor=None, zorder=2)
            self.ax.add_patch(rect)
            self.bars.append(rect)

        # Minimal pill "button" in footer (centered)
        self.btn_w = 0.16
        self.btn_h = 0.50 * self.FOOTER_H  # in full-axes coordinates
        self.btn_x = 0.5 - self.btn_w / 2
        self.btn_y = (self.FOOTER_H - self.btn_h) / 2

        self.button = FancyBboxPatch(
            (self.btn_x, self.btn_y), self.btn_w, self.btn_h,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor=(0.18, 0.18, 0.20, 1.0), edgecolor=(1, 1, 1, 0.12), linewidth=0.9,
            zorder=3
        )
        self.ax.add_patch(self.button)
        self.label = self.ax.text(
            self.btn_x + self.btn_w/2, self.btn_y + self.btn_h/2,
            "Toggle Theme (T)",
            ha="center", va="center", color=(1, 1, 1, 0.9), fontsize=10, zorder=4
        )

        self._hover = False

    def _in_button(self, x, y):
        return (self.btn_x <= x <= self.btn_x + self.btn_w) and (self.btn_y <= y <= self.btn_y + self.btn_h)

    def on_motion(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            if self._hover:
                self._hover = False
                self._restyle_button(hover=False)
            return
        inside = self._in_button(event.xdata, event.ydata)
        if inside != self._hover:
            self._hover = inside
            self._restyle_button(hover=inside)

    def on_click(self, event):
        if event.inaxes is self.ax and event.xdata is not None and self._in_button(event.xdata, event.ydata):
            self.toggle_theme()

    def on_key(self, event):
        # Keyboard shortcut: T/t to toggle
        if event.key and event.key.lower() == "t":
            self.toggle_theme()

    def toggle_theme(self):
        self.body_is_black = not self.body_is_black
        self.body_rect.set_facecolor(self.COLOR_BLACK if self.body_is_black else self.COLOR_BEIGE)
        self.fig.canvas.draw_idle()

    def _restyle_button(self, hover: bool):
        if hover:
            self.button.set_facecolor((0.26, 0.26, 0.30, 1.0))
            self.button.set_linewidth(1.2)
            self.label.set_color((1, 1, 1, 1.0))
        else:
            self.button.set_facecolor((0.18, 0.18, 0.20, 1.0))
            self.button.set_linewidth(0.9)
            self.label.set_color((1, 1, 1, 0.9))
        self.fig.canvas.draw_idle()

    def update_bars(self, db_values: np.ndarray):
        """db_values in [-60..0]; map to heights within plot area."""
        # map -60..0 to 0..1 (relative height)
        heights01 = (db_values + 60.0) / 60.0
        # plot-space height range:
        H = self.plot_y1 - self.plot_y0
        for rect, h01, dbv in zip(self.bars, heights01, db_values):
            rect.set_height(max(0.0, float(h01) * H))
            rect.set_y(self.plot_y0)
            rect.set_facecolor(self.cmap(self.norm(float(dbv))))

def main():
    ap = argparse.ArgumentParser(description="Real-time loopback spectrum (sleek UI).")
    ap.add_argument("--device", help="Loopback device index (int) or name substring (str).")
    ap.add_argument("--probe", action="store_true", help="Force reprobe of device format.")
    ap.add_argument("--bars", type=int, default=48, help="Number of spectrum bars (default: 48)")
    ap.add_argument("--fmax", type=int, default=20000, help="Max frequency to display (default: 20000)")
    args = ap.parse_args()

    # Audio
    pa = pyaudio.PyAudio()
    dev_arg = int(args.device) if (args.device and str(args.device).isdigit()) else args.device
    dev = resolve_device(pa, dev_arg)
    print(f"Opening: [{dev['index']}] {dev['name']} …")
    stream, channels, sr = get_stream(pa, dev, force_probe=args.probe)
    print(f"✅ Stream open: {channels} ch @ {sr} Hz — close the window to stop.\n")
    gen = chunks_numpy_mono(stream, channels)

    # Figure
    plt.rcParams["figure.dpi"] = 110
    fig = plt.figure(figsize=(12, 5), facecolor=(0, 0, 0, 0))  # figure bg irrelevant; we draw full-canvas axes
    ui = SleekUI(fig, nbars=args.bars)

    # Events (button + keyboard)
    cid_move = fig.canvas.mpl_connect("motion_notify_event", ui.on_motion)
    cid_click = fig.canvas.mpl_connect("button_press_event", ui.on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", ui.on_key)

    # Animation state
    smoothing = 0.6
    smoothed = np.full(args.bars, -60.0, dtype=np.float32)

    def update(_i):
        nonlocal smoothed
        try:
            mono = next(gen)
        except Exception:
            return []

        spec = magnitude_bars(mono, sr, nfft=4096, fmax=args.fmax, nbars=args.bars)
        smoothed = smoothing * smoothed + (1.0 - smoothing) * spec
        ui.update_bars(smoothed)
        return []

    ani = FuncAnimation(
        fig, update,
        interval=15,
        blit=False,
        cache_frame_data=False,
        save_count=300
    )

    try:
        plt.show()
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        pa.terminate()
        # disconnect callbacks
        for cid in (cid_move, cid_click, cid_key):
            try: fig.canvas.mpl_disconnect(cid)
            except Exception: pass

if __name__ == "__main__":
    main()
