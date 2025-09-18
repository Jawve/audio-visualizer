# audio_hub.py (smooth, log-spaced bars with rolling buffer)
from __future__ import annotations
import math, time, threading
from typing import List, Optional
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from audio_core import (
    list_loopback_devices,
    open_loopback_stream,
    chunks_mono_float32,
    FRAMES_PER_BUFFER,
)

class AudioHub(QObject):
    """
    - Scans WASAPI loopback devices and emits devicesChanged(list[str]).
    - Opens a real loopback stream and emits levelsUpdated(list[0..1]) ~30 FPS.
    - Bars are log-spaced across 20–20k Hz, computed from a rolling buffer.
    - Sensitivity and EQ window [low..high] are applied smoothly.
    """

    devicesChanged = Signal(list)     # device names (with 'Default Output' first)
    currentDeviceChanged = Signal(str)
    levelsUpdated = Signal(list)      # list[float] in [0..1], length == nbars

    def __init__(self, parent=None, nbars: int = 48):
        super().__init__(parent)
        self._nbars = nbars
        self._sens = 1.0
        self._eq_lo = 0.0
        self._eq_hi = 1.0

        self._dev_names: List[str] = []
        self._device_name: Optional[str] = None

        # stream state
        self._stream = None
        self._stream_channels = 0
        self._stream_sr = 48000
        self._reader_thread = None
        self._reader_alive = False

        # rolling buffer for smoother FFTs (≈8192 samples)
        self._buf_size = 8192
        self._buf = np.zeros(self._buf_size, dtype=np.float32)
        self._buf_w = 0  # write position (ring)

        # tick painter
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30 FPS

        # async scan
        QTimer.singleShot(0, self._scan_devices)

    # ---------------- public controls ----------------
    def set_device(self, name: str):
        if not name:
            return
        if self._device_name == name:
            return
        self._device_name = name
        self.currentDeviceChanged.emit(name)
        self._restart_stream()

    def set_sensitivity(self, v: float):
        try: v = float(v)
        except Exception: return
        self._sens = max(0.05, min(4.0, v))

    def set_eq_range(self, low: float, high: float):
        lo = max(0.0, min(1.0, float(low)))
        hi = max(0.0, min(1.0, float(high)))
        if lo > hi: lo, hi = hi, lo
        self._eq_lo, self._eq_hi = lo, hi

    def terminate(self):
        self._stop_reader()
        self._timer.stop()

    # ---------------- device scan ----------------
    def _scan_devices(self):
        devs = list_loopback_devices()
        names = ["Default Output"]
        names.extend([d["name"] for d in devs])
        # dedup keep order
        seen, clean = set(), []
        for n in names:
            if n not in seen:
                seen.add(n); clean.append(n)
        self._dev_names = clean
        self.devicesChanged.emit(clean)
        if self._device_name is None:
            self.set_device("Default Output")

    # ---------------- stream lifecycle ----------------
    def _restart_stream(self):
        self._stop_reader()
        try:
            st, ch, sr, _dev = open_loopback_stream(self._device_name)
            self._stream = st
            self._stream_channels = ch
            self._stream_sr = sr
            # reset buffer
            self._buf[:] = 0.0
            self._buf_w = 0
            self._start_reader()
        except Exception:
            self._stream = None
            self._stream_channels = 0

    def _start_reader(self):
        self._reader_alive = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _stop_reader(self):
        self._reader_alive = False
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.5)
        self._reader_thread = None
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        self._stream = None

    def _reader_loop(self):
        gen = chunks_mono_float32(self._stream, self._stream_channels)
        B = self._buf_size
        w = self._buf_w
        while self._reader_alive and self._stream:
            try:
                chunk = next(gen)  # float32 [-1,1], length = FRAMES_PER_BUFFER
            except Exception:
                break
            n = int(chunk.size)
            if n >= B:
                # take the last B samples of this chunk
                self._buf[:] = chunk[-B:]
                w = 0
            else:
                end = w + n
                if end <= B:
                    self._buf[w:end] = chunk
                else:
                    k = B - w
                    self._buf[w:] = chunk[:k]
                    self._buf[:n-k] = chunk[k:]
                w = (w + n) % B
            self._buf_w = w

    # ---------------- tick → compute bars ----------------
    def _tick(self):
        if self._stream is None or self._stream_channels <= 0:
            # idle “breathing” wave (left→right low→high)
            t = time.time()
            n = self._nbars
            vals = []
            for i in range(n):
                f = i / max(1, n - 1)
                v = 0.22 + 0.22*(0.5 + 0.5*math.sin(6.0*f + 0.9*t + 1.2*math.sin(0.35*t)))
                vals.append(v)
            self.levelsUpdated.emit(vals)
            return

        # make a contiguous copy of latest buffer (oldest→newest)
        B, w = self._buf_size, self._buf_w
        if w == 0:
            buf = self._buf.copy()
        else:
            buf = np.concatenate((self._buf[w:], self._buf[:w]))

        bars = self._bars_from_buffer(buf, self._stream_sr, self._nbars, self._eq_lo, self._eq_hi)
        bars = np.clip(bars * self._sens, 0.0, 1.0)
        self.levelsUpdated.emit(bars.tolist())

    # ---- DSP helpers ----
    def _bars_from_buffer(self, x: np.ndarray, sr: int, nbars: int, lo: float, hi: float) -> np.ndarray:
        """
        Log-spaced magnitude sampling at fixed center freqs with interpolation,
        then light spatial smoothing across bars. Returns values in [0..1].
        """
        if x.size < 512:
            return np.zeros(nbars, dtype=np.float32)

        # Use up to 4096 points from end (enough resolution; fast)
        NFFT = 4096 if x.size >= 4096 else int(2**np.ceil(np.log2(x.size)))
        xw = x[-NFFT:]
        # Hann window
        w = np.hanning(xw.size).astype(np.float32)
        X = np.fft.rfft(xw * w)
        mag = np.abs(X).astype(np.float32)
        freqs = np.fft.rfftfreq(xw.size, 1.0/sr).astype(np.float32)

        # Keep 20–20k
        fmin, fmax = 20.0, 20000.0
        m = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(m):
            return np.zeros(nbars, dtype=np.float32)
        freqs = freqs[m]; mag = mag[m]

        # Log-spaced centers across [fmin..fmax]
        centers = np.logspace(np.log10(fmin), np.log10(fmax), nbars).astype(np.float32)

        # Interpolate magnitude at centers (linear in Hz domain gives smooth shape)
        # Avoid zeros by minimum floor
        interp = np.interp(centers, freqs, mag, left=mag[0], right=mag[-1]).astype(np.float32)
        interp = np.maximum(interp, 1e-8)

        # Apply smooth EQ window on 0..1 log-frequency coordinate
        f01 = (np.log10(centers) - np.log10(fmin)) / (np.log10(fmax) - np.log10(fmin))
        # soft edges: a simple raised-cosine on 10% margins of the selected region
        edge = 0.10
        w_lo = _raised_cosine_edge(f01, lo, edge, start=True)
        w_hi = _raised_cosine_edge(f01, hi, edge, start=False)
        weight = w_lo * w_hi

        interp *= (0.25 + 0.75*weight)  # dim out-of-range instead of hard zero

        # Convert to dB-like then to 0..1
        db = 20.0*np.log10(interp)  # ≈ [-inf..0]
        # normalize last 60 dB
        bars = np.clip((db + 60.0)/60.0, 0.0, 1.0)

        # Light spatial smoothing (remove “teeth”)
        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        bars = np.convolve(bars, kernel, mode="same")

        return bars.astype(np.float32)

def _raised_cosine_edge(f01: np.ndarray, edge_pos: float, width: float, start: bool) -> np.ndarray:
    """
    Smooth step that is ~0 below/above edge with a cosine ramp of given width.
    If start=True: ramps from 0→1 across [edge_pos - width, edge_pos].
    If start=False: ramps from 1→0 across [edge_pos, edge_pos + width].
    """
    edge_pos = float(np.clip(edge_pos, 0.0, 1.0))
    width = max(1e-6, float(width))
    y = np.ones_like(f01, dtype=np.float32)
    if start:
        # below (edge_pos - width) → 0, above edge_pos → 1
        a = edge_pos - width
        y = np.where(f01 <= a, 0.0, y)
        mask = (f01 > a) & (f01 < edge_pos)
        phase = (f01[mask] - a) / (edge_pos - a)
        y[mask] = 0.5 - 0.5*np.cos(np.pi*phase)
    else:
        # below edge_pos → 1, above (edge_pos + width) → 0
        b = edge_pos + width
        y = np.where(f01 >= b, 0.0, y)
        mask = (f01 > edge_pos) & (f01 < b)
        phase = (b - f01[mask]) / (b - edge_pos)
        y[mask] = 0.5 - 0.5*np.cos(np.pi*phase)
    return y
