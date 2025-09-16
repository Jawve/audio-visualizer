# visualizer.py — real-time spectrum using audio_core + matplotlib
# pip install matplotlib
import argparse, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyaudiowpatch as pyaudio

from audio_core import (
    list_loopbacks, resolve_device, get_stream, chunks_numpy_mono, dbfs
)

def magnitude_bars(x: np.ndarray, sr: int, nfft=4096, fmax=20000, nbars=48) -> np.ndarray:
    """Windowed FFT → magnitude (dBFS-like bars)."""
    N = min(len(x), nfft)
    if N < 64:
        return np.full(nbars, -60.0, dtype=np.float32)
    w = np.hanning(N)
    X = np.fft.rfft(x[:N] * w)
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    mags = np.abs(X)
    mask = freqs <= fmax
    mags = mags[mask]
    splits = np.array_split(mags, nbars)
    bars = np.array([s.mean() for s in splits], dtype=np.float32)
    bars_db = 20*np.log10(bars + 1e-9)
    return np.clip(bars_db, -60, 0)

def main():
    ap = argparse.ArgumentParser(description="Real-time loopback spectrum viewer.")
    ap.add_argument("--device", help="Index or substring of loopback device.")
    ap.add_argument("--probe", action="store_true", help="Force reprobe.")
    ap.add_argument("--bars", type=int, default=48, help="Number of bars.")
    args = ap.parse_args()

    pa = pyaudio.PyAudio()
    dev_arg = int(args.device) if (args.device and str(args.device).isdigit()) else args.device
    dev = resolve_device(pa, dev_arg)
    print(f"Opening: [{dev['index']}] {dev['name']} …")
    stream, ch, sr = get_stream(pa, dev, force_probe=args.probe)
    print(f"✅ {ch} ch @ {sr} Hz")
    gen = chunks_numpy_mono(stream, ch)

    # --- Matplotlib UI ---
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_ylim(-60, 0); ax.set_xlim(0, args.bars)
    ax.set_ylabel("dBFS"); ax.set_xlabel("Frequency →")
    ax.set_title(f"Spectrum — {dev['name']}  ({sr} Hz)")
    bars = ax.bar(range(args.bars), [-60]*args.bars)
    rms_text = ax.text(0.01, 0.95, "RMS: -- dBFS", transform=ax.transAxes, va="top")

    def update(_i):
        try:
            mono = next(gen)  # small chunk
        except Exception:
            return bars
        rms_text.set_text(f"RMS: {dbfs(mono):6.1f} dBFS")
        spec = magnitude_bars(mono, sr, nfft=4096, nbars=args.bars)
        for rect, h in zip(bars, spec):
            rect.set_height(h)
        return bars

    ani = FuncAnimation(
        fig, update,
        interval=15,          # ~60fps target; real fps depends on CPU/GPU
        blit=False,
        cache_frame_data=False,  # <-- suppresses your warning
        save_count=300           # <-- cap internal caching
    )

    try:
        plt.show()
    finally:
        try:
            stream.stop_stream(); stream.close()
        except Exception:
            pass
        pa.terminate()

if __name__ == "__main__":
    main()
