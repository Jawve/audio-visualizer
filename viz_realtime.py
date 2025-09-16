# viz_realtime.py — simple real-time loopback spectrum (Windows, PyAudioWPatch)
# pip install PyAudioWPatch numpy matplotlib

import os, json, time, argparse, sys
import numpy as np
import pyaudiowpatch as pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

FRAMES_PER_BUFFER = 1024
CACHE_FILE = "device_cache.json"

# ---------- cache utils (compatible with your previous script) ----------
def load_cache(path=CACHE_FILE):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache, path=CACHE_FILE):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, path)

def cache_key(dev: dict) -> str:
    return dev["name"].lower()

# ---------- device selection / probing ----------
def list_loopbacks(pa):
    devs = list(pa.get_loopback_device_info_generator())
    print("=== WASAPI LOOPBACK DEVICES ===")
    for d in devs:
        print(f"[{d['index']}] {d['name']} | default SR: {d.get('defaultSampleRate')} | maxInCh: {d.get('maxInputChannels')}")
    return devs

def resolve_device(pa, device_arg):
    # device_arg: None -> prompt, int index, or substring
    devs = list(pa.get_loopback_device_info_generator())
    if device_arg is None:
        list_loopbacks(pa)
        while True:
            s = input("\nEnter device index or substring (q to quit): ").strip()
            if s.lower() in ("q","quit","exit"): sys.exit(0)
            if s.isdigit():
                try: return pa.get_device_info_by_index(int(s))
                except Exception: print("Invalid index.")
            else:
                matches = [d for d in devs if s.lower() in d["name"].lower()]
                if len(matches) == 1: return matches[0]
                if len(matches) > 1:
                    print("Multiple matches:")
                    for d in matches: print(f"  [{d['index']}] {d['name']}")
                else:
                    print("No match, try again.")
    else:
        # passed param: try index then substring
        if isinstance(device_arg, int):
            return pa.get_device_info_by_index(device_arg)
        s = str(device_arg).lower()
        for d in devs:
            if s in d["name"].lower(): return d
        raise RuntimeError(f"No loopback device matching: {device_arg!r}")

def probe_device(pa, dev):
    idx = dev["index"]
    default_sr = int(round(dev.get("defaultSampleRate") or 48000))
    max_in = int(dev.get("maxInputChannels") or 0)

    rates = []
    for r in (default_sr, 48000, 44100, 96000):
        if r and r not in rates: rates.append(r)
    chs = []
    for c in (2, max_in, 1, 8):
        if c and (max_in == 0 or c <= max_in) and c not in chs:
            chs.append(c)
    if not chs: chs = [2, 1]

    last_err = None
    for ch in chs:
        for sr in rates:
            try:
                st = pa.open(format=pyaudio.paInt16, channels=ch, rate=sr,
                             input=True, input_device_index=idx,
                             frames_per_buffer=FRAMES_PER_BUFFER)
                st.stop_stream(); st.close()
                print(f"[probe] ✅ {ch}ch @ {sr} Hz works")
                return ch, sr
            except Exception as e:
                last_err = e
                print(f"[probe] ✗ {ch}ch @ {sr} Hz -> {e}")
    raise RuntimeError(f"Probe failed: {last_err}")

def open_stream_with_cache(pa, dev, force_probe=False):
    cache = load_cache()
    key = cache_key(dev)
    cfg = cache.get(key)

    # force re-probe or cache miss or caps changed
    caps = {"defaultSampleRate": dev.get("defaultSampleRate"),
            "maxInputChannels": dev.get("maxInputChannels")}
    need_probe = force_probe or (cfg is None) \
        or cfg.get("caps",{}) != caps

    if need_probe:
        ch, sr = probe_device(pa, dev)
        cfg = {"channels": ch, "samplerate": sr, "caps": caps, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        cache[key] = cfg
        save_cache(cache)

    # open with cached working config
    st = pa.open(format=pyaudio.paInt16,
                 channels=int(cfg["channels"]),
                 rate=int(cfg["samplerate"]),
                 input=True,
                 input_device_index=dev["index"],
                 frames_per_buffer=FRAMES_PER_BUFFER)
    return st, int(cfg["channels"]), int(cfg["samplerate"])

# ---------- simple DSP helpers ----------
def dbfs(x):
    # x: float32 mono signal in [-1,1], return RMS dBFS
    rms = np.sqrt(np.mean(np.square(x)) + 1e-12)
    return 20*np.log10(rms + 1e-12)

def int16_to_float32(buf, channels):
    # Convert bytes -> int16 -> float32 [-1,1]; downmix to mono
    a = np.frombuffer(buf, dtype=np.int16)
    if channels > 1:
        a = a.reshape(-1, channels).mean(axis=1)
    return (a.astype(np.float32) / 32768.0)

def magnitude_spectrum(x, sr, nfft=2048, fmax=20000, nbars=48):
    # Windowed FFT, linear bins collapsed to nbars (approx log spacing optional later)
    N = min(len(x), nfft)
    w = np.hanning(N)
    X = np.fft.rfft(x[:N] * w)
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    mags = np.abs(X)

    # limit to fmax
    mask = freqs <= fmax
    freqs = freqs[mask]; mags = mags[mask]

    # group into nbars by equal-size slices (simple & fast)
    splits = np.array_split(mags, nbars)
    bars = np.array([s.mean() for s in splits], dtype=np.float32)
    # normalize a bit for display stability
    bars_db = 20*np.log10(bars + 1e-9)
    # shift so floor ~ -60 dB
    bars_db = np.clip(bars_db, -60, 0)
    return bars_db

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Real-time loopback spectrum (PyAudioWPatch + matplotlib).")
    ap.add_argument("--device", help="Loopback device index (int) or substring (str).")
    ap.add_argument("--probe", action="store_true", help="Force reprobe of device config.")
    ap.add_argument("--bars", type=int, default=48, help="Number of spectrum bars (default 48)")
    args = ap.parse_args()

    pa = pyaudio.PyAudio()
    # parse device arg
    dev_arg = None
    if args.device is not None and str(args.device).isdigit():
        dev_arg = int(args.device)
    else:
        dev_arg = args.device

    dev = resolve_device(pa, dev_arg)
    print(f"\nOpening [{dev['index']}] {dev['name']} …")
    stream, channels, sr = open_stream_with_cache(pa, dev, force_probe=args.probe)
    print(f"✅ Stream open: {channels} ch @ {sr} Hz")
    print("Close the plot window to stop.\n")

    # --- matplotlib setup ---
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_ylim(-60, 0)
    ax.set_xlim(0, args.bars)
    ax.set_ylabel("dBFS")
    ax.set_xlabel("Frequency →")
    ax.set_title(f"Spectrum — {dev['name']}  ({sr} Hz, {channels} ch)")
    bars = ax.bar(range(args.bars), [-60]*args.bars)
    rms_text = ax.text(0.01, 0.95, "RMS: -- dBFS", transform=ax.transAxes, va="top")

    # --- animation/update loop ---
    def update(_frame):
        try:
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        except Exception:
            return bars

        mono = int16_to_float32(data, channels=channels)
        # show simple meters
        level = dbfs(mono)
        rms_text.set_text(f"RMS: {level:6.1f} dBFS")

        spec = magnitude_spectrum(mono, sr=sr, nfft=4096, fmax=20000, nbars=args.bars)
        for rect, h in zip(bars, spec):
            rect.set_height(h)
        return bars

    ani = FuncAnimation(fig, update, interval=15, blit=False)

    try:
        plt.show()
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        pa.terminate()

if __name__ == "__main__":
    main()
