# audio_core.py — shared loopback device selection, probing, caching, streaming
import os, json, time, sys
from typing import Dict, Any, Tuple, Optional, Iterable
import numpy as np
import pyaudiowpatch as pyaudio

CACHE_FILE = "device_cache.json"
FRAMES_PER_BUFFER = 1024

# ---------------- cache utils ----------------
def _load_cache() -> Dict[str, Any]:
    if not os.path.exists(CACHE_FILE): return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(cache: Dict[str, Any]) -> None:
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, CACHE_FILE)

def _cache_key(dev: dict) -> str:
    return dev["name"].lower()  # names are more stable than indices

# ---------------- device utils ----------------
def list_loopbacks(pa: pyaudio.PyAudio) -> list[dict]:
    devs = list(pa.get_loopback_device_info_generator())
    for d in devs:
        print(f"[{d['index']}] {d['name']} | default SR: {d.get('defaultSampleRate')} | maxInCh: {d.get('maxInputChannels')}")
    return devs

def resolve_device(pa: pyaudio.PyAudio, device_arg: Optional[str|int]) -> dict:
    """device_arg can be None (prompt), an int index, or a substring."""
    devs = list(pa.get_loopback_device_info_generator())
    if device_arg is None:
        list_loopbacks(pa)
        while True:
            s = input("\nEnter device index or substring (q to quit): ").strip()
            if s.lower() in ("q","quit","exit"): sys.exit(0)
            if s.isdigit():
                try:
                    return pa.get_device_info_by_index(int(s))
                except Exception:
                    print("Invalid index.")
            else:
                matches = [d for d in devs if s.lower() in d["name"].lower()]
                if len(matches) == 1: return matches[0]
                if len(matches) > 1:
                    print("Multiple matches:")
                    for d in matches: print(f"  [{d['index']}] {d['name']}")
                else:
                    print("No match, try again.")
    else:
        if isinstance(device_arg, int):
            return pa.get_device_info_by_index(device_arg)
        s = str(device_arg).lower()
        for d in devs:
            if s in d["name"].lower(): return d
        raise RuntimeError(f"No loopback device matching: {device_arg!r}")

# ---------------- probing & opening ----------------
def probe_device(pa: pyaudio.PyAudio, dev: dict) -> Tuple[int, int]:
    """Return (channels, samplerate) that opens successfully."""
    idx = dev["index"]
    default_sr = int(round(dev.get("defaultSampleRate") or 48000))
    max_in = int(dev.get("maxInputChannels") or 0)

    rates = []
    for r in (default_sr, 48000, 44100, 96000):
        if r and r not in rates: rates.append(r)

    chs = []
    for c in (2, max_in, 1, 8):
        if c and (max_in == 0 or c <= max_in) and c not in chs: chs.append(c)
    if not chs: chs = [2, 1]

    last_err = None
    for ch in chs:
        for sr in rates:
            try:
                st = pa.open(format=pyaudio.paInt16, channels=ch, rate=sr,
                             input=True, input_device_index=idx,
                             frames_per_buffer=FRAMES_PER_BUFFER)
                st.stop_stream(); st.close()
                print(f"[probe] ✅ {ch}ch @ {sr} Hz")
                return ch, sr
            except Exception as e:
                last_err = e
                print(f"[probe] ✗ {ch}ch @ {sr} Hz → {e}")
    raise RuntimeError(f"Probe failed for '{dev['name']}': {last_err}")

def get_stream(pa: pyaudio.PyAudio, dev: dict, force_probe: bool=False):
    """Open and return (stream, channels, samplerate). Uses cache, auto-reprobes if needed."""
    cache = _load_cache()
    key = _cache_key(dev)
    cfg = cache.get(key)
    caps_now = {"defaultSampleRate": dev.get("defaultSampleRate"),
                "maxInputChannels": dev.get("maxInputChannels")}

    need_probe = force_probe or (cfg is None) or (cfg.get("caps") != caps_now)
    if need_probe:
        ch, sr = probe_device(pa, dev)
        cfg = {"channels": ch, "samplerate": sr, "caps": caps_now, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        cache[key] = cfg
        _save_cache(cache)

    try:
        st = pa.open(format=pyaudio.paInt16,
                     channels=int(cfg["channels"]),
                     rate=int(cfg["samplerate"]),
                     input=True,
                     input_device_index=dev["index"],
                     frames_per_buffer=FRAMES_PER_BUFFER)
    except Exception:
        # one-shot auto-reprobe if cached config no longer works
        ch, sr = probe_device(pa, dev)
        cfg.update({"channels": ch, "samplerate": sr, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        cache[key] = cfg; _save_cache(cache)
        st = pa.open(format=pyaudio.paInt16,
                     channels=ch, rate=sr,
                     input=True, input_device_index=dev["index"],
                     frames_per_buffer=FRAMES_PER_BUFFER)
    return st, int(cfg["channels"]), int(cfg["samplerate"])

# ---------------- chunk helpers for UI/DSP ----------------
def chunks_numpy_mono(stream, channels: int) -> Iterable[np.ndarray]:
    """Yield mono float32 arrays in [-1,1] from the stream, one chunk per read."""
    import numpy as np
    while True:
        buf = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        a = np.frombuffer(buf, dtype=np.int16)
        if channels > 1:
            a = a.reshape(-1, channels).mean(axis=1)
        yield (a.astype(np.float32) / 32768.0)

def dbfs(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x*x) + 1e-12)
    return 20*np.log10(rms + 1e-12)
