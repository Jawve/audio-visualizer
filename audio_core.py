# audio_core.py
# WASAPI loopback helpers using PyAudioWPatch (+ tiny config cache)
import os, json, time
from typing import Dict, Any, Tuple, Optional, List, Iterable
import numpy as np
import pyaudiowpatch as pyaudio

CACHE_FILE = "device_cache.json"
FRAMES_PER_BUFFER = 1024

# ---------------- cache ----------------
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

def _key_for(dev: dict) -> str:
    return dev["name"].lower()

# ---------------- devices ----------------
def list_loopback_devices() -> List[dict]:
    """
    Return list of WASAPI loopback-capable output devices (dicts from PyAudioWPatch).
    """
    pa = pyaudio.PyAudio()
    try:
        return list(pa.get_loopback_device_info_generator())
    finally:
        pa.terminate()

def get_default_loopback_index(pa: pyaudio.PyAudio) -> Optional[int]:
    """
    Find the loopback device that matches the system default output, if possible.
    """
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = wasapi.get("defaultOutputDevice", None)
        if default_out is None or default_out < 0:
            return None
        default_out_name = pa.get_device_info_by_index(default_out)["name"]
        for d in pa.get_loopback_device_info_generator():
            if d["name"] == default_out_name:
                return d["index"]
    except Exception:
        return None
    return None

def resolve_device_by_name(name: str) -> Optional[dict]:
    """
    name can be exact device name or the pseudo 'Default Output'.
    """
    pa = pyaudio.PyAudio()
    try:
        devs = list(pa.get_loopback_device_info_generator())
        if not devs:
            return None
        if name.strip().lower() == "default output":
            idx = get_default_loopback_index(pa)
            if idx is not None:
                return pa.get_device_info_by_index(idx)
            # fallback to first as "default"
            return devs[0]
        # substring / exact match
        s = name.lower()
        for d in devs:
            if s in d["name"].lower():
                return d
        return None
    finally:
        pa.terminate()

# ---------------- probing/opening ----------------
def _probe_format(pa: pyaudio.PyAudio, dev: dict) -> Tuple[int, int]:
    """
    Pick (channels, samplerate) that opens successfully for this loopback device.
    """
    idx = dev["index"]
    default_sr = int(round(dev.get("defaultSampleRate") or 48000))
    max_in = int(dev.get("maxInputChannels") or 0)

    rate_try = []
    for r in (default_sr, 48000, 44100, 96000):
        if r and r not in rate_try:
            rate_try.append(r)

    ch_try = []
    for c in (2, max_in, 1, 8):  # common layouts; Sonar/Voicemeeter often 8
        if c and (max_in == 0 or c <= max_in) and c not in ch_try:
            ch_try.append(c)
    if not ch_try:
        ch_try = [2, 1]

    last_err = None
    for ch in ch_try:
        for sr in rate_try:
            try:
                st = pa.open(format=pyaudio.paInt16, channels=ch, rate=sr,
                             input=True, input_device_index=idx,
                             frames_per_buffer=FRAMES_PER_BUFFER)
                st.stop_stream(); st.close()
                return ch, sr
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Probe failed for '{dev['name']}': {last_err}")

def open_loopback_stream(device_name: str) -> Tuple[object, int, int, dict]:
    """
    Open a WASAPI loopback stream for the named device (or 'Default Output').
    Returns: (stream, channels, samplerate, device_info)
    Uses a small cache keyed by device name.
    """
    pa = pyaudio.PyAudio()
    dev = resolve_device_by_name(device_name)
    if dev is None:
        # If still not found, try default list head as last resort
        all_devs = list(pa.get_loopback_device_info_generator())
        if not all_devs:
            pa.terminate()
            raise RuntimeError("No WASAPI loopback devices found.")
        dev = all_devs[0]

    cache = _load_cache()
    key = _key_for(dev)
    cfg = cache.get(key)
    caps = {
        "defaultSampleRate": dev.get("defaultSampleRate"),
        "maxInputChannels": dev.get("maxInputChannels")
    }

    need_probe = (cfg is None) or (cfg.get("caps") != caps)
    if need_probe:
        ch, sr = _probe_format(pa, dev)
        cfg = {"channels": ch, "samplerate": sr, "caps": caps, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        cache[key] = cfg
        _save_cache(cache)

    # open with (maybe) cached cfg; if fails, reprobe once
    try:
        st = pa.open(format=pyaudio.paInt16,
                     channels=int(cfg["channels"]),
                     rate=int(cfg["samplerate"]),
                     input=True, input_device_index=dev["index"],
                     frames_per_buffer=FRAMES_PER_BUFFER)
    except Exception:
        ch, sr = _probe_format(pa, dev)
        cfg.update({"channels": ch, "samplerate": sr, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        cache[key] = cfg; _save_cache(cache)
        st = pa.open(format=pyaudio.paInt16,
                     channels=ch, rate=sr, input=True, input_device_index=dev["index"],
                     frames_per_buffer=FRAMES_PER_BUFFER)

    return st, int(cfg["channels"]), int(cfg["samplerate"]), dev

# ---------------- chunk helpers ----------------
def chunks_mono_float32(stream, channels: int) -> Iterable[np.ndarray]:
    """
    Yield mono float32 arrays normalized to [-1, 1] from the loopback stream.
    """
    while True:
        buf = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        a = np.frombuffer(buf, dtype=np.int16)
        if channels > 1:
            a = a.reshape(-1, channels).mean(axis=1)
        yield (a.astype(np.float32) / 32768.0)
