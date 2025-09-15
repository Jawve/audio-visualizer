# loopback_cached.py — WASAPI loopback with resilient caching & auto-reprobe
# pip install PyAudioWPatch
import os, sys, json, wave, argparse, time
from typing import Optional, Tuple, Dict, Any
import pyaudiowpatch as pyaudio

FRAMES_PER_BUFFER = 1024
DEFAULT_SECONDS   = 3
CACHE_FILE        = "device_cache.json"

def sanitize(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (" ", "_", "-", ".")).strip().replace(" ", "_")

def load_cache() -> Dict[str, Any]:
    if not os.path.exists(CACHE_FILE): return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_cache(cache: Dict[str, Any]) -> None:
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, CACHE_FILE)

def list_loopbacks(pa: pyaudio.PyAudio):
    devs = list(pa.get_loopback_device_info_generator())
    if not devs:
        print("No devices found (using wasapi)")
        return []
    print("=== WASAPI LOOPBACK DEVICES ===")
    for d in devs:
        print(f"[{d['index']}] {d['name']} | default SR: {d.get('defaultSampleRate')} | maxInCh: {d.get('maxInputChannels')}")
    return devs

def resolve_device(pa: pyaudio.PyAudio, device_arg: Optional[str|int]) -> dict:
    """device_arg: None -> prompt; int index; str substring match (case-insensitive)."""
    all_devs = list(pa.get_loopback_device_info_generator())
    if device_arg is None:
        list_loopbacks(pa)
        while True:
            choice = input("\nEnter device index or substring (or 'q' to quit): ").strip()
            if choice.lower() in ("q","quit","exit"): sys.exit(0)
            # try int index first
            if choice.isdigit():
                try:
                    return pa.get_device_info_by_index(int(choice))
                except Exception:
                    print("Invalid index, try again.")
                    continue
            # substring match
            s = choice.lower()
            matches = [d for d in all_devs if s in d["name"].lower()]
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                print("Multiple matches:")
                for d in matches: print(f"  [{d['index']}] {d['name']}")
            else:
                print("No match; try again.")
    else:
        if isinstance(device_arg, int):
            return pa.get_device_info_by_index(device_arg)
        # substring
        s = str(device_arg).lower()
        for d in all_devs:
            if s in d["name"].lower():
                return d
        # fallback: search all devices by index/name if not in loopback set yet
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if s in d["name"].lower():
                return d
        raise RuntimeError(f"No loopback device matching: {device_arg!r}")

def cache_key(dev: dict) -> str:
    # Prefer name-based key (indices can shuffle). Lowercase for stability.
    return dev["name"].lower()

def probe_device(pa: pyaudio.PyAudio, dev: dict) -> Tuple[int, int]:
    """Find a working (channels, samplerate)."""
    idx = dev["index"]
    default_sr = int(round(dev.get("defaultSampleRate") or 48000))
    max_in = int(dev.get("maxInputChannels") or 0)

    rate_candidates = []
    for r in (default_sr, 48000, 44100, 96000):
        if r and r not in rate_candidates:
            rate_candidates.append(r)

    ch_candidates = []
    for c in (2, max_in, 1, 8):  # try common layouts; 8 for virtual buses like Sonar/Voicemeeter
        if c and (max_in == 0 or c <= max_in) and c not in ch_candidates:
            ch_candidates.append(c)

    last_err = None
    for ch in ch_candidates:
        for sr in rate_candidates:
            try:
                print(f"[probe] trying {ch}ch @ {sr} Hz …")
                stream = pa.open(format=pyaudio.paInt16, channels=ch, rate=sr,
                                 input=True, input_device_index=idx, frames_per_buffer=FRAMES_PER_BUFFER)
                stream.stop_stream(); stream.close()
                print(f"[probe] ✅ works with {ch}ch @ {sr} Hz")
                return ch, sr
            except Exception as e:
                last_err = e
                print(f"[probe] ✗ {ch}ch @ {sr} Hz → {e}")
    raise RuntimeError(f"Probe failed for '{dev['name']}': {last_err}")

def open_with_cfg(pa: pyaudio.PyAudio, dev: dict, ch: int, sr: int):
    return pa.open(format=pyaudio.paInt16, channels=ch, rate=sr,
                   input=True, input_device_index=dev["index"],
                   frames_per_buffer=FRAMES_PER_BUFFER)

def save_wav(frames, ch, sr, devname, outfile=None) -> str:
    path = os.path.join(os.getcwd(), outfile) if outfile else os.path.join(
        os.getcwd(), f"loopback_{sanitize(devname)}_{sr}Hz_{ch}ch.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(ch); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(b"".join(frames))
    return path

def main():
    ap = argparse.ArgumentParser(description="WASAPI loopback recorder with resilient caching (PyAudioWPatch).")
    ap.add_argument("--device", help="Device index or name substring. If omitted, you’ll be prompted.")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    ap.add_argument("--outfile", type=str)
    ap.add_argument("--probe", action="store_true", help="Force re-probe for the selected device.")
    ap.add_argument("--clear-cache", action="store_true")
    args = ap.parse_args()

    if args.clear_cache:
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE); print("Deleted", CACHE_FILE)
        else: print("No cache file to delete.")
        return

    pa = pyaudio.PyAudio()
    # parse device arg as int if numeric
    dev_arg = int(args.device) if (args.device and args.device.isdigit()) else args.device
    dev = resolve_device(pa, dev_arg)

    print(f"\nTarget: [{dev['index']}] {dev['name']} | default SR: {dev.get('defaultSampleRate')} | maxInCh: {dev.get('maxInputChannels')}")
    cache = load_cache()
    key = cache_key(dev)
    cfg = cache.get(key)

    # auto-reprobe conditions:
    need_probe = args.probe or (cfg is None)
    if cfg:
        # if reported caps changed since we cached, re-probe
        if ("caps" not in cfg
            or cfg["caps"].get("defaultSampleRate") != dev.get("defaultSampleRate")
            or cfg["caps"].get("maxInputChannels") != dev.get("maxInputChannels")):
            print("[cache] Device caps changed; re-probing…")
            need_probe = True

    if need_probe:
        try:
            ch, sr = probe_device(pa, dev)
            cfg = {
                "channels": ch,
                "samplerate": sr,
                "caps": {
                    "defaultSampleRate": dev.get("defaultSampleRate"),
                    "maxInputChannels": dev.get("maxInputChannels")
                },
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            cache[key] = cfg
            save_cache(cache)
            print(f"[cache] Saved: {cfg['channels']}ch @ {cfg['samplerate']} Hz → {CACHE_FILE}")
        except Exception as e:
            print("Probe failed:", e); pa.terminate(); sys.exit(1)

    # open with cached config; if it fails, auto-reprobe once
    try:
        stream = open_with_cfg(pa, dev, int(cfg["channels"]), int(cfg["samplerate"]))
    except Exception as e:
        print("[open] Cached config failed:", e)
        print("[open] Auto re-probing…")
        ch, sr = probe_device(pa, dev)
        cfg.update({"channels": ch, "samplerate": sr, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        cache[key] = cfg; save_cache(cache)
        stream = open_with_cfg(pa, dev, ch, sr)

    ch, sr = int(cfg["channels"]), int(cfg["samplerate"])
    print(f"\nRecording {args.seconds:.2f}s at {sr} Hz, {ch} ch … (ensure audio is playing)")
    frames = []
    for _ in range(int(sr / FRAMES_PER_BUFFER * args.seconds)):
        frames.append(stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False))

    stream.stop_stream(); stream.close(); pa.terminate()
    outpath = save_wav(frames, ch, sr, dev["name"], args.outfile)
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()
