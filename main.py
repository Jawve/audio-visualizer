# main.py — keeps the “record to WAV” spike, reusing audio_core
import os, sys, wave, argparse
import pyaudiowpatch as pyaudio
from audio_core import (
    list_loopbacks, resolve_device, get_stream, CACHE_FILE, FRAMES_PER_BUFFER
)

def sanitize(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (" ", "_", "-", ".")).strip().replace(" ", "_")

def save_wav(frames, ch, sr, devname, outfile=None) -> str:
    path = os.path.join(os.getcwd(), outfile) if outfile else os.path.join(
        os.getcwd(), f"loopback_{sanitize(devname)}_{sr}Hz_{ch}ch.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(ch); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(b"".join(frames))
    return path

def main():
    ap = argparse.ArgumentParser(description="WASAPI loopback recorder (PyAudioWPatch) using audio_core.")
    ap.add_argument("--device", help="Device index or name substring. If omitted, prompt.")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--outfile", type=str)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    args = ap.parse_args()

    if args.clear_cache and os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("Deleted", CACHE_FILE)
        return

    pa = pyaudio.PyAudio()
    dev_arg = int(args.device) if (args.device and str(args.device).isdigit()) else args.device
    dev = resolve_device(pa, dev_arg)

    print(f"\nTarget: [{dev['index']}] {dev['name']} | default SR: {dev.get('defaultSampleRate')} | maxInCh: {dev.get('maxInputChannels')}")
    stream, ch, sr = get_stream(pa, dev, force_probe=args.probe)
    print(f"Opened: {ch} ch @ {sr} Hz")

    frames = []
    for _ in range(int(sr / FRAMES_PER_BUFFER * args.seconds)):
        frames.append(stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False))

    stream.stop_stream(); stream.close(); pa.terminate()
    outpath = save_wav(frames, ch, sr, dev["name"], args.outfile)
    print("Saved:", outpath)

if __name__ == "__main__":
    main()
