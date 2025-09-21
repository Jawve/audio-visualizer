using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Text.Json;
using System.Windows.Threading;
using MathNet.Numerics.IntegralTransforms;
using NAudio.CoreAudioApi; // requires NAudio package

namespace test
{
    /// <summary>
    /// AudioHub (WPF)
    ///  • Lower-latency visualizer: ~50ms FFT window (adaptive per sample rate) @ ~60 FPS UI updates.
    ///  • Still computes 20 Hz → 20 kHz log-spaced bands via fractional-bin power integration (no bin gaps).
    ///  • Emphasizes beats/transients with fast-vs-slow dB envelopes per band.
    ///  • Sensitivity controls responsiveness (temporal/spectral smoothing + dB floor), not raw amplitude.
    ///  • Volume (±24 dB) pre-FFT; device cache + diagnostics log preserved.
    /// Estimated perceived delay ≈ NFFT/(2·SR) + small smoothing (typically ~30–80 ms now).
    /// </summary>
    public sealed class AudioHub : IDisposable
    {
        private readonly AudioCore _core = new();
        private readonly DispatcherTimer _tick;

        public event Action<List<DeviceInfo>>? DevicesChanged;
        public event Action<string?>? CurrentDeviceChanged;
        public event Action<List<float>>? LevelsUpdated;

        private List<DeviceInfo> _devices = new();
        private DeviceInfo? _current;
        public bool IsCapturing { get; private set; }

        private readonly float[] _ring;   // rolling mono buffer
        private int _wpos;
        private readonly object _lock = new();
        private readonly int _nbars;

        // Sensitivity → responsiveness parameters
        private float _sens = 1.0f;             // slider value (0.05..4)
        private float _alphaAttack = 0.25f;     // temporal EMA coefficients for output (amplitude domain)
        private float _alphaRelease = 0.55f;
        private double _dbFloor = -80.0;        // normalization floor in dB
        private float _spatialLoose = 0.5f;     // 0 tight .. 1 loose smoothing between neighbor bands

        // Beat/transient emphasis envelopes (in dB domain)
        private double[]? _fastDb;
        private double[]? _slowDb;
        private float[]? _prevBars;             // previous output (for temporal smoothing)

        // Volume (linear gain factor derived from dB)
        private double _volumeDb = 0.0;         // -24 .. +24 dB typical
        private double LinearGain => Math.Pow(10.0, _volumeDb / 20.0);

        // Device format
        private int _sr = DefaultSampleRate;
        private int _ch = DefaultChannels;
        private const int DefaultSampleRate = 48000;
        private const int DefaultChannels = 2;

        // Diagnostics/cache
        private readonly string _baseDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "test");
        private readonly string _logPath;
        private readonly string _cachePath;
        private bool _dumpedEdges;

        public AudioHub(int nbars = 48)
        {
            _nbars = Math.Max(8, nbars);
            _ring = new float[32768]; // ~0.68s @ 48 kHz (ample for any window we choose)

            Directory.CreateDirectory(_baseDir);
            Directory.CreateDirectory(Path.Combine(_baseDir, "logs"));
            _logPath = Path.Combine(_baseDir, "logs", "visualizer.log");
            _cachePath = Path.Combine(_baseDir, "device_cache.json");

            _core.PcmBytes += OnPcm;
            _core.DevicesChanged += OnDevices;
            _core.DefaultDeviceChanged += d =>
            {
                if (_current != null && _current.IsDefault)
                    CurrentDeviceChanged?.Invoke(d.Name);
                RequestRefresh();
            };

            // Default sensitivity mapping tuned for ~60 FPS + ~50 ms windows
            SetSensitivity(1.0f);

            RequestRefresh();

            _tick = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(16) }; // ~60 FPS for snappier feel
            _tick.Tick += (s, e) => { try { ComputeAndEmit(); } catch { } };
            _tick.Start();
        }

        public void RequestRefresh() => OnDevices(_core.ListLoopbackCapable());
        private void OnDevices(List<DeviceInfo> list) { _devices = list; DevicesChanged?.Invoke(list); }

        private void OnPcm(byte[] buf)
        {
            if (!IsCapturing) return;

            const int bytesPerSample = 4; // 32-bit float PCM
            int totalSamples = buf.Length / bytesPerSample;
            int ch = _ch <= 0 ? 1 : _ch;
            double gain = LinearGain;

            lock (_lock)
            {
                if (ch == 1)
                {
                    for (int i = 0; i < totalSamples; i++)
                    {
                        float m = BitConverter.ToSingle(buf, i * 4);
                        _ring[_wpos] = (float)(m * gain);
                        _wpos = (_wpos + 1) % _ring.Length;
                    }
                }
                else
                {
                    for (int i = 0; i + ch - 1 < totalSamples; i += ch)
                    {
                        double acc = 0;
                        for (int c = 0; c < ch; c++) acc += BitConverter.ToSingle(buf, (i + c) * 4);
                        float m = (float)(acc / ch * gain);
                        _ring[_wpos] = m;
                        _wpos = (_wpos + 1) % _ring.Length;
                    }
                }
            }
        }

        public void SetDeviceById(string? id)
        {
            if (string.IsNullOrWhiteSpace(id))
            {
                StopCapture(); _current = null; CurrentDeviceChanged?.Invoke(null);
                return;
            }
            var dev = _devices.FirstOrDefault(d => d.Id == id);
            if (dev == null) return;
            _current = dev;

            // Real format via NAudio (robust)
            (_sr, _ch) = QueryDeviceFormat(dev.Id) ?? (_sr, _ch);
            PersistDeviceSnapshot(dev, _sr, _ch);

            _core.OpenLoopback(dev);
            IsCapturing = true;
            CurrentDeviceChanged?.Invoke(dev.Name);
            _dumpedEdges = false; // re-dump edges for this device/sample rate
            Log($"[SetDevice] {dev.Name} ({dev.Id}) SR={_sr}Hz Ch={_ch}");
        }

        public void StopCapture() { _core.Stop(); IsCapturing = false; }

        /// <summary>
        /// Sensitivity controls responsiveness (not amplitude):
        ///  - higher sens → faster temporal smoothing, tighter spatial smoothing, lower dB floor
        ///  - lower sens → slower & steadier, looser spatial smoothing, higher dB floor
        /// Tuned for ~60 FPS updates and ~50ms FFT windows.
        /// </summary>
        public void SetSensitivity(float v)
        {
            _sens = Math.Clamp(v, 0.05f, 4.0f);
            double n = (_sens - 0.05) / (4.0 - 0.05); // 0..1
            n = Math.Clamp(n, 0.0, 1.0);

            // Output temporal smoothing (amplitude domain)
            double tAttack = Lerp(0.060, 0.010, n);  // 60 → 10 ms
            double tRelease = Lerp(0.200, 0.040, n); // 200 → 40 ms
            double dt = 0.016; // ~60 FPS
            _alphaAttack = (float)Math.Exp(-dt / tAttack);
            _alphaRelease = (float)Math.Exp(-dt / tRelease);

            // Normalization floor (dB)
            _dbFloor = -Lerp(55.0, 92.0, n); // -55 → -92 dB

            // Spatial smoothing tightness
            _spatialLoose = (float)(1.0 - n); // 1 at low sens, 0 at high sens

            // Reset fast/slow envelopes to avoid big jumps when user drags slider
            _fastDb = null; _slowDb = null; _prevBars = null;
        }

        public double SetVolumeDb(double db)
        {
            _volumeDb = Math.Clamp(db, -24.0, 24.0);
            return _volumeDb;
        }
        public double NudgeVolumeDb(double deltaDb) => SetVolumeDb(_volumeDb + deltaDb);
        public double GetVolumeDb() => _volumeDb;

        private void ComputeAndEmit()
        {
            float[] buf;
            lock (_lock)
            {
                buf = new float[_ring.Length];
                int tail = _ring.Length - _wpos;
                Array.Copy(_ring, _wpos, buf, 0, tail);
                Array.Copy(_ring, 0, buf, tail, _wpos);
            }

            int sr = _sr > 0 ? _sr : GuessSampleRate();

            // Choose a low-latency FFT length around ~50ms, bounded 2048..8192 and to buffer length
            int nfftPref = ChooseFftLen(sr);

            // 1) Compute raw dB per band from current buffer
            var dbs = BandsDbFromBuffer(buf, sr, _nbars, out double[] edges, out int nfftUsed, nfftPref);

            if (!_dumpedEdges)
            {
                _dumpedEdges = true;
                DumpEdges(sr, _ch, nfftUsed, edges);
            }

            // 2) Beat emphasis in dB domain: fast vs slow envelopes
            EnsureEnvArrays(_nbars, dbs);
            double n = (_sens - 0.05) / (4.0 - 0.05); n = Math.Clamp(n, 0.0, 1.0);
            double dt = 0.016; // ~60 FPS
            double tFast = Lerp(0.015, 0.008, n); // 15 → 8 ms
            double tSlow = Lerp(0.15, 0.08, n);  // 150 → 80 ms
            double aFast = Math.Exp(-dt / tFast);
            double aSlow = Math.Exp(-dt / tSlow);
            double boostPerDiff = Lerp(2.5, 6.5, n); // slightly stronger at high sens

            var adjDb = new double[_nbars];
            for (int i = 0; i < _nbars; i++)
            {
                _fastDb![i] = aFast * _fastDb[i] + (1 - aFast) * dbs[i];
                _slowDb![i] = aSlow * _slowDb[i] + (1 - aSlow) * dbs[i];
                double diff = Math.Max(0.0, _fastDb[i] - _slowDb[i]);
                double boost = Math.Min(boostPerDiff, diff * boostPerDiff); // clamp by slope
                adjDb[i] = dbs[i] + boost;
            }

            // 3) Normalize to 0..1 using dbFloor
            var norm = new float[_nbars];
            for (int i = 0; i < _nbars; i++)
            {
                double v = (adjDb[i] - _dbFloor) / (0.0 - _dbFloor);
                norm[i] = (float)Math.Clamp(v, 0.0, 1.0);
            }

            // 4) Spatial smoothing between neighbors (tightness depends on sensitivity)
            float nb = 0.15f + 0.25f * _spatialLoose; // 0.15 .. 0.40
            float cb = 1f - 2f * nb;                 // 0.70 .. 0.20
            var spatial = new float[_nbars];
            for (int i = 0; i < _nbars; i++)
            {
                float acc = norm[i] * cb; float ws = cb;
                if (i > 0) { acc += norm[i - 1] * nb; ws += nb; }
                if (i + 1 < _nbars) { acc += norm[i + 1] * nb; ws += nb; }
                spatial[i] = acc / Math.Max(1e-6f, ws);
            }

            // 5) Temporal attack/release smoothing per band (output domain)
            _prevBars ??= new float[spatial.Length];
            var outv = new float[spatial.Length];
            for (int i = 0; i < spatial.Length; i++)
            {
                float target = spatial[i];
                float prev = _prevBars[i];
                float alpha = target > prev ? _alphaAttack : _alphaRelease;
                float value = alpha * prev + (1 - alpha) * target;
                _prevBars[i] = value;
                outv[i] = value;
            }

            LevelsUpdated?.Invoke(new List<float>(outv));
        }

        /// <summary>
        /// Compute per-band dB values from a mono buffer using log-spaced edges with fractional-bin integration.
        /// </summary>
        private static double[] BandsDbFromBuffer(float[] x, int sr, int nbars, out double[] edges, out int nfftUsed, int? preferredNfft = null)
        {
            if (x.Length < 1024) { edges = Array.Empty<double>(); nfftUsed = 1024; return Enumerable.Repeat(-120.0, nbars).ToArray(); }

            int nfft = preferredNfft.HasValue ? preferredNfft.Value : ChooseFftLen(sr);
            nfft = Math.Min(nfft, x.Length & ~1); // even and within buffer
            nfft = Math.Clamp(nfft, 1024, 16384); // absolute safety bounds
            nfftUsed = nfft;

            var seg = x[^nfft..];

            // Hann window
            var w = new float[seg.Length]; for (int i = 0; i < w.Length; i++) w[i] = 0.5f * (1f - (float)Math.Cos(2 * Math.PI * i / (w.Length - 1)));
            var re = new Complex[seg.Length]; for (int i = 0; i < seg.Length; i++) re[i] = new Complex(seg[i] * w[i], 0);
            Fourier.Forward(re, FourierOptions.Matlab);

            int bins = re.Length / 2 + 1; // include Nyquist
            double df = sr / (double)re.Length;

            // Power spectrum
            var power = new double[bins];
            for (int i = 0; i < bins; i++) { double mag = re[i].Magnitude; power[i] = mag * mag; }

            // Band edges (cap at 0.98 × Nyquist and ≤ 20 kHz)
            double nyq = sr * 0.5;
            double fmax = Math.Min(20000.0, 0.98 * nyq);
            const double fmin = 20.0;
            edges = LogSpace(fmin, fmax, nbars + 1);

            var dbs = new double[nbars];
            for (int b = 0; b < nbars; b++)
            {
                double lo = edges[b], hi = edges[b + 1];
                double klo = lo / df, khi = hi / df;
                int kFirst = (int)Math.Floor(klo);
                int kLast = (int)Math.Floor(khi);

                double sum = 0.0, wsum = 0.0;
                for (int k = Math.Max(0, kFirst); k <= Math.Min(bins - 1, kLast); k++)
                {
                    double left = Math.Max(k, klo);
                    double right = Math.Min(k + 1, khi);
                    double weight = Math.Max(0.0, right - left); // fractional overlap of bin in this band
                    if (weight > 0) { sum += power[k] * weight; wsum += weight; }
                }
                if (wsum <= 1e-12)
                {
                    int k = (int)Math.Round((klo + khi) * 0.5);
                    k = Math.Clamp(k, 0, bins - 1);
                    sum = power[k]; wsum = 1.0;
                }

                double mean = sum / wsum;
                dbs[b] = 10.0 * Math.Log10(Math.Max(1e-20, mean));
            }

            return dbs;
        }

        private void EnsureEnvArrays(int n, double[] seed)
        {
            if (_fastDb == null || _fastDb.Length != n) _fastDb = (double[])seed.Clone();
            if (_slowDb == null || _slowDb.Length != n) _slowDb = (double[])seed.Clone();
        }

        // --- Device query, cache, and diagnostics ---

        private (int sr, int ch)? QueryDeviceFormat(string deviceId)
        {
            try
            {
                using var en = new MMDeviceEnumerator();
                using var dev = en.GetDevice(deviceId);
                var mix = dev.AudioClient.MixFormat;
                int sr = mix.SampleRate;
                int ch = mix.Channels;
                return (sr, ch);
            }
            catch (Exception ex)
            {
                Log("[QueryDeviceFormat] " + ex.Message);
                int sr = GuessSampleRate();
                int ch = GuessChannels();
                return (sr, ch);
            }
        }

        private int GuessSampleRate()
        {
            try
            {
                var p = _core.GetType().GetProperty("SampleRate", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (p != null && p.GetValue(_core) is int i && i > 0) return i;

                var wfProp = _core.GetType().GetProperty("WaveFormat", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                            ?? _core.GetType().GetProperty("Format", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (wfProp != null)
                {
                    var wf = wfProp.GetValue(_core);
                    var srProp = wf?.GetType().GetProperty("SampleRate");
                    if (srProp != null && srProp.GetValue(wf) is int j && j > 0) return j;
                }
            }
            catch { }
            return DefaultSampleRate;
        }

        private int GuessChannels()
        {
            try
            {
                var p = _core.GetType().GetProperty("Channels", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (p != null && p.GetValue(_core) is int i && i > 0) return i;

                var wfProp = _core.GetType().GetProperty("WaveFormat", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                            ?? _core.GetType().GetProperty("Format", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (wfProp != null)
                {
                    var wf = wfProp.GetValue(_core);
                    var chProp = wf?.GetType().GetProperty("Channels");
                    if (chProp != null && chProp.GetValue(wf) is int j && j > 0) return j;
                }
            }
            catch { }
            return DefaultChannels;
        }

        private void PersistDeviceSnapshot(DeviceInfo dev, int sr, int ch)
        {
            try
            {
                var snap = new DeviceSnapshot { Id = dev.Id, Name = dev.Name, SampleRate = sr, Channels = ch, TimestampUtc = DateTime.UtcNow };
                var dict = LoadCache();
                dict[dev.Id] = snap;
                var json = JsonSerializer.Serialize(dict, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(_cachePath, json);
            }
            catch (Exception ex) { Log("[PersistDeviceSnapshot] " + ex.Message); }
        }

        private Dictionary<string, DeviceSnapshot> LoadCache()
        {
            try
            {
                if (File.Exists(_cachePath))
                {
                    var json = File.ReadAllText(_cachePath);
                    var d = JsonSerializer.Deserialize<Dictionary<string, DeviceSnapshot>>(json);
                    if (d != null) return d;
                }
            }
            catch (Exception ex) { Log("[LoadCache] " + ex.Message); }
            return new Dictionary<string, DeviceSnapshot>();
        }

        private void DumpEdges(int sr, int ch, int nfft, double[] edges)
        {
            try
            {
                using var sw = new StreamWriter(_logPath, append: true);
                sw.WriteLine($"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] SR={sr}Hz CH={ch} NFFT={nfft}");
                sw.WriteLine("BandEdges(Hz): " + string.Join(", ", edges.Select(e => e.ToString("0.##"))));
            }
            catch { }
        }

        private void Log(string msg)
        {
            try { using var sw = new StreamWriter(_logPath, append: true); sw.WriteLine($"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {msg}"); }
            catch { }
        }

        private static double[] LogSpace(double a, double b, int n)
        { var arr = new double[n]; double la = Math.Log10(a), lb = Math.Log10(b); for (int i = 0; i < n; i++) arr[i] = Math.Pow(10, la + (lb - la) * i / Math.Max(1, n - 1)); return arr; }

        private static double Lerp(double a, double b, double t) => a + (b - a) * t;

        /// <summary>
        /// Choose a low-latency NFFT close to ~50 ms of audio at the given sample rate; bounded 2048..8192.
        /// </summary>
        private static int ChooseFftLen(int sr)
        {
            // target ≈ 0.05 s of audio
            double target = Math.Max(0.03, Math.Min(0.06, 0.05));
            double targetSamples = sr * target;

            // nearest power of two
            int pow2 = 1; while (pow2 < targetSamples) pow2 <<= 1; int below = pow2 >> 1;
            int chosen = Math.Abs(pow2 - targetSamples) < Math.Abs(targetSamples - below) ? pow2 : below;
            // safety and floor for low-end resolution
            chosen = Math.Clamp(chosen, 2048, 8192);
            return chosen;
        }

        public void Dispose() { _core.Dispose(); }
    }

    public record DeviceSnapshot
    {
        public string Id { get; init; } = string.Empty;
        public string Name { get; init; } = string.Empty;
        public int SampleRate { get; init; }
        public int Channels { get; init; }
        public DateTime TimestampUtc { get; init; }
    }
}