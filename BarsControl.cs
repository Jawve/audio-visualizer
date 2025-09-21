using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;

namespace test
{
    public class BarsControl : FrameworkElement
    {
        public enum RenderMode { Live, Idle }

        private readonly List<float> _levels;
        private readonly List<float> _smoothed;
        private int _barCount;
        private float _smooth;
        private RenderMode _mode = RenderMode.Idle;

        private readonly DispatcherTimer _anim;
        private float _phase;

        // Appearance
        public Color BarColor { get; set; } = Color.FromArgb(215, 128, 58, 180); // solid purple-ish
        /// <summary>Lower = thicker bars (0..1). Ex: 0.20 = 80% of slot width.</summary>
        public double BarSpacing { get; set; } = 0.20; // thicker default
        /// <summary>Scales maximum bar height (0..1). 0.95 ≈ top of content area.</summary>
        public double HeightScale { get; set; } = 0.95;

        // XAML parameterless ctor
        public BarsControl() : this(48, 0.65f) { }

        public BarsControl(int barCount, float smoothing)
        {
            SnapsToDevicePixels = true;
            _barCount = Math.Max(8, barCount);
            _smooth = Math.Clamp(smoothing, 0f, 0.99f);
            _levels = new List<float>(new float[_barCount]);
            _smoothed = new List<float>(new float[_barCount]);

            _anim = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(16) };
            _anim.Tick += (s, e) => { _phase += 0.06f; if (_phase > 1000f) _phase -= 1000f; InvalidateVisual(); };
            _anim.Start();

            Focusable = true; // to receive mouse wheel if desired
        }

        public void SetBarCount(int n)
        {
            _barCount = Math.Max(8, n);
            _levels.Clear(); for (int i = 0; i < _barCount; i++) _levels.Add(0);
            _smoothed.Clear(); for (int i = 0; i < _barCount; i++) _smoothed.Add(0);
            InvalidateVisual();
        }
        public void SetSmoothing(float s) { _smooth = Math.Clamp(s, 0f, 0.99f); }
        public void SetIdle(bool idle) { _mode = idle ? RenderMode.Idle : RenderMode.Live; }

        public void UpdateLevels(IReadOnlyList<float> values)
        {
            if (_mode == RenderMode.Idle) return;
            int n = Math.Min(values.Count, _barCount);
            for (int i = 0; i < n; i++) _levels[i] = Math.Clamp(values[i], 0f, 1f);
            InvalidateVisual();
        }

        protected override void OnRender(DrawingContext dc)
        {
            base.OnRender(dc);
            double w = ActualWidth, h = ActualHeight; if (w <= 0 || h <= 0) return;

            double xpad = Math.Max(6, 0.01 * w);
            double ypad = Math.Max(6, 0.05 * h);
            double contentH = Math.Max(6, h - 2 * ypad);
            double usableH = contentH * Math.Clamp(HeightScale, 0.05, 1.0); // height limiter
            double baseY = ypad + contentH; // anchor to bottom of content area

            double slot = (w - 2 * xpad) / Math.Max(1, _barCount);
            double bw = slot * (1.0 - Math.Clamp(BarSpacing, 0.0, 0.95));
            if (bw < 1) bw = 1;

            var brush = new SolidColorBrush(BarColor);

            if (_mode == RenderMode.Idle)
            {
                for (int i = 0; i < _barCount; i++)
                {
                    double x0 = xpad + i * slot + (slot - bw) / 2.0;
                    double full = usableH * 0.6; // idle amplitude ceiling
                    double k = 2.2 * Math.PI / _barCount;
                    double v = 0.25 + 0.75 * (0.5 + 0.5 * Math.Sin(_phase + i * k));
                    double hbar = Math.Max(1, v * full);
                    var rect = new Rect(x0, baseY - hbar, bw, hbar);
                    dc.DrawRectangle(brush, null, rect);
                }
                return;
            }

            for (int i = 0; i < _barCount; i++)
                _smoothed[i] = _smooth * _smoothed[i] + (1 - _smooth) * _levels[i];

            for (int i = 0; i < _barCount; i++)
            {
                double v = _smoothed[i];
                double x0 = xpad + i * slot + (slot - bw) / 2.0;
                double hbar = Math.Max(1, v * usableH);
                var rect = new Rect(x0, baseY - hbar, bw, hbar);
                dc.DrawRectangle(brush, null, rect);
            }
        }
    }
}