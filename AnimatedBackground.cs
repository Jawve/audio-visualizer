using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;

namespace test
{
    public class AnimatedBackground : FrameworkElement
    {
        private readonly DispatcherTimer _timer;
        private readonly List<Blob> _blobs = new();
        private readonly List<Dot> _dots = new();
        private readonly Random _rng = new();
        private DateTime _last = DateTime.UtcNow;

        public AnimatedBackground()
        {
            IsHitTestVisible = false;
            _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(16) };
            _timer.Tick += (s, e) => InvalidateVisual();
            _timer.Start();
        }

        protected override void OnRender(DrawingContext dc)
        {
            base.OnRender(dc);
            var w = ActualWidth; var h = ActualHeight;
            if (w <= 0 || h <= 0) return;

            var now = DateTime.UtcNow;
            var dt = (now - _last).TotalSeconds; if (dt < 0 || dt > 1) dt = 0.016; _last = now;

            // Gradient background
            var bg = new LinearGradientBrush(Color.FromRgb(10, 8, 14), Color.FromRgb(18, 14, 28), new Point(0, 0), new Point(0, 1));
            dc.DrawRectangle(bg, null, new Rect(0, 0, w, h));

            EnsureContent(w, h);

            // Blobs (soft radial gradients)
            foreach (var b in _blobs)
            {
                var center = new Point(b.X, b.Y);
                var radius = b.R;
                // Fix: FromArgb expects (A,R,G,B), not (A, Color)
                var cA = Color.FromArgb(b.A, b.Color.R, b.Color.G, b.Color.B);
                var c0 = Color.FromArgb(0, b.Color.R, b.Color.G, b.Color.B);
                var brush = new RadialGradientBrush(cA, c0)
                {
                    GradientOrigin = new Point(0.5, 0.5),
                    Center = new Point(0.5, 0.5),
                    RadiusX = 1,
                    RadiusY = 1
                };
                dc.PushOpacity(0.9);
                dc.DrawEllipse(brush, null, center, radius, radius);
                dc.Pop();
                b.X += b.Vx * dt * 60; b.Y += b.Vy * dt * 60;
                Wrap(b, w, h);
            }

            // Dots along swoops
            var dotBrush = new SolidColorBrush(Color.FromArgb(30, 255, 255, 255));
            foreach (var d in _dots.ToArray())
            {
                d.U += d.Speed * dt; if (d.U > 1) { ResetDot(d, w, h); continue; }
                var p = Cubic(d.P0, d.P1, d.P2, d.P3, Ease(d.U));
                dc.DrawEllipse(dotBrush, null, p, d.Size, d.Size);
            }
        }

        private void EnsureContent(double w, double h)
        {
            while (_blobs.Count < 7) _blobs.Add(RandomBlob(w, h));
            while (_dots.Count < 90) _dots.Add(RandomDot(w, h));
        }

        private Blob RandomBlob(double w, double h)
        {
            bool purple = _rng.NextDouble() < 0.7;
            var col = purple ? Color.FromRgb(128, 58, 180) : Color.FromRgb(255, 140, 0);
            byte a = (byte)(purple ? 90 + _rng.Next(60) : 60 + _rng.Next(40));
            double r = (_rng.NextDouble() * 0.18 + 0.10) * Math.Max(w, h);
            return new Blob { X = _rng.NextDouble() * w, Y = _rng.NextDouble() * h, R = r, Color = col, A = a, Vx = _rng.NextDouble() * 0.4 - 0.2, Vy = _rng.NextDouble() * 0.4 - 0.2 };
        }

        private Dot RandomDot(double w, double h)
        {
            var p0 = new Point(_rng.NextDouble() * -0.1 * w, _rng.NextDouble() * 0.6 * h);
            var p3 = new Point(w * (0.8 + _rng.NextDouble() * 0.4), _rng.NextDouble() * 0.9 * h);
            var p1 = new Point(_rng.NextDouble() * 0.6 * w, _rng.NextDouble() * -0.2 * h);
            var p2 = new Point(w * (0.4 + _rng.NextDouble() * 0.6), h * (0.6 + _rng.NextDouble() * 0.6));
            double speed = 0.05 + _rng.NextDouble() * 0.35;
            double size = 1.0 + _rng.NextDouble() * 2.5;
            return new Dot { P0 = p0, P1 = p1, P2 = p2, P3 = p3, U = _rng.NextDouble(), Speed = speed, Size = size };
        }

        private void ResetDot(Dot d, double w, double h)
        {
            var nd = RandomDot(w, h);
            d.P0 = nd.P0; d.P1 = nd.P1; d.P2 = nd.P2; d.P3 = nd.P3; d.U = 0; d.Speed = nd.Speed; d.Size = nd.Size;
        }

        private static void Wrap(Blob b, double w, double h)
        {
            if (b.X < -b.R) b.X = w + b.R; if (b.X > w + b.R) b.X = -b.R;
            if (b.Y < -b.R) b.Y = h + b.R; if (b.Y > h + b.R) b.Y = -b.R;
        }

        private static Point Cubic(Point p0, Point p1, Point p2, Point p3, double u)
        {
            double v = 1 - u;
            double x = v * v * v * p0.X + 3 * v * v * u * p1.X + 3 * v * u * u * p2.X + u * u * u * p3.X;
            double y = v * v * v * p0.Y + 3 * v * v * u * p1.Y + 3 * v * u * u * p2.Y + u * u * u * p3.Y;
            return new Point(x, y);
        }

        private static double Ease(double u) => 0.5 * (1 - Math.Cos(Math.PI * Math.Clamp(u, 0, 1)));

        private class Blob { public double X, Y, R, Vx, Vy; public Color Color; public byte A; }
        private class Dot { public Point P0, P1, P2, P3; public double U, Speed, Size; }
    }
}