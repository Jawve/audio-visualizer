using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace test
{
    public class SandboxLayer
    {
        public Guid Id { get; } = Guid.NewGuid();
        public string? SourcePath { get; private set; }
        public ImageSource? Image { get; private set; }

        // Placement
        public Point Position { get; set; } = new Point(80, 80);
        public double BaseScaleX { get; set; } = 1.0;
        public double BaseScaleY { get; set; } = 1.0;
        public bool FlipX { get; set; }
        public bool FlipY { get; set; }
        public double RotationDeg { get; set; } = 0.0; // reserved, not exposed yet

        // Audio mapping (per-layer)
        public double FreqMinHz { get; set; } = 60;
        public double FreqMaxHz { get; set; } = 1200;
        public double Sensitivity { get; set; } = 1.0; // multiplier on energy
        public double Gain { get; set; } = 1.0;        // extra scale factor

        public bool VisualizeGrowUp { get; set; } = true; // height-only, grows upward

        public void LoadFromPath(string path)
        {
            SourcePath = path;
            var bi = new BitmapImage();
            bi.BeginInit();
            bi.CacheOption = BitmapCacheOption.OnLoad; // file can be released
            bi.UriSource = new Uri(path, UriKind.Absolute);
            bi.EndInit();
            bi.Freeze();
            Image = bi;
        }

        public Transform BuildTransform(double extraScaleY)
        {
            // Anchor at bottom center for grow-up effect
            var tg = new TransformGroup();
            double sx = BaseScaleX * (FlipX ? -1.0 : 1.0);
            double sy = BaseScaleY * (FlipY ? -1.0 : 1.0);
            if (VisualizeGrowUp) sy *= extraScaleY;
            tg.Children.Add(new TranslateTransform(-0.5, -1.0)); // move origin from center/1.0? We'll use LayoutTransformOrigin
            tg.Children.Add(new ScaleTransform(sx, sy));
            tg.Children.Add(new TranslateTransform(Position.X, Position.Y));
            return tg;
        }
    }
}