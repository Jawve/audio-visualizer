using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media;

namespace test
{
    public partial class RangeSlider : UserControl
    {
        public static readonly DependencyProperty MinimumProperty = DependencyProperty.Register(
            nameof(Minimum), typeof(double), typeof(RangeSlider), new PropertyMetadata(20.0, OnLayoutChanged));
        public static readonly DependencyProperty MaximumProperty = DependencyProperty.Register(
            nameof(Maximum), typeof(double), typeof(RangeSlider), new PropertyMetadata(20000.0, OnLayoutChanged));
        public static readonly DependencyProperty LowerValueProperty = DependencyProperty.Register(
            nameof(LowerValue), typeof(double), typeof(RangeSlider), new PropertyMetadata(60.0, OnLayoutChanged));
        public static readonly DependencyProperty UpperValueProperty = DependencyProperty.Register(
            nameof(UpperValue), typeof(double), typeof(RangeSlider), new PropertyMetadata(1200.0, OnLayoutChanged));

        public double Minimum { get => (double)GetValue(MinimumProperty); set => SetValue(MinimumProperty, value); }
        public double Maximum { get => (double)GetValue(MaximumProperty); set => SetValue(MaximumProperty, value); }
        public double LowerValue { get => (double)GetValue(LowerValueProperty); set => SetValue(LowerValueProperty, value); }
        public double UpperValue { get => (double)GetValue(UpperValueProperty); set => SetValue(UpperValueProperty, value); }

        private readonly double[] _snapHz = new[] { 20.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0 };

        public event Action<double, double>? RangeChanged;

        public RangeSlider()
        {
            InitializeComponent();
            Loaded += (_, __) => Redraw();
            SizeChanged += (_, __) => Redraw();

            ThumbMin.DragDelta += (s, e) => MoveThumb(ref _lowerX, e.HorizontalChange);
            ThumbMax.DragDelta += (s, e) => MoveThumb(ref _upperX, e.HorizontalChange);
            ThumbMin.DragCompleted += (s, e) => Snap();
            ThumbMax.DragCompleted += (s, e) => Snap();
        }

        private double _lowerX, _upperX;

        private static void OnLayoutChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        { (d as RangeSlider)?.Redraw(); }

        private void Redraw()
        {
            double w = Math.Max(1, Track.ActualWidth);
            double min = Minimum, max = Maximum;
            Func<double, double> norm = f => Math.Clamp((Math.Log10(f) - Math.Log10(min)) / (Math.Log10(max) - Math.Log10(min)), 0, 1);

            _lowerX = norm(LowerValue) * w; _upperX = norm(UpperValue) * w;
            if (_upperX < _lowerX) (_lowerX, _upperX) = (_upperX, _lowerX);

            Canvas.SetLeft(Sel, _lowerX); Sel.Width = Math.Max(2, _upperX - _lowerX);
            Canvas.SetLeft(ThumbMin, Math.Max(0, _lowerX - ThumbMin.Width / 2)); Canvas.SetTop(ThumbMin, -5);
            Canvas.SetLeft(ThumbMax, Math.Max(0, _upperX - ThumbMax.Width / 2)); Canvas.SetTop(ThumbMax, -5);
        }

        private void MoveThumb(ref double field, double dx)
        {
            field = Math.Clamp(field + dx, 0, Math.Max(0, Track.ActualWidth));
            if (_upperX < _lowerX) (_lowerX, _upperX) = (_upperX, _lowerX);
            UpdateValuesFromPixels();
            Redraw();
        }

        private void UpdateValuesFromPixels()
        {
            double w = Math.Max(1, Track.ActualWidth);
            Func<double, double> denorm = t => Math.Pow(10, Math.Log10(Minimum) + t * (Math.Log10(Maximum) - Math.Log10(Minimum)));
            double lo = denorm(_lowerX / w), hi = denorm(_upperX / w);
            LowerValue = Math.Round(lo);
            UpperValue = Math.Round(hi);
            RangeChanged?.Invoke(LowerValue, UpperValue);
        }

        private void Snap()
        {
            double bestLo = LowerValue, bestHi = UpperValue;
            foreach (var s in _snapHz)
            {
                if (Math.Abs(s - LowerValue) / s < 0.08) bestLo = s;
                if (Math.Abs(s - UpperValue) / s < 0.08) bestHi = s;
            }
            LowerValue = bestLo; UpperValue = bestHi; Redraw(); RangeChanged?.Invoke(LowerValue, UpperValue);
        }
    }
}