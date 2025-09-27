using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace test
{
    public partial class SandboxControl : UserControl
    {
        // Expose workspace size to XAML (MainWindow sets 1600x900)
        public double WorkspaceWidth
        {
            get => Space?.Width ?? 0;
            set { if (Space != null) Space.Width = value; }
        }
        public double WorkspaceHeight
        {
            get => Space?.Height ?? 0;
            set { if (Space != null) Space.Height = value; }
        }

        private readonly List<(SandboxLayer layer, Image img,
                               ScaleTransform growY,   // dynamic height growth (audio)
                               ScaleTransform baseScale,
                               TranslateTransform translate)> _visuals = new();

        private SandboxLayer? _selected;

        // Drag state (selected layer)
        private bool _dragging;
        private Point _dragStart;
        private Point _origPos;

        // Pan state (middle mouse)
        private bool _panning;
        private Point _panStart;
        private Point _scrollAtPanStart;

        // Zoom
        private double _zoom = 1.0; // 0.5..3.0

        public SandboxControl()
        {
            InitializeComponent();

            // Layer editing gestures (selection is driven by MainWindow via Select(Guid))
            Space.MouseLeftButtonDown += OnMouseLeftDown;
            Space.MouseLeftButtonUp += OnMouseLeftUp;
            Space.MouseMove += OnMouseMove;
            Space.MouseWheel += OnMouseWheel;

            // Middle-button pan start/stop
            Space.MouseDown += (s, e) =>
            {
                if (e.MiddleButton == MouseButtonState.Pressed)
                {
                    _panning = true;
                    _panStart = e.GetPosition(this);
                    _scrollAtPanStart = new Point(Scroller.HorizontalOffset, Scroller.VerticalOffset);
                    Space.Cursor = Cursors.ScrollAll;
                    Space.CaptureMouse();
                }
            };
            Space.MouseUp += (s, e) =>
            {
                if (_panning && e.ChangedButton == MouseButton.Middle)
                {
                    _panning = false;
                    Space.ReleaseMouseCapture();
                    Space.Cursor = Cursors.Arrow;
                }
            };
        }

        public IReadOnlyList<SandboxLayer> Layers => _visuals.Select(v => v.layer).ToList();

        // ---- Public API used by MainWindow ----

        public void AddLayer(SandboxLayer layer)
        {
            if (layer.Image == null) return;

            var img = new Image
            {
                Source = layer.Image,
                RenderTransformOrigin = new Point(0.5, 1.0), // bottom-center anchor for grow-up
                SnapsToDevicePixels = true,
                IsHitTestVisible = false // selection comes from the list
            };

            var baseScale = new ScaleTransform(layer.BaseScaleX * (layer.FlipX ? -1 : 1),
                                               layer.BaseScaleY * (layer.FlipY ? -1 : 1));
            var grow = new ScaleTransform(1, 1);
            var translate = new TranslateTransform(layer.Position.X, layer.Position.Y);

            var tg = new TransformGroup();
            tg.Children.Add(baseScale);
            tg.Children.Add(grow);
            tg.Children.Add(translate);
            img.RenderTransform = tg;

            _visuals.Add((layer, img, grow, baseScale, translate));
            Space.Children.Add(img);
        }

        public void RemoveSelected()
        {
            if (_selected == null) return;
            var v = _visuals.FirstOrDefault(x => x.layer.Id == _selected.Id);
            if (v.layer != null)
            {
                Space.Children.Remove(v.img);
                _visuals.Remove(v);
            }
            _selected = null;
        }

        public void Select(Guid id)
        {
            _selected = _visuals.Select(v => v.layer).FirstOrDefault(l => l.Id == id);
        }

        public void MoveSelectedToIndex(int newIndex)
        {
            if (_visuals.Count == 0 || _selected == null) return;
            newIndex = Math.Clamp(newIndex, 0, _visuals.Count - 1);

            var current = _visuals.FirstOrDefault(x => x.layer.Id == _selected.Id);
            if (current.layer == null) return;

            Space.Children.Remove(current.img);
            _visuals.Remove(current);
            _visuals.Insert(newIndex, current);
            Space.Children.Insert(newIndex, current.img);
        }

        public void FlipSelected(bool horizontal)
        {
            if (_selected == null) return;
            var entry = _visuals.First(x => x.layer.Id == _selected.Id);

            if (horizontal) _selected.FlipX = !_selected.FlipX;
            else _selected.FlipY = !_selected.FlipY;

            entry.baseScale.ScaleX = Math.Abs(entry.baseScale.ScaleX) * (_selected.FlipX ? -1 : 1);
            entry.baseScale.ScaleY = Math.Abs(entry.baseScale.ScaleY) * (_selected.FlipY ? -1 : 1);
        }

        // Called by MainWindow every spectrum tick
        public void UpdateBandEnergy(float[] bars, Func<double, int> freqToIndex)
        {
            foreach (var (layer, _, grow, _, _) in _visuals)
            {
                int i0 = freqToIndex(layer.FreqMinHz);
                int i1 = freqToIndex(layer.FreqMaxHz);
                if (i1 < i0) (i0, i1) = (i1, i0);
                i0 = Math.Clamp(i0, 0, bars.Length - 1);
                i1 = Math.Clamp(i1, 0, bars.Length - 1);

                float acc = 0; int count = 0;
                for (int i = i0; i <= i1; i++) { acc += bars[i]; count++; }
                float avg = count > 0 ? acc / count : 0f;

                double mapped = Math.Clamp(layer.Gain * layer.Sensitivity * avg, 0.0, 1.0);
                double scaleY = 1.0 + 0.4 * mapped;  // +40% at peak
                grow.ScaleY = scaleY;                // grows upward thanks to origin = (0.5, 1.0)
            }
        }

        // ---- Input handling ----

        private void OnMouseLeftDown(object? sender, MouseButtonEventArgs e)
        {
            if (_selected == null) return;
            _dragging = true;
            _dragStart = e.GetPosition(Space);   // coordinates in workspace space
            _origPos = _selected.Position;
            Space.CaptureMouse();
            Cursor = Cursors.SizeAll;
        }

        private void OnMouseLeftUp(object? sender, MouseButtonEventArgs e)
        {
            if (!_dragging) return;
            _dragging = false;
            Space.ReleaseMouseCapture();
            Cursor = Cursors.Arrow;
        }

        private void OnMouseMove(object? sender, MouseEventArgs e)
        {
            if (_panning)
            {
                var p = e.GetPosition(this);
                Vector d = p - _panStart;
                Scroller.ScrollToHorizontalOffset(_scrollAtPanStart.X - d.X);
                Scroller.ScrollToVerticalOffset(_scrollAtPanStart.Y - d.Y);
                return;
            }

            if (_dragging && _selected != null)
            {
                var p = e.GetPosition(Space);
                Vector d = p - _dragStart;
                var pos = new Point(_origPos.X + d.X, _origPos.Y + d.Y);
                if ((Keyboard.Modifiers & ModifierKeys.Shift) == ModifierKeys.Shift)
                {
                    pos = new Point(Math.Round(pos.X / 10.0) * 10.0,
                                    Math.Round(pos.Y / 10.0) * 10.0);
                }
                _selected.Position = pos;

                var entry = _visuals.First(x => x.layer.Id == _selected.Id);
                entry.translate.X = pos.X;
                entry.translate.Y = pos.Y;
            }
        }

        private void OnMouseWheel(object? sender, MouseWheelEventArgs e)
        {
            if ((Keyboard.Modifiers & ModifierKeys.Control) == ModifierKeys.Control)
            {
                // Zoom workspace
                double step = e.Delta > 0 ? 0.1 : -0.1;
                _zoom = Math.Clamp(_zoom + step, 0.5, 3.0);
                Zoom.ScaleX = _zoom; Zoom.ScaleY = _zoom;
                e.Handled = true;
                return;
            }

            if (_selected != null)
            {
                // Scale selected layer ±10%
                double factor = e.Delta > 0 ? 1.1 : 0.9;
                _selected.BaseScaleX *= factor;
                _selected.BaseScaleY *= factor;

                var entry = _visuals.First(x => x.layer.Id == _selected.Id);
                entry.baseScale.ScaleX *= factor;
                entry.baseScale.ScaleY *= factor;
                e.Handled = true;
            }
        }
    }
}
