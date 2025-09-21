using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media; // for VisualTreeHelper

namespace test
{
    public partial class MainWindow : Window
    {
        private readonly AudioHub _hub;

        // We don't assume your XAML has exact names — we try to find them.
        private BarsControl? _bars;
        private ComboBox? _deviceCombo;
        private Slider? _sensSlider;
        private Slider? _volSlider; // optional, if you add <Slider x:Name="VolSlider" .../>

        private bool _popOut; // for XAML Click="PopOut" handlers

        public MainWindow()
        {
            InitializeComponent();

            // First pass: try by Name from XAML
            _bars = FindName("Bars") as BarsControl;
            _deviceCombo = FindName("DeviceCombo") as ComboBox;
            _sensSlider = FindName("SensSlider") as Slider;
            _volSlider = FindName("VolSlider") as Slider; // only wires if present

            // Second pass (after loaded), resolve by walking the visual tree
            Loaded += (_, __) =>
            {
                _bars ??= FindChild<BarsControl>(this, "Bars") ?? FindChild<BarsControl>(this, null);
                _deviceCombo ??= FindChild<ComboBox>(this, "DeviceCombo") ?? FindChild<ComboBox>(this, null);
                _sensSlider ??= FindChild<Slider>(this, "SensSlider");
                _volSlider ??= FindChild<Slider>(this, "VolSlider");

                WireUi();

                // Force a refresh *after* events are subscribed
                _hub.RequestRefresh();
            };

            _hub = new AudioHub(nbars: 48);
            _hub.DevicesChanged += OnDevicesChanged;
            _hub.LevelsUpdated += levels => Dispatcher.Invoke(() => _bars?.UpdateLevels(levels));

            // Also issue a refresh once here (in case devices arrive very fast)
            _hub.RequestRefresh();
        }

        private void WireUi()
        {
            if (_bars != null)
            {
                _bars.SetIdle(true);            // idle until a device is chosen
                _bars.BarSpacing = 0.22;        // thicker bars
                _bars.HeightScale = 0.95;       // reach near the top
                _bars.MouseWheel += Bars_MouseWheel_AdjustVolume; // wheel to nudge volume dB
                _bars.Focusable = true;
            }

            if (_sensSlider != null)
            {
                _sensSlider.Minimum = 0.05; _sensSlider.Maximum = 4.0; _sensSlider.Value = 1.0;
                _sensSlider.ValueChanged += (s, e) =>
                {
                    float v = (float)_sensSlider.Value;
                    _hub.SetSensitivity(v);
                    if (_bars != null)
                        _bars.SetSmoothing(MapSensToVisualSmooth(v)); // presentation smoothing tied to sensitivity
                };
            }

            if (_volSlider != null)
            {
                _volSlider.Minimum = -24; _volSlider.Maximum = 24; _volSlider.Value = 0; _volSlider.TickFrequency = 1; _volSlider.IsSnapToTickEnabled = true;
                _volSlider.ValueChanged += (s, e) =>
                {
                    double db = _hub.SetVolumeDb(_volSlider.Value);
                    UpdateTitleVolume(db);
                };
            }

            if (_deviceCombo != null)
            {
                _deviceCombo.SelectionChanged += (s, e) =>
                {
                    var item = _deviceCombo.SelectedItem as DeviceItem;
                    if (item == null || item.IsPlaceholder)
                    {
                        _hub.SetDeviceById(null);
                        if (_bars != null) _bars.SetIdle(true);
                    }
                    else
                    {
                        _hub.SetDeviceById(item.Id);
                        if (_bars != null) _bars.SetIdle(false);
                    }
                };
            }
        }

        // XAML can bind Click="PopOut" on a Button/MenuItem; this satisfies that handler.
        public void PopOut(object sender, RoutedEventArgs e)
        {
            _popOut = !_popOut;
            Topmost = _popOut;
        }

        private void Bars_MouseWheel_AdjustVolume(object sender, MouseWheelEventArgs e)
        {
            double step = e.Delta > 0 ? 1.0 : -1.0; // ±1 dB per notch
            double db = _hub.NudgeVolumeDb(step);
            _volSlider?.Dispatcher.Invoke(() => _volSlider.Value = db);
            UpdateTitleVolume(db);
        }

        private void UpdateTitleVolume(double db)
        {
            this.Title = $"Visualizer  ({db:+0;-0;0} dB)";
        }

        private void OnDevicesChanged(List<DeviceInfo> list)
        {
            if (_deviceCombo == null) return;
            Dispatcher.Invoke(() =>
            {
                // Preserve current selection if possible
                string? selectedId = (_deviceCombo.SelectedItem as DeviceItem)?.Id;

                var items = new List<DeviceItem>();
                items.Add(DeviceItem.Placeholder());

                foreach (var d in list)
                    items.Add(new DeviceItem { Id = d.Id, Name = d.Name });

                // Handle no devices case with a special placeholder
                if (list.Count == 0)
                    items.Add(new DeviceItem { Id = string.Empty, Name = "(No audio devices found)", IsPlaceholder = true });

                _deviceCombo.ItemsSource = items;
                _deviceCombo.DisplayMemberPath = nameof(DeviceItem.Name);

                // Try to reselect previously chosen device; otherwise select placeholder
                int idx = Math.Max(0, items.FindIndex(it => !it.IsPlaceholder && it.Id == selectedId));
                _deviceCombo.SelectedIndex = idx >= 0 ? idx : 0;
            });
        }

        private static float MapSensToVisualSmooth(double v)
        {
            // Normalize slider 0.05..4 → 0..1
            double n = (v - 0.05) / (4.0 - 0.05);
            n = Math.Clamp(n, 0.0, 1.0);
            // High sensitivity → smaller visual smoothing; Low sensitivity → bigger smoothing
            double s = Lerp(0.10, 0.85, 1.0 - n);
            return (float)s;
        }

        private static double Lerp(double a, double b, double t) => a + (b - a) * t;

        protected override void OnClosed(EventArgs e)
        {
            base.OnClosed(e);
            _hub.Dispose();
        }

        // Visual tree search helper
        private static T? FindChild<T>(DependencyObject parent, string? name) where T : FrameworkElement
        {
            if (parent == null) return null;
            int count = VisualTreeHelper.GetChildrenCount(parent);
            for (int i = 0; i < count; i++)
            {
                var child = VisualTreeHelper.GetChild(parent, i);
                if (child is T fe)
                {
                    if (string.IsNullOrEmpty(name) || fe.Name == name) return fe;
                }
                var result = FindChild<T>(child, name);
                if (result != null) return result;
            }
            return null;
        }

        private class DeviceItem
        {
            public string Id { get; set; } = string.Empty;
            public string Name { get; set; } = string.Empty;
            public bool IsPlaceholder { get; set; }

            public static DeviceItem Placeholder() => new DeviceItem { Id = string.Empty, Name = "— Select Audio Device —", IsPlaceholder = true };
        }
    }
}
