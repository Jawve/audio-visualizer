using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace test
{
    public partial class MainWindow : Window
    {
        private readonly AudioHub _hub = new AudioHub(nbars: 48);

        private BarsControl? _bars;
        private ComboBox? _deviceCombo;
        private Slider? _sensSlider;
        private Slider? _volSlider;

        private readonly List<SandboxLayer> _layers = new();
        private SandboxLayer? _selected;

        public MainWindow()
        {
            InitializeComponent();
            this.Title = "Window Visualizer";

            _bars = FindName("Bars") as BarsControl;
            _deviceCombo = FindName("DeviceCombo") as ComboBox;
            _sensSlider = FindName("SensSlider") as Slider;
            _volSlider = FindName("VolSlider") as Slider;

            _hub.DevicesChanged += OnDevicesChanged;
            _hub.LevelsUpdated += levels =>
            {
                Dispatcher.Invoke(() =>
                {
                    _bars?.UpdateLevels(levels);
                    Sandbox_UpdateFromSpectrum(levels);
                });
            };

            Loaded += (_, __) =>
            {
                WireUi();
                HookLayerUi();
                _hub.RequestRefresh();
                LblMin.Text = "60"; LblMax.Text = "1200";
            };
        }

        private void WireUi()
        {
            if (_bars != null)
            {
                _bars.SetIdle(true);
                _bars.BarSpacing = 0.22;
                _bars.HeightScale = 0.95;
                _bars.MouseWheel += Bars_MouseWheel_AdjustVolume;
                _bars.Focusable = true;
            }

            if (_sensSlider != null)
            {
                _sensSlider.Minimum = 0.05; _sensSlider.Maximum = 4.0; _sensSlider.Value = 1.0;
                _sensSlider.ValueChanged += (s, e) =>
                {
                    float v = (float)_sensSlider.Value;
                    _hub.SetSensitivity(v);
                    _bars?.SetSmoothing(MapSensToVisualSmooth(v));
                };
            }

            if (_volSlider != null)
            {
                _volSlider.Minimum = -24; _volSlider.Maximum = 24;
                _volSlider.TickFrequency = 0.5; _volSlider.IsSnapToTickEnabled = true;
                _volSlider.LargeChange = 2; _volSlider.SmallChange = 0.5;
                _volSlider.Value = _hub.GetVolumeDb();
                _volSlider.ValueChanged += (s, e) => { if (IsLoaded) _hub.SetVolumeDb(_volSlider.Value); };
            }

            if (_deviceCombo != null)
            {
                _deviceCombo.SelectionChanged += (s, e) =>
                {
                    var item = _deviceCombo.SelectedItem as DeviceItem;
                    if (item == null || item.IsPlaceholder)
                    { _hub.SetDeviceById(null); _bars?.SetIdle(true); }
                    else { _hub.SetDeviceById(item.Id); _bars?.SetIdle(false); }
                };
            }
        }

        // EXPLICIT handler wired in XAML (BtnHelp_Click)
        private void BtnHelp_Click(object sender, RoutedEventArgs e)
        {
            MessageBox.Show(
                "Sandbox Controls:\n\n" +
                "• Select a layer in the list to edit it.\n" +
                "• Left-drag inside sandbox = move the selected layer.\n" +
                "• Mouse Wheel = scale selected layer (±10%).\n" +
                "• Shift + Drag = snap to 10px grid.\n" +
                "• Ctrl + Wheel = zoom workspace.\n" +
                "• Middle-click + Drag = pan workspace.\n",
                "Controls", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        private void HookLayerUi()
        {
            BtnAddLayer.Click += (_, __) => AddLayerFromDialog();
            BtnDeleteLayer.Click += (_, __) => DeleteSelectedLayer();
            BtnUp.Click += (_, __) => MoveSelected(-1);
            BtnDown.Click += (_, __) => MoveSelected(+1);

            LayerList.SelectionChanged += (s, e) =>
            {
                _selected = LayerList.SelectedItem as SandboxLayer;
                if (_selected != null)
                {
                    Sandbox.Select(_selected.Id);
                    FreqSlider.LowerValue = _selected.FreqMinHz;
                    FreqSlider.UpperValue = _selected.FreqMaxHz;
                    LayerSens.Value = _selected.Sensitivity;
                    LayerGain.Value = _selected.Gain;
                    LblMin.Text = ((int)_selected.FreqMinHz).ToString();
                    LblMax.Text = ((int)_selected.FreqMaxHz).ToString();
                }
            };

            FreqSlider.RangeChanged += (lo, hi) =>
            {
                if (_selected == null) return;
                _selected.FreqMinHz = lo; _selected.FreqMaxHz = hi;
                LblMin.Text = ((int)lo).ToString(); LblMax.Text = ((int)hi).ToString();
            };

            LayerSens.ValueChanged += (s, e) => { if (_selected != null) _selected.Sensitivity = LayerSens.Value; };
            LayerGain.ValueChanged += (s, e) => { if (_selected != null) _selected.Gain = LayerGain.Value; };

            BtnFlipH.Click += (_, __) => { if (_selected != null) Sandbox.FlipSelected(true); };
            BtnFlipV.Click += (_, __) => { if (_selected != null) Sandbox.FlipSelected(false); };
        }

        private void AddLayerFromDialog()
        {
            var dlg = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.jpeg;*.gif", Multiselect = false };
            if (dlg.ShowDialog() == true)
            {
                var layer = new SandboxLayer();
                layer.LoadFromPath(dlg.FileName);
                _layers.Add(layer);
                Sandbox.AddLayer(layer);
                RefreshLayerList(selectId: layer.Id);
            }
        }

        private void DeleteSelectedLayer()
        {
            if (_selected == null) return;
            _layers.RemoveAll(l => l.Id == _selected.Id);
            Sandbox.RemoveSelected();
            RefreshLayerList();
        }

        private void MoveSelected(int delta)
        {
            if (_selected == null) return;
            int idx = _layers.FindIndex(l => l.Id == _selected.Id);
            if (idx < 0) return;
            int newIdx = Math.Clamp(idx + delta, 0, _layers.Count - 1);
            if (newIdx == idx) return;
            var l = _layers[idx];
            _layers.RemoveAt(idx); _layers.Insert(newIdx, l);
            Sandbox.MoveSelectedToIndex(newIdx);
            RefreshLayerList(selectId: l.Id);
        }

        private void RefreshLayerList(Guid? selectId = null)
        {
            LayerList.ItemsSource = null;
            LayerList.ItemsSource = _layers;
            if (selectId.HasValue)
            {
                var sel = _layers.FirstOrDefault(x => x.Id == selectId.Value);
                LayerList.SelectedItem = sel;
            }
        }

        // Map frequency to bar index (48 log-spaced bars, 20..20000 Hz)
        private int FreqToIndex(double f)
        {
            f = Math.Clamp(f, 20, 20000);
            int n = 48;
            double t = (Math.Log10(f) - Math.Log10(20.0)) / (Math.Log10(20000.0) - Math.Log10(20.0));
            int idx = (int)Math.Round(t * (n - 1));
            return Math.Clamp(idx, 0, n - 1);
        }

        private void Sandbox_UpdateFromSpectrum(IReadOnlyList<float> levels)
        {
            var arr = levels as float[] ?? levels.ToArray();
            Sandbox.UpdateBandEnergy(arr, freq => FreqToIndex(freq));
        }

        private void OnDevicesChanged(List<DeviceInfo> list)
        {
            if (DeviceCombo == null) return;
            Dispatcher.Invoke(() =>
            {
                string? selectedId = (DeviceCombo.SelectedItem as DeviceItem)?.Id;
                var items = new List<DeviceItem> { DeviceItem.Placeholder() };
                foreach (var d in list) items.Add(new DeviceItem { Id = d.Id, Name = d.Name });
                if (list.Count == 0)
                    items.Add(new DeviceItem { Id = string.Empty, Name = "(No audio devices found)", IsPlaceholder = true });

                DeviceCombo.ItemsSource = items; DeviceCombo.DisplayMemberPath = nameof(DeviceItem.Name);
                int idx = Math.Max(0, items.FindIndex(it => !it.IsPlaceholder && it.Id == selectedId));
                DeviceCombo.SelectedIndex = idx >= 0 ? idx : 0;
            });
        }

        private void Bars_MouseWheel_AdjustVolume(object sender, MouseWheelEventArgs e)
        {
            double step = e.Delta > 0 ? 1.0 : -1.0;
            double db = _hub.NudgeVolumeDb(step);
            _volSlider?.Dispatcher.Invoke(() => _volSlider.Value = db);
        }

        private static float MapSensToVisualSmooth(double v)
        {
            double n = (v - 0.05) / (4.0 - 0.05); n = Math.Clamp(n, 0.0, 1.0);
            double s = Lerp(0.10, 0.85, 1.0 - n);
            return (float)s;
        }

        // Scroll wheel should work anywhere over the control panel
        private void LeftScroll_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            // Scroll by a few lines per wheel notch
            double delta = e.Delta > 0 ? -48 : 48; // pixels
            LeftScroll.ScrollToVerticalOffset(LeftScroll.VerticalOffset + delta);
            e.Handled = true;
        }

        private static double Lerp(double a, double b, double t) => a + (b - a) * t;

        private class DeviceItem
        {
            public string Id { get; set; } = string.Empty;
            public string Name { get; set; } = string.Empty;
            public bool IsPlaceholder { get; set; }
            public static DeviceItem Placeholder() => new DeviceItem { Id = string.Empty, Name = "— Select Audio Device —", IsPlaceholder = true };
        }

        protected override void OnClosed(EventArgs e)
        {
            base.OnClosed(e);
            _hub.Dispose();
        }
    }
}
