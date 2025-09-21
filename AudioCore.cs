using System;
using System.Collections.Generic;
using NAudio.CoreAudioApi;
using NAudio.CoreAudioApi.Interfaces;
using NAudio.Wave;

namespace test
{
    public sealed class AudioCore : IMMNotificationClient, IDisposable
    {
        private readonly MMDeviceEnumerator _enum = new();
        private WasapiCapture? _capture;

        public event Action<List<DeviceInfo>>? DevicesChanged;
        public event Action<DeviceInfo>? DefaultDeviceChanged;
        public event Action<byte[]>? PcmBytes;

        public AudioCore()
        {
            _enum.RegisterEndpointNotificationCallback(this);
        }

        public List<DeviceInfo> ListLoopbackCapable()
        {
            var result = new List<DeviceInfo>();
            var def = _enum.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
            foreach (var dev in _enum.EnumerateAudioEndPoints(DataFlow.Render, DeviceState.Active))
            {
                result.Add(new DeviceInfo
                {
                    Id = dev.ID,
                    Name = dev.FriendlyName,
                    IsDefault = dev.ID == def.ID
                });
            }
            // Keep ordering: default first, then others in original order (no de-dup so every device shows)
            result.Sort((a, b) => (a.IsDefault ? -1 : 0) - (b.IsDefault ? -1 : 0));
            return result;
        }

        public DeviceInfo GetDefault()
        {
            var def = _enum.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
            return new DeviceInfo { Id = def.ID, Name = def.FriendlyName, IsDefault = true };
        }

        public void OpenLoopback(DeviceInfo dev)
        {
            Stop();
            var endpoint = _enum.GetDevice(dev.Id);
            _capture = new WasapiLoopbackCapture(endpoint) { ShareMode = AudioClientShareMode.Shared };
            _capture.DataAvailable += (s, e) =>
            {
                var buf = new byte[e.BytesRecorded];
                Array.Copy(e.Buffer, buf, e.BytesRecorded);
                PcmBytes?.Invoke(buf);
            };
            _capture.StartRecording();
        }

        public void Stop()
        {
            if (_capture != null)
            {
                try { _capture.StopRecording(); } catch { }
                _capture.Dispose();
                _capture = null;
            }
        }

        public void OnDeviceStateChanged(string deviceId, DeviceState newState) => DevicesChanged?.Invoke(ListLoopbackCapable());
        public void OnDeviceAdded(string pwstrDeviceId) => DevicesChanged?.Invoke(ListLoopbackCapable());
        public void OnDeviceRemoved(string deviceId) => DevicesChanged?.Invoke(ListLoopbackCapable());
        public void OnDefaultDeviceChanged(DataFlow flow, Role role, string defaultDeviceId)
        {
            if (flow == DataFlow.Render && role == Role.Multimedia)
                DefaultDeviceChanged?.Invoke(GetDefault());
        }
        public void OnPropertyValueChanged(string pwstrDeviceId, PropertyKey key) { }

        public void Dispose()
        {
            Stop();
            try { _enum.UnregisterEndpointNotificationCallback(this); } catch { }
            _enum.Dispose();
        }
    }

    public record DeviceInfo
    {
        public string Id { get; init; } = string.Empty;
        public string Name { get; init; } = string.Empty;
        public bool IsDefault { get; init; }
        public string Label => IsDefault ? $"Default Output — {Name}" : Name;
        public override string ToString() => Label;
    }
}
