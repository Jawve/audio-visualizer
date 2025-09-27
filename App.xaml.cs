using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Markup;   // XamlParseException
using System.Windows.Threading;

namespace test
{
    public partial class App : Application
    {
        private string LogPath =>
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                         "WindowVisualizer", "boot.log");

        protected override void OnStartup(StartupEventArgs e)
        {
            // Global handlers so nothing dies silently
            AppDomain.CurrentDomain.UnhandledException += CurrentDomain_UnhandledException;
            DispatcherUnhandledException += App_DispatcherUnhandledException;
            TaskScheduler.UnobservedTaskException += TaskScheduler_UnobservedTaskException;

            Directory.CreateDirectory(Path.GetDirectoryName(LogPath)!);
            Log("=== App starting ===");

            try
            {
                // Attempt to create/show the main window explicitly so we can catch parse errors
                var win = new MainWindow();
                MainWindow = win;
                win.Show();
                Log("MainWindow shown.");
            }
            catch (XamlParseException xpe)
            {
                var msg = BuildXamlMessage("XAML Parse error creating MainWindow", xpe);
                Log(msg + Environment.NewLine + xpe.ToString());
                MessageBox.Show(msg, "Startup Error (XAML)", MessageBoxButton.OK, MessageBoxImage.Error);
                Shutdown(1);
            }
            catch (Exception ex)
            {
                var msg = "Unhandled error during startup:\n" + ex.Message;
                Log(msg + Environment.NewLine + ex.ToString());
                MessageBox.Show(msg, "Startup Error", MessageBoxButton.OK, MessageBoxImage.Error);
                Shutdown(1);
            }

            base.OnStartup(e);
        }

        private void App_DispatcherUnhandledException(object? sender, DispatcherUnhandledExceptionEventArgs e)
        {
            var msg = "Unhandled UI exception:\n" + e.Exception.Message;
            Log(msg + Environment.NewLine + e.Exception.ToString());
            MessageBox.Show(msg, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            e.Handled = true;
        }

        private void CurrentDomain_UnhandledException(object? sender, UnhandledExceptionEventArgs e)
        {
            var ex = e.ExceptionObject as Exception;
            var msg = "Unhandled domain exception:\n" + (ex?.Message ?? e.ExceptionObject.ToString());
            Log(msg + Environment.NewLine + ex);
            try { MessageBox.Show(msg, "Fatal Error", MessageBoxButton.OK, MessageBoxImage.Error); } catch { }
        }

        private void TaskScheduler_UnobservedTaskException(object? sender, UnobservedTaskExceptionEventArgs e)
        {
            var msg = "Unobserved task exception:\n" + e.Exception.Message;
            Log(msg + Environment.NewLine + e.Exception.ToString());
            e.SetObserved();
        }

        private string BuildXamlMessage(string header, XamlParseException xpe)
        {
            var sb = new StringBuilder();
            sb.AppendLine(header);
            if (xpe.LineNumber > 0 || xpe.LinePosition > 0)
                sb.AppendLine($"Line {xpe.LineNumber}, Position {xpe.LinePosition}");
            sb.AppendLine(xpe.Message);
            if (xpe.InnerException != null)
            {
                sb.AppendLine();
                sb.AppendLine("Inner: " + xpe.InnerException.Message);
            }
            sb.AppendLine();
            sb.AppendLine("A log was written to:");
            sb.AppendLine(LogPath);
            return sb.ToString();
        }

        private void Log(string text)
        {
            try
            {
                File.AppendAllText(LogPath, $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {text}{Environment.NewLine}");
            }
            catch { /* ignore logging failure */ }
        }
    }
}
