// CaptureView.xaml.cs
//
// WinUI 3 (không phải UWP cũ, không phải WPF) — framework UI hiện đại nhất
// hiện tại của Microsoft cho desktop, hỗ trợ Mica/acrylic material tương
// đương .ultraThinMaterial của SwiftUI, giữ đồng bộ ngôn ngữ thiết kế với
// bản iOS/macOS và Android.
//
// CHƯA BUILD/TEST trong sandbox này (không có Windows/Visual Studio
// toolchain) — khác với phần Rust core đã build+test thật. Cần mở bằng
// Visual Studio (Windows App SDK) để build.
//
// Camera dùng MediaCapture API (Windows.Media.Capture) — đây là API hiện
// đại thay thế Media Foundation viết tay mà tôi có nhắc ở câu trả lời
// trước; MediaCapture bọc sẵn Media Foundation bên dưới nên vẫn đúng tinh
// thần "Media Foundation" nhưng đỡ code hơn nhiều, phù hợp hơn cho phần
// UI/orchestration (core Rust vẫn lo phần tính toán nặng).

using System;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;

namespace Wifakey.Capture
{
    public enum CaptureState
    {
        Searching, Adjusting, Ready, Processing, Success, Failed
    }

    public enum FlowKind { Enroll, Verify }

    public sealed partial class CaptureView : Page
    {
        private readonly MediaCapture _mediaCapture = new();
        private CaptureState _state = CaptureState.Searching;
        public string SessionId { get; set; } = string.Empty;
        public FlowKind Flow { get; set; }

        public CaptureView()
        {
            this.InitializeComponent();
            Loaded += async (_, _) => await InitializeCameraAsync();
            Unloaded += (_, _) => _mediaCapture.Dispose();
        }

        private async System.Threading.Tasks.Task InitializeCameraAsync()
        {
            var settings = new MediaCaptureInitializationSettings
            {
                StreamingCaptureMode = StreamingCaptureMode.Video,
            };
            await _mediaCapture.InitializeAsync(settings);
            PreviewElement.Source = _mediaCapture;
            await _mediaCapture.StartPreviewAsync();

            // TODO: thêm FrameReader ở đây, mỗi frame gọi vào core (qua P/Invoke
            // tới wifakey_core.dll) để đánh giá chất lượng khuôn mặt, tương tự
            // CameraController.swift (iOS/macOS) và CaptureViewModel.kt (Android).
        }

        private void UpdateState(CaptureState newState)
        {
            _state = newState;
            StatusText.Text = StatusTextFor(newState);
            StatusDot.Fill = new SolidColorBrush(ColorFor(newState));
            InstructionText.Text = InstructionTextFor(newState, Flow);
            ProcessingRing.Visibility = newState == CaptureState.Processing
                ? Visibility.Visible : Visibility.Collapsed;

            if (newState == CaptureState.Ready)
            {
                _ = TriggerCaptureAsync();
            }
        }

        /// Gọi vào core để chạy pipeline enroll/verify, rồi POST thẳng lên
        /// server — không đưa payload nhạy cảm qua bất kỳ lớp UI nào khác,
        /// đúng nguyên tắc đã thống nhất ở các câu trả lời trước.
        private async System.Threading.Tasks.Task TriggerCaptureAsync()
        {
            UpdateState(CaptureState.Processing);
            // TODO: gọi hàm FFI thật khi core.enroll()/verify() sẵn sàng, vd:
            //   var result = WifakeyCoreInterop.Enroll(embedding, SessionId);
            //   await NetworkClient.PostDirectlyToServerAsync(result, SessionId);
            //   UpdateState(CaptureState.Success);
            await System.Threading.Tasks.Task.CompletedTask;
        }

        private static string StatusTextFor(CaptureState s) => s switch
        {
            CaptureState.Searching => "Đang tìm khuôn mặt",
            CaptureState.Adjusting => "Đang căn chỉnh",
            CaptureState.Ready => "Đã sẵn sàng",
            CaptureState.Processing => "Đang xử lý",
            CaptureState.Success => "Thành công",
            CaptureState.Failed => "Thất bại",
            _ => "",
        };

        private static string InstructionTextFor(CaptureState s, FlowKind flow) => s switch
        {
            CaptureState.Searching => "Đưa khuôn mặt vào khung hình",
            CaptureState.Adjusting => "Giữ yên và nhìn thẳng vào camera",
            CaptureState.Ready => "Giữ yên...",
            CaptureState.Processing => "Đang xác thực, vui lòng đợi trong giây lát",
            CaptureState.Success => flow == FlowKind.Enroll ? "Đăng ký thành công" : "Xác thực thành công",
            CaptureState.Failed => "Vui lòng thử lại",
            _ => "",
        };

        private static Windows.UI.Color ColorFor(CaptureState s) => s switch
        {
            CaptureState.Searching => Windows.UI.Color.FromArgb(255, 255, 167, 38),
            CaptureState.Adjusting => Windows.UI.Color.FromArgb(255, 255, 238, 88),
            CaptureState.Ready or CaptureState.Success => Windows.UI.Color.FromArgb(255, 102, 187, 106),
            CaptureState.Processing => Windows.UI.Color.FromArgb(255, 66, 165, 245),
            CaptureState.Failed => Windows.UI.Color.FromArgb(255, 239, 83, 80),
            _ => Windows.UI.Color.FromArgb(255, 255, 255, 255),
        };
    }
}
