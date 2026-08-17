using Microsoft.UI.Xaml;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Graphics.Imaging;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Media.Core;
using Windows.Media.Playback;

namespace WifakeyApp;

public enum FlowKind { Enroll, Verify }

public sealed partial class MainWindow : Window
{
    [DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr wifakey_pipeline_load(string inputJson);

    [DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr wifakey_pipeline_process_frame(
        ulong handle, byte[] frameBgr, UIntPtr frameLen, uint width, uint height);

    [DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void wifakey_pipeline_free(ulong handle);

    [DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void wifakey_free_string(IntPtr s);

    [DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr wifakey_enroll(string inputJson);

    [DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr wifakey_verify_prepare(string inputJson);

    private ulong _pipelineHandle;
    private MediaCapture? _mediaCapture;
    private MediaFrameReader? _frameReader;
    private MediaPlayer? _mediaPlayer;

    private volatile bool _captureCompleted;
    private int _isProcessingFlag;
    private DateTime _lastProcessedAt = DateTime.MinValue;
    private static readonly TimeSpan MinProcessInterval = TimeSpan.FromMilliseconds(200);

    private readonly FlowKind _flowKind;
    private readonly string? _sessionId;

    private static readonly string LocalTestEnrollDataPath =
        Path.Combine(AppContext.BaseDirectory, "local_test_enroll_data.json");

    public MainWindow(FlowKind flowKind = FlowKind.Enroll, string? sessionId = null)
    {
        _flowKind = flowKind;
        _sessionId = sessionId;

        InitializeComponent();
        ResultText.Text = "Đang khởi tạo...";
        this.Closed += (_, _) => CleanupPipeline();
        _ = InitializePipelineAndCameraAsync();
    }

    private async Task InitializePipelineAndCameraAsync()
    {
        try
        {
            string baseDir = AppContext.BaseDirectory;
            var loadRequest = new
            {
                det_onnx_path = Path.Combine(baseDir, "det_10g.onnx"),
                liveness_onnx_path = Path.Combine(baseDir, "anti-spoofing.onnx"),
                embedding_onnx_path = Path.Combine(baseDir, "adaface_ir101.onnx"),
                confidence_threshold = 0.6,
                liveness_threshold_prob = 0.8,
                bbox_expansion_factor = 1.5
            };

            IntPtr resultPtr = wifakey_pipeline_load(JsonSerializer.Serialize(loadRequest));
            string? resultJson = Marshal.PtrToStringUTF8(resultPtr);
            wifakey_free_string(resultPtr);

            using var doc = JsonDocument.Parse(resultJson!);
            if (doc.RootElement.TryGetProperty("error", out var err))
            {
                ResultText.Text = "Pipeline load lỗi: " + err.GetString();
                return;
            }
            _pipelineHandle = doc.RootElement.GetProperty("handle").GetUInt64();
            ResultText.Text = "Pipeline OK, đang mở camera...";

            await StartCameraAsync();
        }
        catch (Exception ex)
        {
            ResultText.Text = "LỖI KHỞI TẠO: " + ex;
        }
    }

    private async Task StartCameraAsync()
    {
        _mediaCapture = new MediaCapture();

        var settings = new MediaCaptureInitializationSettings
        {
            StreamingCaptureMode = StreamingCaptureMode.Video,
            MemoryPreference = MediaCaptureMemoryPreference.Cpu,
            SharingMode = MediaCaptureSharingMode.ExclusiveControl
        };
        await _mediaCapture.InitializeAsync(settings);

        var frameSource = _mediaCapture.FrameSources.Values
            .FirstOrDefault(s => s.Info.SourceKind == MediaFrameSourceKind.Color);
        if (frameSource == null)
        {
            ResultText.Text = "Không tìm thấy camera màu";
            return;
        }

        _mediaPlayer = new MediaPlayer
        {
            Source = MediaSource.CreateFromMediaFrameSource(frameSource)
        };
        PreviewPlayer.SetMediaPlayer(_mediaPlayer);
        _mediaPlayer.Play();

        _frameReader = await _mediaCapture.CreateFrameReaderAsync(frameSource);
        _frameReader.FrameArrived += OnFrameArrived;
        var startStatus = await _frameReader.StartAsync();

        if (startStatus != MediaFrameReaderStartStatus.Success)
        {
            ResultText.Text = "Frame reader start thất bại: " + startStatus;
            return;
        }

        ResultText.Text = "Camera đã mở, đang xử lý...";
    }

    private void OnFrameArrived(MediaFrameReader sender, MediaFrameArrivedEventArgs args)
    {
        if (_captureCompleted) return;
        if (DateTime.UtcNow - _lastProcessedAt < MinProcessInterval) return;
        if (Interlocked.CompareExchange(ref _isProcessingFlag, 1, 0) != 0) return;

        try
        {
            using var frame = sender.TryAcquireLatestFrame();
            var bitmap = frame?.VideoMediaFrame?.SoftwareBitmap;
            if (bitmap == null)
            {
                System.Diagnostics.Debug.WriteLine("[Wifakey] SoftwareBitmap null");
                return;
            }

            using var bgra8Bitmap = SoftwareBitmap.Convert(
                bitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);

            _lastProcessedAt = DateTime.UtcNow;

            byte[] bgraBytes = new byte[4 * bgra8Bitmap.PixelWidth * bgra8Bitmap.PixelHeight];
            bgra8Bitmap.CopyToBuffer(bgraBytes.AsBuffer());

            int w = bgra8Bitmap.PixelWidth;
            int h = bgra8Bitmap.PixelHeight;
            byte[] bgrBytes = new byte[3 * w * h];
            for (int i = 0; i < w * h; i++)
            {
                bgrBytes[i * 3] = bgraBytes[i * 4];
                bgrBytes[i * 3 + 1] = bgraBytes[i * 4 + 1];
                bgrBytes[i * 3 + 2] = bgraBytes[i * 4 + 2];
            }

            IntPtr resultPtr = wifakey_pipeline_process_frame(
                _pipelineHandle, bgrBytes, (UIntPtr)bgrBytes.Length, (uint)w, (uint)h);
            string? resultJson = Marshal.PtrToStringUTF8(resultPtr);
            wifakey_free_string(resultPtr);

            DispatcherQueue.TryEnqueue(() =>
            {
                try { HandleProcessResult(resultJson!); }
                catch (Exception ex) { ResultText.Text = "LỖI CẬP NHẬT UI: " + ex; }
            });
        }
        catch (Exception ex)
        {
            DispatcherQueue.TryEnqueue(() => ResultText.Text = "LỖI XỬ LÝ FRAME: " + ex);
        }
        finally
        {
            Interlocked.Exchange(ref _isProcessingFlag, 0);
        }
    }

    private async void HandleProcessResult(string resultJson)
    {
        try
        {
            using var doc = JsonDocument.Parse(resultJson);
            var root = doc.RootElement;

            if (root.TryGetProperty("error", out var err))
            {
                ResultText.Text = "Lỗi: " + err.GetString();
                return;
            }

            string status = root.GetProperty("status").GetString()!;

            if (status == "success")
            {
                _captureCompleted = true;

                var embedding = root.GetProperty("embedding")
                    .EnumerateArray().Select(e => e.GetSingle()).ToList();

                if (_frameReader != null)
                {
                    await _frameReader.StopAsync();
                    _frameReader.FrameArrived -= OnFrameArrived;
                }

                ResultText.Text = "Đang xử lý sinh trắc học...";
                await CompleteBiometricFlowAsync(embedding);
            }
            else
            {
                ResultText.Text = status switch
                {
                    "face_out_of_frame" => "Vui lòng đưa toàn bộ khuôn mặt vào khung hình",
                    "no_face" => "Đang tìm khuôn mặt...",
                    "low_confidence" => "Đang căn chỉnh...",
                    "spoof_detected" => "Không xác thực được — vui lòng dùng khuôn mặt thật",
                    _ => "Trạng thái: " + status
                };
            }
        }
        catch (Exception ex)
        {
            ResultText.Text = "LỖI XỬ LÝ KẾT QUẢ: " + ex;
        }
    }

    private static string CallNative(Func<string, IntPtr> nativeFn, string requestJson)
    {
        IntPtr resultPtr = nativeFn(requestJson);
        string? result = Marshal.PtrToStringUTF8(resultPtr);
        wifakey_free_string(resultPtr);
        return result ?? throw new InvalidOperationException("native trả về null pointer");
    }

    private async Task CompleteBiometricFlowAsync(List<float> embedding)
    {
        try
        {
            string baseDir = AppContext.BaseDirectory;
            string mMatrixPath = Path.Combine(baseDir, "M_matrix.txt");
            string gMatrixPath = Path.Combine(baseDir, "generator_matrix_G.txt");
            string llrTablePath = Path.Combine(baseDir, "empirical_llr_table.txt");

            if (_flowKind == FlowKind.Enroll)
                await RunEnrollAsync(embedding, mMatrixPath, gMatrixPath, llrTablePath);
            else
                await RunVerifyAsync(embedding, mMatrixPath, gMatrixPath, llrTablePath);
        }
        catch (Exception ex)
        {
            ResultText.Text = "LỖI: " + ex;
        }
    }

    private async Task RunEnrollAsync(List<float> embedding, string mMatrixPath, string gMatrixPath, string llrTablePath)
    {
        // TODO: user_secret/service_salt phải lấy từ POST /enroll/init
        // {session_id} khi có server thật — dùng giá trị test tạm.
        var userSecret = Enumerable.Range(1, 20).ToList();
        var serviceSalt = Enumerable.Range(1, 8).ToList();

        var enrollRequest = new
        {
            embedding, user_secret = userSecret, service_salt = serviceSalt,
            m_matrix_path = mMatrixPath, generator_matrix_g_path = gMatrixPath,
            empirical_llr_table_path = llrTablePath
        };

        string resultJson = CallNative(wifakey_enroll, JsonSerializer.Serialize(enrollRequest));
        using var doc = JsonDocument.Parse(resultJson);

        if (doc.RootElement.TryGetProperty("error", out var err))
        {
            ResultText.Text = "Enroll lỗi: " + err.GetString();
            return;
        }

        // TODO: khi có FastAPI, thay bằng POST /enroll/complete
        // {session_id, helper_data, reliability_mask, key_hash, service_salt}.
        var testStorage = new
        {
            user_secret = userSecret, service_salt = serviceSalt,
            helper_data = doc.RootElement.GetProperty("helper_data"),
            reliability_mask = doc.RootElement.GetProperty("reliability_mask")
        };
        await File.WriteAllTextAsync(LocalTestEnrollDataPath, JsonSerializer.Serialize(testStorage));

        // Chỉ hiện 4 byte đầu key_hash — đủ để biết pipeline chạy ra dữ liệu
        // thật (không phải lỗi im lặng), không in tràn toàn bộ dữ liệu.
        byte[] keyHashPrefix = doc.RootElement.GetProperty("key_hash")
            .EnumerateArray().Take(4).Select(e => (byte)e.GetInt32()).ToArray();
        string debugInfo = $"key_hash: {BitConverter.ToString(keyHashPrefix)}...";

        await ShowSuccessAndCloseAsync("Đăng ký thành công!", debugInfo);
    }

    private async Task RunVerifyAsync(List<float> embedding, string mMatrixPath, string gMatrixPath, string llrTablePath)
    {
        if (!File.Exists(LocalTestEnrollDataPath))
        {
            ResultText.Text = "Chưa có dữ liệu enroll local để test verify.";
            return;
        }

        // TODO: thay khối đọc file này bằng POST /verify/challenge {username}
        // lấy dữ liệu thật từ server.
        string savedJson = await File.ReadAllTextAsync(LocalTestEnrollDataPath);
        using var savedDoc = JsonDocument.Parse(savedJson);
        var saved = savedDoc.RootElement;

        var userSecret = saved.GetProperty("user_secret").EnumerateArray().Select(e => e.GetInt32()).ToList();
        var serviceSalt = saved.GetProperty("service_salt").EnumerateArray().Select(e => e.GetInt32()).ToList();
        var helperData = saved.GetProperty("helper_data").EnumerateArray().Select(e => e.GetBoolean()).ToList();
        var reliabilityMask = saved.GetProperty("reliability_mask").EnumerateArray().Select(e => e.GetBoolean()).ToList();

        var verifyRequest = new
        {
            embedding, user_secret = userSecret, helper_data = helperData,
            reliability_mask = reliabilityMask, service_salt = serviceSalt,
            m_matrix_path = mMatrixPath, generator_matrix_g_path = gMatrixPath,
            empirical_llr_table_path = llrTablePath
        };

        string resultJson = CallNative(wifakey_verify_prepare, JsonSerializer.Serialize(verifyRequest));
        using var doc = JsonDocument.Parse(resultJson);

        if (doc.RootElement.TryGetProperty("error", out var err))
        {
            ResultText.Text = "Verify lỗi: " + err.GetString();
            return;
        }

        // TODO: hiện chỉ dừng ở llr — decode + so hash cuối cùng cần server
        // thật (đã chuyển sang server theo thiết kế kiến trúc đã chốt).
        int llrCount = doc.RootElement.GetProperty("llr").GetArrayLength();
        await ShowSuccessAndCloseAsync("Xác thực thành công!", $"llr: {llrCount} phần tử");
    }

    private async Task ShowSuccessAndCloseAsync(string message, string? debugInfo = null)
    {
        ResultText.Text = debugInfo != null ? $"{message}\n{debugInfo}" : message;
        await Task.Delay(1500);
        Application.Current.Exit();
    }

    private void CleanupPipeline()
    {
        _mediaPlayer?.Dispose();
        _mediaPlayer = null;

        if (_frameReader != null)
            _frameReader.FrameArrived -= OnFrameArrived;
        _mediaCapture?.Dispose();
        _mediaCapture = null;

        if (_pipelineHandle != 0)
        {
            wifakey_pipeline_free(_pipelineHandle);
            _pipelineHandle = 0;
        }
    }
}