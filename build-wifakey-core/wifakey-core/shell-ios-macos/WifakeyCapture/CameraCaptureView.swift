// CameraCaptureView.swift
//
// Giao diện chụp khuôn mặt dùng CHUNG cho cả iOS và macOS (SwiftUI hỗ trợ
// cả 2 nền tảng qua #if os(...)) — đây là lợi thế thật của cặp iOS/macOS so
// với Windows/Android, đã nhắc ở câu trả lời trước.
//
// CHƯA BUILD/TEST được trong sandbox này (không có Xcode toolchain) — khác
// với phần Rust core đã build+test thật. Bạn cần mở bằng Xcode để build.
//
// Thiết kế hiện đại: khung oval mờ dần theo trạng thái, animation pulse nhẹ
// khi đang tìm khuôn mặt, chuyển màu xanh + rung nhẹ (haptic) khi bắt được
// khuôn mặt đúng vị trí, dùng .ultraThinMaterial cho lớp nền mờ kiểu iOS 17+.

import SwiftUI
import AVFoundation

/// Trạng thái capture — driven bởi liveness/face-detection chạy trong core
/// (qua FFI), KHÔNG phải logic UI tự quyết định "đúng/sai".
enum CaptureState: Equatable {
    case searching          // chưa thấy khuôn mặt
    case adjusting          // thấy khuôn mặt nhưng chưa đúng vị trí/khoảng cách
    case ready               // đúng vị trí, sẵn sàng chụp
    case processing          // đã chụp, đang chạy pipeline (AdaFace, liveness...)
    case success
    case failed(reason: String)
}

struct CameraCaptureView: View {
    @StateObject private var cameraController = CameraController()
    @State private var captureState: CaptureState = .searching

    /// session_id nhận từ deep-link/Native Messaging lúc launch (bàn ở các
    /// câu trả lời trước) — truyền vào để biết gửi kết quả POST cho đúng phiên.
    let sessionId: String
    let flowKind: FlowKind // .enroll hoặc .verify

    var body: some View {
        ZStack {
            // Camera preview full màn hình
            CameraPreviewLayer(session: cameraController.session)
                .ignoresSafeArea()

            // Lớp tối nhẹ phủ ngoài khung oval để làm nổi bật vùng chụp
            FaceGuideOverlay(state: captureState)

            VStack {
                topStatusBar
                Spacer()
                bottomInstructionPanel
            }
            .padding()
        }
        .onAppear { cameraController.start() }
        .onDisappear { cameraController.stop() }
        .onChange(of: cameraController.detectedFaceQuality) { quality in
            captureState = deriveState(from: quality)
            if case .ready = captureState {
                Task { await triggerCapture() }
            }
        }
    }

    private var topStatusBar: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            Text(statusText)
                .font(.system(.subheadline, design: .rounded, weight: .medium))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private var bottomInstructionPanel: some View {
        VStack(spacing: 12) {
            Text(instructionText)
                .font(.system(.body, design: .rounded))
                .foregroundStyle(.white)
                .multilineTextAlignment(.center)

            if case .processing = captureState {
                ProgressView()
                    .tint(.white)
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .padding(.horizontal)
    }

    private var statusColor: Color {
        switch captureState {
        case .searching: .orange
        case .adjusting: .yellow
        case .ready, .success: .green
        case .processing: .blue
        case .failed: .red
        }
    }

    private var statusText: String {
        switch captureState {
        case .searching: "Đang tìm khuôn mặt"
        case .adjusting: "Đang căn chỉnh"
        case .ready: "Đã sẵn sàng"
        case .processing: "Đang xử lý"
        case .success: "Thành công"
        case .failed: "Thất bại"
        }
    }

    private var instructionText: String {
        switch captureState {
        case .searching: "Đưa khuôn mặt vào khung hình"
        case .adjusting: "Giữ yên và nhìn thẳng vào camera"
        case .ready: "Giữ yên..."
        case .processing: "Đang xác thực, vui lòng đợi trong giây lát"
        case .success: flowKind == .enroll ? "Đăng ký thành công" : "Xác thực thành công"
        case .failed(let reason): reason
        }
    }

    private func deriveState(from quality: FaceQuality?) -> CaptureState {
        guard let quality else { return .searching }
        if quality.isWellPositioned && quality.livenessPassed {
            return .ready
        }
        return .adjusting
    }

    /// Gọi vào core (qua FFI) để chạy pipeline enroll/verify, rồi POST thẳng
    /// kết quả lên server — KHÔNG trả payload nhạy cảm về qua bất kỳ closure
    /// nào ở lớp UI, đúng nguyên tắc "native app gửi trực tiếp cho server"
    /// đã thống nhất ở câu trả lời trước.
    private func triggerCapture() async {
        captureState = .processing
        // TODO: gọi hàm FFI thật, vd:
        //   let frame = cameraController.currentFrameBuffer()
        //   let result = WifakeyCore.enroll(embedding: ..., sessionId: sessionId)
        //   await NetworkClient.postDirectlyToServer(result, sessionId: sessionId)
        // Xem README phần "Bước 5" của core để biết khi nào enroll()/verify()
        // thật sẽ sẵn sàng để gọi ở đây.
    }
}

enum FlowKind { case enroll, verify }

/// Khung oval hướng dẫn vị trí khuôn mặt — animate mượt theo trạng thái.
struct FaceGuideOverlay: View {
    let state: CaptureState

    var body: some View {
        GeometryReader { geo in
            let ovalSize = CGSize(width: geo.size.width * 0.68, height: geo.size.width * 0.68 * 1.3)
            ZStack {
                // Nền tối phủ ngoài, khoét oval ở giữa (kỹ thuật mask ngược)
                Rectangle()
                    .fill(.black.opacity(0.45))
                    .mask(
                        Rectangle()
                            .overlay(
                                Ellipse()
                                    .frame(width: ovalSize.width, height: ovalSize.height)
                                    .blendMode(.destinationOut)
                            )
                    )
                    .compositingGroup()

                Ellipse()
                    .strokeBorder(borderColor, lineWidth: 3)
                    .frame(width: ovalSize.width, height: ovalSize.height)
                    .scaleEffect(pulseScale)
                    .animation(
                        state == .searching
                            ? .easeInOut(duration: 1.2).repeatForever(autoreverses: true)
                            : .spring(response: 0.35, dampingFraction: 0.7),
                        value: state
                    )
            }
            .position(x: geo.size.width / 2, y: geo.size.height / 2)
        }
    }

    private var borderColor: Color {
        switch state {
        case .searching: .white.opacity(0.6)
        case .adjusting: .yellow
        case .ready, .success: .green
        case .processing: .blue
        case .failed: .red
        }
    }

    private var pulseScale: CGFloat {
        state == .searching ? 1.03 : 1.0
    }
}

/// Bọc AVCaptureVideoPreviewLayer cho cả 2 nền tảng — đây là chỗ duy nhất
/// thật sự khác nhau giữa iOS (UIViewRepresentable) và macOS
/// (NSViewRepresentable), phần logic overlay/state phía trên dùng chung
/// nguyên vẹn.
#if os(iOS)
struct CameraPreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }
    func updateUIView(_ uiView: PreviewUIView, context: Context) {}
}

final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}
#elseif os(macOS)
struct CameraPreviewLayer: NSViewRepresentable {
    let session: AVCaptureSession
    func makeNSView(context: Context) -> PreviewNSView {
        let view = PreviewNSView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }
    func updateNSView(_ nsView: PreviewNSView, context: Context) {}
}

final class PreviewNSView: NSView {
    let previewLayer = AVCaptureVideoPreviewLayer()
    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer = previewLayer
    }
    required init?(coder: NSCoder) { fatalError() }
}
#endif
