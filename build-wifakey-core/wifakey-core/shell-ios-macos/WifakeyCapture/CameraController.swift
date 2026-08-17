// CameraController.swift
// Quản lý AVCaptureSession + kết nối tới core (qua FFI) để đánh giá chất
// lượng khuôn mặt mỗi frame. Phần detection thật (liveness, vị trí khuôn
// mặt) chạy trong core Rust — file này chỉ có nhiệm vụ lấy frame và gọi vào
// core, KHÔNG tự làm face-detection ở tầng Swift.

import AVFoundation
import Combine

struct FaceQuality {
    let isWellPositioned: Bool
    let livenessPassed: Bool
}

@MainActor
final class CameraController: NSObject, ObservableObject {
    let session = AVCaptureSession()
    @Published var detectedFaceQuality: FaceQuality?

    private let videoOutput = AVCaptureVideoDataOutput()
    private let processingQueue = DispatchQueue(label: "wifakey.camera.processing")

    func start() {
        Task.detached(priority: .userInitiated) { [weak self] in
            await self?.configureSession()
            self?.session.startRunning()
        }
    }

    func stop() {
        session.stopRunning()
    }

    private func configureSession() async {
        guard await AVCaptureDevice.requestAccess(for: .video) else {
            return
        }
        session.beginConfiguration()
        session.sessionPreset = .high

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        session.commitConfiguration()
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // TODO: lấy CVPixelBuffer từ sampleBuffer, gọi vào core qua FFI để
        // đánh giá chất lượng/liveness, vd:
        //   let quality = WifakeyCore.assessFrame(pixelBuffer)
        //   Task { @MainActor in self.detectedFaceQuality = quality }
        //
        // Chưa implement vì phụ thuộc hàm FFI thật (core.enroll()/verify()
        // chưa hoàn thiện — xem README của core).
    }
}
