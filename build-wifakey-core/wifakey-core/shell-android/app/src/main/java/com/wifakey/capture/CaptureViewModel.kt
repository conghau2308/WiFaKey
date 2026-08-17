// CaptureViewModel.kt
// Tương ứng CameraController.swift bên iOS/macOS về mặt vai trò, nhưng
// tách theo đúng kiến trúc MVVM chuẩn của Android thay vì gộp vào 1 file.

package com.wifakey.capture

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class CaptureViewModel : ViewModel() {
    private val _captureState = MutableStateFlow<CaptureState>(CaptureState.Searching)
    val captureState: StateFlow<CaptureState> = _captureState

    /// Gọi từ callback đánh giá chất lượng khuôn mặt mỗi frame (từ core qua
    /// JNI) — TODO: nối vào ImageAnalysis use case trong CameraCaptureScreen.kt.
    fun onFaceQualityUpdated(wellPositioned: Boolean, livenessPassed: Boolean) {
        _captureState.value = when {
            wellPositioned && livenessPassed -> CaptureState.Ready
            wellPositioned -> CaptureState.Adjusting
            else -> CaptureState.Searching
        }
    }

    /// Gọi vào core (qua JNI/UniFFI) để chạy pipeline enroll/verify, rồi
    /// POST thẳng lên server — không đưa payload nhạy cảm ra ngoài ViewModel
    /// này, đúng nguyên tắc "native app gửi trực tiếp cho server" đã bàn.
    fun triggerCapture(sessionId: String, flowKind: FlowKind) {
        _captureState.value = CaptureState.Processing
        viewModelScope.launch {
            // TODO: gọi hàm FFI thật khi core.enroll()/verify() sẵn sàng, vd:
            //   val result = WifakeyCore.enroll(embedding, sessionId)
            //   NetworkClient.postDirectlyToServer(result, sessionId)
            //   _captureState.value = CaptureState.Success
        }
    }
}
