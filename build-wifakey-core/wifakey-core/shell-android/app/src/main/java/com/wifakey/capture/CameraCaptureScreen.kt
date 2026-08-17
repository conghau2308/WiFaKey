// CameraCaptureScreen.kt
//
// Đồng bộ thiết kế với bản iOS/macOS (CameraCaptureView.swift) — cùng
// state machine (searching/adjusting/ready/processing/success/failed),
// cùng khung oval overlay, cùng status pill mờ ở trên. Material3 làm nền
// tảng UI hiện đại cho Android tương đương .ultraThinMaterial của SwiftUI.
//
// CHƯA BUILD/TEST trong sandbox này (không có Android SDK/Gradle) — khác
// với phần Rust core đã build+test thật.

package com.wifakey.capture

import androidx.camera.core.CameraSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner

sealed class CaptureState {
    data object Searching : CaptureState()
    data object Adjusting : CaptureState()
    data object Ready : CaptureState()
    data object Processing : CaptureState()
    data object Success : CaptureState()
    data class Failed(val reason: String) : CaptureState()
}

enum class FlowKind { ENROLL, VERIFY }

@Composable
fun CameraCaptureScreen(
    sessionId: String,
    flowKind: FlowKind,
    viewModel: CaptureViewModel = androidx.lifecycle.viewmodel.compose.viewModel(),
) {
    val state by viewModel.captureState.collectAsState()
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    LaunchedEffect(state) {
        if (state is CaptureState.Ready) {
            viewModel.triggerCapture(sessionId, flowKind)
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = androidx.camera.core.Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                    val selector = CameraSelector.DEFAULT_FRONT_CAMERA
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(lifecycleOwner, selector, preview)
                    // TODO: thêm ImageAnalysis use case ở đây, gọi vào core
                    // qua JNI mỗi frame để đánh giá chất lượng khuôn mặt,
                    // tương tự CameraController.swift bản iOS/macOS.
                }, androidx.core.content.ContextCompat.getMainExecutor(ctx))
                previewView
            },
            modifier = Modifier.fillMaxSize(),
        )

        FaceGuideOverlay(state = state, modifier = Modifier.fillMaxSize())

        Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
            StatusPill(state = state)
            Spacer(modifier = Modifier.weight(1f))
            InstructionPanel(state = state, flowKind = flowKind)
        }
    }
}

@Composable
private fun StatusPill(state: CaptureState) {
    Surface(
        shape = CircleShape,
        color = Color.Black.copy(alpha = 0.35f),
        modifier = Modifier.align(Alignment.Start),
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .background(statusColor(state), CircleShape)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = statusText(state),
                color = Color.White,
                fontWeight = FontWeight.Medium,
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

@Composable
private fun InstructionPanel(state: CaptureState, flowKind: FlowKind) {
    Surface(
        shape = RoundedCornerShape(24.dp),
        color = Color.Black.copy(alpha = 0.35f),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = instructionText(state, flowKind),
                color = Color.White,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                style = MaterialTheme.typography.bodyLarge,
            )
            if (state is CaptureState.Processing) {
                Spacer(modifier = Modifier.height(12.dp))
                CircularProgressIndicator(color = Color.White)
            }
        }
    }
}

@Composable
private fun FaceGuideOverlay(state: CaptureState, modifier: Modifier = Modifier) {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1.0f,
        targetValue = if (state is CaptureState.Searching) 1.03f else 1.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "pulseScale",
    )
    val borderColor = statusColor(state)

    Canvas(modifier = modifier) {
        val ovalWidth = size.width * 0.68f
        val ovalHeight = ovalWidth * 1.3f
        val center = Offset(size.width / 2f, size.height / 2f)

        // Nền tối phủ ngoài, khoét oval ở giữa (blend mode xoá vùng oval)
        drawRect(color = Color.Black.copy(alpha = 0.45f))
        drawOval(
            color = Color.Transparent,
            topLeft = Offset(center.x - ovalWidth / 2, center.y - ovalHeight / 2),
            size = androidx.compose.ui.geometry.Size(ovalWidth, ovalHeight),
            blendMode = BlendMode.Clear,
        )
        drawOval(
            color = borderColor,
            topLeft = Offset(
                center.x - (ovalWidth * pulseScale) / 2,
                center.y - (ovalHeight * pulseScale) / 2,
            ),
            size = androidx.compose.ui.geometry.Size(ovalWidth * pulseScale, ovalHeight * pulseScale),
            style = Stroke(width = 6f),
        )
    }
}

private fun statusColor(state: CaptureState): Color = when (state) {
    is CaptureState.Searching -> Color(0xFFFFA726)
    is CaptureState.Adjusting -> Color(0xFFFFEE58)
    is CaptureState.Ready, is CaptureState.Success -> Color(0xFF66BB6A)
    is CaptureState.Processing -> Color(0xFF42A5F5)
    is CaptureState.Failed -> Color(0xFFEF5350)
}

private fun statusText(state: CaptureState): String = when (state) {
    is CaptureState.Searching -> "Đang tìm khuôn mặt"
    is CaptureState.Adjusting -> "Đang căn chỉnh"
    is CaptureState.Ready -> "Đã sẵn sàng"
    is CaptureState.Processing -> "Đang xử lý"
    is CaptureState.Success -> "Thành công"
    is CaptureState.Failed -> "Thất bại"
}

private fun instructionText(state: CaptureState, flowKind: FlowKind): String = when (state) {
    is CaptureState.Searching -> "Đưa khuôn mặt vào khung hình"
    is CaptureState.Adjusting -> "Giữ yên và nhìn thẳng vào camera"
    is CaptureState.Ready -> "Giữ yên..."
    is CaptureState.Processing -> "Đang xác thực, vui lòng đợi trong giây lát"
    is CaptureState.Success -> if (flowKind == FlowKind.ENROLL) "Đăng ký thành công" else "Xác thực thành công"
    is CaptureState.Failed -> state.reason
}
