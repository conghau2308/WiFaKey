//! Hỗ trợ luồng chữ ký thiết bị đã bàn ở các câu trả lời trước.
//!
//! QUAN TRỌNG: core này KHÔNG BAO GIỜ tự ký và KHÔNG BAO GIỜ chạm vào
//! private key của thiết bị. Private key sống trong Secure Enclave
//! (iOS/macOS) / StrongBox Keymaster (Android) / TPM qua CNG (Windows) —
//! chỉ shell (Swift/Kotlin/C#) mới có quyền gọi API hệ điều hành để ký,
//! core chỉ dựng ĐÚNG chuỗi byte cần ký (canonical message), để đảm bảo
//! shell trên mọi platform ký cùng 1 định dạng, tránh lệch mismatch.
//!
//! CHỈ HỖ TRỢ HARDWARE-BACKED THEO YÊU CẦU HIỆN TẠI: không có module
//! software-signing nào ở đây. Khi bạn nghiên cứu xong phương án software
//! (cho Linux desktop), thêm module riêng `device_signature_software.rs`,
//! đừng sửa file này — giữ 2 đường tách biệt rõ ràng để không lẫn lộn mức
//! độ tin cậy giữa 2 loại (như đã bàn: software-tier không được dùng để
//! bỏ qua bước sinh trắc học, phải luôn phân biệt được ở tầng server).

use sha2::{Digest, Sha256};

/// Dựng canonical message để shell ký, dùng cho luồng Verify (challenge-response
/// đã thiết kế ở các câu trước): msg = user_id || nonce || SHA256(K').
///
/// Trả về bytes đã hash sẵn (SHA256 của message ghép) — hầu hết API ký
/// hardware-backed (SecKeyCreateSignature, Android Keystore Signature,
/// NCryptSignHash) nhận digest sẵn thay vì message thô, nên hash ở đây cho
/// tiện dùng thẳng.
pub fn build_verify_signing_payload(user_id: &str, nonce: &[u8], k_prime: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(user_id.as_bytes());
    hasher.update(nonce);
    hasher.update(Sha256::digest(k_prime));
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Dựng canonical message cho lúc đăng ký thiết bị mới (enroll device key,
/// khác với enroll biometric) — msg = user_id || device_pubkey, để server
/// xác nhận đúng public key này thuộc đúng user lúc thêm vào danh sách
/// thiết bị tin cậy (đã bàn ở phần "onboard thiết bị mới").
pub fn build_device_registration_payload(user_id: &str, device_pubkey: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(user_id.as_bytes());
    hasher.update(device_pubkey);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_inputs_give_same_payload() {
        let a = build_verify_signing_payload("user-1", b"nonce-abc", b"k-prime-bytes");
        let b = build_verify_signing_payload("user-1", b"nonce-abc", b"k-prime-bytes");
        assert_eq!(a, b);
    }

    #[test]
    fn different_nonce_gives_different_payload() {
        // Đúng tính chất chống replay: nonce khác -> payload ký khác ->
        // signature cũ không tái sử dụng được
        let a = build_verify_signing_payload("user-1", b"nonce-abc", b"k-prime-bytes");
        let b = build_verify_signing_payload("user-1", b"nonce-xyz", b"k-prime-bytes");
        assert_ne!(a, b);
    }
}
