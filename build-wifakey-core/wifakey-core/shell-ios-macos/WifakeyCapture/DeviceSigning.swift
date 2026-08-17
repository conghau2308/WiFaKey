// DeviceSigning.swift
//
// CHỈ hardware-backed (Secure Enclave) theo đúng yêu cầu hiện tại — không có
// software fallback trong file này. Nếu Secure Enclave không khả dụng
// (thiết bị quá cũ, hoặc chạy trên Simulator), hàm sẽ throw lỗi thay vì âm
// thầm rơi về phương án yếu hơn — tránh tự động hạ cấp bảo mật mà không ai
// biết.

import Security
import Foundation

enum DeviceSigningError: Error {
    case secureEnclaveUnavailable
    case keyGenerationFailed(OSStatus)
    case signingFailed(OSStatus)
}

enum DeviceSigning {
    private static let keyTag = "com.wifakey.device.signingkey".data(using: .utf8)!

    /// Sinh cặp khoá trong Secure Enclave (chỉ chạy 1 lần lúc đăng ký thiết
    /// bị mới — bước "Enroll (bổ sung)" đã bàn ở câu trả lời trước).
    /// Private key KHÔNG BAO GIỜ rời khỏi Secure Enclave — kể cả code này
    /// cũng không đọc được nó, chỉ dùng được thông qua SecKey reference.
    static func generateDeviceKeyPair() throws -> SecKey {
        let access = SecAccessControlCreateWithFlags(
            nil,
            kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            .privateKeyUsage,
            nil
        )!

        let attributes: [String: Any] = [
            kSecAttrKeyType as String: kSecAttrKeyTypeECSECPrimeRandom,
            kSecAttrKeySizeInBits as String: 256,
            kSecAttrTokenID as String: kSecAttrTokenIDSecureEnclave, // <- bắt buộc hardware-backed
            kSecPrivateKeyAttrs as String: [
                kSecAttrIsPermanent as String: true,
                kSecAttrApplicationTag as String: keyTag,
                kSecAttrAccessControl as String: access,
            ],
        ]

        var error: Unmanaged<CFError>?
        guard let privateKey = SecKeyCreateRandomKey(attributes as CFDictionary, &error) else {
            throw DeviceSigningError.keyGenerationFailed(errSecParam)
        }
        return privateKey
    }

    /// Ký payload (đã hash sẵn — dùng build_verify_signing_payload() từ
    /// core Rust để đảm bảo cùng định dạng trên mọi platform).
    static func sign(digest: [UInt8], using privateKey: SecKey) throws -> Data {
        var error: Unmanaged<CFError>?
        let signature = SecKeyCreateSignature(
            privateKey,
            .ecdsaSignatureDigestX962SHA256,
            Data(digest) as CFData,
            &error
        )
        guard let signature else {
            throw DeviceSigningError.signingFailed(errSecParam)
        }
        return signature as Data
    }

    /// Lấy public key dạng bytes để gửi lên server lúc đăng ký thiết bị
    /// (server lưu, dùng để verify chữ ký các lần sau — KHÔNG cần bí mật).
    static func publicKeyBytes(from privateKey: SecKey) throws -> Data {
        guard let publicKey = SecKeyCopyPublicKey(privateKey) else {
            throw DeviceSigningError.secureEnclaveUnavailable
        }
        var error: Unmanaged<CFError>?
        guard let data = SecKeyCopyExternalRepresentation(publicKey, &error) else {
            throw DeviceSigningError.secureEnclaveUnavailable
        }
        return data as Data
    }
}
