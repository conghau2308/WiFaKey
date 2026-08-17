// DeviceSigning.kt
//
// CHỈ hardware-backed (StrongBox Keymaster) theo đúng yêu cầu hiện tại.
// setIsStrongBoxBacked(true) sẽ THROW StrongBoxUnavailableException nếu
// thiết bị không có chip StrongBox riêng — cố ý KHÔNG catch rồi âm thầm
// fallback về software keystore, để tránh tự hạ cấp bảo mật mà không ai biết
// (đúng tinh thần "chưa dùng software, để sau" bạn đã chốt).
//
// LƯU Ý THỰC TẾ: không phải mọi thiết bị Android đều có StrongBox (chỉ các
// dòng flagship, Pixel 3+ trở lên...) — thiết bị không có sẽ không đăng ký
// được thiết bị tin cậy cho tới khi bạn bổ sung software-tier sau này.

package com.wifakey.capture

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.security.keystore.StrongBoxUnavailableException
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.PrivateKey
import java.security.PublicKey
import java.security.Signature

object DeviceSigning {
    private const val KEYSTORE_ALIAS = "com.wifakey.device.signingkey"
    private const val ANDROID_KEYSTORE = "AndroidKeyStore"

    class HardwareKeyUnavailableException(cause: Throwable) :
        Exception("Thiết bị không có StrongBox — chưa hỗ trợ software fallback theo yêu cầu hiện tại", cause)

    /// Sinh cặp khoá trong StrongBox (chỉ chạy 1 lần lúc đăng ký thiết bị mới).
    fun generateDeviceKeyPair(): PublicKey {
        val spec = KeyGenParameterSpec.Builder(
            KEYSTORE_ALIAS,
            KeyProperties.PURPOSE_SIGN,
        )
            .setAlgorithmParameterSpec(java.security.spec.ECGenParameterSpec("secp256r1"))
            .setDigests(KeyProperties.DIGEST_SHA256)
            .setIsStrongBoxBacked(true) // <- bắt buộc hardware-backed, không fallback
            .setUserAuthenticationRequired(false) // xác thực người dùng đã do biometric pipeline lo
            .build()

        return try {
            val generator = KeyPairGenerator.getInstance(
                KeyProperties.KEY_ALGORITHM_EC,
                ANDROID_KEYSTORE,
            )
            generator.initialize(spec)
            generator.generateKeyPair().public
        } catch (e: StrongBoxUnavailableException) {
            throw HardwareKeyUnavailableException(e)
        }
    }

    /// Ký digest (đã hash sẵn từ build_verify_signing_payload() bên core Rust
    /// — cùng định dạng trên mọi platform).
    fun sign(digest: ByteArray): ByteArray {
        val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
        val privateKey = keyStore.getKey(KEYSTORE_ALIAS, null) as PrivateKey

        val signature = Signature.getInstance("SHA256withECDSA")
        signature.initSign(privateKey)
        signature.update(digest)
        return signature.sign()
    }

    /// Lấy public key để gửi lên server lúc đăng ký thiết bị.
    fun getPublicKeyBytes(): ByteArray {
        val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
        val cert = keyStore.getCertificate(KEYSTORE_ALIAS)
        return cert.publicKey.encoded
    }
}
