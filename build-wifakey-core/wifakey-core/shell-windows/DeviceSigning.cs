// DeviceSigning.cs
//
// CHỈ hardware-backed qua TPM (Microsoft Platform Crypto Provider) theo
// đúng yêu cầu hiện tại — không có software fallback. Nếu máy không có TPM
// 2.0 hoạt động đúng chuẩn, việc tạo key sẽ throw exception thay vì âm thầm
// rơi về Microsoft Software Key Storage Provider (đây chính là điểm khác
// biệt quan trọng nhất — 2 provider trông giống hệt nhau ở API, chỉ khác
// tên chuỗi provider, nên rất dễ vô tình dùng nhầm software provider nếu
// không cẩn thận, đã lưu ý ở lần bàn trước về việc Windows có TPM không
// đồng đều giữa các máy).
//
// CHƯA BUILD/TEST trong sandbox này — cần Windows + .NET để build.

using System;
using System.Security.Cryptography;

namespace Wifakey.Capture
{
    public static class DeviceSigning
    {
        private const string KeyName = "wifakey-device-signing-key";

        // Chuỗi provider CHÍNH XÁC này là điểm mấu chốt đảm bảo hardware-backed:
        private const string HardwareProviderName = "Microsoft Platform Crypto Provider";

        public class HardwareKeyUnavailableException : Exception
        {
            public HardwareKeyUnavailableException(Exception inner)
                : base("Máy không có TPM 2.0 khả dụng — chưa hỗ trợ software fallback theo yêu cầu hiện tại", inner) { }
        }

        /// Sinh cặp khoá trong TPM (chỉ chạy 1 lần lúc đăng ký thiết bị mới).
        public static CngKey GenerateDeviceKeyPair()
        {
            var keyParams = new CngKeyCreationParameters
            {
                Provider = new CngProvider(HardwareProviderName), // <- bắt buộc TPM, KHÔNG dùng "Microsoft Software Key Storage Provider"
                KeyUsage = CngKeyUsages.Signing,
                ExportPolicy = CngExportPolicies.None, // private key không được export dưới bất kỳ hình thức nào
            };
            keyParams.Parameters.Add(
                new CngProperty("Length", BitConverter.GetBytes(256), CngPropertyOptions.None));

            try
            {
                return CngKey.Create(CngAlgorithm.ECDsaP256, KeyName, keyParams);
            }
            catch (CryptographicException e)
            {
                throw new HardwareKeyUnavailableException(e);
            }
        }

        /// Ký digest (đã hash sẵn từ build_verify_signing_payload() bên core
        /// Rust — cùng định dạng trên mọi platform).
        public static byte[] Sign(byte[] digest)
        {
            using var key = CngKey.Open(KeyName, new CngProvider(HardwareProviderName));
            using var ecdsa = new ECDsaCng(key);
            return ecdsa.SignHash(digest);
        }

        /// Lấy public key để gửi lên server lúc đăng ký thiết bị.
        public static byte[] GetPublicKeyBytes()
        {
            using var key = CngKey.Open(KeyName, new CngProvider(HardwareProviderName));
            using var ecdsa = new ECDsaCng(key);
            return ecdsa.ExportSubjectPublicKeyInfo();
        }
    }
}
