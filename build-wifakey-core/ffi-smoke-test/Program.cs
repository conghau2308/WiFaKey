using System.Runtime.InteropServices;
using System.Text.Json;

[DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
static extern IntPtr wifakey_enroll(string inputJson);

[DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
static extern IntPtr wifakey_verify_prepare(string inputJson);

[DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
static extern void wifakey_free_string(IntPtr s);

static string CallNative(Func<string, IntPtr> nativeFn, string requestJson)
{
    IntPtr resultPtr = nativeFn(requestJson);
    string? result = Marshal.PtrToStringUTF8(resultPtr);
    wifakey_free_string(resultPtr);
    return result ?? throw new InvalidOperationException("native trả về null pointer");
}

Console.OutputEncoding = System.Text.Encoding.UTF8;

string baseDir = AppContext.BaseDirectory;
string mMatrixPath = Path.Combine(baseDir, "M_matrix.txt");
string gMatrixPath = Path.Combine(baseDir, "generator_matrix_G.txt");
string llrTablePath = Path.Combine(baseDir, "empirical_llr_table.txt");

var embedding = Enumerable.Repeat(0.5, 512).ToList();
var userSecret = Enumerable.Range(1, 20).ToList();
var serviceSalt = Enumerable.Range(1, 8).ToList();

// ---- 1. Enroll ----
var enrollRequest = new
{
    embedding,
    user_secret = userSecret,
    service_salt = serviceSalt,
    m_matrix_path = mMatrixPath,
    generator_matrix_g_path = gMatrixPath,
    empirical_llr_table_path = llrTablePath
};
string enrollResultJson = CallNative(wifakey_enroll, JsonSerializer.Serialize(enrollRequest));
Console.WriteLine("Enroll result: " + enrollResultJson);

using var enrollDoc = JsonDocument.Parse(enrollResultJson);
var enrollRoot = enrollDoc.RootElement;

if (enrollRoot.TryGetProperty("error", out var enrollError))
{
    Console.WriteLine($"Enroll lỗi: {enrollError.GetString()}");
    return;
}

// Lấy TRỰC TIẾP từ kết quả enroll, không copy tay
var helperData = enrollRoot.GetProperty("helper_data").EnumerateArray().Select(e => e.GetBoolean()).ToList();
var reliabilityMask = enrollRoot.GetProperty("reliability_mask").EnumerateArray().Select(e => e.GetBoolean()).ToList();

// ---- 2. Verify (dùng lại chính output vừa enroll) ----
var verifyRequest = new
{
    embedding, // cùng embedding -> mô phỏng verify đúng người
    user_secret = userSecret, // PHẢI giống hệt user_secret lúc enroll
    helper_data = helperData,
    reliability_mask = reliabilityMask,
    service_salt = serviceSalt,
    m_matrix_path = mMatrixPath,
    generator_matrix_g_path = gMatrixPath,
    empirical_llr_table_path = llrTablePath
};
string verifyResultJson = CallNative(wifakey_verify_prepare, JsonSerializer.Serialize(verifyRequest));

using var verifyDoc = JsonDocument.Parse(verifyResultJson);
var verifyRoot = verifyDoc.RootElement;

if (verifyRoot.TryGetProperty("error", out var verifyError))
{
    Console.WriteLine($"Verify lỗi: {verifyError.GetString()}");
    return;
}

var llr = verifyRoot.GetProperty("llr").EnumerateArray().Select(e => e.GetDouble()).ToList();
Console.WriteLine($"Verify OK — llr có {llr.Count} phần tử (kỳ vọng 832, khớp helper_data.Count = {helperData.Count})");