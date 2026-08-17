using System.Runtime.InteropServices;

[DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
static extern IntPtr wifakey_enroll(string inputJson);

[DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
static extern void wifakey_free_string(IntPtr s);

Console.OutputEncoding = System.Text.Encoding.UTF8;

// 512 phần tử, đúng bằng test Rust (vec![0.5; 512]) để so sánh chéo được
var embedding = Enumerable.Repeat(0.5, 512).ToList();

var request = new
{
    embedding = embedding,
    user_secret = Enumerable.Range(1, 20).ToList(),
    service_salt = Enumerable.Range(1, 8).ToList(),
    m_matrix_path = "M_matrix.txt",
    generator_matrix_g_path = "generator_matrix_G.txt",
    empirical_llr_table_path = "empirical_llr_table.txt"
};
string requestJson = System.Text.Json.JsonSerializer.Serialize(request);

IntPtr resultPtr = wifakey_enroll(requestJson);
string? result = Marshal.PtrToStringUTF8(resultPtr);
wifakey_free_string(resultPtr);

Console.WriteLine(result);