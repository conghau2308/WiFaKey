using System.Runtime.InteropServices;

namespace WifakeyApp;

internal static class WifakeyCoreInterop
{
    [DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr wifakey_enroll(string inputJson);

    [DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr wifakey_verify_prepare(string inputJson);

    [DllImport("wifakey_core.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void wifakey_free_string(IntPtr s);

    private static string CallNative(Func<string, IntPtr> nativeFn, string requestJson)
    {
        IntPtr resultPtr = nativeFn(requestJson);
        string? result = Marshal.PtrToStringUTF8(resultPtr);
        wifakey_free_string(resultPtr);
        return result ?? throw new InvalidOperationException("native trả về null pointer");
    }

    public static string Enroll(string requestJson) => CallNative(wifakey_enroll, requestJson);
    public static string VerifyPrepare(string requestJson) => CallNative(wifakey_verify_prepare, requestJson);
}