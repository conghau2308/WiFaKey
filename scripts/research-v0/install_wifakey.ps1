#Requires -Version 5.1
<#
.SYNOPSIS
  Cài đặt WiFaKey Auth vào hệ thống.
  Chạy script này MỘT LẦN sau khi build xong.

.USAGE
  PowerShell -ExecutionPolicy Bypass -File install_wifakey.ps1
#>

Set-Location $PSScriptRoot

$src  = Join-Path $PSScriptRoot "dist\WiFaKeyAuth"
$dest = Join-Path $env:LOCALAPPDATA "WiFaKeyAuth"
$exe  = Join-Path $dest "WiFaKeyAuth.exe"
$regKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"

# ── 1. Kiểm tra build tồn tại ─────────────────────────────────────────────────
if (-not (Test-Path "$src\WiFaKeyAuth.exe")) {
    Write-Error "Chưa build! Chạy .\build_native.ps1 trước."
    exit 1
}

# ── 2. Copy vào LocalAppData ──────────────────────────────────────────────────
Write-Host ">> Cài đặt vào $dest ..."
if (Test-Path $dest) {
    # Dừng tiến trình cũ nếu đang chạy
    Get-Process "WiFaKeyAuth" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 800
    Remove-Item -Recurse -Force $dest
}
Copy-Item -Recurse $src $dest
Write-Host "   OK — $([math]::Round((Get-ChildItem -Recurse $dest | Measure-Object -Property Length -Sum).Sum/1MB, 0)) MB"

# ── 3. Đăng ký Windows startup ────────────────────────────────────────────────
Write-Host ">> Thêm vào khởi động cùng Windows..."
Set-ItemProperty -Path $regKey -Name "WiFaKeyAuth" -Value "`"$exe`""
Write-Host "   OK"

# ── 4. Khởi động ngay ────────────────────────────────────────────────────────
Write-Host ">> Khởi động WiFaKey Auth..."
Start-Process $exe

Write-Host ""
Write-Host "Cài đặt hoàn tất!"
Write-Host "  - App chạy ngầm, icon xuất hiện ở system tray."
Write-Host "  - Cài đặt: $dest"
Write-Host "  - Khởi động cùng Windows: Có (có thể tắt trong tray menu)"
