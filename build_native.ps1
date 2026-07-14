Set-Location $PSScriptRoot

# ── 1. Ensure venv exists ─────────────────────────────────────────────────────
if (-not (Test-Path ".venv-native\Scripts\python.exe")) {
    Write-Host ">> Venv not found — running setup_native_venv.ps1 first..."
    & ".\setup_native_venv.ps1"
    if (-not $?) { Write-Error "Venv setup failed"; exit 1 }
}

# ── 2. Activate venv ──────────────────────────────────────────────────────────
& ".\.venv-native\Scripts\Activate.ps1"

# ── 3. Install / upgrade PyInstaller ─────────────────────────────────────────
Write-Host ">> Installing PyInstaller..."
pip install "pyinstaller>=6.0" --quiet
if (-not $?) { Write-Error "PyInstaller install failed"; exit 1 }

# ── 4. Clean previous build ───────────────────────────────────────────────────
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist\WiFaKeyAuth") { Remove-Item -Recurse -Force "dist\WiFaKeyAuth" }

# ── 5. Build ──────────────────────────────────────────────────────────────────
Write-Host ">> Building WiFaKeyAuth.exe (this may take several minutes)..."
pyinstaller WiFaKeyAuth.spec --noconfirm

if (-not $?) {
    Write-Error "PyInstaller build failed."
    exit 1
}

# ── 6. Verify output ──────────────────────────────────────────────────────────
$exe = "dist\WiFaKeyAuth\WiFaKeyAuth.exe"
if (Test-Path $exe) {
    $size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
    $dir  = [math]::Round((Get-ChildItem -Recurse "dist\WiFaKeyAuth" |
             Measure-Object -Property Length -Sum).Sum / 1MB, 0)
    Write-Host ""
    Write-Host "Build succeeded!"
    Write-Host "  Exe : $exe ($size MB)"
    Write-Host "  Dir : dist\WiFaKeyAuth\ ($dir MB total)"
    Write-Host ""
    Write-Host "Run:  .\dist\WiFaKeyAuth\WiFaKeyAuth.exe"
    Write-Host "      .\dist\WiFaKeyAuth\WiFaKeyAuth.exe --port 7825 --origin https://auth.yourdomain.com"
} else {
    Write-Error "Build finished but exe not found at $exe"
    exit 1
}
