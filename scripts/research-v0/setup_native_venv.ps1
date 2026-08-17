param(
    [string]$PythonExe = "python"
)

Set-Location $PSScriptRoot

Write-Host "Creating .venv-native..."
& $PythonExe -m venv .venv-native
if (-not $?) { Write-Error "Failed to create venv"; exit 1 }

Write-Host "Activating venv..."
& .\.venv-native\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing packages..."
pip install -r requirements-native.txt
if (-not $?) { Write-Error "pip install failed"; exit 1 }

Write-Host ""
Write-Host "Verifying install..."
python -c "import onnxruntime; print('  onnxruntime', onnxruntime.__version__)"
python -c "import cv2; print('  opencv', cv2.__version__)"
python -c "import skimage; print('  scikit-image', skimage.__version__)"
python -c "import PySide6; from PySide6.QtWidgets import QApplication; print('  PySide6 OK')"
python -c "import websockets; print('  websockets', websockets.__version__)"
python -c "import requests; print('  requests', requests.__version__)"

Write-Host ""
Write-Host "Done. Activate with: .\.venv-native\Scripts\Activate.ps1"
