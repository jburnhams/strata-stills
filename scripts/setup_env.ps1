# Build the stratastills environment from scratch on Windows (Python 3.13, NVIDIA GPU).
#   scripts\setup_env.ps1 [-Cuda cu128] [-WithVision]
# Prerequisites: winget install Python.Python.3.13 OpenJS.NodeJS
# -Cuda is the PyTorch wheel index suffix (cu128 or newer is needed for RTX 50-series / Blackwell); -Cuda cpu installs the CPU build.
param([string]$Cuda = 'cu128', [switch]$WithVision)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot; $Venv = Join-Path $Root '.venv'; $Py = Join-Path $Venv 'Scripts\python.exe'
function Say($m) { Write-Host "`n==> $m" }
Say "virtual environment: $Venv"
py -3.13 -m venv $Venv
& $Py -m pip install --upgrade pip | Out-Null
Say "torch ($Cuda)"
& $Py -m pip install torch --index-url "https://download.pytorch.org/whl/$Cuda"
Say 'requirements'
& $Py -m pip install -r (Join-Path $Root 'requirements.txt')
if ($WithVision) {
  Say 'detector and denoising models'
  & $Py -m pip install -r (Join-Path $Root 'requirements-vision.txt')
  & $Py -m pip install --force-reinstall --no-deps torchvision --index-url "https://download.pytorch.org/whl/$Cuda"      # the detector package pulls the CPU torchvision
}
$env:PYTHONPATH = Join-Path $Root 'src'
Say 'checking'
& $Py -m stratastills doctor
if (Get-Command npm -ErrorAction SilentlyContinue) { Push-Location (Join-Path $Root 'web'); npm install --silent; npm run build; Pop-Location; Write-Host 'web app built' }
else { Write-Host 'npm not found: winget install OpenJS.NodeJS, then cd web; npm install; npm run build' }
