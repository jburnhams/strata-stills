# Build the stratastills environment from scratch on Windows (Python 3.13, NVIDIA GPU).
#   scripts\setup_env.ps1 [-Cuda cu128] [-WithVision] [-Report]
# Prerequisites: winget install Python.Python.3.13 OpenJS.NodeJS
# torch: scripts/setup_torch.py looks at the NVIDIA card and installs the newest build it can run (-Cuda cu128|cu126|cpu overrides it).
param([string]$Cuda = 'auto', [switch]$WithVision, [switch]$Report)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot; $Venv = Join-Path $Root '.venv'; $Py = Join-Path $Venv 'Scripts\python.exe'
function Say($m) { Write-Host "`n==> $m" }
Say "virtual environment: $Venv"
py -3.13 -m venv $Venv
& $Py -m pip install --upgrade pip | Out-Null
Say "torch ($Cuda)"
& $Py (Join-Path $PSScriptRoot 'setup_torch.py') --python $Py --cuda $Cuda
Say 'requirements'
& $Py -m pip install -r (Join-Path $Root 'requirements.txt')
Say 'learned denoise (Real-ESRGAN weights, 9 MB, into models/): spandrel without its dependencies, so that it cannot replace the torch just installed'
& $Py -m pip install einops safetensors
& $Py -m pip install --no-deps spandrel==0.4.2
$env:PYTHONPATH = Join-Path $Root 'src'; & $Py -c "from stratastills import denoise_ai; denoise_ai.fetch()"
if ($WithVision) {
  Say 'detector and denoising models'
  & $Py -m pip install -r (Join-Path $Root 'requirements-vision.txt')
  & $Py (Join-Path $PSScriptRoot 'setup_torch.py') --python $Py --cuda $Cuda      # the detector package pulls the CPU torchvision: the right one again
}
$env:PYTHONPATH = Join-Path $Root 'src'
Say 'checking'
& $Py -m stratastills doctor
if (Get-Command npm -ErrorAction SilentlyContinue) { Push-Location (Join-Path $Root 'web'); npm install --silent; npm run build; Pop-Location; Write-Host 'web app built' }
else { Write-Host 'npm not found: winget install OpenJS.NodeJS, then cd web; npm install; npm run build' }

if ($Report) {
    & $Py -c "import sys; sys.path.insert(0, r'$Root/src'); from stratastills import externaltools as E; E.require()"; if ($LASTEXITCODE) { exit 1 }      # darktable and RawTherapee are needed: the report is a comparison with them
    if (Test-Path (Join-Path $Root 'candidates')) { & $Py (Join-Path $Root 'scripts/build_report.py'); Write-Host 'report: out/report/index.html' }
    else { Write-Host 'no candidates folder: report skipped (scripts/copy_candidates.py)' }
}
