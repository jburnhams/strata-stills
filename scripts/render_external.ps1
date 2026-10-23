# Render the candidate raws with another program's command line, for scripts/parity_external.py to score against the camera JPEGs.
#   ./scripts/render_external.ps1 darktable            (darktable-cli, default settings: its scene-referred workflow)
#   ./scripts/render_external.ps1 rawtherapee [-Pp3 file.pp3]
# Reads candidates/a7c/*.ARW (scripts/copy_candidates.py), writes candidates/renders/<program>/<stem>.jpg; the originals are never touched. darktable-cli wants forward slashes.
param([Parameter(Mandatory)][ValidateSet('darktable', 'rawtherapee')][string]$Program, [string]$Pp3 = '', [string]$Set = 'a7c')
$root = (Resolve-Path "$PSScriptRoot/..").Path -replace '\\', '/'; $out = "$root/candidates/renders/$Program" + $(if ($Pp3) { '-' + [IO.Path]::GetFileNameWithoutExtension($Pp3) } else { '' }); New-Item -ItemType Directory -Force $out | Out-Null
$stems = (Get-Content "$root/docs/${Set}_candidates.json" | ConvertFrom-Json).stem
foreach ($s in $stems) {
    $raw = "$root/candidates/$Set/$s.ARW"; if (-not (Test-Path $raw)) { Write-Host "missing $raw"; continue }
    if ($Program -eq 'darktable') { & "$env:LOCALAPPDATA/Programs/darktable/bin/darktable-cli.exe" $raw "$out/$s.jpg" --width 3000 --height 3000 --core --configdir "$root/candidates/dtconf" --library ':memory:' 2>&1 | Out-Null }
    else {
        $exe = (Get-Command rawtherapee-cli -ErrorAction SilentlyContinue).Source; if (-not $exe) { throw 'rawtherapee-cli is not installed' }
        $args = @('-o', "$out/$s.jpg", '-q', '-Y'); if ($Pp3) { $args += @('-p', $Pp3) }; $args += @('-c', $raw); & $exe @args 2>&1 | Out-Null
    }
    Write-Host $s
}
