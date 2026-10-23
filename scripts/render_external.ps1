# Render the candidate raws with another program's command line, for scripts/parity_external.py to score against the camera JPEGs.
#   ./scripts/render_external.ps1 darktable            (darktable-cli, default settings: its scene-referred workflow)
#   ./scripts/render_external.ps1 rawtherapee [-Pp3 file.pp3 | -Pp3 auto-matched]   ('auto-matched' = RawTherapee's own Auto-Matched Curve profile for the picture's ISO: Low up to 400, Medium up to 1600, else High; no -Pp3: its default profile)
# Reads candidates/a7c/*.ARW (scripts/copy_candidates.py), writes candidates/renders/<program>/<stem>.jpg; the originals are never touched. darktable-cli wants forward slashes.
param([Parameter(Mandatory)][ValidateSet('darktable', 'rawtherapee')][string]$Program, [string]$Pp3 = '', [string]$Set = 'a7c')
$rtDir = 'C:/Program Files/RawTherapee/5.13'
$root = (Resolve-Path "$PSScriptRoot/..").Path -replace '\\', '/'; $out = "$root/candidates/renders/$Program" + $(if ($Pp3) { '-' + [IO.Path]::GetFileNameWithoutExtension($Pp3) } else { '' }); New-Item -ItemType Directory -Force $out | Out-Null
$stems = (Get-Content "$root/docs/${Set}_candidates.json" | ConvertFrom-Json).stem
foreach ($s in $stems) {
    $raw = "$root/candidates/$Set/$s.ARW"; if (-not (Test-Path $raw)) { Write-Host "missing $raw"; continue }
    if ($Program -eq 'darktable') { & "$env:LOCALAPPDATA/Programs/darktable/bin/darktable-cli.exe" $raw "$out/$s.jpg" --width 3000 --height 3000 --core --configdir "$root/candidates/dtconf" --library ':memory:' 2>&1 | Out-Null }
    else {
        $exe = (Get-Command rawtherapee-cli -ErrorAction SilentlyContinue).Source; if (-not $exe) { $exe = "$rtDir/rawtherapee-cli.exe" }; if (-not (Test-Path $exe)) { throw 'rawtherapee-cli is not installed' }
        $prof = $Pp3; if ($Pp3 -eq 'auto-matched') { $iso = [int](& "$root/.venv/Scripts/python.exe" -I -c "import sys, tifffile; print(tifffile.TiffFile(sys.argv[1]).pages[0].tags['ExifTag'].value['ISOSpeedRatings'])" $raw); $prof = "$rtDir/profiles/Auto-Matched Curve - ISO " + $(if ($iso -le 400) { 'Low' } elseif ($iso -le 1600) { 'Medium' } else { 'High' }) + '.pp3' }; $args = @('-o', "$out/$s.jpg", '-Y', '-j92'); if ($prof) { $args += @('-p', $prof) }; $args += @('-c', $raw); & $exe @args 2>&1 | Out-Null
    }
    Write-Host $s
}

