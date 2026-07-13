$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = Split-Path -Parent $root
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    throw 'Python 3.10 or later was not found in PATH.'
}

$venv = Join-Path $root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    & $python -m venv --system-site-packages $venv
}
& $venvPython -m pip install --disable-pip-version-check -q -r (Join-Path $root 'server\requirements.txt')

$existing = Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue
if (-not $existing) {
    $env:WAVEST_WORKSPACE_ROOT = $workspace
    $env:WAVEST_CHECKPOINT = Join-Path $workspace 'results\nature_main\cytassist_rep2_radius55\checkpoint.pt'
    $env:WAVEST_DEVICE = 'cuda'
    $env:WAVEST_BATCH_SIZE = '4'
    $process = Start-Process -FilePath $venvPython `
        -ArgumentList '-m', 'uvicorn', 'server.app:app', '--host', '127.0.0.1', '--port', '8010' `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath (Join-Path $root '.wavest-mdt-api.pid') -Value $process.Id
    Start-Sleep -Seconds 2
}

Write-Host 'WaveST-MDT local inference API is running at http://127.0.0.1:8010/api/health' -ForegroundColor Green
