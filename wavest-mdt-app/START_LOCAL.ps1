$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$node = (Get-Command node -ErrorAction SilentlyContinue).Source
if (-not $node) {
    throw 'Node.js was not found in PATH. Install Node.js 20 or later before starting WaveST-MDT.'
}

$existing = Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue
if (-not $existing) {
    $process = Start-Process -FilePath $node `
        -ArgumentList 'node_modules\vite\bin\vite.js', 'preview', '--host', '0.0.0.0', '--port', '4173' `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath (Join-Path $root '.wavest-mdt-server.pid') -Value $process.Id
    Start-Sleep -Seconds 2
}

Start-Process 'http://localhost:4173'
Write-Host 'WaveST-MDT is running at http://localhost:4173' -ForegroundColor Green
Write-Host 'LAN access uses this computer IP with port 4173.'
