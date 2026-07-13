$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $root '.wavest-mdt-server.pid'
if (Test-Path $pidFile) {
    $serverPid = [int](Get-Content -LiteralPath $pidFile -Raw)
    Stop-Process -Id $serverPid -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $pidFile -Force
}
$apiPidFile = Join-Path $root '.wavest-mdt-api.pid'
if (Test-Path $apiPidFile) {
    $apiPid = [int](Get-Content -LiteralPath $apiPidFile -Raw)
    Stop-Process -Id $apiPid -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $apiPidFile -Force
}
Write-Host 'WaveST-MDT local server and inference API stopped.' -ForegroundColor Green
