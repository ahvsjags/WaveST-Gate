$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $root '.wavest-mdt-server.pid'
if (Test-Path $pidFile) {
    $serverPid = [int](Get-Content -LiteralPath $pidFile -Raw)
    Stop-Process -Id $serverPid -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $pidFile -Force
}
Write-Host 'WaveST-MDT local server stopped.' -ForegroundColor Green
