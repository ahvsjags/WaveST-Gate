$root = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $root 'START_WAVEST_MDT_BACKEND.ps1')
& (Join-Path $root 'START_LOCAL.ps1')
