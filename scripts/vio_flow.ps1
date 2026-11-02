# Build the VIO-instrumented Lyra zone_isolation bitstream (Arty S7). Same
# paren-free-path workaround as scripts/vivado_flow.ps1.
param(
    [int]$Freq = 50,
    [string]$Part = "xc7s50csga324-1"
)

$vivadoBin = "D:\AMDDesignTools\2026.1\Vivado\bin"
$vivadoExe = Join-Path $vivadoBin "vivado.bat"

$junctionRoot = "D:\lyra_fpga"
$projectRoot = Split-Path -Parent $PSScriptRoot
$tclScript = Join-Path $junctionRoot "vivado\build_vio.tcl"
$logDir = Join-Path $projectRoot "build\vivado"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $junctionRoot "build\vivado\lyra_vio_build.log"
$logReal = Join-Path $logDir "lyra_vio_build.log"

Push-Location $vivadoBin
try {
    & $vivadoExe -mode batch -source $tclScript -tclargs $Freq $Part -log $log -nojournal
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Host "exit code: $exitCode  (log: $logReal)"
Select-String -Path $logReal -Pattern "ERROR|CRITICAL WARNING|DONE lyra_vio" | Select-Object -Last 40
