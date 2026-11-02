# Vivado implementation flow for the Lyra zone_isolation SECDED core (Arty S7).
# Run from Windows PowerShell, any cwd. Usage:
#   scripts\vivado_flow.ps1 -Freq 100 -Part xc7s50csga324-1
#
# Same parenthesized-path workaround as Project-08's scripts/vivado_flow.ps1:
# vivado.bat's setupEnv.bat breaks on "(Lyra-FPGA)" in the path the same way
# it broke on "(FPGA)" there. Fix: launch from the paren-free Vivado bin dir,
# pass only paren-free argument paths (D:\lyra_fpga junction), let build_vivado.tcl
# cd back into the real project root itself (Tcl's cd tolerates parens fine).
param(
    [int]$Freq = 100,
    [string]$Part = "xc7s50csga324-1"
)

$vivadoBin = "D:\AMDDesignTools\2026.1\Vivado\bin"
$vivadoExe = Join-Path $vivadoBin "vivado.bat"

$junctionRoot = "D:\lyra_fpga"
$projectRoot = Split-Path -Parent $PSScriptRoot
$tclScript = Join-Path $junctionRoot "vivado\build_vivado.tcl"
$tag = "lyra_zone_isolation_secded"
$logDir = Join-Path $projectRoot "build\vivado"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $junctionRoot "build\vivado\${tag}_vivado.log"
$logReal = Join-Path $logDir "${tag}_vivado.log"

Push-Location $vivadoBin
try {
    & $vivadoExe -mode batch -source $tclScript -tclargs $Freq $Part -log $log -nojournal
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Host "exit code: $exitCode  (log: $logReal)"
Select-String -Path $logReal -Pattern "WNS|Slack|LUT|FF |DONE tag" | Select-Object -Last 20
