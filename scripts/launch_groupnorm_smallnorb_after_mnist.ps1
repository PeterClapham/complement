$repoDir = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repoDir "results\logs"
$statusLog = Join-Path $logDir "smallnorb_groupnorm_handoff.log"
$runLog = Join-Path $logDir "smallnorb_groupnorm_seed0_200ep.out.log"
$errorLog = Join-Path $logDir "smallnorb_groupnorm_seed0_200ep.err.log"
$completionArtifact = Join-Path $repoDir (
    "results\experiments\beta_grids\variational_gon_groupnorm_seed0_batch128_150ep\" +
    "mnist\seed-0\beta-inf-10p0__beta-opt-10p0\model.pt"
)

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Add-Content -Path $statusLog -Value "$(Get-Date -Format s) waiting for final GroupNorm MNIST model"

while (-not (Test-Path $completionArtifact)) {
    Start-Sleep -Seconds 60
}

Add-Content -Path $statusLog -Value "$(Get-Date -Format s) starting GroupNorm smallNORB grid"
Set-Location $repoDir
& "$repoDir\.venv\Scripts\python.exe" -u scripts\run_grid.py --config configs\smallnorb_groupnorm_seed0_200ep.yaml 1>> $runLog 2>> $errorLog
$exitCode = $LASTEXITCODE
Add-Content -Path $statusLog -Value "$(Get-Date -Format s) finished GroupNorm smallNORB grid with exit code $exitCode"
