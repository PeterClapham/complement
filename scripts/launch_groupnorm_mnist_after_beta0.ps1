$repoDir = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repoDir "results\logs"
$statusLog = Join-Path $logDir "mnist_groupnorm_handoff.log"
$runLog = Join-Path $logDir "mnist_groupnorm_seed0_150ep.out.log"
$errorLog = Join-Path $logDir "mnist_groupnorm_seed0_150ep.err.log"
$completionArtifact = Join-Path $repoDir (
    "results\evaluations\beta_grids\smallnorb_seed0_batch128_80ep\" +
    "final-200ep-beta0\posterior_mixed_fraction_heatmap.png"
)

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Add-Content -Path $statusLog -Value "$(Get-Date -Format s) waiting for beta-zero completion artifact"

while (-not (Test-Path $completionArtifact)) {
    Start-Sleep -Seconds 60
}

Add-Content -Path $statusLog -Value "$(Get-Date -Format s) starting GroupNorm MNIST grid"
Set-Location $repoDir
& "$repoDir\.venv\Scripts\python.exe" -u scripts\run_grid.py --config configs\mnist_groupnorm_seed0_150ep.yaml 1>> $runLog 2>> $errorLog
$exitCode = $LASTEXITCODE
Add-Content -Path $statusLog -Value "$(Get-Date -Format s) finished GroupNorm MNIST grid with exit code $exitCode"
