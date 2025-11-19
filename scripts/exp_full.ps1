# ===============================================
# exp_full.ps1  (stable final, ASCII only)
# Example:
# powershell -ExecutionPolicy Bypass -File scripts\exp_full.ps1 `
#   -Root "." -Py "D:\Anaconda3\envs\chorus\python.exe" `
#   -Seeds 0,1,2,3,4 -Gpu 0 -Regenerate 0
# ===============================================

param(
  [string]$Root = ".",
  [string]$Py   = "python",
  [int[]] $Seeds = @(0),
  [int]   $Gpu   = 0,
  [int]   $Regenerate = 0,

  # Common hparams
  [int]$EmbSize = 64,
  [int]$NumWorkers = 0,

  # TopK
  [int]$EpochTopK = 50,
  [int]$EarlyStopTopK = 10,
  [int]$BatchTopK = 256,
  [int]$EvalBatchTopK = 1024,

  # CTR
  [int]$EpochCTR = 10,
  [int]$EarlyStopCTR = 2,
  [int]$BatchCTR = 512,
  [int]$EvalBatchCTR = 4096
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-Path($p, $desc) {
  if (-not (Test-Path -LiteralPath $p)) {
    throw ("Path not found for {0}: {1}" -f $desc, $p)
  }
}

# Normalize root to absolute path
$Root = (Resolve-Path -LiteralPath $Root).Path
Write-Host ("ROOT = {0}" -f $Root) -ForegroundColor Yellow
Write-Host ("PY   = {0}" -f $Py)   -ForegroundColor Yellow

# Basic checks
Assert-Path $Root "project root"
Assert-Path (Join-Path $Root "src\main.py") "src\main.py"
Assert-Path (Join-Path $Root "data") "data dir"

# Job lists
$TopKJobs = @(
  @{ Model = "BPRMF";          Dataset = "MovieLens_1M/ML_1MTOPK" }
  @{ Model = "BPRMF";          Dataset = "Grocery_and_Gourmet_Food/GGFTOPK" }
  @{ Model = "NeuMF";          Dataset = "MovieLens_1M/ML_1MTOPK" }
  @{ Model = "NeuMF";          Dataset = "Grocery_and_Gourmet_Food/GGFTOPK" }
  @{ Model = "FinalMLPReImpl"; Dataset = "MovieLens_1M/ML_1MTOPK" }
  @{ Model = "FinalMLPReImpl"; Dataset = "Grocery_and_Gourmet_Food/GGFTOPK" }
)

$CTRJobs = @(
  @{ Model = "FM";             Dataset = "MovieLens_1M/ML_1MCTR" }
  @{ Model = "FM";             Dataset = "Grocery_and_Gourmet_Food/GGFCTR" }
  @{ Model = "DeepFM";         Dataset = "MovieLens_1M/ML_1MCTR" }
  @{ Model = "DeepFM";         Dataset = "Grocery_and_Gourmet_Food/GGFCTR" }
  @{ Model = "FinalMLPReImpl"; Dataset = "MovieLens_1M/ML_1MCTR" }
  @{ Model = "FinalMLPReImpl"; Dataset = "Grocery_and_Gourmet_Food/GGFCTR" }
)

function Run-TopK {
  param([string]$Model, [string]$Dataset, [int]$Seed)

  Write-Host ("--> [TopK] {0} - {1} - seed={2}" -f $Model,$Dataset,$Seed) -ForegroundColor Cyan

  $args = @(
    (Join-Path $Root "src\main.py"),
    "--model_name", $Model,
    "--model_mode", "TopK",
    "--dataset",    $Dataset,
    "--batch_size", $BatchTopK,
    "--epoch",      $EpochTopK,
    "--early_stop", $EarlyStopTopK,
    "--emb_size",   $EmbSize,
    "--eval_batch_size", $EvalBatchTopK,
    "--num_workers", $NumWorkers,
    "--seed",        $Seed,
    "--regenerate",  $Regenerate,
    "--gpu",         $Gpu
  )

  & $Py @args
  if ($LASTEXITCODE -ne 0) {
    throw ("TopK run failed: {0} - {1} - seed={2}, exit={3}" -f $Model,$Dataset,$Seed,$LASTEXITCODE)
  }
}

function Run-CTR {
  param([string]$Model, [string]$Dataset, [int]$Seed)

  Write-Host ("--> [CTR ] {0} - {1} - seed={2}" -f $Model,$Dataset,$Seed) -ForegroundColor Green

  $args = @(
    (Join-Path $Root "src\main.py"),
    "--model_name", $Model,
    "--model_mode", "CTR",
    "--dataset",    $Dataset,
    "--metrics",    "AUC,LOG_LOSS",
    "--main_metric","AUC",
    "--batch_size", $BatchCTR,
    "--epoch",      $EpochCTR,
    "--early_stop", $EarlyStopCTR,
    "--emb_size",   $EmbSize,
    "--eval_batch_size", $EvalBatchCTR,
    "--num_workers", $NumWorkers,
    "--seed",        $Seed,
    "--loss_n",      "BCE",
    "--regenerate",  $Regenerate,
    "--gpu",         $Gpu
  )

  & $Py @args
  if ($LASTEXITCODE -ne 0) {
    throw ("CTR run failed: {0} - {1} - seed={2}, exit={3}" -f $Model,$Dataset,$Seed,$LASTEXITCODE)
  }
}

Push-Location $Root
try {
  foreach ($s in $Seeds) {
    foreach ($job in $TopKJobs) { Run-TopK -Model $job.Model -Dataset $job.Dataset -Seed $s }
  }
  foreach ($s in $Seeds) {
    foreach ($job in $CTRJobs)  { Run-CTR  -Model $job.Model -Dataset $job.Dataset -Seed $s }
  }
  Write-Host "[ALL DONE] all training jobs finished." -ForegroundColor Yellow
}
catch {
  Write-Host ("[ERROR] {0}" -f $_.Exception.Message) -ForegroundColor Red
  exit 1
}
finally {
  Pop-Location
}
