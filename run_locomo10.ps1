$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot "venv\Scripts\python.exe"
$envFile = Join-Path $repoRoot ".env"
$dataset = Join-Path $repoRoot "data\locomo10.json"

Write-Host "[1/5] Switching to repo root: $repoRoot"

if (-not (Test-Path $pythonExe)) {
    throw "Python virtual environment not found: $pythonExe"
}

if (-not (Test-Path $envFile)) {
    throw ".env not found. Create it from .env.example and set OPENAI_API_KEY first."
}

if (-not (Test-Path $dataset)) {
    throw "Dataset not found: $dataset"
}

$env:PYTHONUNBUFFERED = "1"

Write-Host "[2/5] Found virtual environment: $pythonExe"
Write-Host "[3/5] Found .env and dataset"
Write-Host "[4/5] Loading Python dependencies and embedding model cache"
Write-Host "      First run can pause here for a while. Do not interrupt unless it fully hangs."
Write-Host "[5/5] Starting LoCoMo full evaluation"

& $pythonExe -u test_fixed_memory.py `
    --dataset $dataset `
    --sample 0 1 2 3 4 5 6 7 8 9 `
    --max-questions 999 `
    --category-to-test 1,2,3,4,5 `
    --model gpt-4o-mini `
    --embedding-model minilm
