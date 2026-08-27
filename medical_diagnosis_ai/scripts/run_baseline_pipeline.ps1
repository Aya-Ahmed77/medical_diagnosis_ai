<#
Baseline-only pipeline wrapper (PowerShell).

Behavior (required sequence):
 1) Verify being executed from the project root (or locate project root automatically).
 2) Verify Python and baseline dependencies (requests, pymongo, scikit-learn, joblib, bs4, numpy, pandas, Flask optionally).
 3) Verify .env configuration contains MONGO_URI.
 4) Verify MongoDB connectivity (ping) — performed at runtime by the script.
 5) Run the real NHS scraper (scripts/run_scraper.py) unless skipped.
 6) Run the real preprocessing pipeline (scripts/run_preprocessing.py).
 7) Train the REAL TF-IDF + Logistic Regression baseline (scripts/train_baseline.py).
 8) The training script persists model artifact and metadata via the project's persistence.
 9) Report the actual evaluation metrics printed by the training script and by reading the Models collection.

Command-line options:
  -ScrapeLimit <int>    : limit passed to scraper (default 50)
  -InstallDeps          : install only baseline deps (does NOT install TF / PyTorch)
  -SkipScrape           : skip scraping step (assumes conditions already in MongoDB)

Important constraints enforced by the wrapper:
 - Does NOT install TensorFlow or PyTorch.
 - Does NOT run RNN/LSTM/Transformer.
 - Stops immediately on first error with non-zero exit code.
 - Uses existing project scripts and modules; does not fake results.
#>
param(
    [int]$ScrapeLimit = 50,
    [switch]$InstallDeps,
    [switch]$SkipScrape
)

Set-StrictMode -Version Latest

# Determine project root: prefer current directory if it looks like the project root,
# otherwise use the script location's parent (scripts/ is expected to live under project root).
$cwdIsRoot = (Test-Path (Join-Path (Get-Location) 'app') -and Test-Path (Join-Path (Get-Location) 'scripts'))
if ($cwdIsRoot) {
    $projectRoot = Resolve-Path . | Select-Object -ExpandProperty Path
} else {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
    $projectRoot = Resolve-Path (Join-Path $scriptDir '..') | Select-Object -ExpandProperty Path
}
Write-Host "Project root resolved to: $projectRoot"
Push-Location $projectRoot

function Fail($msg, $code=1) {
    Write-Error $msg
    Pop-Location
    exit $code
}

function Run-Cmd($cmd, $args = '') {
    Write-Host ">> $cmd $args"
    & $cmd $args
    if ($LASTEXITCODE -ne 0) { Fail("Command failed: $cmd $args", $LASTEXITCODE) }
}

# 1) Verify python exists
try {
    & python --version > $null 2>&1
} catch {
    Fail("Python is not available on PATH. Install Python 3.10+ and re-run.")
}

# 2) Install baseline dependencies if requested (explicit, lightweight set)
if ($InstallDeps) {
    Write-Host "Installing baseline dependencies (no TF / PyTorch)..."
    Run-Cmd pip "install --upgrade pip"
    $pkgs = @(
        'requests==2.32.3',
        'beautifulsoup4==4.12.3',
        'pymongo==4.8.0',
        'pandas==2.2.2',
        'numpy==1.26.4',
        'scikit-learn==1.5.1',
        'joblib==1.4.2',
        'python-dotenv==1.0.1',
        'Flask==3.0.3'
    )
    foreach ($p in $pkgs) { Run-Cmd pip "install $p" }
} else {
    Write-Host "Skipping automatic dependency installation. Ensure baseline dependencies are installed." 
}

# 3) Verify .env exists and contains MONGO_URI
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item .env.example .env -Force
        Write-Host ".env created from .env.example. Please edit .env now to set MONGO_URI and other values, then re-run the script."
        Fail(".env was missing; created from .env.example. Edit .env and re-run.")
    } else {
        Fail(".env missing and .env.example not found. Cannot proceed.")
    }
}

# Read .env and ensure MONGO_URI appears non-empty
$envLines = Get-Content .env | Where-Object { $_ -and ($_ -match '\S') }
$mongoLine = $envLines | Where-Object { $_ -match '^[\s#]*MONGO_URI\s*=' }
if (-not $mongoLine) { Fail("MONGO_URI not set in .env. Please set MONGO_URI and re-run.") }
$mongoVal = $mongoLine -replace '^[\s#]*MONGO_URI\s*=\s*',''
if (-not $mongoVal) { Fail("MONGO_URI appears empty in .env. Please set a valid URI and re-run.") }
Write-Host "Found MONGO_URI in .env (masked): $($mongoVal.Substring(0,[Math]::Min(20,$mongoVal.Length)))..."

# 4) Verify baseline Python packages are importable (lightweight check)
$checkModules = @('requests','bs4','pymongo','sklearn','joblib','numpy','pandas')
$missing = @()
foreach ($m in $checkModules) {
    try {
        # Use Python to check import without executing project code
        $cmd = "python -c ""import importlib,sys; sys.exit(0 if importlib.util.find_spec('$m') else 1)"""
        iex $cmd
        if ($LASTEXITCODE -ne 0) { $missing += $m }
    } catch {
        $missing += $m
    }
}
if ($missing.Count -gt 0) {
    Fail("Missing required Python packages: $($missing -join ', '). Install them (use -InstallDeps) and re-run.")
}
Write-Host "Baseline Python packages present."

# 5) Verify required scripts exist
$requiredScripts = @('scripts\run_scraper.py','scripts\run_preprocessing.py','scripts\train_baseline.py')
foreach ($s in $requiredScripts) {
    if (-not (Test-Path $s)) { Fail("Required script not found: $s") }
}

# 6) Verify important app modules exist (static check)
$requiredModules = @('app\services\scraping_service.py','app\services\training_service.py','app\database\schemas.py','app\database\gridfs_store.py')
foreach ($m in $requiredModules) { if (-not (Test-Path $m)) { Fail("Required module file missing: $m") } }

Write-Host "All required scripts and module files found."

# 7) Verify MongoDB connectivity (runtime check). This invokes app code to ping.
Write-Host "Checking MongoDB connectivity via app.database.connection.check_connection()..."
$pingScript = @'
from app.database.connection import check_connection
import sys
ok = check_connection()
print('OK' if ok else 'NOTOK')
if not ok:
    sys.exit(2)
'@
python - <<PY
$pingScript
PY
if ($LASTEXITCODE -ne 0) { Fail('MongoDB connectivity check failed. Ensure MongoDB is running and MONGO_URI in .env is correct.') }
Write-Host "MongoDB connectivity OK."

# 8) Run scraper unless skipped
if ($SkipScrape) { Write-Host "Skipping scraper as requested (-SkipScrape)." } else {
    Write-Host "Running real NHS scraper (this will make network requests). Limit: $ScrapeLimit"
    Run-Cmd python "scripts/run_scraper.py --limit $ScrapeLimit"
}

# 9) Run preprocessing
Write-Host "Running preprocessing pipeline (build dataset, fit pipeline, save artifacts)"
Run-Cmd python "scripts/run_preprocessing.py"

# 10) Train baseline TF-IDF + Logistic Regression
Write-Host "Training baseline TF-IDF + Logistic Regression (this will persist model to GridFS)."
$trainOutput = & python scripts/train_baseline.py 2>&1
if ($LASTEXITCODE -ne 0) { Fail('Baseline training failed. See output below:`n' + $trainOutput) }
Write-Host "Baseline training output:`n$trainOutput"

# 11) Parse training output JSON if present and report metrics
try {
    $json = $trainOutput | Out-String
    $parsed = python - <<PY
import sys, json
s = sys.stdin.read()
try:
    j = json.loads(s)
    print(json.dumps(j, indent=2))
    sys.exit(0)
except Exception:
    # training might have printed additional text; attempt to extract last JSON object
    import re
    m = re.search(r'\{.*\}\s*$', s, re.DOTALL)
    if m:
        j = json.loads(m.group(0))
        print(json.dumps(j, indent=2))
        sys.exit(0)
    print('NO_JSON')
    sys.exit(3)
PY
} catch {
    Write-Host "Could not parse training output as JSON. The training script printed: `n$trainOutput"
}

# 12) Query Models collection for the saved baseline model and print stored metrics
Write-Host "Fetching stored baseline model metadata from Models collection..."
python - <<PY
from app.database.schemas import ModelsRepository
import json
repo = ModelsRepository()
models = repo.list_all()
baseline = [m for m in models if m.get('type')=='tfidf_logreg']
print(json.dumps({'found': len(baseline), 'models': baseline}, indent=2))
if not baseline:
    raise SystemExit(4)
PY
if ($LASTEXITCODE -ne 0) { Fail('Could not find baseline model metadata in Models collection after training.') }

Write-Host "Baseline pipeline completed successfully. Metrics and model metadata printed above."

Pop-Location
exit 0
