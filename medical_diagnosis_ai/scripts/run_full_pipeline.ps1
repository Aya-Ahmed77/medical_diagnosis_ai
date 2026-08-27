<#
Full local pipeline script (PowerShell).
Usage examples:
  # Quick smoke-run (install minimal deps manually first):
  ./run_full_pipeline.ps1 -ScrapeLimit 20 -BaselineOnly

  # Full run including deep learning (may take long, requires TF/PyTorch):
  ./run_full_pipeline.ps1 -ScrapeLimit 200 -InstallDeps -TrainDL -TrainTransformer -StartFlask

Notes:
 - Run this from the scripts/ folder (the script computes project root automatically).
 - This script attempts to be helpful with checks, but installing full requirements may take a long time.
 - MongoDB must be running and reachable via MONGO_URI in .env (see README).
 - Transformer training requires network access to the HuggingFace Hub and significant compute.
#>

param(
    [int]$ScrapeLimit = 50,
    [switch]$InstallDeps,             # Install baseline (lightweight) dependencies only
    [switch]$InstallDL,               # Install TensorFlow and related DL deps (heavy)
    [switch]$InstallTransformer,      # Install torch + transformers (heavy)
    [switch]$BaselineOnly,
    [switch]$TrainDL,
    [switch]$TrainTransformer,
    [switch]$StartFlask
)

Set-StrictMode -Version Latest
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $scriptDir
$projectRoot = Resolve-Path ".." | Select-Object -ExpandProperty Path
Set-Location $projectRoot
Write-Host "Project root: $projectRoot"

function Run-Command($cmd) {
    Write-Host "-> $cmd"
    & cmd /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
}

# 1) Virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
}
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
if ($InstallDeps) {
    Write-Host "Installing baseline (lightweight) dependencies only. This excludes heavy DL packages (TensorFlow/PyTorch) by default."
    pip install --upgrade pip
    # Baseline packages required to run scraper, preprocessing and baseline model
    $baselinePkgs = @(
        'Flask==3.0.3',
        'python-dotenv==1.0.1',
        'pymongo==4.8.0',
        'requests==2.32.3',
        'beautifulsoup4==4.12.3',
        'pandas==2.2.2',
        'numpy==1.26.4',
        'scikit-learn==1.5.1',
        'joblib==1.4.2',
        'pytest==8.3.2',
        'mongomock==4.1.2'
    )
    foreach ($p in $baselinePkgs) { pip install $p }

    if ($InstallDL) {
        Write-Host "Installing TensorFlow (RNN/LSTM) as requested (may be large)."
        pip install tensorflow==2.17.0
    } else {
        Write-Host "TensorFlow not requested. To install DL deps pass -InstallDL."
    }

    if ($InstallTransformer) {
        Write-Host "Installing PyTorch + Transformers (heavy)."
        pip install torch==2.4.0 transformers==4.44.2
    } else {
        Write-Host "Transformers not requested. To install transformer deps pass -InstallTransformer."
    }
} else {
    Write-Host "Skipping automatic dependency installation. You can install baseline deps with -InstallDeps, and heavy DL deps with -InstallDL / -InstallTransformer." 
}

# 3) Ensure .env exists
if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env -Force
    Write-Host ".env file created from .env.example. Edit .env to set MONGO_URI and other values before proceeding if needed."
} else {
    Write-Host ".env already exists."
}

# 4) Check MongoDB connectivity (best-effort)
Write-Host "Checking MongoDB connection (uses MONGO_URI from .env via app.config)."
try {
    python - <<'PY'
from app.config import get_config
from app.database.connection import get_client
cfg = get_config()
try:
    c = get_client()
    c.admin.command('ping')
    print('MongoDB reachable')
except Exception as e:
    print('MongoDB check failed:', e)
    raise SystemExit(2)
PY
} catch {
    Write-Host "MongoDB is not reachable from this script. Start MongoDB (e.g. 'docker run -d -p 27017:27017 --name mongo mongo:7') and set MONGO_URI in .env."
    throw
}

# 5) Ensure DB indexes
Write-Host "Ensuring required MongoDB indexes..."
python - <<'PY'
from app.database.connection import ensure_indexes
ensure_indexes()
print('Indexes ensured')
PY

# 6) Run scraper
Write-Host "Running scraper (limit=$ScrapeLimit). This will contact NHS Inform; obey rate limits in .env."
python scripts/run_scraper.py --limit $ScrapeLimit

# 7) Prepare dataset / preprocessing
Write-Host "Preparing dataset and saving preprocessing pipeline..."
python scripts/run_preprocessing.py

# 8) Train baseline model (TF-IDF + Logistic Regression)
Write-Host "Training baseline TF-IDF + Logistic Regression model..."
python scripts/train_baseline.py

# 9) Optionally train RNN/LSTM (requires TensorFlow)
if ($TrainDL) {
    Write-Host "Checking for TensorFlow..."
    try {
        python - <<'PY'
import importlib
if importlib.util.find_spec('tensorflow') is None:
    raise SystemExit(2)
print('tensorflow available')
PY
        Write-Host "Training RNN..."
        python scripts/train_rnn.py
        Write-Host "Training LSTM..."
        python scripts/train_lstm.py
    } catch {
        Write-Host "TensorFlow not available. Install TensorFlow to train RNN/LSTM. Skipping DL training."
    }
} else {
    Write-Host "Skipping RNN/LSTM training (TrainDL flag not set)."
}

# 10) Optionally train transformer (requires torch, transformers and HF access)
if ($TrainTransformer) {
    Write-Host "Checking for PyTorch and transformers..."
    try {
        python - <<'PY'
import importlib
if importlib.util.find_spec('torch') is None or importlib.util.find_spec('transformers') is None:
    raise SystemExit(2)
print('torch and transformers available')
PY
        Write-Host "Fine-tuning transformer (may require internet access and GPU)."
        python scripts/train_transformer.py
    } catch {
        Write-Host "PyTorch or transformers not available. Install them to train transformer. Skipping transformer training."
    }
} else {
    Write-Host "Skipping transformer training (TrainTransformer flag not set)."
}

# 11) Evaluate all models (runs training+evaluation pipeline if needed)
Write-Host "Running model evaluation / comparison (evaluate_models.py)..."
python scripts/evaluate_models.py

# 12) List saved models (metadata)
Write-Host "Listing saved model metadata from MongoDB (models collection):"
python - <<'PY'
from app.database.schemas import ModelsRepository
import json
for m in ModelsRepository().list_all():
    m['_id'] = str(m.get('_id'))
    print(json.dumps({'name': m.get('name'), 'type': m.get('type'), 'metrics': m.get('metrics')}, indent=2))
PY

# 13) Optionally start Flask server
if ($StartFlask) {
    Write-Host "Starting Flask API (run.py). Use Ctrl+C to stop."
    python run.py
} else {
    Write-Host "Done. Flask not started (StartFlask flag not set)."
}

Pop-Location
Write-Host "Full pipeline script completed. Review output above for errors and next actions."