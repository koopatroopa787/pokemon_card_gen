# Pokemon Card Generator - Quick Setup (PowerShell/Windows)
# Run with: .\quick_setup.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Pokemon Card Generator - Quick Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Check for Kaggle credentials
$kaggleDir = "$env:USERPROFILE\.kaggle"
$kaggleJson = "$kaggleDir\kaggle.json"

if (-not (Test-Path $kaggleJson)) {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Write-Host "KAGGLE API SETUP REQUIRED" -ForegroundColor Red
    Write-Host "======================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "To download the Pokemon cards dataset, you need Kaggle API credentials." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Steps:" -ForegroundColor Cyan
    Write-Host "  1. Go to: https://www.kaggle.com/settings" -ForegroundColor White
    Write-Host "  2. Scroll to 'API' section" -ForegroundColor White
    Write-Host "  3. Click 'Create New Token'" -ForegroundColor White
    Write-Host "  4. Download kaggle.json" -ForegroundColor White
    Write-Host ""
    Write-Host "Then run:" -ForegroundColor Cyan
    Write-Host "  New-Item -ItemType Directory -Force -Path $kaggleDir" -ForegroundColor White
    Write-Host "  Move-Item $env:USERPROFILE\Downloads\kaggle.json $kaggleDir\" -ForegroundColor White
    Write-Host ""
    Write-Host "After setting up kaggle.json, run this script again." -ForegroundColor Yellow
    Write-Host "======================================" -ForegroundColor Red
    exit 1
}

# Download and prepare dataset
Write-Host ""
Write-Host "Downloading Pokemon cards dataset from Kaggle..." -ForegroundColor Yellow
python setup_kaggle_dataset.py

if ($LASTEXITCODE -eq 0) {
    # Done
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "Setup Complete!" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "The Pokemon card dataset is ready for training." -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Train the model:" -ForegroundColor White
    Write-Host "     python train_pokemon.py --data_dir data/pokemon_cards/training/train" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  2. After training, start the API:" -ForegroundColor White
    Write-Host "     python run_server.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  3. Open the UI:" -ForegroundColor White
    Write-Host "     http://localhost:8000/ui" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "✗ Setup failed. Please check the error messages above." -ForegroundColor Red
    exit 1
}
