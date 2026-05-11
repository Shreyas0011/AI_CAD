# This script launches the AI CAD Backend using the Anaconda Python environment
$PYTHON_PATH = "C:\Users\Shreyas\anaconda3\python.exe"
if (Test-Path $PYTHON_PATH) {
    Write-Host "Starting AI CAD Backend using Anaconda..." -ForegroundColor Cyan
    & $PYTHON_PATH backend/main.py
} else {
    Write-Host "Error: Anaconda Python not found at $PYTHON_PATH" -ForegroundColor Red
    Write-Host "Please ensure Anaconda is installed in your home directory." -ForegroundColor Yellow
}
