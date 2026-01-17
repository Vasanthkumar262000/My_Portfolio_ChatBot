# PowerShell script to start backend (bypasses execution policy)
Write-Host "Starting Backend Server..." -ForegroundColor Green
Write-Host ""

# Use direct python path to avoid activation issues
$pythonPath = ".\venv\Scripts\python.exe"
$uvicornPath = ".\venv\Scripts\uvicorn.exe"

if (Test-Path $pythonPath) {
    Write-Host "Starting FastAPI server on port 8000..." -ForegroundColor Yellow
    Write-Host ""
    & $pythonPath -m uvicorn main:app --host 0.0.0.0 --port 8000
} else {
    Write-Host "Error: Python not found at $pythonPath" -ForegroundColor Red
    Write-Host "Make sure virtual environment is set up correctly." -ForegroundColor Red
    pause
}






