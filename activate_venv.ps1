# PowerShell script to activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1
Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run:" -ForegroundColor Yellow
Write-Host "  python main.py" -ForegroundColor Cyan
Write-Host "  uvicorn main:app --reload" -ForegroundColor Cyan







