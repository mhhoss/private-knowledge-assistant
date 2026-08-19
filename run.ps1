# Windows launcher: starts the API in the background, waits for it to be ready, then
# runs Streamlit in the foreground. Thin deployment layer only — no application logic
# lives here (see STRATEGY.md's "platform-specific details belong only in thin
# deployment layers or scripts, not in core business logic").
#
# Usage: right-click > "Run with PowerShell", or from a PowerShell prompt:
#   .\run.ps1

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
    Write-Error "No .env file found. Copy .env.example to .env and fill in your provider credentials first."
    exit 1
}

Write-Host "Starting the API (http://127.0.0.1:8000) ..."
$api = Start-Process -FilePath "uv" -ArgumentList "run", "uvicorn", "app.main:app" -PassThru -NoNewWindow

$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/documents" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        if ($api.HasExited) {
            Write-Error "The API process exited unexpectedly. Check the output above for the reason (e.g. missing credentials in .env)."
            exit 1
        }
    }
}

if (-not $ready) {
    Write-Error "The API did not become ready within 60 seconds. Check the output above."
    Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "API is ready. Starting the UI (http://127.0.0.1:8501) ..."
try {
    uv run streamlit run streamlit_app.py
} finally {
    Write-Host "Stopping the API ..."
    Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
}
