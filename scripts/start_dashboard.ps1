param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$DataDir = Join-Path $Root "data"
$LogDir = Join-Path $Root "logs"
$PidFile = Join-Path $DataDir "dashboard.pid"
$OutLog = Join-Path $LogDir "dashboard.out.log"
$ErrLog = Join-Path $LogDir "dashboard.err.log"

if (-not (Test-Path $Python)) {
    Write-Error "Virtual environment not found at '$Root\.venv'. Run: python -m venv .venv && .venv\Scripts\pip install -r requirements.txt"
    exit 1
}

New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

function Get-CommandLine {
    param([int]$ProcessId)

    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
    if ($processInfo) {
        return [string]$processInfo.CommandLine
    }
    return ""
}

function Test-DashboardProcess {
    param([int]$ProcessId)

    $commandLine = Get-CommandLine -ProcessId $ProcessId
    return ($commandLine -match "streamlit" -and $commandLine -match "src[\\/]+dashboard\.py")
}

$RunningProcessId = $null

if (Test-Path $PidFile) {
    $rawPid = (Get-Content $PidFile -Raw).Trim()
    if ($rawPid -match "^\d+$") {
        $existingProcessId = [int]$rawPid
        if (Test-DashboardProcess -ProcessId $existingProcessId) {
            $RunningProcessId = $existingProcessId
        } else {
            Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        }
    }
}

if (-not $RunningProcessId) {
    $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -gt 0) {
        $ownerProcessId = [int]$listeners[0].OwningProcess
        if (Test-DashboardProcess -ProcessId $ownerProcessId) {
            $RunningProcessId = $ownerProcessId
            Set-Content -Path $PidFile -Value $ownerProcessId
        } else {
            Write-Error "Port $Port is already in use by PID $ownerProcessId. Stop that process or change the port in scripts\start_dashboard.ps1."
            exit 1
        }
    }
}

if ($RunningProcessId) {
    Write-Host "Dashboard already running. PID: $RunningProcessId"
    Write-Host "URL: http://localhost:$Port"
    exit 0
}

$arguments = @(
    "-m", "streamlit",
    "run", "src/dashboard.py",
    "--server.headless", "true",
    "--server.port", [string]$Port
)

$process = Start-Process `
    -FilePath $Python `
    -ArgumentList $arguments `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru

Set-Content -Path $PidFile -Value $process.Id

Write-Host "Started dashboard in background. PID: $($process.Id)"
Write-Host "URL: http://localhost:$Port"
Write-Host "Logs: $OutLog and $ErrLog"
