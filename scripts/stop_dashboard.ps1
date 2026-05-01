param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PidFile = Join-Path $Root "data\dashboard.pid"
$ProcessIds = @()

function Add-ProcessId {
    param([int]$ProcessId)

    if ($ProcessId -gt 0 -and -not ($script:ProcessIds -contains $ProcessId)) {
        $script:ProcessIds += $ProcessId
    }
}

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

function Stop-ProcessTree {
    param([int]$ProcessId)

    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue)
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId ([int]$child.ProcessId)
    }

    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped PID $ProcessId"
    }
}

if (Test-Path $PidFile) {
    $rawPid = (Get-Content $PidFile -Raw).Trim()
    if ($rawPid -match "^\d+$") {
        Add-ProcessId -ProcessId ([int]$rawPid)
    }
}

$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
foreach ($listener in $listeners) {
    if ($listener.OwningProcess) {
        Add-ProcessId -ProcessId ([int]$listener.OwningProcess)
    }
}

if ($ProcessIds.Count -eq 0) {
    Write-Host "No dashboard process found on port $Port."
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    exit 0
}

$stopped = 0
foreach ($processId in $ProcessIds) {
    if (Test-DashboardProcess -ProcessId $processId) {
        Stop-ProcessTree -ProcessId $processId
        $stopped += 1
    } else {
        Write-Host "Skipping PID $processId because it does not look like this dashboard."
    }
}

Remove-Item $PidFile -Force -ErrorAction SilentlyContinue

if ($stopped -eq 0) {
    Write-Host "No matching dashboard process was stopped."
} else {
    Write-Host "Dashboard stopped."
}
