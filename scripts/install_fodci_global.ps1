[CmdletBinding()]
param(
    [switch]$SkipCheckpointDownload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvRoot = Join-Path $HOME ".fodci"
$venvPath = Join-Path $venvRoot "venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvScripts = Join-Path $venvPath "Scripts"

$py = Get-Command "py.exe" -ErrorAction SilentlyContinue
if ($null -eq $py) {
    throw "Python Launcher (py.exe) was not found. Install Python 3.12 and enable the Python Launcher, then run this script again."
}

$pythonVersion = (& $py.Source -3.12 --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 was not found. Install Python 3.12, then verify with: py -0p"
}
Write-Host "Using $pythonVersion"

New-Item -ItemType Directory -Force -Path $venvRoot | Out-Null
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "Creating dedicated Fodci environment: $venvPath"
    & $py.Source -3.12 -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Fodci virtual environment."
    }
}

Write-Host "Updating pip and installing Fodci with the CPU model extra..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed."
}
$installSpec = "$repoRoot[model]"
& $venvPython -m pip install --editable $installSpec
if ($LASTEXITCODE -ne 0) {
    throw "Fodci installation failed."
}

$checkpoint = Join-Path $repoRoot "artifacts\checkpoints\fodci-testing-qa-v1.pt"
if (-not $SkipCheckpointDownload -and -not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    $downloadScript = Join-Path $repoRoot "scripts\download_phase1312_checkpoint.ps1"
    Write-Host "Downloading and verifying the stable Fodci checkpoint..."
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $downloadScript
    if ($LASTEXITCODE -ne 0) {
        throw "Stable checkpoint download failed."
    }
}
elseif (-not $SkipCheckpointDownload) {
    Write-Host "Stable checkpoint already exists: $checkpoint"
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$pathEntries = @()
if (-not [string]::IsNullOrWhiteSpace($userPath)) {
    $pathEntries = @($userPath -split [IO.Path]::PathSeparator | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}
$normalizedScripts = [IO.Path]::GetFullPath($venvScripts).TrimEnd('\').ToLowerInvariant()
$alreadyPresent = $pathEntries | Where-Object {
    try {
        ([IO.Path]::GetFullPath($_).TrimEnd('\').ToLowerInvariant()) -eq $normalizedScripts
    }
    catch {
        $false
    }
}
if ($null -eq $alreadyPresent) {
    $pathEntries += $venvScripts
    [Environment]::SetEnvironmentVariable("Path", ($pathEntries -join [IO.Path]::PathSeparator), "User")
    Write-Host "Added $venvScripts to the user PATH."
}

# Make the command available immediately in this PowerShell session as well.
if (-not (($env:Path -split [IO.Path]::PathSeparator) -contains $venvScripts)) {
    $env:Path = "$venvScripts$([IO.Path]::PathSeparator)$env:Path"
}

$command = Get-Command "fodci" -ErrorAction SilentlyContinue
if ($null -eq $command) {
    throw "The fodci command was not found after installation. Open a new terminal and run: Get-Command fodci"
}
& $venvPython -c "import importlib.util; assert importlib.util.find_spec('backend_ai') is not None; print('backend_ai import: OK')"
if ($LASTEXITCODE -ne 0) {
    throw "The backend_ai package import verification failed."
}

Write-Host ""
Write-Host "Fodci installation completed successfully."
Write-Host "Command: $($command.Source)"
Write-Host "Stable checkpoint: $checkpoint"
Write-Host "Open a new terminal, then run: fodci"
