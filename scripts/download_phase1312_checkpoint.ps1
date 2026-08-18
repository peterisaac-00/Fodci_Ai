$ErrorActionPreference = "Stop"

$repository = "peterisaac-00/Fodci_Ai"
$tag = "v13.12-stable"
$asset = "fodci-testing-qa-v1.pt"
$expectedSha256 = "3af5d2b6009f5a0fd0ff98644d9666bd0c30f0dfe8994f91524ae6df11433bfa"
$checkpointDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) "artifacts\checkpoints"
$destination = Join-Path $checkpointDirectory $asset
$temporary = "$destination.download"
$url = "https://github.com/$repository/releases/download/$tag/$asset"

New-Item -ItemType Directory -Force -Path $checkpointDirectory | Out-Null
Write-Host "Downloading $asset from the GitHub Release..."
Invoke-WebRequest -Uri $url -OutFile $temporary -UseBasicParsing

$actualSha256 = (Get-FileHash -Path $temporary -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    Remove-Item -Force $temporary
    throw "Checkpoint SHA-256 mismatch. Expected $expectedSha256 but received $actualSha256."
}

Move-Item -Force $temporary $destination
Write-Host "Checkpoint downloaded and verified: $destination"
Write-Host "SHA-256: $actualSha256"
