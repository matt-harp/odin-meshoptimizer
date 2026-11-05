param([string]$Version = "")

$ErrorActionPreference = "Stop"
$rootDir = Get-Location

$Version = (Get-Content "meshoptimizer_tag" -Raw).Trim()

New-Item -ItemType Directory -Force -Path build | Out-Null
Set-Location build

if (-not (Test-Path "meshoptimizer")) {
	git clone -b $Version https://github.com/zeux/meshoptimizer.git -c advice.detachedHead=false
}

Write-Host "Compiling meshoptimizer ($Version)..."
Set-Location meshoptimizer

cmake -B cmake-build -S .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build cmake-build --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (Test-Path "cmake-build\Release\meshoptimizer.lib") {
	Move-Item -Path "cmake-build\Release\meshoptimizer.lib" -Destination "$rootDir\meshoptimizer.lib" -Force
}

Set-Location $rootDir
