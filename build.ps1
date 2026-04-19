$ErrorActionPreference = "Stop"
$rootDir = Get-Location

New-Item -ItemType Directory -Force -Path build | Out-Null
Set-Location build

if (-not (Test-Path "meshoptimizer")) {
	git clone https://github.com/zeux/meshoptimizer.git --revision b22872835dbabc56a6e4a366ea9917f62b7daf1a -c advice.detachedHead=false
}

Write-Host "Compiling meshoptimizer..."
Set-Location meshoptimizer

cmake -B cmake-build -S .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build cmake-build --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (Test-Path "cmake-build\Release\meshoptimizer.lib") {
    Move-Item -Path "cmake-build\Release\meshoptimizer.lib" -Destination "$rootDir\meshoptimizer.lib" -Force
}

Set-Location $rootDir
