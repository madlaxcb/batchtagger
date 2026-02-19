$ErrorActionPreference = "Stop"

Write-Host "Checking Python..."

function Get-PythonCommand {
    # 1. Check for local python_env
    if (Test-Path ".\python_env\python.exe") {
        Write-Host "Found local python_env."
        return ".\python_env\python.exe"
    }

    # 2. Check for system python
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        return "python"
    }
    if (Get-Command "py" -ErrorAction SilentlyContinue) {
        return "py"
    }
    return $null
}

$pythonCmd = Get-PythonCommand

if (-not $pythonCmd) {
    Write-Warning "Python not found in PATH or python_env. Please ensure Python 3.10+ is installed."
    exit 1
}

# Verify version
try {
    # Use Invoke-Expression or & operator carefully
    if ($pythonCmd -match "python_env") {
        $verOutput = & $pythonCmd --version 2>&1
    } else {
        $verOutput = & $pythonCmd --version 2>&1
    }
    Write-Host "Using Python: $verOutput"
    Write-Host "Path: $pythonCmd"
} catch {
    Write-Error "Failed to execute python command: $pythonCmd"
    exit 1
}

Write-Host "Installing dependencies..."
try {
    & $pythonCmd -m pip install --upgrade pip
    & $pythonCmd -m pip install -r requirements.txt
} catch {
    Write-Error "Failed to install dependencies."
    exit 1
}

Write-Host "Cleaning previous builds..."
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

Write-Host "Building with PyInstaller..."
try {
    & $pythonCmd -m PyInstaller BatchTagger.spec
} catch {
    Write-Error "PyInstaller build failed."
    exit 1
}

Write-Host "Verifying build..."
if (-not (Test-Path "dist\BatchTagger\BatchTagger.exe")) {
    Write-Error "Build failed: BatchTagger.exe not found."
    exit 1
}

Write-Host "Running Smoke Test..."
try {
    & $pythonCmd smoke_test.py
} catch {
    Write-Error "Smoke test execution failed."
    exit 1
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build and Verification Successful!" -ForegroundColor Green
    Write-Host "Executable is located at: dist\BatchTagger\BatchTagger.exe"
} else {
    Write-Error "Smoke Test Failed!"
    exit 1
}

