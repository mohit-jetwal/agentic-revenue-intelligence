<#
.SYNOPSIS
    Developer task runner.

.EXAMPLE
    .\tasks.ps1 setup
    .\tasks.ps1 check
    .\tasks.ps1 api
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'lint', 'format', 'typecheck', 'test', 'check', 'security', 'api', 'ui', 'health', 'config', 'clean')]
    [string]$Task = 'check',

    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

switch ($Task) {
    'setup' {
        Invoke-Step 'uv sync' { uv sync --all-extras }
        if (-not (Test-Path '.env')) {
            Copy-Item '.env.example' '.env'
            Write-Host 'Created .env from .env.example' -ForegroundColor Green
        }
    }
    'lint'      { Invoke-Step 'ruff check'  { uv run ruff check . } }
    'format'    { Invoke-Step 'ruff format' { uv run ruff format . } }
    'typecheck' { Invoke-Step 'mypy'        { uv run mypy } }
    'test'      { Invoke-Step 'pytest'      { uv run pytest } }
    'security'  { Invoke-Step 'bandit'      { uv run bandit -c pyproject.toml -r app ml data -q } }
    'check' {
        Invoke-Step 'ruff check' { uv run ruff check . }
        Invoke-Step 'mypy'       { uv run mypy }
        Invoke-Step 'pytest'     { uv run pytest }
        Write-Host "`nAll checks passed." -ForegroundColor Green
    }
    'api'    { uv run uvicorn app.main:app --reload --port $Port }
    'ui'     { uv run streamlit run app/ui/streamlit_app.py }
    'health' { uv run ari health }
    'config' { uv run ari config }
    'clean' {
        Get-ChildItem -Recurse -Directory -Include '__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache' -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host 'Removed tool caches.' -ForegroundColor Green
    }
}
