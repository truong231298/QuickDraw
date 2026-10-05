$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bundlePath = Join-Path $projectRoot "quickdraw-colab-source.zip"
$include = @("pyproject.toml", "requirements.txt", "config", "scripts", "src")
$paths = $include | ForEach-Object { Join-Path $projectRoot $_ }
Compress-Archive -Path $paths -DestinationPath $bundlePath -Force
Write-Host "Created $bundlePath"
Write-Host "Upload this ZIP in the first cell of notebooks/QuickDraw_Colab.ipynb"
