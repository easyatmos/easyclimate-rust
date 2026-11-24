# build_manywindows_wheel.ps1

$pythonVersions = @("3.10", "3.11", "3.12", "3.13", "3.14", "3.15")

# 清理函数
function Clean-BuildArtifacts {
    Write-Host "🧹 Cleaning old build artifacts..." -ForegroundColor Yellow
    
    # 清理 python 包目录下的 .pyd, .so, .dll 文件
    Get-ChildItem -Path "." -Recurse -Include "*.pyd", "*.so", "*.dll" |
        Where-Object { $_.FullName -notmatch "\\target\\" } |
        Remove-Item -Force -ErrorAction SilentlyContinue

    if (Test-Path "python\__pycache__") {
        Remove-Item -Path "python\__pycache__" -Recurse -Force
    }
    
    # 清理 target/maturin 目录
    if (Test-Path "target/maturin") {
        Remove-Item -Path "target/maturin" -Recurse -Force
    }

    # 清理 .venv 目录
    if (Test-Path ".venv") {
        Remove-Item -Path ".venv" -Recurse -Force
    }
    
    Write-Host "✅ Cleanup completed" -ForegroundColor Green
}

# 在开始构建前清理一次
Clean-BuildArtifacts

foreach ($version in $pythonVersions) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Building wheel for Python $version..." -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan

    Clean-BuildArtifacts
    
    # 使用 uv run 构建
    uv run --python $version --with maturin maturin build --release -o dist/
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Completed wheel for Python $version" -ForegroundColor Green
        
        # 每次构建后清理，避免影响下一个版本
        Clean-BuildArtifacts
    } else {
        Write-Host "❌ Build failed for Python $version" -ForegroundColor Red
        Clean-BuildArtifacts
        exit 1
    }
}

Write-Host "`n🎉 All wheels built successfully!" -ForegroundColor Green
Write-Host "`n📦 Generated wheels:" -ForegroundColor Cyan
Get-ChildItem dist/*.whl | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Yellow }