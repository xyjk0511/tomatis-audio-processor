# 快速配置指南
# 运行此脚本来一键安装所有依赖

Write-Host "=== DSP 开发环境快速配置 ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "此脚本将安装:" -ForegroundColor Yellow
Write-Host "  1. Miniconda (Python 环境管理器)"
Write-Host "  2. Python 3.11 + 必需的包"
Write-Host "  3. FFmpeg (音频处理工具)"
Write-Host ""

$confirm = Read-Host "是否继续? (Y/N)"
if ($confirm -ne 'Y' -and $confirm -ne 'y') {
    Write-Host "已取消" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "步骤 1/2: 安装 Miniconda 和 Python 环境" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan

.\setup_miniconda.ps1

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ Miniconda 安装失败，请检查错误信息" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "步骤 2/2: 安装 FFmpeg" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan

.\setup_ffmpeg.ps1

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ FFmpeg 安装失败，请检查错误信息" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=" * 50 -ForegroundColor Green
Write-Host "🎉 所有组件安装完成！" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green
Write-Host ""

Write-Host "下一步:" -ForegroundColor Yellow
Write-Host "1. 关闭并重新打开终端"
Write-Host "2. 运行: conda activate dsp"
Write-Host "3. 运行: python test_environment.py"
Write-Host ""

Write-Host "Python 解释器路径:" -ForegroundColor Cyan
Write-Host "$env:USERPROFILE\miniconda3\envs\dsp\python.exe"
Write-Host ""
