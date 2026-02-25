# Miniconda 自动安装脚本
# 此脚本会下载并安装 Miniconda，然后创建 dsp 环境

Write-Host "=== Miniconda 安装脚本 ===" -ForegroundColor Cyan
Write-Host ""

# 设置下载路径
$minicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$installerPath = "$env:TEMP\Miniconda3-latest-Windows-x86_64.exe"
$installPath = "$env:USERPROFILE\miniconda3"

# 检查是否已安装
if (Test-Path "$installPath\Scripts\conda.exe") {
    Write-Host "✓ Miniconda 已安装在: $installPath" -ForegroundColor Green
    $conda = "$installPath\Scripts\conda.exe"
} else {
    Write-Host "正在下载 Miniconda..." -ForegroundColor Yellow
    
    try {
        # 下载 Miniconda
        Invoke-WebRequest -Uri $minicondaUrl -OutFile $installerPath -UseBasicParsing
        Write-Host "✓ 下载完成" -ForegroundColor Green
        
        Write-Host "正在安装 Miniconda（这可能需要几分钟）..." -ForegroundColor Yellow
        
        # 静默安装
        Start-Process -FilePath $installerPath -ArgumentList "/InstallationType=JustMe", "/RegisterPython=1", "/S", "/D=$installPath" -Wait
        
        Write-Host "✓ Miniconda 安装完成" -ForegroundColor Green
        
        # 清理安装文件
        Remove-Item $installerPath -ErrorAction SilentlyContinue
        
        $conda = "$installPath\Scripts\conda.exe"
    } catch {
        Write-Host "✗ 安装失败: $_" -ForegroundColor Red
        exit 1
    }
}

# 添加到 PATH（当前会话）
$env:Path = "$installPath;$installPath\Scripts;$installPath\Library\bin;" + $env:Path

Write-Host ""
Write-Host "=== 创建 dsp 环境 ===" -ForegroundColor Cyan

# 初始化 conda
& $conda init powershell 2>&1 | Out-Null

# 创建 dsp 环境
Write-Host "正在创建 Python 3.11 环境..." -ForegroundColor Yellow
& $conda create -n dsp python=3.11 -y

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ dsp 环境创建成功" -ForegroundColor Green
} else {
    Write-Host "✗ 环境创建失败" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== 安装 Python 包 ===" -ForegroundColor Cyan

# 激活环境并安装包
$pythonPath = "$installPath\envs\dsp\python.exe"
$pipPath = "$installPath\envs\dsp\Scripts\pip.exe"

if (Test-Path $pipPath) {
    Write-Host "正在安装: numpy scipy soundfile librosa pandas matplotlib..." -ForegroundColor Yellow
    
    & $pipPath install numpy scipy soundfile librosa pandas matplotlib
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ 所有包安装成功" -ForegroundColor Green
    } else {
        Write-Host "✗ 包安装失败" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✗ 找不到 pip: $pipPath" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== 验证安装 ===" -ForegroundColor Cyan

# 验证包
& $pythonPath -c 'import numpy, scipy, soundfile, librosa, pandas, matplotlib; print("所有包导入成功！")'

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ 验证通过" -ForegroundColor Green
} else {
    Write-Host "✗ 验证失败" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== 安装完成 ===" -ForegroundColor Green
Write-Host ""
Write-Host "Python 解释器路径: $pythonPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步:" -ForegroundColor Yellow
Write-Host "1. 重启终端或运行: conda init powershell" 
Write-Host "2. 激活环境: conda activate dsp"
Write-Host "3. 运行 setup_ffmpeg.ps1 安装 FFmpeg"
Write-Host ""
