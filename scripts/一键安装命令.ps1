# 一键安装命令
# 复制下面的命令到 PowerShell 中执行

# ============================================
# 步骤 1: 下载并安装 Miniconda
# ============================================

# 下载 Miniconda 安装程序
$url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$output = "$env:TEMP\Miniconda3-Installer.exe"
Write-Host "正在下载 Miniconda..." -ForegroundColor Yellow
Invoke-WebRequest -Uri $url -OutFile $output
Write-Host "下载完成，正在启动安装程序..." -ForegroundColor Green
Start-Process $output -Wait

# 安装完成后，关闭并重新打开 PowerShell，然后继续执行下面的命令

# ============================================
# 步骤 2: 创建 Python 环境并安装包
# ============================================

# 创建 dsp 环境
conda create -n dsp python=3.11 -y

# 激活环境
conda activate dsp

# 安装所有必需的包
pip install numpy scipy soundfile librosa pandas matplotlib

# 验证安装
python -c "import numpy, scipy, soundfile, librosa, pandas, matplotlib; print('所有包安装成功!')"

# ============================================
# 步骤 3: 下载并安装 FFmpeg
# ============================================

# 下载 FFmpeg
$url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$output = "$env:TEMP\ffmpeg.zip"
Write-Host "正在下载 FFmpeg..." -ForegroundColor Yellow
Invoke-WebRequest -Uri $url -OutFile $output

# 解压
Write-Host "正在解压..." -ForegroundColor Yellow
Expand-Archive -Path $output -DestinationPath "$env:TEMP\ffmpeg_temp" -Force

# 移动到目标位置
$extracted = Get-ChildItem "$env:TEMP\ffmpeg_temp" -Directory | Select-Object -First 1
New-Item -ItemType Directory -Path "C:\ffmpeg" -Force | Out-Null
Copy-Item -Path "$($extracted.FullName)\*" -Destination "C:\ffmpeg" -Recurse -Force

# 添加到 PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*C:\ffmpeg\bin*") {
    $newPath = "$currentPath;C:\ffmpeg\bin"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "FFmpeg 已添加到 PATH" -ForegroundColor Green
}

# 清理临时文件
Remove-Item "$env:TEMP\ffmpeg.zip" -Force
Remove-Item "$env:TEMP\ffmpeg_temp" -Recurse -Force

Write-Host "FFmpeg 安装完成！" -ForegroundColor Green

# 重启 PowerShell 后验证
# ffmpeg -version

# ============================================
# 步骤 4: 验证完整环境
# ============================================

# 在新的 PowerShell 中运行:
# cd F:\TOMATIS
# conda activate dsp
# python test_environment.py
